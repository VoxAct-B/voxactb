import importlib
from os.path import exists, dirname, abspath, join
from typing import Type, List
import pickle

from pyrep import PyRep
from pyrep.objects import VisionSensor
from pyrep.const import RenderMode
from pyrep.robots.arms.panda import Panda

from rlbench import utils
from rlbench.action_modes.action_mode import ActionMode
from rlbench.backend.const import *
from rlbench.backend.robot import Robot
from rlbench.backend.scene_two_robots import Scene2Robots
from rlbench.backend.task_two_robots import Task2Robots
from rlbench.backend.utils import get_quaternion_from_euler
from rlbench.const import SUPPORTED_ROBOTS
from rlbench.demo import Demo
from rlbench.observation_config_two_robots import ObservationConfig2Robots
from rlbench.sim2real.domain_randomization import RandomizeEvery, \
    VisualRandomizationConfig, DynamicsRandomizationConfig
from rlbench.sim2real.domain_randomization_scene import DomainRandomizationScene
from rlbench.task_environment_two_robots import TaskEnvironment2Robots

DIR_PATH = dirname(abspath(__file__))


class Environment2Robots(object):
    """Each environment has a scene."""

    def __init__(self,
                 action_mode: ActionMode,
                 dataset_root: str = '',
                 obs_config: ObservationConfig2Robots = ObservationConfig2Robots(),
                 headless: bool = False,
                 static_positions: bool = False,
                 robot_setup: str = 'panda',
                 randomize_every: RandomizeEvery = None,
                 frequency: int = 1,
                 visual_randomization_config: VisualRandomizationConfig = None,
                 dynamics_randomization_config: DynamicsRandomizationConfig = None,
                 attach_grasped_objects: bool = True,
                 mode: str = 'default',
                 task_name = None,
                 ):

        self._dataset_root = dataset_root
        self._action_mode = action_mode
        self._obs_config = obs_config
        self._headless = headless
        self._static_positions = static_positions
        self._robot_setup = robot_setup.lower()
        self._mode = mode
        self._task_name = task_name

        self._randomize_every = randomize_every
        self._frequency = frequency
        self._visual_randomization_config = visual_randomization_config
        self._dynamics_randomization_config = dynamics_randomization_config
        self._attach_grasped_objects = attach_grasped_objects

        if robot_setup not in SUPPORTED_ROBOTS.keys():
            raise ValueError('robot_configuration must be one of %s' %
                             str(SUPPORTED_ROBOTS.keys()))

        if (randomize_every is not None and
                visual_randomization_config is None and
                dynamics_randomization_config is None):
            raise ValueError(
                'If domain randomization is enabled, must supply either '
                'visual_randomization_config or dynamics_randomization_config')

        if task_name == 'OpenJar':
            self._TTT_FILE = 'task_design_open_jar.ttt'
        elif task_name == 'OpenDrawer':
            self._TTT_FILE = 'task_design_open_drawer.ttt'
        elif task_name == 'SweepToDustpan':
            self._TTT_FILE = 'task_design_sweep_to_dustpan.ttt'
        elif task_name == 'PutItemInDrawer':
            self._TTT_FILE = 'task_design_put_item_in_drawer.ttt'
        elif task_name == 'HandOverItem':
            self._TTT_FILE = 'task_design_hand_over_item.ttt'
        else:
            raise NotImplementedError

        self._check_dataset_structure()
        self._pyrep = None
        self._robot_right = None
        self._robot_left = None
        self._scene = None
        self._prev_task = None

    def _check_dataset_structure(self):
        if len(self._dataset_root) > 0 and not exists(self._dataset_root):
            raise RuntimeError(
                'Data set root does not exists: %s' % self._dataset_root)

    def _string_to_task(self, task_name: str):
        task_name = task_name.replace('.py', '')
        try:
            class_name = ''.join(
                [w[0].upper() + w[1:] for w in task_name.split('_')])
            mod = importlib.import_module("rlbench.tasks.%s" % task_name)
        except Exception as e:
            raise RuntimeError(
                'Tried to interpret %s as a task, but failed. Only valid tasks '
                'should belong in the tasks/ folder' % task_name) from e
        return getattr(mod, class_name)

    def launch(self):
        if self._pyrep is not None:
            raise RuntimeError('Already called launch!')
        print('Launching ', self._TTT_FILE)
        self._pyrep = PyRep()
        self._pyrep.launch(join(DIR_PATH, self._TTT_FILE), headless=self._headless)

        arm_class, gripper_class, _ = SUPPORTED_ROBOTS[
            self._robot_setup]

        # We assume the panda is already loaded in the scene.
        if self._robot_setup != 'panda':
            raise NotImplementedError
            # Remove the panda from the scene
            # panda_arm = Panda()
            # panda_pos = panda_arm.get_position()
            # panda_arm.remove()
            # arm_path = join(DIR_PATH, 'robot_ttms', self._robot_setup + '.ttm')
            # self._pyrep.import_model(arm_path)
            # arm, gripper = arm_class(), gripper_class()
            # arm.set_position(panda_pos)
        else:
            arm_right, gripper_right = arm_class(), gripper_class()
            arm_left, gripper_left = arm_class(1), gripper_class(1)

        self._robot_right = Robot(arm_right, gripper_right)
        self._robot_left = Robot(arm_left, gripper_left)
        if self._randomize_every is None:
            self._scene = Scene2Robots(
                self._pyrep, self._robot_right, self._robot_left, self._obs_config, self._robot_setup, self._mode)
        else:
            raise NotImplementedError
            # self._scene = DomainRandomizationScene(
            #     self._pyrep, self._robot, self._obs_config, self._robot_setup,
            #     self._randomize_every, self._frequency,
            #     self._visual_randomization_config,
            #     self._dynamics_randomization_config)

        self._action_mode.arm_action_mode.set_control_mode(self._robot_right)
        self._action_mode.arm_action_mode.set_control_mode(self._robot_left)

    def shutdown(self):
        print('Environment2Robots is shutting down')
        if self._pyrep is not None:
            self._pyrep.shutdown()
        self._pyrep = None


    def add_highres_front_cam_for_llm(self):
        """
        This camera is used for determining the acting arm in the scene.
        """
        self._highres_front_cam = VisionSensor.create([512, 512])
        self._highres_front_cam.set_explicit_handling(True)

        # manually determined camera position (directly looking at the workspace)
        quat = get_quaternion_from_euler(115, 0, 0)
        cam_pose = [0.275, 2.125, 1.98, *quat] # original implementation
        # cam_pose = [0.275, 1.375, 1.98, *quat] # closer to the table
        # self._highres_front_cam.set_pose(self._scene._cam_front.get_pose()) # front camera view
        self._highres_front_cam.set_pose(cam_pose) # direct view at the workspace
        self._highres_front_cam.set_render_mode(RenderMode.OPENGL)

    def get_task(self, task_class: Type[Task2Robots], dominant: str = 'right') -> TaskEnvironment2Robots:

        # If user hasn't called launch, implicitly call it.
        if self._pyrep is None:
            self.launch()

        self._scene.unload()
        task = task_class(self._pyrep, self._robot_right, self._robot_left)
        self._prev_task = task
        return TaskEnvironment2Robots(
            self._pyrep, self._robot_right, self._robot_left, self._scene, task,
            self._action_mode, self._dataset_root, self._obs_config,
            self._static_positions, self._attach_grasped_objects, self._mode, dominant)

    @property
    def action_shape(self):
        import sys
        import pdb

        class ForkedPdb(pdb.Pdb):
            """A Pdb subclass that may be used
            from a forked multiprocessing child

            """
            def interaction(self, *args, **kwargs):
                _stdin = sys.stdin
                try:
                    sys.stdin = open('/dev/stdin')
                    pdb.Pdb.interaction(self, *args, **kwargs)
                finally:
                    sys.stdin = _stdin
        ForkedPdb().set_trace()
        return self._action_mode.action_shape(self._scene),

    def get_demos(self, task_name: str, amount: int,
                  variation_number=0,
                  image_paths=False,
                  random_selection: bool = True,
                  from_episode_number: int = 0) -> List[Demo]:
        if self._dataset_root is None or len(self._dataset_root) == 0:
            raise RuntimeError(
                "Can't ask for a stored demo when no dataset root provided.")
        demos = utils.get_stored_demos(
            amount, image_paths, self._dataset_root, variation_number,
            task_name, self._obs_config, random_selection, from_episode_number)
        return demos

    def get_task_descriptions_with_episode(self, task_name: str,
                                           episode_number: int) -> List[str]:
        episode_description_pkl_file = join(self._dataset_root,
                                            f'{task_name}',
                                            VARIATIONS_ALL_FOLDER,
                                            EPISODES_FOLDER,
                                            EPISODE_FOLDER % episode_number,
                                            VARIATION_DESCRIPTIONS)
        with open(episode_description_pkl_file, 'rb') as f:
            episode_description = pickle.load(f)

        return episode_description

    def get_scene_data(self) -> dict:
        """Get the data of various scene/camera information.

        This temporarily starts the simulator in headless mode.

        :return: A dictionary containing scene data.
        """

        def _get_cam_info(cam: VisionSensor):
            if not cam.still_exists():
                return None
            intrinsics = cam.get_intrinsic_matrix()
            return dict(
                intrinsics=intrinsics,
                near_plane=cam.get_near_clipping_plane(),
                far_plane=cam.get_far_clipping_plane(),
                extrinsics=cam.get_matrix())

        headless = self._headless
        self._headless = True
        self.launch()
        d = dict(
            front_camera=_get_cam_info(self._scene._cam_front),
            wrist_camera=_get_cam_info(self._scene._cam_wrist),
            wrist2_camera=_get_cam_info(self._scene._cam_wrist2),
        )
        self.shutdown()
        self._headless = headless
        return d
