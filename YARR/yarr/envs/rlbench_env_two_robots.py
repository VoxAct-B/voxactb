from typing import Type, List

import numpy as np
try:
    from rlbench import ObservationConfig2Robots, Environment2Robots, CameraConfig
except (ModuleNotFoundError, ImportError) as e:
    print("You need to install RLBench: 'https://github.com/stepjam/RLBench'")
    raise e
from rlbench.action_modes.action_mode import ActionMode
from rlbench.backend.observation_two_robots import Observation2Robots
from rlbench.backend.task_two_robots import Task2Robots
from helpers import utils
from clip import tokenize

from yarr.envs.env import Env, MultiTaskEnv
from yarr.utils.observation_type import ObservationElement
from yarr.utils.transition import Transition
from yarr.utils.process_str import change_case

# VoxPoser imports
import openai
from voxposer.envs.rlbench_env import VoxPoserRLBench2Robots
from voxposer.visualizers import ValueMapVisualizer
from voxposer.arguments import get_config
from voxposer.interfaces import setup_LMP
from rlbench import tasks

ROBOT_STATE_KEYS = ['joint_velocities_right', 'joint_positions_right', 'joint_forces_right',
                    'joint_velocities_left', 'joint_positions_left', 'joint_forces_left',
                        'gripper_right_open', 'gripper_right_pose',
                        'gripper_right_joint_positions', 'gripper_right_touch_forces',
                        'gripper_left_open', 'gripper_left_pose',
                        'gripper_left_joint_positions', 'gripper_left_touch_forces',
                        'task_low_dim_state', 'misc']

def _extract_obs(obs: Observation2Robots, channels_last: bool, observation_config, train_cfg):
    obs_dict = vars(obs)
    obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
    robot_state_right = obs.get_low_dim_data(which_arm='right')
    robot_state_left = obs.get_low_dim_data(which_arm='left')
    # Remove all of the individual state elements
    obs_dict = {k: v for k, v in obs_dict.items()
                if k not in ROBOT_STATE_KEYS}
    if not channels_last:
        # Swap channels from last dim to 1st dim
        obs_dict = {k: np.transpose(
            v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
                    for k, v in obs_dict.items()}
    else:
        # Add extra dim to depth data
        obs_dict = {k: v if v.ndim == 3 else np.expand_dims(v, -1)
                    for k, v in obs_dict.items()}
    obs_dict['low_dim_state_right_arm'] = np.array(robot_state_right, dtype=np.float32)
    obs_dict['low_dim_state_left_arm'] = np.array(robot_state_left, dtype=np.float32)
    obs_dict['ignore_collisions'] = np.array([obs.ignore_collisions], dtype=np.float32)
    for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
        obs_dict[k] = v.astype(np.float32)

    for config, name in [
        (observation_config.front_camera, 'front'),
        (observation_config.wrist_camera, 'wrist'),
        (observation_config.wrist2_camera, 'wrist2')]:
        if config.point_cloud:
            obs_dict['%s_camera_extrinsics' % name] = obs.misc['%s_camera_extrinsics' % name]
            obs_dict['%s_camera_intrinsics' % name] = obs.misc['%s_camera_intrinsics' % name]
    return obs_dict


def _get_cam_observation_elements(camera: CameraConfig, prefix: str, channels_last):
    elements = []
    img_s = list(camera.image_size)
    shape = img_s + [3] if channels_last else [3] + img_s
    if camera.rgb:
        elements.append(
            ObservationElement('%s_rgb' % prefix, shape, np.uint8))
    if camera.point_cloud:
        elements.append(
            ObservationElement('%s_point_cloud' % prefix, shape, np.float32))
        elements.append(
            ObservationElement('%s_camera_extrinsics' % prefix, (4, 4),
                               np.float32))
        elements.append(
            ObservationElement('%s_camera_intrinsics' % prefix, (3, 3),
                               np.float32))
    if camera.depth:
        shape = img_s + [1] if schannels_last else [1] + img_s
        elements.append(
            ObservationElement('%s_depth' % prefix, shape, np.float32))
    if camera.mask:
        raise NotImplementedError()

    return elements


def _observation_elements(observation_config, channels_last, train_cfg) -> List[ObservationElement]:
    elements = []
    robot_state_len = 0

    if train_cfg.method.which_arm == 'right':
        if observation_config.joint_velocities_right:
            robot_state_len += 7
        if observation_config.joint_positions_right:
            robot_state_len += 7
        if observation_config.joint_forces_right:
            robot_state_len += 7
        if observation_config.gripper_right_open:
            robot_state_len += 1
        if observation_config.gripper_right_pose:
            robot_state_len += 7
        if observation_config.gripper_right_joint_positions:
            robot_state_len += 2
        if observation_config.gripper_right_touch_forces:
            robot_state_len += 2
        if observation_config.task_low_dim_state:
            raise NotImplementedError()

        if robot_state_len > 0:
            elements.append(ObservationElement(
                'low_dim_state', (robot_state_len,), np.float32))
    elif train_cfg.method.which_arm == 'left':
        if observation_config.joint_velocities_left:
            robot_state_len += 7
        if observation_config.joint_positions_left:
            robot_state_len += 7
        if observation_config.joint_forces_left:
            robot_state_len += 7
        if observation_config.gripper_left_open:
            robot_state_len += 1
        if observation_config.gripper_left_pose:
            robot_state_len += 7
        if observation_config.gripper_left_joint_positions:
            robot_state_len += 2
        if observation_config.gripper_left_touch_forces:
            robot_state_len += 2
        if observation_config.task_low_dim_state:
            raise NotImplementedError()

        if robot_state_len > 0:
            elements.append(ObservationElement(
                'low_dim_state', (robot_state_len,), np.float32))
    else:
        robot_state_len_right = 0
        robot_state_len_left = 0
        if observation_config.joint_velocities_right:
            robot_state_len_right += 7
        if observation_config.joint_positions_right:
            robot_state_len_right += 7
        if observation_config.joint_forces_right:
            robot_state_len_right += 7
        if observation_config.gripper_right_open:
            robot_state_len_right += 1
        if observation_config.gripper_right_pose:
            robot_state_len_right += 7
        if observation_config.gripper_right_joint_positions:
            robot_state_len_right += 2
        if observation_config.gripper_right_touch_forces:
            robot_state_len_right += 2
        if observation_config.task_low_dim_state:
            raise NotImplementedError()

        if observation_config.joint_velocities_left:
            robot_state_len_left += 7
        if observation_config.joint_positions_left:
            robot_state_len_left += 7
        if observation_config.joint_forces_left:
            robot_state_len_left += 7
        if observation_config.gripper_left_open:
            robot_state_len_left += 1
        if observation_config.gripper_left_pose:
            robot_state_len_left += 7
        if observation_config.gripper_left_joint_positions:
            robot_state_len_left += 2
        if observation_config.gripper_left_touch_forces:
            robot_state_len_left += 2
        if observation_config.task_low_dim_state:
            raise NotImplementedError()

        if robot_state_len_right > 0:
            elements.append(ObservationElement(
                'low_dim_state_right_arm', (robot_state_len_right,), np.float32))
        if robot_state_len_left > 0:
            elements.append(ObservationElement(
                'low_dim_state_left_arm', (robot_state_len_left,), np.float32))

    elements.extend(_get_cam_observation_elements(
        observation_config.front_camera, 'front', channels_last))
    elements.extend(_get_cam_observation_elements(
        observation_config.wrist_camera, 'wrist', channels_last))
    elements.extend(_get_cam_observation_elements(
        observation_config.wrist2_camera, 'wrist2', channels_last))
    return elements

class RLBenchEnv2Robots(Env):

    def __init__(self, task_class: Type[Task2Robots],
                 observation_config: ObservationConfig2Robots,
                 action_mode: ActionMode,
                 dataset_root: str = '',
                 channels_last=False,
                 headless=True,
                 include_lang_goal_in_obs=False,
                 train_cfg=None,
                 voxposer_only_eval=False,
                 eval_which_arm='',
                 custom_ttt_file=''):
        super(RLBenchEnv2Robots, self).__init__()
        self._task_class = task_class
        self._observation_config = observation_config
        self._channels_last = channels_last
        self._include_lang_goal_in_obs = include_lang_goal_in_obs
        self._crop_target_obj_voxel = train_cfg.method.crop_target_obj_voxel
        self._voxposer_only_eval = voxposer_only_eval
        self._eval_which_arm = eval_which_arm

        if train_cfg.method.which_arm in ['dominant', 'assistive'] or voxposer_only_eval or self._eval_which_arm == 'dominant_assistive':
            self.dominant_assitive_policy = True
        else:
            self.dominant_assitive_policy = False

        if train_cfg.method.crop_target_obj_voxel or voxposer_only_eval or self.dominant_assitive_policy:
            ############## VoxPoser + PerAct ##############
            # setup
            openai.api_key = 'REPLACE-MET'  # set your API key here
            voxposer_config = ''
            if self._voxposer_only_eval:
                voxposer_config = '../../../../voxposer/configs/voxposer_only_config.yaml'
            else:
                voxposer_config = '../../../../voxposer/configs/rlbench_config.yaml'
            print('VoxPoser config: ', voxposer_config)
            self.voxposer_config = get_config('rlbench', voxposer_config)
            self.voxposer_cache_dir = '../../../../voxposer/cache'
            # uncomment this if you'd like to change the language model (e.g., for faster speed or lower cost)
            for lmp_name, cfg in self.voxposer_config['lmp_config']['lmps'].items():
                cfg['model'] = 'gpt-3.5-turbo'
                # cfg['model'] = 'gpt-4' # very expensive...

            # initialize env and voxposer ui
            visualizer = ValueMapVisualizer(self.voxposer_config['visualizer'])
            self._rlbench_env = VoxPoserRLBench2Robots(
                visualizer=visualizer, observation_config=observation_config,
                dataset_root=dataset_root, headless=headless, task_name=task_class.__name__, dominant_assitive_policy=self.dominant_assitive_policy, custom_ttt_file=custom_ttt_file)
            self.lmps, self.lmp_env = setup_LMP(self._rlbench_env, self.voxposer_config, debug=False, cache_dir=self.voxposer_cache_dir, voxposer_only_eval=self._voxposer_only_eval)
            self.voxposer_ui = self.lmps['plan_ui']

            # load appropriate task
            # NOTE: modify this when new task is added
            if task_class.__name__ == 'OpenJar':
                self._rlbench_env.load_task(tasks.OpenJar)
            elif task_class.__name__ == 'OpenDrawer':
                self._rlbench_env.load_task(tasks.OpenDrawer)
            elif task_class.__name__ == 'SweepToDustpan':
                self._rlbench_env.load_task(tasks.SweepToDustpan)
            elif task_class.__name__ == 'PutItemInDrawer':
                self._rlbench_env.load_task(tasks.PutItemInDrawer)
            elif task_class.__name__ == 'HandOverItem':
                self._rlbench_env.load_task(tasks.HandOverItem)
            else:
                print(f'{task_class.__name__} is not implemented in RLBenchEnv2Robots')
                raise NotImplementedError
            self._rlbench_env.task._dataset_root = dataset_root
        else:
            self._rlbench_env = Environment2Robots(
                action_mode=action_mode, obs_config=observation_config,
                dataset_root=dataset_root, headless=headless, task_name=task_class.__name__)
        self._task = None
        self._lang_goal = 'unknown goal'
        self._train_cfg = train_cfg

    def reload_voxposer_variables(self):
        self.lmps, self.lmp_env = setup_LMP(self._rlbench_env, self.voxposer_config, debug=False, cache_dir=self.voxposer_cache_dir)
        self.voxposer_ui = self.lmps['plan_ui']

    def extract_obs(self, obs: Observation2Robots):
        extracted_obs = _extract_obs(obs, self._channels_last, self._observation_config, self._train_cfg)
        if self._include_lang_goal_in_obs:
            if self._train_cfg.method.which_arm == 'multiarm':
                left_arm_description, right_arm_description = utils.extract_left_and_right_arm_instruction(self._lang_goal)
                extracted_obs['lang_goal_tokens_left'] = tokenize([left_arm_description])[0].numpy()
                extracted_obs['lang_goal_tokens_right'] = tokenize([right_arm_description])[0].numpy()
            else:
                extracted_obs['lang_goal_tokens'] = tokenize([self._lang_goal])[0].numpy()
        return extracted_obs

    def launch(self):
        if self._crop_target_obj_voxel or self._voxposer_only_eval or self.dominant_assitive_policy:
            # already called launch when VoxPoserRLBench2Robots is instantitated
            self._task = self._rlbench_env.task
        else:
            self._rlbench_env.launch()
            self._task = self._rlbench_env.get_task(self._task_class)

    def shutdown(self):
        if self._crop_target_obj_voxel or self._voxposer_only_eval or self.dominant_assitive_policy:
            self._rlbench_env.rlbench_env.shutdown()

            self._rlbench_env = None
            self.lmps = None
            self.lmp_env = None
            self.voxposer_ui = None
        else:
            self._rlbench_env.shutdown()

    def reset(self) -> dict:
        descriptions, obs = self._task.reset()
        self._lang_goal = descriptions[0] # first description variant
        extracted_obs = self.extract_obs(obs)
        return extracted_obs

    def step(self, action: np.ndarray, which_arm) -> Transition:
        obs, reward, terminal = self._task.step(action, which_arm)
        obs = self.extract_obs(obs)
        return Transition(obs, reward, terminal)

    @property
    def observation_elements(self) -> List[ObservationElement]:
        return _observation_elements(self._observation_config, self._channels_last, self._train_cfg)

    @property
    def action_shape(self):
        return (self._rlbench_env.action_size, )

    @property
    def env(self) -> Environment2Robots:
        return self._rlbench_env


class MultiTaskRLBenchEnv(MultiTaskEnv):

    def __init__(self,
                 task_classes: List[Type[Task2Robots]],
                 observation_config: ObservationConfig2Robots,
                 action_mode: ActionMode,
                 dataset_root: str = '',
                 channels_last=False,
                 headless=True,
                 swap_task_every: int = 1,
                 include_lang_goal_in_obs=False):
        super(MultiTaskRLBenchEnv, self).__init__()
        # self._task_classes = task_classes
        # self._observation_config = observation_config
        # self._channels_last = channels_last
        # self._include_lang_goal_in_obs = include_lang_goal_in_obs
        # self._rlbench_env = Environment2Robots(
        #     action_mode=action_mode, obs_config=observation_config,
        #     dataset_root=dataset_root, headless=headless)
        # self._task = None
        # self._task_name = ''
        # self._lang_goal = 'unknown goal'
        # self._swap_task_every = swap_task_every
        # self._rlbench_env
        # self._episodes_this_task = 0
        # self._active_task_id = -1

        # self._task_name_to_idx = {change_case(tc.__name__):i for i, tc in enumerate(self._task_classes)}
        raise NotImplementedError

    def _set_new_task(self, shuffle=False):
        if shuffle:
            self._active_task_id = np.random.randint(0, len(self._task_classes))
        else:
            self._active_task_id = (self._active_task_id + 1) % len(self._task_classes)
        task = self._task_classes[self._active_task_id]
        self._task = self._rlbench_env.get_task(task)

    def set_task(self, task_name: str):
        self._active_task_id = self._task_name_to_idx[task_name]
        task = self._task_classes[self._active_task_id]
        self._task = self._rlbench_env.get_task(task)

        descriptions, _ = self._task.reset()
        self._lang_goal = descriptions[0] # first description variant

    def extract_obs(self, obs: Observation2Robots):
        extracted_obs = _extract_obs(obs, self._channels_last, self._observation_config)
        if self._include_lang_goal_in_obs:
            extracted_obs['lang_goal_tokens'] = tokenize([self._lang_goal])[0].numpy()
        return extracted_obs

    def launch(self):
        self._rlbench_env.launch()
        self._set_new_task()

    def shutdown(self):
        self._rlbench_env.shutdown()

    def reset(self) -> dict:
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        descriptions, obs = self._task.reset()
        self._lang_goal = descriptions[0] # first description variant
        extracted_obs = self.extract_obs(obs)

        return extracted_obs

    def step(self, action: np.ndarray) -> Transition:
        obs, reward, terminal = self._task.step(action)
        obs = self.extract_obs(obs)
        return Transition(obs, reward, terminal)

    @property
    def observation_elements(self) -> List[ObservationElement]:
        return _observation_elements(self._observation_config, self._channels_last)

    @property
    def action_shape(self):
        return (self._rlbench_env.action_size, )

    @property
    def env(self) -> Environment2Robots:
        return self._rlbench_env

    @property
    def num_tasks(self) -> int:
        return len(self._task_classes)