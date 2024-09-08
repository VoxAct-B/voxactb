import os
import numpy as np
import open3d as o3d
import json
from rlbench.action_modes.action_mode import MoveArmThenGripper, MoveArmThenGripper2Robots
from rlbench.action_modes.arm_action_modes import ArmActionMode, EndEffectorPoseViaPlanning, EndEffectorPoseViaPlanning2Robots
from rlbench.action_modes.gripper_action_modes import Discrete, GripperActionMode, Discrete2Robots
from rlbench.environment import Environment
from rlbench.environments_two_robots import Environment2Robots
import rlbench.tasks as tasks
from pyrep.const import ObjectType
from pyrep.objects.shape import Shape
from ..utils import normalize_vector, bcolors
import math
from PIL import Image
import io
import base64
import requests
import time
from ..LLM_cache import DiskCache


class CustomMoveArmThenGripper(MoveArmThenGripper):
    """
    A potential workaround for the default MoveArmThenGripper as we frequently run into zero division errors and failed path.
    TODO: check the root cause of it.
    Ignore arm action if it fails.

    Attributes:
        _prev_arm_action (numpy.ndarray): Stores the previous arm action.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prev_arm_action = None

    def action(self, scene, action):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        # if the arm action is the same as the previous action, skip it
        if self._prev_arm_action is not None and np.allclose(arm_action, self._prev_arm_action):
            self.gripper_action_mode.action(scene, ee_action)
        else:
            try:
                self.arm_action_mode.action(scene, arm_action)
            except Exception as e:
                print(f'{bcolors.FAIL}[rlbench_env.py] Ignoring failed arm action; Exception: "{str(e)}"{bcolors.ENDC}')
            self.gripper_action_mode.action(scene, ee_action)
        self._prev_arm_action = arm_action.copy()

class VoxPoserRLBench():
    def __init__(self, visualizer=None):
        """
        Initializes the VoxPoserRLBench environment.

        Args:
            visualizer: Visualization interface, optional.
        """
        action_mode = CustomMoveArmThenGripper(arm_action_mode=EndEffectorPoseViaPlanning(),
                                        gripper_action_mode=Discrete())
        self.rlbench_env = Environment(action_mode)
        self.rlbench_env.launch()
        self.task = None

        self.workspace_bounds_min = np.array([self.rlbench_env._scene._workspace_minx, self.rlbench_env._scene._workspace_miny, self.rlbench_env._scene._workspace_minz])
        self.workspace_bounds_max = np.array([self.rlbench_env._scene._workspace_maxx, self.rlbench_env._scene._workspace_maxy, self.rlbench_env._scene._workspace_maxz])
        self.visualizer = visualizer
        if self.visualizer is not None:
            self.visualizer.update_bounds(self.workspace_bounds_min, self.workspace_bounds_max)
        self.camera_names = ['front', 'left_shoulder', 'right_shoulder', 'overhead', 'wrist']
        # calculate lookat vector for all cameras (for normal estimation)
        name2cam = {
            'front': self.rlbench_env._scene._cam_front,
            'left_shoulder': self.rlbench_env._scene._cam_over_shoulder_left,
            'right_shoulder': self.rlbench_env._scene._cam_over_shoulder_right,
            'overhead': self.rlbench_env._scene._cam_overhead,
            'wrist': self.rlbench_env._scene._cam_wrist,
        }
        forward_vector = np.array([0, 0, 1])
        self.lookat_vectors = {}
        for cam_name in self.camera_names:
            extrinsics = name2cam[cam_name].get_matrix()
            lookat = extrinsics[:3, :3] @ forward_vector
            self.lookat_vectors[cam_name] = normalize_vector(lookat)
        # load file containing object names for each task
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task_object_names.json')
        with open(path, 'r') as f:
            self.task_object_names = json.load(f)

        self._reset_task_variables()

    def get_object_names(self):
        """
        Returns the names of all objects in the current task environment.

        Returns:
            list: A list of object names.
        """
        name_mapping = self.task_object_names[self.task.get_name()]
        exposed_names = [names[0] for names in name_mapping]
        return exposed_names

    def load_task(self, task):
        """
        Loads a new task into the environment and resets task-related variables.
        Records the mask IDs of the robot, gripper, and objects in the scene.

        Args:
            task (str or rlbench.tasks.Task): Name of the task class or a task object.
        """
        self._reset_task_variables()
        if isinstance(task, str):
            task = getattr(tasks, task)
        self.task = self.rlbench_env.get_task(task)
        self.arm_mask_ids = [obj.get_handle() for obj in self.task._robot.arm.get_objects_in_tree(exclude_base=False)]
        self.gripper_mask_ids = [obj.get_handle() for obj in self.task._robot.gripper.get_objects_in_tree(exclude_base=False)]
        self.robot_mask_ids = self.arm_mask_ids + self.gripper_mask_ids
        self.obj_mask_ids = [obj.get_handle() for obj in self.task._task.get_base().get_objects_in_tree(exclude_base=False)]
        # store (object name <-> object id) mapping for relevant task objects
        try:
            name_mapping = self.task_object_names[self.task.get_name()]
        except KeyError:
            raise KeyError(f'Task {self.task.get_name()} not found in "envs/task_object_names.json" (hint: make sure the task and the corresponding object names are added to the file)')
        exposed_names = [names[0] for names in name_mapping]
        internal_names = [names[1] for names in name_mapping]
        scene_objs = self.task._task.get_base().get_objects_in_tree(object_type=ObjectType.SHAPE,
                                                                      exclude_base=False,
                                                                      first_generation_only=False)
        for scene_obj in scene_objs:
            if scene_obj.get_name() in internal_names:
                exposed_name = exposed_names[internal_names.index(scene_obj.get_name())]
                self.name2ids[exposed_name] = [scene_obj.get_handle()]
                self.id2name[scene_obj.get_handle()] = exposed_name
                for child in scene_obj.get_objects_in_tree():
                    self.name2ids[exposed_name].append(child.get_handle())
                    self.id2name[child.get_handle()] = exposed_name

    def get_3d_obs_by_name(self, query_name):
        """
        Retrieves 3D point cloud observations and normals of an object by its name.

        Args:
            query_name (str): The name of the object to query.

        Returns:
            tuple: A tuple containing object points and object normals.
        """
        assert query_name in self.name2ids, f"Unknown object name: {query_name}"
        obj_ids = self.name2ids[query_name]
        # gather points and masks from all cameras
        points, masks, normals = [], [], []
        for cam in self.camera_names:
            points.append(getattr(self.latest_obs, f"{cam}_point_cloud").reshape(-1, 3))
            masks.append(getattr(self.latest_obs, f"{cam}_mask").reshape(-1))
            # estimate normals using o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[-1])
            pcd.estimate_normals()
            cam_normals = np.asarray(pcd.normals)
            # use lookat vector to adjust normal vectors
            flip_indices = np.dot(cam_normals, self.lookat_vectors[cam]) > 0
            cam_normals[flip_indices] *= -1
            normals.append(cam_normals)
        points = np.concatenate(points, axis=0)
        masks = np.concatenate(masks, axis=0)
        normals = np.concatenate(normals, axis=0)
        # get object points
        obj_points = points[np.isin(masks, obj_ids)]
        if len(obj_points) == 0:
            raise ValueError(f"Object {query_name} not found in the scene")
        obj_normals = normals[np.isin(masks, obj_ids)]
        # voxel downsample using o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points)
        pcd.normals = o3d.utility.Vector3dVector(obj_normals)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        obj_points = np.asarray(pcd_downsampled.points)
        obj_normals = np.asarray(pcd_downsampled.normals)
        return obj_points, obj_normals

    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        """
        Retrieves the entire scene's 3D point cloud observations and colors.

        Args:
            ignore_robot (bool): Whether to ignore points corresponding to the robot.
            ignore_grasped_obj (bool): Whether to ignore points corresponding to grasped objects.

        Returns:
            tuple: A tuple containing scene points and colors.
        """
        points, colors, masks = [], [], []
        for cam in self.camera_names:
            points.append(getattr(self.latest_obs, f"{cam}_point_cloud").reshape(-1, 3))
            colors.append(getattr(self.latest_obs, f"{cam}_rgb").reshape(-1, 3))
            masks.append(getattr(self.latest_obs, f"{cam}_mask").reshape(-1))
        points = np.concatenate(points, axis=0)
        colors = np.concatenate(colors, axis=0)
        masks = np.concatenate(masks, axis=0)

        # only keep points within workspace
        chosen_idx_x = (points[:, 0] > self.workspace_bounds_min[0]) & (points[:, 0] < self.workspace_bounds_max[0])
        chosen_idx_y = (points[:, 1] > self.workspace_bounds_min[1]) & (points[:, 1] < self.workspace_bounds_max[1])
        chosen_idx_z = (points[:, 2] > self.workspace_bounds_min[2]) & (points[:, 2] < self.workspace_bounds_max[2])
        points = points[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]
        colors = colors[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]
        masks = masks[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]

        if ignore_robot:
            robot_mask = np.isin(masks, self.robot_mask_ids)
            points = points[~robot_mask]
            colors = colors[~robot_mask]
            masks = masks[~robot_mask]
        if self.grasped_obj_ids and ignore_grasped_obj:
            grasped_mask = np.isin(masks, self.grasped_obj_ids)
            points = points[~grasped_mask]
            colors = colors[~grasped_mask]
            masks = masks[~grasped_mask]

        # voxel downsample using o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        points = np.asarray(pcd_downsampled.points)
        colors = np.asarray(pcd_downsampled.colors).astype(np.uint8)

        return points, colors

    def reset(self):
        """
        Resets the environment and the task. Also updates the visualizer.

        Returns:
            tuple: A tuple containing task descriptions and initial observations.
        """
        assert self.task is not None, "Please load a task first"
        self.task.sample_variation()
        descriptions, obs = self.task.reset()
        obs = self._process_obs(obs)
        self.init_obs = obs
        self.latest_obs = obs
        self._update_visualizer()
        return descriptions, obs

    def apply_action(self, action):
        """
        Applies an action in the environment and updates the state.

        Args:
            action: The action to apply.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        assert self.task is not None, "Please load a task first"
        action = self._process_action(action)
        obs, reward, terminate = self.task.step(action)
        obs = self._process_obs(obs)
        self.latest_obs = obs
        self.latest_reward = reward
        self.latest_terminate = terminate
        self.latest_action = action
        self._update_visualizer()
        grasped_objects = self.rlbench_env._scene.robot.gripper.get_grasped_objects()
        if len(grasped_objects) > 0:
            self.grasped_obj_ids = [obj.get_handle() for obj in grasped_objects]
        return obs, reward, terminate

    def move_to_pose(self, pose, speed=None):
        """
        Moves the robot arm to a specific pose.

        Args:
            pose: The target pose.
            speed: The speed at which to move the arm. Currently not implemented.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        if self.latest_action is None:
            action = np.concatenate([pose, [self.init_obs.gripper_open]])
        else:
            action = np.concatenate([pose, [self.latest_action[-1]]])
        return self.apply_action(action)

    def open_gripper(self):
        """
        Opens the gripper of the robot.
        """
        action = np.concatenate([self.latest_obs.gripper_pose, [1.0]])
        return self.apply_action(action)

    def close_gripper(self):
        """
        Closes the gripper of the robot.
        """
        action = np.concatenate([self.latest_obs.gripper_pose, [0.0]])
        return self.apply_action(action)

    def set_gripper_state(self, gripper_state):
        """
        Sets the state of the gripper.

        Args:
            gripper_state: The target state for the gripper.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        action = np.concatenate([self.latest_obs.gripper_pose, [gripper_state]])
        return self.apply_action(action)

    def reset_to_default_pose(self):
        """
        Resets the robot arm to its default pose.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        if self.latest_action is None:
            action = np.concatenate([self.init_obs.gripper_pose, [self.init_obs.gripper_open]])
        else:
            action = np.concatenate([self.init_obs.gripper_pose, [self.latest_action[-1]]])
        return self.apply_action(action)

    def get_ee_pose(self):
        assert self.latest_obs is not None, "Please reset the environment first"
        return self.latest_obs.gripper_pose

    def get_ee_pos(self):
        return self.get_ee_pose()[:3]

    def get_ee_quat(self):
        return self.get_ee_pose()[3:]

    def get_last_gripper_action(self):
        """
        Returns the last gripper action.

        Returns:
            float: The last gripper action.
        """
        if self.latest_action is not None:
            return self.latest_action[-1]
        else:
            return self.init_obs.gripper_open

    def _reset_task_variables(self):
        """
        Resets variables related to the current task in the environment.

        Note: This function is generally called internally.
        """
        self.init_obs = None
        self.latest_obs = None
        self.latest_reward = None
        self.latest_terminate = None
        self.latest_action = None
        self.grasped_obj_ids = None
        # scene-specific helper variables
        self.arm_mask_ids = None
        self.gripper_mask_ids = None
        self.robot_mask_ids = None
        self.obj_mask_ids = None
        self.name2ids = {}  # first_generation name -> list of ids of the tree
        self.id2name = {}  # any node id -> first_generation name

    def _update_visualizer(self):
        """
        Updates the scene in the visualizer with the latest observations.

        Note: This function is generally called internally.
        """
        if self.visualizer is not None:
            points, colors = self.get_scene_3d_obs(ignore_robot=False, ignore_grasped_obj=False)
            self.visualizer.update_scene_points(points, colors)

    def _process_obs(self, obs):
        """
        Processes the observations, specifically converts quaternion format from xyzw to wxyz.

        Args:
            obs: The observation to process.

        Returns:
            The processed observation.
        """
        quat_xyzw = obs.gripper_pose[3:]
        quat_wxyz = np.concatenate([quat_xyzw[-1:], quat_xyzw[:-1]])
        obs.gripper_pose[3:] = quat_wxyz
        return obs

    def _process_action(self, action):
        """
        Processes the action, specifically converts quaternion format from wxyz to xyzw.

        Args:
            action: The action to process.

        Returns:
            The processed action.
        """
        quat_wxyz = action[3:7]
        quat_xyzw = np.concatenate([quat_wxyz[1:], quat_wxyz[:1]])
        action[3:7] = quat_xyzw
        return action



"""
Bimanual Manipulation
"""
class CustomMoveArmThenGripper2Robots(MoveArmThenGripper2Robots):
    """
    A potential workaround for the default MoveArmThenGripper as we frequently run into zero division errors and failed path.
    TODO: check the root cause of it.
    Ignore arm action if it fails.

    Attributes:
        _prev_arm_action (numpy.ndarray): Stores the previous arm action.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prev_arm_action = None

    def action(self, scene, action, which_arm):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])

        if which_arm == 'right hand':
            which_arm = 'right'
        if which_arm == 'left hand':
            which_arm = 'left'
        if which_arm not in ['right hand', 'left hand', 'left', 'right']:
            raise NotImplementedError

        # print(f'{which_arm} executing in CustomMoveArmThenGripper2Robots action function.') # for debugging
        # if the arm action is the same as the previous action, skip it
        if self._prev_arm_action is not None and np.allclose(arm_action, self._prev_arm_action):
            self.gripper_action_mode.action(scene, ee_action, which_arm)
        else:
            try:
                self.arm_action_mode.action(scene, arm_action, which_arm=which_arm)
            except Exception as e:
                print(f'{bcolors.FAIL}[rlbench_env.py] Ignoring failed arm action; Exception: "{str(e)}"{bcolors.ENDC}')
            self.gripper_action_mode.action(scene, ee_action, which_arm)
        self._prev_arm_action = arm_action.copy()

    def action_peract(self, scene, action, which_arm):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:arm_act_size+1])
        ignore_collisions = bool(action[arm_act_size+1:arm_act_size+2])
        self.arm_action_mode.action(scene, arm_action, ignore_collisions, which_arm)
        self.gripper_action_mode.action(scene, ee_action, which_arm)

class VoxPoserRLBench2Robots():
    def __init__(self, visualizer=None, observation_config=None, dataset_root=None, headless=None, task_name=None, dominant_assitive_policy=False, custom_ttt_file=''):
        """
        Initializes the VoxPoserRLBench environment.

        Args:
            visualizer: Visualization interface, optional.
        """
        action_mode = CustomMoveArmThenGripper2Robots(arm_action_mode=EndEffectorPoseViaPlanning2Robots(),
                                        gripper_action_mode=Discrete2Robots())
        if observation_config is not None:
            # VoxPoser + PerAct
            self.rlbench_env = Environment2Robots(action_mode=action_mode, obs_config=observation_config, dataset_root=dataset_root, headless=headless, task_name=task_name)
        else:
            self.rlbench_env = Environment2Robots(action_mode, task_name=task_name)
        if custom_ttt_file != '':
            self.rlbench_env._TTT_FILE = custom_ttt_file
        self.rlbench_env.launch()
        self.task = None
        self.task_name = task_name

        self.workspace_bounds_min = np.array([self.rlbench_env._scene._workspace_minx, self.rlbench_env._scene._workspace_miny, self.rlbench_env._scene._workspace_minz])
        self.workspace_bounds_max = np.array([self.rlbench_env._scene._workspace_maxx, self.rlbench_env._scene._workspace_maxy, self.rlbench_env._scene._workspace_maxz])
        self.visualizer = visualizer
        if self.visualizer is not None:
            self.visualizer.update_bounds(self.workspace_bounds_min, self.workspace_bounds_max)
        self.camera_names = ['front', 'wrist', 'wrist2']
        # calculate lookat vector for all cameras (for normal estimation)
        name2cam = {
            'front': self.rlbench_env._scene._cam_front,
            'wrist': self.rlbench_env._scene._cam_wrist,
            'wrist2': self.rlbench_env._scene._cam_wrist2,
        }
        forward_vector = np.array([0, 0, 1])
        self.lookat_vectors = {}
        for cam_name in self.camera_names:
            extrinsics = name2cam[cam_name].get_matrix()
            lookat = extrinsics[:3, :3] @ forward_vector
            self.lookat_vectors[cam_name] = normalize_vector(lookat)
        # load file containing object names for each task
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task_object_names.json')
        with open(path, 'r') as f:
            self.task_object_names = json.load(f)

        self._reset_task_variables()

        # dominant, assistive policies
        self._dominant_assitive_policy = dominant_assitive_policy
        self._dominant_arm = ''
        self._dominant_arm_for_ep_reset = ''

        ##### ChatGPT for determining acting arm
        # self.rlbench_env.add_highres_front_cam_for_llm()
        # self._image_cache = DiskCache(cache_dir='../../../../voxposer/cache', load_cache=True)
        # self._env_image_folder_path = f'../../../../voxposer/env_images'
        # self._env_image_saved_path = f'{self._env_image_folder_path}/direct_view.jpg'
        # if not os.path.exists(self._env_image_folder_path):
        #     os.makedirs(self._env_image_folder_path)


    def set_dominant_hand_for_ep_reset(self, ep_number):
        """
        Set _dominant_arm_for_ep_reset variable based on current episode number. This function is used to properly set up the environment, specifically the base_rotation_bounds for each task.
        """
        if self._dominant_assitive_policy:
            # episode numbers < 12 are all left-armed dominant, and episode numbers  >= 12 to 24 are all right-armed dominant
            if ep_number < 12:
                self._dominant_arm_for_ep_reset = 'left'
            else:
                self._dominant_arm_for_ep_reset = 'right'

    def determine_dominant_hand(self):
        if self.task_name == 'OpenDrawer':
            ##### VLM for determining acting arm
            # compute the angle between the look-at vector and the bottom drawer handle (normals vector)
            bottom_drawer_handle_normals = self.get_3d_normals_by_name('bottom drawer handle')
            bottom_drawer_handle_normals_avg = np.mean(bottom_drawer_handle_normals, axis=0)
            self.angle_bottom_drawer_handle_and_lookat = math.degrees(np.arccos(self.lookat_vectors['front'][0] * bottom_drawer_handle_normals_avg[0] + self.lookat_vectors['front'][1] * bottom_drawer_handle_normals_avg[1] + self.lookat_vectors['front'][2] * bottom_drawer_handle_normals_avg[2]))

            # 135 is the threshold angle used to determine which arm to use
            if self.angle_bottom_drawer_handle_and_lookat >= 135:
                self._dominant_arm = 'right'
            else:
                self._dominant_arm = 'left'

            # for debugging... self._dominant_arm_for_ep_reset contains the ground truth dominant arm
            # self._dominant_arm = self._dominant_arm_for_ep_reset

            print(f'Chosen dominant arm is {self._dominant_arm}')

            ##### ChatGPT for determining acting arm
            # self.rlbench_env._highres_front_cam.handle_explicitly()
            # highres_front_rgb = self.rlbench_env._highres_front_cam.capture_rgb()
            # highres_front_rgb = np.clip((highres_front_rgb * 255.).astype(np.uint8), 0, 255)
            # # highres_front_rgb = Image.fromarray(highres_front_rgb) # for debugging
            # # highres_front_rgb.show() # for debugging

            # prompt = "This picture shows a simulation environment with two robotic arms and a drawer in a tabletop environment. The image is taken by a front camera looking at the two robotic arms, a drawer with three handles (top, middle, and bottom), and a table. One robotic arm is positioned on the left side of the table. The other robotic arm is placed on the right side of the table. The drawer is positioned in between the two robotic arms. The robotic arms are fixed onto the table, but the drawer can randomly spawn on the table between the two robotic arms. The drawer could spawn in a different location with a different orientation. Ignore the background walls and floor. Pay attention to the orientation of the drawer with respect to the two robotic arms and the table. Describe what's in the image in detail. Then, tell me in a new sentence that which robotic arm is the front of the drawer (with top, middle, and bottom drawer handles) facing without other texts."
            # # obs = self.rlbench_env._scene.get_observation()
            # # dominant_arm = self._determine_dominant_hand_LLM_helper(obs.front_rgb, prompt) # front rgb image (128 x 128)
            # dominant_arm = self._determine_dominant_hand_LLM_helper(highres_front_rgb, prompt) # high-res front rgb image (512 x 512)
            # self._dominant_arm = dominant_arm
            # print(f'\n\n !!!!!! ChatGPT dominant arm prediction: {dominant_arm}')
        elif self.task_name == 'PutItemInDrawer':
            ##### VLM for determining acting arm
            # compute the angle between the look-at vector and the top drawer handle (normals vector)
            top_drawer_handle_normals = self.get_3d_normals_by_name('top drawer handle')
            top_drawer_handle_normals_avg = np.mean(top_drawer_handle_normals, axis=0)
            self.angle_top_drawer_handle_and_lookat = math.degrees(np.arccos(self.lookat_vectors['front'][0] * top_drawer_handle_normals_avg[0] + self.lookat_vectors['front'][1] * top_drawer_handle_normals_avg[1] + self.lookat_vectors['front'][2] * top_drawer_handle_normals_avg[2]))

            # 134 is the threshold angle used to determine which arm to use
            if self.angle_top_drawer_handle_and_lookat >= 134:
                self._dominant_arm = 'left'
            else:
                self._dominant_arm = 'right'
            # print('\n\n\n\n\n!!!!!!!!!! self.angle_top_drawer_handle_and_lookat: ', self.angle_top_drawer_handle_and_lookat)

            # for debugging... self._dominant_arm_for_ep_reset contains the ground truth dominant arm
            # self._dominant_arm = self._dominant_arm_for_ep_reset

            print(f'Chosen dominant arm is {self._dominant_arm}')
        elif self.task_name == 'OpenJar':
            jar_points = self.get_3d_points_by_name('jar')
            jar_points_avg = np.mean(jar_points, axis=0)
            robot_right_position, robot_left_position = self.get_robot_arms_position()

            jar_to_right_arm_dist = math.dist(jar_points_avg, robot_right_position)
            jar_to_left_arm_dist = math.dist(jar_points_avg, robot_left_position)

            if jar_to_right_arm_dist < jar_to_left_arm_dist:
                # jar is closer to the robot arm on the right
                self._dominant_arm = 'right'
            else:
                # jar is closer to the robot arm on the left
                self._dominant_arm = 'left'
            print('jar_to_right_arm_dist in determine_dominant_hand: ', jar_to_right_arm_dist)
            print('jar_to_left_arm_dist in determine_dominant_hand: ', jar_to_left_arm_dist)
            print('determine_dominant_hand: ', self._dominant_arm)
        elif self.task_name == 'HandOverItem':
            cube_points = self.get_3d_points_by_name('cube')
            cube_points_avg = np.mean(cube_points, axis=0)
            robot_right_position, robot_left_position = self.get_robot_arms_position()

            cube_to_right_arm_dist = math.dist(cube_points_avg, robot_right_position)
            cube_to_left_arm_dist = math.dist(cube_points_avg, robot_left_position)

            if cube_to_right_arm_dist < cube_to_left_arm_dist:
                # cube is closer to the robot arm on the right
                self._dominant_arm = 'left'
            else:
                # cube is closer to the robot arm on the left
                self._dominant_arm = 'right'
            print('cube_to_right_arm_dist in determine_dominant_hand: ', cube_to_right_arm_dist)
            print('cube_to_left_arm_dist in determine_dominant_hand: ', cube_to_left_arm_dist)
            print('determine_dominant_hand: ', self._dominant_arm)
        else:
            raise NotImplementedError

    def _determine_dominant_hand_LLM_helper(self, front_rgb_numpy, prompt):
        # openAI API Key
        api_key = "REPLACE-ME"

        # Function to encode the image
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        # Getting the base64 string
        front_rgb = Image.fromarray(front_rgb_numpy)
        front_rgb.save(self._env_image_saved_path)
        base64_image = encode_image(self._env_image_saved_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"{prompt}"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }

        if payload in self._image_cache:
            print('(using image cache in _determine_dominant_hand_LLM_helper)')
            return self._image_cache[payload]
        else:
            try:
                content = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()['choices'][0]['message']['content']
            except:
                print('Retry OpenAI API in 5 seconds...')
                time.sleep(5) # sleep for 5 seconds
                content = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()['choices'][0]['message']['content']
            print('\n\nChatGPT full response: ', content)

            # extract the last two sentences from the response
            acting_arm = '.'.join(content.split('.')[-2:])
            if 'left' in acting_arm:
                acting_arm = 'left'
            elif 'right' in acting_arm:
                acting_arm = 'right'
            else:
                print('!!!!!!!!!!!!!!! Incorrect response from ChatGPT in _determine_dominant_hand_LLM_helper: ', content)
                # randomly assign acting arm
                rand = np.random.randint(2)
                if rand == 0:
                    acting_arm = 'right'
                else:
                    acting_arm = 'left'
                print(f'Randomly assign {content} as the acting arm')
            self._image_cache[payload] = acting_arm

        time.sleep(2) # for debugging... easier to read the prints from the terminal
        return acting_arm

    def get_target_object_world_coords(self, gt_target_object_world_coords=False, auto_crop=False):
        """
        NOTE: target_object_world_coords should be close to or the same as target_object_pos in scene_two_robots.py
        """
        if gt_target_object_world_coords:
            if self.task_name in ['OpenDrawer', 'PutItemInDrawer']:
                object_handle = Shape('drawer_middle').get_handle()
            elif self.task_name == 'OpenJar':
                object_handle = [Shape('jar_lid0').get_handle(), Shape('jar0').get_handle()]
            else:
                raise NotImplementedError
            target_object_world_coords = self.get_target_object_pos_by_obj_handle_front_camera(object_handle)
            return target_object_world_coords

        # option 1: get image from the front camera
        obs = self.rlbench_env._scene.get_observation()
        front_rgb = Image.fromarray(obs.front_rgb)
        front_rgb = front_rgb.resize((1024, 1024))
        points = self.rlbench_env._scene._cam_front.capture_pointcloud()

        # option 2: get image from a higher-resolution front camera
        # front_rgb = self.rlbench_env._scene.get_highres_front_image_in_pil()
        # get object points
        # points = self.rlbench_env._scene._highres_front_cam.capture_pointcloud()

        # target_object_world_coords, auto_crop_radius = self.rlbench_env._scene.vlm.get_target_object_world_coords(front_rgb, points, self.task_name, debug=True, auto_crop=auto_crop)
        target_object_world_coords, auto_crop_radius = self.rlbench_env._scene.vlm.get_target_object_world_coords(front_rgb, points, self.task_name, debug=False, auto_crop=auto_crop)

        return target_object_world_coords, auto_crop_radius

    def get_object_names(self):
        """
        Returns the names of all objects in the current task environment.

        Returns:
            list: A list of object names.
        """
        name_mapping = self.task_object_names[self.task.get_name()]
        exposed_names = [names[0] for names in name_mapping]
        return exposed_names

    def load_task(self, task):
        """
        Loads a new task into the environment and resets task-related variables.
        Records the mask IDs of the robot, gripper, and objects in the scene.

        Args:
            task (str or rlbench.tasks.Task): Name of the task class or a task object.
        """
        self._reset_task_variables()
        if isinstance(task, str):
            task = getattr(tasks, task)
        self.task = self.rlbench_env.get_task(task)
        self.arm_right_mask_ids = [obj.get_handle() for obj in self.task._robot_right.arm.get_objects_in_tree(exclude_base=False)]
        self.gripper_right_mask_ids = [obj.get_handle() for obj in self.task._robot_right.gripper.get_objects_in_tree(exclude_base=False)]
        self.robot_right_mask_ids = self.arm_right_mask_ids + self.gripper_right_mask_ids

        self.arm_left_mask_ids = [obj.get_handle() for obj in self.task._robot_left.arm.get_objects_in_tree(exclude_base=False)]
        self.gripper_left_mask_ids = [obj.get_handle() for obj in self.task._robot_left.gripper.get_objects_in_tree(exclude_base=False)]
        self.robot_left_mask_ids = self.arm_left_mask_ids + self.gripper_left_mask_ids

        self.obj_mask_ids = [obj.get_handle() for obj in self.task._task.get_base().get_objects_in_tree(exclude_base=False)]
        # store (object name <-> object id) mapping for relevant task objects
        try:
            name_mapping = self.task_object_names[self.task.get_name()]
        except KeyError:
            raise KeyError(f'Task {self.task.get_name()} not found in "envs/task_object_names.json" (hint: make sure the task and the corresponding object names are added to the file)')
        exposed_names = [names[0] for names in name_mapping]
        internal_names = [names[1] for names in name_mapping]
        scene_objs = self.task._task.get_base().get_objects_in_tree(object_type=ObjectType.SHAPE,
                                                                      exclude_base=False,
                                                                      first_generation_only=False)
        for scene_obj in scene_objs:
            if scene_obj.get_name() in internal_names:
                exposed_name = exposed_names[internal_names.index(scene_obj.get_name())]
                self.name2ids[exposed_name] = [scene_obj.get_handle()]
                self.id2name[scene_obj.get_handle()] = exposed_name
                for child in scene_obj.get_objects_in_tree():
                    self.name2ids[exposed_name].append(child.get_handle())
                    self.id2name[child.get_handle()] = exposed_name

    def update_env_variables(self):
        self._reset_task_variables()
        self.arm_right_mask_ids = [obj.get_handle() for obj in self.task._robot_right.arm.get_objects_in_tree(exclude_base=False)]
        self.gripper_right_mask_ids = [obj.get_handle() for obj in self.task._robot_right.gripper.get_objects_in_tree(exclude_base=False)]
        self.robot_right_mask_ids = self.arm_right_mask_ids + self.gripper_right_mask_ids

        self.arm_left_mask_ids = [obj.get_handle() for obj in self.task._robot_left.arm.get_objects_in_tree(exclude_base=False)]
        self.gripper_left_mask_ids = [obj.get_handle() for obj in self.task._robot_left.gripper.get_objects_in_tree(exclude_base=False)]
        self.robot_left_mask_ids = self.arm_left_mask_ids + self.gripper_left_mask_ids

        self.obj_mask_ids = [obj.get_handle() for obj in self.task._task.get_base().get_objects_in_tree(exclude_base=False)]
        # store (object name <-> object id) mapping for relevant task objects
        try:
            name_mapping = self.task_object_names[self.task.get_name()]
        except KeyError:
            raise KeyError(f'Task {self.task.get_name()} not found in "envs/task_object_names.json" (hint: make sure the task and the corresponding object names are added to the file)')
        exposed_names = [names[0] for names in name_mapping]
        internal_names = [names[1] for names in name_mapping]
        scene_objs = self.task._task.get_base().get_objects_in_tree(object_type=ObjectType.SHAPE,
                                                                      exclude_base=False,
                                                                      first_generation_only=False)
        for scene_obj in scene_objs:
            if scene_obj.get_name() in internal_names:
                exposed_name = exposed_names[internal_names.index(scene_obj.get_name())]
                self.name2ids[exposed_name] = [scene_obj.get_handle()]
                self.id2name[scene_obj.get_handle()] = exposed_name
                for child in scene_obj.get_objects_in_tree():
                    self.name2ids[exposed_name].append(child.get_handle())
                    self.id2name[child.get_handle()] = exposed_name

    def load_objects(self):
        self.arm_right_mask_ids = [obj.get_handle() for obj in self.task._robot_right.arm.get_objects_in_tree(exclude_base=False)]
        self.gripper_right_mask_ids = [obj.get_handle() for obj in self.task._robot_right.gripper.get_objects_in_tree(exclude_base=False)]
        self.robot_right_mask_ids = self.arm_right_mask_ids + self.gripper_right_mask_ids

        self.arm_left_mask_ids = [obj.get_handle() for obj in self.task._robot_left.arm.get_objects_in_tree(exclude_base=False)]
        self.gripper_left_mask_ids = [obj.get_handle() for obj in self.task._robot_left.gripper.get_objects_in_tree(exclude_base=False)]
        self.robot_left_mask_ids = self.arm_left_mask_ids + self.gripper_left_mask_ids

        self.obj_mask_ids = [obj.get_handle() for obj in self.task._task.get_base().get_objects_in_tree(exclude_base=False)]
        # store (object name <-> object id) mapping for relevant task objects
        try:
            name_mapping = self.task_object_names[self.task.get_name()]
        except KeyError:
            raise KeyError(f'Task {self.task.get_name()} not found in "envs/task_object_names.json" (hint: make sure the task and the corresponding object names are added to the file)')
        exposed_names = [names[0] for names in name_mapping]
        internal_names = [names[1] for names in name_mapping]
        scene_objs = self.task._task.get_base().get_objects_in_tree(object_type=ObjectType.SHAPE,
                                                                      exclude_base=False,
                                                                      first_generation_only=False)
        for scene_obj in scene_objs:
            if scene_obj.get_name() in internal_names:
                exposed_name = exposed_names[internal_names.index(scene_obj.get_name())]
                self.name2ids[exposed_name] = [scene_obj.get_handle()]
                self.id2name[scene_obj.get_handle()] = exposed_name
                for child in scene_obj.get_objects_in_tree():
                    self.name2ids[exposed_name].append(child.get_handle())
                    self.id2name[child.get_handle()] = exposed_name

    def get_target_object_pos_by_obj_handle_front_camera(self, object_handle):
        """
        Retrieves 3D position of an object by its object handle.

        Args:
            object_handle (int): Object handle.

        Returns:
            tuple: A tuple containing object position (x, y, z)
        """
        points = getattr(self.latest_obs, f"front_point_cloud")
        mask = getattr(self.latest_obs, f"front_mask")
        obj_points = points[np.isin(mask, object_handle)]
        if len(obj_points) == 0:
            raise ValueError(f"Object {object_handle} not found in the scene")

        # for debugging
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 10))
        # front_rgb = getattr(self.latest_obs, f"front_rgb")
        # plt.imshow(front_rgb)
        # object_mask = np.isin(mask, object_handle)
        # plt.imshow(object_mask, cmap='gray', alpha=0.7)
        # y_indices, x_indices = np.nonzero(object_mask)
        # cX = np.mean(x_indices).astype(int)
        # cY = np.mean(y_indices).astype(int)
        # plt.plot(cX, cY, marker='v', color="red")
        # plt.savefig('debug_gt_mask_in_rlbench_env.png')
        # plt.close()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        obj_points = np.asarray(pcd_downsampled.points)
        target_object_pos = np.mean(obj_points, axis=0)
        return target_object_pos

    def get_3d_obs_by_name(self, query_name):
        """
        Retrieves 3D point cloud observations and normals of an object by its name.

        Args:
            query_name (str): The name of the object to query.

        Returns:
            tuple: A tuple containing object points and object normals.
        """
        assert query_name in self.name2ids, f"Unknown object name: {query_name}"
        obj_ids = self.name2ids[query_name]
        # gather points and masks from all cameras
        points, masks, normals = [], [], []
        for cam in self.camera_names:
            points.append(getattr(self.latest_obs, f"{cam}_point_cloud").reshape(-1, 3))
            masks.append(getattr(self.latest_obs, f"{cam}_mask").reshape(-1))
            # estimate normals using o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[-1])
            pcd.estimate_normals()
            cam_normals = np.asarray(pcd.normals)
            # use lookat vector to adjust normal vectors
            flip_indices = np.dot(cam_normals, self.lookat_vectors[cam]) > 0
            cam_normals[flip_indices] *= -1
            normals.append(cam_normals)
        points = np.concatenate(points, axis=0)
        masks = np.concatenate(masks, axis=0)
        normals = np.concatenate(normals, axis=0)
        # get object points
        obj_points = points[np.isin(masks, obj_ids)]
        if len(obj_points) == 0:
            raise ValueError(f"Object {query_name} not found in the scene")
        obj_normals = normals[np.isin(masks, obj_ids)]
        # voxel downsample using o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points)
        pcd.normals = o3d.utility.Vector3dVector(obj_normals)

        # for debugging
        # if query_name == 'jar':
        #     print('Visualizing....')
        #     # visualize point clouds using o3d
        #     pcd_visual = o3d.geometry.PointCloud()
        #     pcd_visual.points = o3d.utility.Vector3dVector(points)
        #     pcd_visual.normals = o3d.utility.Vector3dVector(normals)
        #     vis = o3d.visualization.Visualizer()
        #     vis.create_window()
        #     vis.add_geometry(pcd_visual)
        #     vis.run()

        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        obj_points = np.asarray(pcd_downsampled.points)
        obj_normals = np.asarray(pcd_downsampled.normals)
        return obj_points, obj_normals

    def get_3d_normals_by_name(self, query_name):
        assert query_name in self.name2ids, f"Unknown object name: {query_name}"
        obj_ids = self.name2ids[query_name]
        # gather points and masks from all cameras
        points, masks, normals = [], [], []
        for cam in self.camera_names:
            points.append(getattr(self.latest_obs, f"{cam}_point_cloud").reshape(-1, 3))
            masks.append(getattr(self.latest_obs, f"{cam}_mask").reshape(-1))
            # estimate normals using o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[-1])
            pcd.estimate_normals()
            cam_normals = np.asarray(pcd.normals)
            # use lookat vector to adjust normal vectors
            flip_indices = np.dot(cam_normals, self.lookat_vectors[cam]) > 0
            cam_normals[flip_indices] *= -1
            normals.append(cam_normals)
        points = np.concatenate(points, axis=0)
        masks = np.concatenate(masks, axis=0)
        normals = np.concatenate(normals, axis=0)
        # get object points
        obj_points = points[np.isin(masks, obj_ids)]
        if len(obj_points) == 0:
            raise ValueError(f"Object {query_name} not found in the scene")
        obj_normals = normals[np.isin(masks, obj_ids)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points)
        pcd.normals = o3d.utility.Vector3dVector(obj_normals)

        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        obj_points = np.asarray(pcd_downsampled.points)
        obj_normals = np.asarray(pcd_downsampled.normals)

        return obj_normals

    def get_3d_points_by_name(self, query_name):
        assert query_name in self.name2ids, f"Unknown object name: {query_name}"
        obj_ids = self.name2ids[query_name]
        # gather points and masks from all cameras
        points, masks, normals = [], [], []
        for cam in self.camera_names:
            points.append(getattr(self.latest_obs, f"{cam}_point_cloud").reshape(-1, 3))
            masks.append(getattr(self.latest_obs, f"{cam}_mask").reshape(-1))
            # estimate normals using o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[-1])
            pcd.estimate_normals()
            cam_normals = np.asarray(pcd.normals)
            # use lookat vector to adjust normal vectors
            flip_indices = np.dot(cam_normals, self.lookat_vectors[cam]) > 0
            cam_normals[flip_indices] *= -1
            normals.append(cam_normals)
        points = np.concatenate(points, axis=0)
        masks = np.concatenate(masks, axis=0)
        normals = np.concatenate(normals, axis=0)
        # get object points
        obj_points = points[np.isin(masks, obj_ids)]
        if len(obj_points) == 0:
            raise ValueError(f"Object {query_name} not found in the scene")
        obj_normals = normals[np.isin(masks, obj_ids)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points)
        pcd.normals = o3d.utility.Vector3dVector(obj_normals)

        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        obj_points = np.asarray(pcd_downsampled.points)

        return obj_points

    def get_robot_arms_position(self):
        # we assume we know where the robot arms are positioned on the table
        robot_right_position = self.rlbench_env._scene.robot_right_arm.arm.get_position()
        robot_left_position = self.rlbench_env._scene.robot_left_arm.arm.get_position()
        return robot_right_position, robot_left_position

    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        """
        Retrieves the entire scene's 3D point cloud observations and colors.

        Args:
            ignore_robot (bool): Whether to ignore points corresponding to the robot.
            ignore_grasped_obj (bool): Whether to ignore points corresponding to grasped objects.

        Returns:
            tuple: A tuple containing scene points and colors.
        """
        points, colors, masks = [], [], []
        for cam in self.camera_names:
            points.append(getattr(self.latest_obs, f"{cam}_point_cloud").reshape(-1, 3))
            colors.append(getattr(self.latest_obs, f"{cam}_rgb").reshape(-1, 3))
            masks.append(getattr(self.latest_obs, f"{cam}_mask").reshape(-1))
        points = np.concatenate(points, axis=0)
        colors = np.concatenate(colors, axis=0)
        masks = np.concatenate(masks, axis=0)

        # only keep points within workspace
        chosen_idx_x = (points[:, 0] > self.workspace_bounds_min[0]) & (points[:, 0] < self.workspace_bounds_max[0])
        chosen_idx_y = (points[:, 1] > self.workspace_bounds_min[1]) & (points[:, 1] < self.workspace_bounds_max[1])
        chosen_idx_z = (points[:, 2] > self.workspace_bounds_min[2]) & (points[:, 2] < self.workspace_bounds_max[2])
        points = points[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]
        colors = colors[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]
        masks = masks[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]

        if ignore_robot:
            robot_right_mask = np.isin(masks, self.robot_right_mask_ids)
            points = points[~robot_right_mask]
            colors = colors[~robot_right_mask]
            masks = masks[~robot_right_mask]

            robot_left_mask = np.isin(masks, self.robot_left_mask_ids)
            points = points[~robot_left_mask]
            colors = colors[~robot_left_mask]
            masks = masks[~robot_left_mask]
        if self.grasped_obj_ids and ignore_grasped_obj:
            grasped_mask = np.isin(masks, self.grasped_obj_ids)
            points = points[~grasped_mask]
            colors = colors[~grasped_mask]
            masks = masks[~grasped_mask]

        # voxel downsample using o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        points = np.asarray(pcd_downsampled.points)
        colors = np.asarray(pcd_downsampled.colors).astype(np.uint8)

        return points, colors

    def reset(self):
        """
        Resets the environment and the task. Also updates the visualizer.

        Returns:
            tuple: A tuple containing task descriptions and initial observations.
        """
        assert self.task is not None, "Please load a task first"
        self.task.sample_variation()
        descriptions, obs = self.task.reset()
        obs = self._process_obs(obs)
        self.init_obs = obs
        self.latest_obs = obs
        self._update_visualizer()
        self.rlbench_env._scene.vlm.reset_image_name_counter()
        return descriptions, obs

    def apply_action(self, action, which_arm):
        """
        Applies an action in the environment and updates the state.

        Args:
            action: The action to apply.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        assert self.task is not None, "Please load a task first"
        action = self._process_action(action)
        obs, reward, terminate = self.task.step(action, which_arm)
        obs = self._process_obs(obs)
        self.latest_obs = obs
        self.latest_reward = reward
        self.latest_terminate = terminate
        self._update_visualizer()
        if which_arm == 'right hand':
            self.latest_action＿right_arm = action
            grasped_objects = self.rlbench_env._scene.robot_right_arm.gripper.get_grasped_objects()
        elif which_arm == 'left hand':
            self.latest_action＿left_arm = action
            grasped_objects = self.rlbench_env._scene.robot_left_arm.gripper.get_grasped_objects()
        else:
            raise NotImplementedError
        if len(grasped_objects) > 0:
            self.grasped_obj_ids = [obj.get_handle() for obj in grasped_objects]
        return obs, reward, terminate

    def move_to_pose(self, pose, which_arm, speed=None):
        """
        Moves the robot arm to a specific pose.

        Args:
            pose: The target pose.
            speed: The speed at which to move the arm. Currently not implemented.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        if which_arm == 'right hand':
            if self.latest_action_right_arm is None:
                action = np.concatenate([pose, [self.init_obs.gripper_right_open]])
            else:
                action = np.concatenate([pose, [self.latest_action_right_arm[-1]]])
        elif which_arm == 'left hand':
            if self.latest_action_left_arm is None:
                action = np.concatenate([pose, [self.init_obs.gripper_left_open]])
            else:
                action = np.concatenate([pose, [self.latest_action_left_arm[-1]]])
        else:
            raise NotImplementedError
        return self.apply_action(action, which_arm)

    def open_gripper(self, which_arm):
        """
        Opens the gripper of the robot.
        """
        if which_arm == 'right hand':
            action = np.concatenate([self.latest_obs.gripper_right_pose, [1.0]])
        elif which_arm == 'left hand':
            action = np.concatenate([self.latest_obs.gripper_left_pose, [1.0]])
        else:
            raise NotImplementedError
        return self.apply_action(action, which_arm)

    def close_gripper(self, which_arm):
        """
        Closes the gripper of the robot.
        """
        if which_arm == 'right hand':
            action = np.concatenate([self.latest_obs.gripper_right_pose, [0.0]])
        elif which_arm == 'left hand':
            action = np.concatenate([self.latest_obs.gripper_left_pose, [0.0]])
        else:
            raise NotImplementedError
        return self.apply_action(action, which_arm)

    def set_gripper_state(self, gripper_state, which_arm):
        """
        Sets the state of the gripper.

        Args:
            gripper_state: The target state for the gripper.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        if which_arm == 'right hand':
            action = np.concatenate([self.latest_obs.gripper_right_pose, [gripper_state]])
        elif which_arm == 'left hand':
            action = np.concatenate([self.latest_obs.gripper_left_pose, [gripper_state]])
        else:
            raise NotImplementedError
        return self.apply_action(action, which_arm)

    def reset_to_default_pose(self):
        """
        Resets the robot arm to its default pose.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        if self.latest_action_right_arm is None:
            action = np.concatenate([self.init_obs.gripper_right_pose, [self.init_obs.gripper_right_open]])
        else:
            action = np.concatenate([self.init_obs.gripper_right_pose, [self.latest_action_right_arm[-1]]])
        obs = self.apply_action(action)

        if self.latest_action_left_arm is None:
            action = np.concatenate([self.init_obs.gripper_left_pose, [self.init_obs.gripper_left_open]])
        else:
            action = np.concatenate([self.init_obs.gripper_left_pose, [self.latest_action_left_arm[-1]]])
        obs = self.apply_action(action)
        return obs

    def get_ee_pose(self, which_arm):
        assert self.latest_obs is not None, "Please reset the environment first"
        if which_arm == 'right hand':
            return self.latest_obs.gripper_right_pose
        elif which_arm == 'left hand':
            return self.latest_obs.gripper_left_pose
        else:
            raise NotImplementedError

    def get_ee_pos(self, which_arm):
        return self.get_ee_pose(which_arm)[:3]

    def get_ee_quat(self, which_arm):
        return self.get_ee_pose(which_arm)[3:]

    def get_last_gripper_action(self, which_arm):
        """
        Returns the last gripper action.

        Returns:
            float: The last gripper action.
        """
        if which_arm == 'right hand':
            if self.latest_action_right_arm is not None:
                return self.latest_action_right_arm[-1]
            else:
                return self.init_obs.gripper_right_open
        elif which_arm == 'left hand':
            if self.latest_action_left_arm is not None:
                return self.latest_action_left_arm[-1]
            else:
                return self.init_obs.gripper_left_open
        else:
            raise NotImplementedError

    def _reset_task_variables(self):
        """
        Resets variables related to the current task in the environment.

        Note: This function is generally called internally.
        """
        self.init_obs = None
        self.latest_obs = None
        self.latest_reward = None
        self.latest_terminate = None
        self.latest_action_right_arm = None
        self.latest_action_left_arm = None
        self.grasped_obj_ids = None
        # scene-specific helper variables
        self.arm_right_mask_ids = None
        self.gripper_right_mask_ids = None
        self.robot_right_mask_ids = None

        self.arm_left_mask_ids = None
        self.gripper_left_mask_ids = None
        self.robot_left_mask_ids = None

        self.obj_mask_ids = None
        self.name2ids = {}  # first_generation name -> list of ids of the tree
        self.id2name = {}  # any node id -> first_generation name

    def _update_visualizer(self):
        """
        Updates the scene in the visualizer with the latest observations.

        Note: This function is generally called internally.
        """
        if self.visualizer is not None:
            points, colors = self.get_scene_3d_obs(ignore_robot=False, ignore_grasped_obj=False)
            self.visualizer.update_scene_points(points, colors)

    def _process_obs(self, obs):
        """
        Processes the observations, specifically converts quaternion format from xyzw to wxyz.

        Args:
            obs: The observation to process.

        Returns:
            The processed observation.
        """
        quat_xyzw_right = obs.gripper_right_pose[3:]
        quat_wxyz_right = np.concatenate([quat_xyzw_right[-1:], quat_xyzw_right[:-1]])
        obs.gripper_right_pose[3:] = quat_wxyz_right

        quat_xyzw_left = obs.gripper_left_pose[3:]
        quat_wxyz_left = np.concatenate([quat_xyzw_left[-1:], quat_xyzw_left[:-1]])
        obs.gripper_left_pose[3:] = quat_wxyz_left
        return obs

    def _process_action(self, action):
        """
        Processes the action, specifically converts quaternion format from wxyz to xyzw.

        Args:
            action: The action to process.

        Returns:
            The processed action.
        """
        quat_wxyz = action[3:7]
        quat_xyzw = np.concatenate([quat_wxyz[1:], quat_wxyz[:1]])
        action[3:7] = quat_xyzw
        return action