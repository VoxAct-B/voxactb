import gc
import logging
import os
import sys

from typing import List
import hydra
import numpy as np
import torch
import pandas as pd
from omegaconf import DictConfig, OmegaConf, ListConfig
from rlbench import CameraConfig, ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper2Robots
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning2Robots
from rlbench.action_modes.gripper_action_modes import Discrete2Robots
from rlbench.backend import task as rlbench_task
from rlbench.backend.observation_two_robots import Observation2Robots
from rlbench.backend.utils import task_file_to_task_class
from yarr.runners.independent_env_runner import IndependentEnvRunner
from yarr.utils.stat_accumulator import SimpleAccumulator
from yarr.utils.peract_utils import get_new_scene_bounds_based_on_crop
from pyrep.objects import VisionSensor
from rlbench.backend.utils import float_array_to_rgb_image, image_to_float_array
from rlbench.backend.const import *
from rlbench.backend.camera_const import *
from PIL import Image

from agents import c2farm_lingunet_bc
from agents import peract_bc
from agents import arm
from agents.baselines import bc_lang, vit_bc_lang

# from helpers.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv
from helpers.custom_rlbench_env_two_robots import CustomRLBenchEnv2Robots, CustomMultiTaskRLBenchEnv
from helpers import utils, demo_loading_utils

from yarr.utils.rollout_generator import RolloutGenerator
from torch.multiprocessing import Process, Manager

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import time

from urx.robot import Robot
from typing import Dict
from clip import tokenize
from rlbench.backend.vlm_real import VLM
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import pickle
from natsort import natsorted
from rlbench.demo import Demo
import copy
import csv
import pandas as pd


ROBOT_STATE_KEYS = ['joint_velocities_right', 'joint_positions_right', 'joint_forces_right',
                    'joint_velocities_left', 'joint_positions_left', 'joint_forces_left',
                        'gripper_right_open', 'gripper_right_pose',
                        'gripper_right_joint_positions', 'gripper_right_touch_forces',
                        'gripper_left_open', 'gripper_left_pose',
                        'gripper_left_joint_positions', 'gripper_left_touch_forces',
                        'task_low_dim_state', 'misc']

class RealSenseCamera():
    def __init__(self, flip = False):
        import pyrealsense2 as rs

        # get device ids
        ctx = rs.context()
        devices = ctx.query_devices()
        device_ids = []
        for dev in devices:
            dev.hardware_reset()
            device_ids.append(dev.get_info(rs.camera_info.serial_number))
        time.sleep(2)
        device_id = device_ids[0]
        self._device_id = device_id

        if device_id is None:
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                dev.hardware_reset()
            time.sleep(2)
            self._pipeline = rs.pipeline()
            config = rs.config()
        else:
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(device_id)

        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipeline_cfg = self._pipeline.start(config)
        self._flip = flip

        # get camera intrinsics: https://github.com/IntelRealSense/librealsense/issues/869
        profile = pipeline_cfg.get_stream(rs.stream.depth)                              # Fetch stream profile for depth stream
        self._camera_intrinsics = profile.as_video_stream_profile().get_intrinsics()    # Downcast to video_stream_profile and fetch intrinsics

    def pad_image(self, image, mode='constant'):
        """
        Copied from RLBench/tools/convert_gello_demo_to_peract.py
        """
        # Use width as the desired output shape
        output_shape = (image.shape[1], image.shape[1], image.shape[2])

        # Calculate the padding sizes for the first two dimensions
        padding_top = (output_shape[0] - image.shape[0]) // 2
        padding_bottom = output_shape[0] - image.shape[0] - padding_top
        padding_left = (output_shape[1] - image.shape[1]) // 2
        padding_right = output_shape[1] - image.shape[1] - padding_left

        # Apply the padding
        if mode == 'constant':
            padded_image = np.pad(image, 
                                    ((padding_top, padding_bottom), 
                                    (padding_left, padding_right), 
                                    (0, 0)), 
                                    mode=mode, constant_values=0)
        else:
            padded_image = np.pad(image, 
                                ((padding_top, padding_bottom), 
                                (padding_left, padding_right), 
                                (0, 0)), 
                                mode=mode)
        return padded_image

    def get_intrinsics(self):
        return np.array([
            [self._camera_intrinsics.fx, 0, self._camera_intrinsics.ppx],
            [0, self._camera_intrinsics.fy, self._camera_intrinsics.ppy],
            [0, 0, 1],
        ])

    def read_rgb_depth_raw(self, img_size = None):
        import cv2
        import pyrealsense2 as rs

        frames = self._pipeline.wait_for_frames()
        # these two lines make sure rgb and depth images are aligned: https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/distance_to_object.ipynb
        align = rs.align(rs.stream.color)
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        # depth_image = cv2.convertScaleAbs(depth_image, alpha=0.03)
        if img_size is None:
            image = color_image[:, :, ::-1]
            depth = depth_image
        else:
            image = cv2.resize(color_image, img_size)[:, :, ::-1]
            depth = cv2.resize(depth_image, img_size)

        # rotate 180 degree's because everything is upside down in order to center the camera
        if self._flip:
            image = cv2.rotate(image, cv2.ROTATE_180)
            depth = cv2.rotate(depth, cv2.ROTATE_180)[:, :, None]
        else:
            depth = depth[:, :, None]
        return image, depth

    def convert_depth_pixels_to_pcd(self, depth_image):
        height, width, _ = depth_image.shape
        fx = self._camera_intrinsics.fx
        fy = self._camera_intrinsics.fy
        cx = self._camera_intrinsics.ppx
        cy = self._camera_intrinsics.ppy

        # numpy version, based on this implementation: https://stackoverflow.com/questions/62871232/speeding-up-depth-image-to-point-cloud-conversion-python
        grid = np.mgrid[0:height, 0:width]
        v, u = grid[0], grid[1]
        z = depth_image[:, :, 0] / 1000.0 # Convert depth to meters
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points = np.stack([x,y,z], axis=-1)

        return points

    def read(
        self,
        img_size = None,  # farthest: float = 0.12
    ):
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.
            farthest: The farthest distance to map to 255.

        Returns:
            np.ndarray: The color image, shape=(H, W, 3)
            np.ndarray: The depth image, shape=(H, W, 1)
        """
        import cv2
        import pyrealsense2 as rs

        frames = self._pipeline.wait_for_frames()
        # these two lines make sure rgb and depth images are aligned: https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/distance_to_object.ipynb
        align = rs.align(rs.stream.color)
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        # depth_image = cv2.convertScaleAbs(depth_image, alpha=0.03)
        if img_size is None:
            image = color_image[:, :, ::-1]
            depth = depth_image
        else:
            image = cv2.resize(color_image, img_size)[:, :, ::-1]
            depth = cv2.resize(depth_image, img_size)

        # rotate 180 degree's because everything is upside down in order to center the camera
        if self._flip:
            image = cv2.rotate(image, cv2.ROTATE_180)
            depth = cv2.rotate(depth, cv2.ROTATE_180)[:, :, None]
        else:
            depth = depth[:, :, None]

        # process rgb and depth images
        front_rgb = self.pad_image(image, mode='constant')
        front_depth = self.pad_image(depth, mode='edge')[:, :, 0]
        front_depth = front_depth / 1000.0 # convert to meters
        front_depth = float_array_to_rgb_image(front_depth, scale_factor=DEPTH_SCALE)
        front_depth = image_to_float_array(front_depth, DEPTH_SCALE)
        
        front_camera_intrinsics = self.get_intrinsics()

        # project point cloud to camera frame
        identity_extrinsics_matrix = np.eye(4)
        front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
            front_depth,
            identity_extrinsics_matrix,
            front_camera_intrinsics)

        # for debugging: make sure point cloud (depth) and RGB look correct
        # # Reshape the data to a 2D array of points (N x 3)
        # point_cloud_data = front_point_cloud.reshape(-1, 3)
        # # Create an Open3D PointCloud object
        # pcd = o3d.geometry.PointCloud()
        # # Assign the points to the PointCloud object
        # pcd.points = o3d.utility.Vector3dVector(point_cloud_data)
        # # here, we want to figure out the scene bounds
        # # NOTE: we can actually these values to determine whether the camera has been moved
        # print('Min x: ', np.min(point_cloud_data[:, 0]))
        # print('Max x: ', np.max(point_cloud_data[:, 0]))
        # print('Range x: ', np.max(point_cloud_data[:, 0]) - np.min(point_cloud_data[:, 0]))
        # print('Min y: ', np.min(point_cloud_data[:, 1]))
        # print('Max y: ', np.max(point_cloud_data[:, 1]))
        # print('Range y: ', np.max(point_cloud_data[:, 1]) - np.min(point_cloud_data[:, 1]))
        # print('Min z: ', np.min(point_cloud_data[:, 2]))
        # print('Max z: ', np.max(point_cloud_data[:, 2]))
        # print('Range z: ', np.max(point_cloud_data[:, 2]) - np.min(point_cloud_data[:, 2]))
        # front_rgb = front_rgb.reshape(-1, 3)
        # front_rgb = front_rgb.astype(np.float) / 255.0
        # pcd.colors = o3d.utility.Vector3dVector(front_rgb)
        # # Visualize the point cloud
        # o3d.visualization.draw_geometries([pcd])

        return front_rgb, front_depth, front_point_cloud


class UR5Robot():
    """
    Mostly copied from GELLO's ur_cb2.py
    """
    def __init__(self, robot_ip):
        self.robot = Robot(robot_ip)
        self._use_gripper = True
        self.prev_time = time.time()
        self.prev_joints = self.get_joint_state()
        self._first_obs = True
        if self._use_gripper:
            self.robot.activate_gripper()

    def _get_gripper_pos(self) -> float:
        if self._use_gripper:
            time.sleep(0.01)
            gripper_pos = self.robot.get_gripper_position()
            assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
            return gripper_pos / 255
        return None
    
    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        robot_joints = self.robot.getj()
        if self._use_gripper:
            gripper_pos = self._get_gripper_pos()
            pos = np.append(robot_joints, gripper_pos)
        else:
            pos = np.array(robot_joints)
        return pos
    
    def get_quaternion_from_euler(self, roll, pitch, yaw):
        """
        Convert an Euler angle to a quaternion.

        Input
        :param roll: The roll (rotation around x-axis) angle in radians.
        :param pitch: The pitch (rotation around y-axis) angle in radians.
        :param yaw: The yaw (rotation around z-axis) angle in radians.

        Output
        :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
        """
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    
        return [qx, qy, qz, qw]
    
    def get_observations(self) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        tip_pos = self.robot.getl()[:3]
        tip_orientation = self.robot.get_orientation()
        euler_angle = tip_orientation.to_euler('ZYX') # roll-pitch-yaw
        quat = self.get_quaternion_from_euler(euler_angle[0], euler_angle[1], euler_angle[2])
        pos_quat = np.array([tip_pos[0], tip_pos[1], tip_pos[2], quat[0], quat[1], quat[2], quat[3]]) # x, y, z, qx, qy, qz, qw
        gripper_pos = np.array([joints[-1]])
        curr_time = time.time()

        if not self._first_obs:
            # compute joint velocities
            dt_joints = joints - self.prev_joints
            dt_time = curr_time - self.prev_time
            joint_velocities = dt_joints / dt_time
            self.prev_joints = joints
            self.prev_time = curr_time

            # for debugging
            # if self.is_left_arm:
            #     arm = 'Left'
            #     # print(f'{arm} joint velocities: ', joint_velocities) 
            #     is_joint_velocities_zero = np.allclose(joint_velocities, 0, atol=0.1)
            #     if is_joint_velocities_zero:
            #         print(f'{arm} joint velocities are zero!!')
            # else:
            #     arm = 'Right'
            #     # print(f'{arm} joint velocities: ', joint_velocities) 
            #     # is_joint_velocities_zero = np.allclose(joint_velocities, 0, atol=0.1)
            #     # if is_joint_velocities_zero:
            #     #     print(f'{arm} joint velocities are zero!!')
        else:
            joint_velocities = joints - joints
            self.prev_joints = joints
            self.prev_time = curr_time
        self._first_obs = False

        return {
            "joint_positions": joints,
            "joint_velocities": joint_velocities,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
        }

class CB2BimanualRobot():
    """
    Based on GELLO's robot.py
    """
    def __init__(self, robot_l, robot_r):
        self._robot_l = UR5Robot(robot_l)
        self._robot_r = UR5Robot(robot_r)
        self.starting_robot_joints_left = [1.3780459778509433, -1.523749890442781, -1.5899893346228173, -1.523211104456275, 1.722199931031265, -0.20266514321065365]
        self.starting_robot_joints_right = [-1.6545815648737472, -1.6381940802802397, 1.795800360316563, -1.7138684258221923, -1.7362808437379504, -0.01120622072583366]

    def get_joint_state(self) -> np.ndarray:
        return np.concatenate(
            (self._robot_l.get_joint_state(), self._robot_r.get_joint_state())
        )

    def get_observations(self) -> Dict[str, np.ndarray]:
        l_obs = self._robot_l.get_observations()
        r_obs = self._robot_r.get_observations()
        assert l_obs.keys() == r_obs.keys()
        return_obs = {}
        for k in l_obs.keys():
            try:
                return_obs[k] = np.concatenate((l_obs[k], r_obs[k]))
            except Exception as e:
                print(e)
                print(k)
                print(l_obs[k])
                print(r_obs[k])
                raise RuntimeError()

        return return_obs
    
    def move_robots_to_starting_states(self):
        for i in range(50):
            self._robot_l.robot.servoj(
                self.starting_robot_joints_left, vel=0.1, acc=0.3, t=0.35, lookahead_time=0.2, gain=100, wait=False
            )
        time.sleep(1)
        print('Finished moving the left arm to starting states.')

        for i in range(50):
            self._robot_r.robot.servoj(
                self.starting_robot_joints_right, vel=0.1, acc=0.3, t=0.35, lookahead_time=0.2, gain=100, wait=False
            )
        time.sleep(1)
        print('Finished moving the right arm to starting states.')

class RobotEnv():
    def __init__(self, train_cfg, eval_cfg):
        self._train_cfg = train_cfg
        self.robot = None

        # initialize camera
        self.camera = RealSenseCamera()
        self.camera_intrinsics = self.camera.get_intrinsics()

        # initialize VLM
        self.vlm = VLM()

        self._time_in_state = eval_cfg.rlbench.time_in_state
        self._episode_length = eval_cfg.rlbench.episode_length
        self._timesteps = 1
        self._i = 0

        self.task_name = eval_cfg.rlbench.tasks[0]
        # NOTE: here we assume each task has fixed arms
        # if self.task_name == 'open_drawer':
        #     self._lang_goal = 'hold the drawer with left hand and open the top drawer with right hand'
        # elif self.task_name == 'open_jar':
        #     self._lang_goal = 'grasp the jar with right hand and grasp the lid of the jar with left hand to unscrew it in an anti_clockwise direction until it is removed from the jar'
        # else:
        #     raise NotImplementedError
        self._lang_goal = None
        self._acting_arm = None

    def initialize_robot(self, robot_left_ip, robot_right_ip):
        self.robot = CB2BimanualRobot(robot_left_ip, robot_right_ip)

    def _extract_obs(self, obs: Observation2Robots):
        """
        Copied from YARR/yarr/envs/rlbench_env_two_robots.py extract_obs and _extract_obs functions
        """
        obs_dict = vars(obs)
        obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
        robot_state_right = obs.get_low_dim_data(which_arm='right')
        robot_state_left = obs.get_low_dim_data(which_arm='left')
        # Remove all of the individual state elements
        obs_dict = {k: v for k, v in obs_dict.items()
                    if k not in ROBOT_STATE_KEYS}

        # Swap channels from last dim to 1st dim
        obs_dict = {k: np.transpose(
            v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
                    for k, v in obs_dict.items()}

        obs_dict['low_dim_state_right_arm'] = np.array(robot_state_right, dtype=np.float32)
        obs_dict['low_dim_state_left_arm'] = np.array(robot_state_left, dtype=np.float32)
        obs_dict['ignore_collisions'] = np.array([obs.ignore_collisions], dtype=np.float32)
        for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
            obs_dict[k] = v.astype(np.float32)

        extracted_obs = obs_dict
     
        if 'descriptions' in obs.misc:
            self._lang_goal = obs.misc['descriptions'][0]
        else:
            acting_arm = self.get_dominant_arm()
            if self.task_name == 'open_jar':
                if acting_arm == 'left':
                    self._lang_goal = 'grasp the jar with right hand and grasp the lid of the jar with left hand to unscrew it in an anti_clockwise direction until it is removed from the jar'
                else:
                    self._lang_goal = 'grasp the jar with left hand and grasp the lid of the jar with right hand to unscrew it in an anti_clockwise direction until it is removed from the jar'
            else:
                raise NotImplementedError
        assert self._lang_goal is not None
        print('Language goal: ', self._lang_goal)

        if self._train_cfg.method.which_arm == 'multiarm':
            left_arm_description, right_arm_description = utils.extract_left_and_right_arm_instruction(self._lang_goal)
            extracted_obs['lang_goal_tokens_left'] = tokenize([left_arm_description])[0].numpy()
            extracted_obs['lang_goal_tokens_right'] = tokenize([right_arm_description])[0].numpy()
        else:
            extracted_obs['lang_goal_tokens'] = tokenize([self._lang_goal])[0].numpy()
        return extracted_obs

    def extract_obs(self, obs, t=None, prev_action=None):
        """
        Copied from peract/helpers/custom_rlbench_env_two_robots.py
        """
        obs.joint_velocities_right = None
        grip_right_mat= obs.gripper_right_matrix
        grip_right_pose = obs.gripper_right_pose
        joint_pos_right = obs.joint_positions_right
        obs.gripper_right_pose = None
        obs.gripper_right_matrix = None
        obs.wrist_camera_matrix = None
        obs.joint_positions_right = None
        if obs.gripper_right_joint_positions is not None:
            obs.gripper_right_joint_positions = np.clip(
                obs.gripper_right_joint_positions, 0., 0.04)

        obs.joint_velocities_left = None
        grip_left_mat = obs.gripper_left_matrix
        grip_left_pose = obs.gripper_left_pose
        joint_pos_left = obs.joint_positions_left
        obs.gripper_left_pose = None
        obs.gripper_left_matrix = None
        obs.wrist2_camera_matrix = None
        obs.joint_positions_left = None
        if obs.gripper_left_joint_positions is not None:
            obs.gripper_left_joint_positions = np.clip(
                obs.gripper_left_joint_positions, 0., 0.04)

        obs_dict = self._extract_obs(obs)

        if self._train_cfg.method.arm_pred_input:
            # arm ID is defined in peract/helpers/demo_loading_utils.py
            obs_dict['low_dim_state_right_arm'] = np.concatenate(
                [obs_dict['low_dim_state_right_arm'], [0]]).astype(np.float32)
            obs_dict['low_dim_state_left_arm'] = np.concatenate(
                [obs_dict['low_dim_state_left_arm'], [1]]).astype(np.float32)
        elif self._train_cfg.method.arm_id_to_proprio:
            if self._time_in_state:
                time = (1. - ((self._i if t is None else t) / float(
                    self._episode_length - 1))) * 2. - 1.
                obs_dict['low_dim_state_right_arm'] = np.concatenate(
                    [obs_dict['low_dim_state_right_arm'], [time], [0]]).astype(np.float32)
                obs_dict['low_dim_state_left_arm'] = np.concatenate(
                    [obs_dict['low_dim_state_left_arm'], [time], [1]]).astype(np.float32)
        else:
            if self._time_in_state:
                time = (1. - ((self._i if t is None else t) / float(
                    self._episode_length - 1))) * 2. - 1.
                obs_dict['low_dim_state_right_arm'] = np.concatenate(
                    [obs_dict['low_dim_state_right_arm'], [time]]).astype(np.float32)
                obs_dict['low_dim_state_left_arm'] = np.concatenate(
                    [obs_dict['low_dim_state_left_arm'], [time]]).astype(np.float32)

        obs.gripper_right_matrix = grip_right_mat
        obs.joint_positions_right = joint_pos_right
        obs.gripper_right_pose = grip_right_pose

        obs.gripper_left_matrix = grip_left_mat
        obs.joint_positions_left = joint_pos_left
        obs.gripper_left_pose = grip_left_pose
        return obs_dict
    
    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def get_observation(self):
        """
        Based on peract/helpers/custom_rlbench_env_two_robots.py get_observation() and YARR/yarr/utils/rollout_generator.py
        """
        front_rgb, front_depth, front_pcd = self.camera.read()
        robot_obs = self.robot.get_observations()
        
        misc_obj = {
            'front_camera_extrinsics': LEFT_ARM_EXTRINSICS, # not used; we're actually not going to use this in our method because we have two transformation matrices (left and right)
            'front_camera_intrinsics': self.camera_intrinsics,
            'front_camera_near': -1, # not used
            'front_camera_far': -1, # not used
            'left_arm_extrinsics': LEFT_ARM_EXTRINSICS,
            'right_arm_extrinsics': RIGHT_ARM_EXTRINSICS,
        }

        obs = Observation2Robots(
            wrist_rgb=None,
            wrist_depth=None,
            wrist_point_cloud=None,
            wrist2_rgb=None,
            wrist2_depth=None,
            wrist2_point_cloud=None,
            front_rgb=front_rgb,
            front_depth=front_depth,
            front_point_cloud=front_pcd,
            wrist_mask=None,
            wrist2_mask=None,
            front_mask=None,
            joint_velocities_right=robot_obs['joint_velocities'][7:],
            joint_positions_right=robot_obs['joint_positions'][7:],
            joint_forces_right=None,
            joint_velocities_left=robot_obs['joint_velocities'][:7],
            joint_positions_left=robot_obs['joint_positions'][:7],
            joint_forces_left=None,
            gripper_right_open=robot_obs['gripper_position'][1],
            gripper_right_pose=robot_obs['ee_pos_quat'][7:],
            gripper_right_matrix=None,
            gripper_right_joint_positions=np.array([robot_obs['gripper_position'][1], robot_obs['gripper_position'][1]]),
            gripper_right_touch_forces=None,
            gripper_left_open=robot_obs['gripper_position'][0],
            gripper_left_pose=robot_obs['ee_pos_quat'][:7],
            gripper_left_matrix=None,
            gripper_left_joint_positions=np.array([robot_obs['gripper_position'][0], robot_obs['gripper_position'][0]]),
            gripper_left_touch_forces=None,
            task_low_dim_state=None,
            ignore_collisions=np.array([1.0]),
            misc=misc_obj,
            target_object_pos=None,
            auto_crop_radius=None)
    
        obs = self.extract_obs(obs)
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * self._timesteps for k, v in obs.items()}

        info = {
            'front_pcd': front_pcd,
            'front_rgb': front_rgb,
            'front_depth': front_depth,
            'misc_obj': misc_obj
        }
        return obs_history, info
    
    def get_new_scene_bounds_using_vlm(self):
        # get point cloud data based on the code in GELLO
        front_rgb, front_depth = self.camera.read_rgb_depth_raw()
        points = self.camera.convert_depth_pixels_to_pcd(front_depth)

        # get new_scene_bounds in camera coordinate frame
        target_object_pos_cam_coordinates = self.vlm.get_target_object_world_coords(front_rgb, points, self.task_name, debug=True)

        # set self._acting_arm based on task-specific rules 
        if self.task_name == 'open_jar':
            """
            jar's position on the line
            jar on the right: array([0.1277224 , 0.07764685, 0.764     ])
            jar in the middle: array([0.07622363, 0.07991821, 0.73      ])
            jar on the left: array([0.03164772, 0.08247737, 0.703     ])
            """
            if target_object_pos_cam_coordinates[0] >= 0.076:
                self._acting_arm = 'right'
            else:
                self._acting_arm = 'left'
        else:
            raise NotImplementedError

        new_scene_bounds = get_new_scene_bounds_based_on_crop(self._train_cfg.method.crop_radius, target_object_pos_cam_coordinates)
        print('new_scene_bounds: ', new_scene_bounds)
        return new_scene_bounds

    def get_dominant_arm(self):
        # NOTE: here we assume each task has fixed arms
        # if self.task_name == 'open_drawer':
        #     return 'right'
        # elif self.task_name == 'open_jar':
        #     return 'left'

        if self._acting_arm is not None:
            return self._acting_arm
        else:
            raise NotImplementedError

    def detect_gimbal_lock_from_quaternion(self, quaternion, seq='xyz'):
        """
        Detect gimbal lock from a quaternion.
        
        :param quaternion: Quaternion [x, y, z, w]
        :param seq: The sequence of Euler angles, default is 'xyz'
        :return: Boolean indicating whether gimbal lock is detected
        """
        # Convert quaternion to Euler angles
        r = R.from_quat(quaternion)
        euler_angles = r.as_euler(seq)
        
        # Check for gimbal lock condition
        tolerance = 1e-6  # Tolerance for detecting critical angle
        gimbal_lock_detected = False
        
        if seq == 'xyz' or seq == 'zyx':
            pitch_angle = euler_angles[1]  # Middle angle in XYZ or ZYX sequence
            if np.isclose(np.abs(pitch_angle), np.pi / 2, atol=tolerance):
                gimbal_lock_detected = True
        
        # Add checks for other sequences if needed

        return gimbal_lock_detected

    def handle_gimbal_lock(self, quaternion, seq='xyz'):
        """
        Handle gimbal lock by slightly perturbing the quaternion.
        
        :param quaternion: Quaternion [x, y, z, w]
        :param seq: The sequence of Euler angles, default is 'xyz'
        :return: Adjusted quaternion to avoid gimbal lock
        """
        if self.detect_gimbal_lock_from_quaternion(quaternion, seq):
            print("Gimbal lock detected, adjusting quaternion...")
            # Apply a small perturbation to the quaternion
            perturbation = 0.001
            quaternion = [q + perturbation for q in quaternion]
            # Normalize the quaternion
            norm = np.linalg.norm(quaternion)
            quaternion = [q / norm for q in quaternion]
        
        return quaternion

    def move_robot(self, act_which_arm, position, quat, gripper_close, info, extrinsics):
        if act_which_arm == 'left':
            robot = self.robot._robot_l.robot
        else:
            robot = self.robot._robot_r.robot

        # quat = self.handle_gimbal_lock(quat, seq='xyz')
        
        # Convert the quaternion to a rotation object
        rotation = R.from_quat(quat)

        # Convert the rotation object to a rotation vector
        rotation_vector = rotation.as_rotvec()

        # Extract rx, ry, rz
        rx, ry, rz = rotation_vector

        visualize_predictions(info, position, quat, extrinsics, robot)

        # notice the rotations are in this order: rz, ry, rx
        pose = [position[0], position[1], position[2], rz, ry, rx]
        current_pose = robot.getl()
        print(f'Current {act_which_arm} TCP: ', current_pose)
        print(f'Target {act_which_arm} TCP: ', pose)
        print(f'Delta TCP: ', pose - current_pose)
        print('Gripper prediction: ', gripper_close)

        key = input("Press c to execute or s to skip this action or anything else to exit:\n")
        if key == 'c':
            if self.task_name == 'open_jar' and ((act_which_arm == 'left' and self._acting_arm == 'left') or (act_which_arm == 'right' and self._acting_arm == 'right')):
                # execute gripper action first and then movel
                if gripper_close == 1:
                    try:
                        # move to postion
                        pose_posonly = current_pose
                        pose_posonly[0] = pose[0]
                        pose_posonly[1] = pose[1]
                        pose_posonly[2] = pose[2]
                        robot.movel(pose_posonly, acc=0.2, vel=0.05)
                    except Exception as error:
                        print(f'{error}, but we will keep going...')
                    
                    # close gripper
                    print('close gripper')
                    robot.set_digital_out(8, True)
                    robot.set_digital_out(9, False)
                    time.sleep(0.05)

                    try:
                        # rotate the gripper
                        robot.movel(pose, acc=0.2, vel=0.05)
                    except Exception as error:
                        print(f'{error}, but we will keep going...')
                else:
                    # moves up and then open the dripper
                    try:
                        robot.movel(pose, acc=0.2, vel=0.05)
                    except Exception as error:
                        print(f'{error}, but we will keep going...')

                    print('open gripper')
                    robot.set_digital_out(8, False)
                    robot.set_digital_out(9, False)
                    time.sleep(0.05)                
            else:
                try:
                    robot.movel(pose, acc=0.2, vel=0.05)
                except Exception as error:
                    print(f'{error}, but we will keep going...')
                if gripper_close == 1:
                    # close gripper
                    print('close gripper')
                    robot.set_digital_out(8, True)
                    robot.set_digital_out(9, False)
                    time.sleep(0.05)
                else:
                    print('open gripper')
                    robot.set_digital_out(8, False)
                    robot.set_digital_out(9, False)
                    time.sleep(0.05)
        elif key == 's':
            print(f'Skip this action for {act_which_arm} arm.')
        else:
            gc.collect()
            sys.exit(0)

    def move_robot_to_starting_states(self):
        self.robot.move_robots_to_starting_states()


def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

def visualize_predictions(info, position, quat, extrinsics, robot):
    front_rgb = info['front_rgb']
    front_depth = info['front_depth']
    front_camera_intrinsics = info['misc_obj']['front_camera_intrinsics']
    
    # project point cloud to the appropriate robot base frame
    front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
        front_depth,
        extrinsics,
        front_camera_intrinsics)

    # Reshape the data to a 2D array of points (N x 3)
    point_cloud_data = front_point_cloud.reshape(-1, 3)
    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    # Assign the points to the PointCloud object
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data)

    front_rgb = front_rgb.reshape(-1, 3)
    front_rgb = front_rgb.astype(np.float) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(front_rgb)

    # Convert quaternion to rotation matrix
    r_pred = R.from_quat(quat)
    rotation_matrix_pred = r_pred.as_matrix()

    # Create a 4x4 transformation matrix
    transformation_matrix_pred = np.eye(4)
    transformation_matrix_pred[:3, :3] = rotation_matrix_pred
    transformation_matrix_pred[:3, 3] = position[:3]

    # add offset to z-axis to account for the 2F-85 gripper
    transformation_matrix_pred[2, 3] += -0.185

    # Create a coordinate frame
    coordinate_frame_pred = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    # Apply the transformation to the coordinate frame
    coordinate_frame_pred.transform(transformation_matrix_pred)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd, coordinate_frame_pred])


    # for debugging: visualize current robot gripper's orientation and predicted gripper's orientation
    # r_pred = R.from_quat(quat)
    # rotation_matrix_pred = r_pred.as_matrix()

    # # Create a 4x4 transformation matrix
    # transformation_matrix_pred = np.eye(4)
    # transformation_matrix_pred[:3, :3] = rotation_matrix_pred
    # transformation_matrix_pred[:3, 3] = position[:3]

    # # add offset to z-axis to account for the 2F-85 gripper
    # transformation_matrix_pred[2, 3] += -0.185

    # # Create a coordinate frame
    # coordinate_frame_pred = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    # # Apply the transformation to the coordinate frame
    # coordinate_frame_pred.transform(transformation_matrix_pred)

    # tip_orientation = robot.get_orientation()
    # euler_angle = tip_orientation.to_euler('ZYX') # roll-pitch-yaw
    # robot_quat = get_quaternion_from_euler(euler_angle[0], euler_angle[1], euler_angle[2])
    # r_robot = R.from_quat(robot_quat)
    # rotation_matrix_robot = r_robot.as_matrix()

    # # Create a 4x4 transformation matrix
    # transformation_matrix_robot = np.eye(4)
    # transformation_matrix_robot[:3, :3] = rotation_matrix_robot
    # transformation_matrix_robot[:3, 3] = position[:3]

    # # add offset to z-axis to account for the 2F-85 gripper
    # transformation_matrix_robot[2, 3] += -0.185

    # # Create a coordinate frame
    # coordinate_frame_robot = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

    # # Apply the transformation to the coordinate frame
    # coordinate_frame_robot.transform(transformation_matrix_robot)

    # # Visualize the point cloud
    # o3d.visualization.draw_geometries([pcd, coordinate_frame_pred, coordinate_frame_robot])


def update_low_dim_time(env, episode_keypoints):
    for i in range(episode_keypoints.shape[0]):
        time = (1. - ((i) / float(env._episode_length - 1))) * 2. - 1.
        low_dim = episode_keypoints[i]['obs_history']['low_dim_state_right_arm'][0]
        episode_keypoints[i]['obs_history']['low_dim_state_right_arm'] = [np.array([low_dim[0], low_dim[1], low_dim[2], time]).astype(np.float32)]
        low_dim = episode_keypoints[i]['obs_history']['low_dim_state_left_arm'][0]
        episode_keypoints[i]['obs_history']['low_dim_state_left_arm'] = [np.array([low_dim[0], low_dim[1], low_dim[2], time]).astype(np.float32)]

def get_observations_from_val(env, demo_path, task_name, train_cfg):
    task_root = os.path.join(demo_path, task_name)

    # All variations
    examples_path = os.path.join(
        task_root, VARIATIONS_ALL_FOLDER,
        EPISODES_FOLDER)
    examples = os.listdir(examples_path)

    # hack: ignore .DS_Store files from macOS zips
    examples = [e for e in examples if '.DS_Store' not in e]
    amount = len(examples)
    selected_examples = natsorted(examples)

    episodes_acting_arm_input, episodes_stabilizing_arm_input = [], []
    episodes_acting_arm_gt_action, episodes_stabilizing_arm_gt_action = [], []
    for example in selected_examples:
        example_path = os.path.join(examples_path, example)
        print('Example path: ', example_path)

        with open(os.path.join(example_path, LOW_DIM_PICKLE), 'rb') as f:
            obs = pickle.load(f)

        episode_descriptions_f = os.path.join(example_path, VARIATION_DESCRIPTIONS)
        if os.path.exists(episode_descriptions_f):
            with open(episode_descriptions_f, 'rb') as f:
                descriptions = pickle.load(f)
        
        front_rgb_f = os.path.join(example_path, FRONT_RGB_FOLDER)
        front_depth_f = os.path.join(example_path, FRONT_DEPTH_FOLDER)

        num_steps = len(obs)
        observations = []
        observations_only = []
        for i in range(num_steps):
            obs[i].misc['descriptions'] = descriptions
            si = IMAGE_FORMAT % i
            obs[i].front_rgb = os.path.join(front_rgb_f, si)
            obs[i].front_depth = os.path.join(front_depth_f, si)

        for i in range(num_steps):
            obs[i].front_rgb = np.array(Image.open(obs[i].front_rgb))
            obs[i].front_depth = image_to_float_array(Image.open(obs[i].front_depth), DEPTH_SCALE)
            identity_extrinsics_matrix = np.eye(4)
            obs[i].front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                                    obs[i].front_depth,
                                    identity_extrinsics_matrix,
                                    obs[i].misc['front_camera_intrinsics'])
            
            # project left gripper position to the camera_color_optical_frame using the inverse of left arm's extrinsics
            t_left_base_frame_to_camera_frame = np.linalg.inv(obs[i].misc['left_arm_extrinsics'])
            gripper_left_pose_base_frame = np.array([obs[i].gripper_left_pose[0], obs[i].gripper_left_pose[1], obs[i].gripper_left_pose[2], 1])
            gripper_left_pose_cam_frame = np.matmul(t_left_base_frame_to_camera_frame, gripper_left_pose_base_frame)
            obs[i].gripper_left_pose[0] = gripper_left_pose_cam_frame[0]
            obs[i].gripper_left_pose[1] = gripper_left_pose_cam_frame[1]
            obs[i].gripper_left_pose[2] = gripper_left_pose_cam_frame[2]

            # project right gripper position to the camera_color_optical_frame using the inverse of right arm's extrinsics
            t_right_base_frame_to_camera_frame = np.linalg.inv(obs[i].misc['right_arm_extrinsics'])
            gripper_right_pose_base_frame = np.array([obs[i].gripper_right_pose[0], obs[i].gripper_right_pose[1], obs[i].gripper_right_pose[2], 1])
            gripper_right_pose_cam_frame = np.matmul(t_right_base_frame_to_camera_frame, gripper_right_pose_base_frame)
            obs[i].gripper_right_pose[0] = gripper_right_pose_cam_frame[0]
            obs[i].gripper_right_pose[1] = gripper_right_pose_cam_frame[1]
            obs[i].gripper_right_pose[2] = gripper_right_pose_cam_frame[2]

            formated_obs = Observation2Robots(
                wrist_rgb=None,
                wrist_depth=None,
                wrist_point_cloud=None,
                wrist2_rgb=None,
                wrist2_depth=None,
                wrist2_point_cloud=None,
                front_rgb=obs[i].front_rgb,
                front_depth=obs[i].front_depth,
                front_point_cloud=obs[i].front_point_cloud,
                wrist_mask=None,
                wrist2_mask=None,
                front_mask=None,
                joint_velocities_right=obs[i].joint_velocities_right,
                joint_positions_right=obs[i].joint_positions_right,
                joint_forces_right=None,
                joint_velocities_left=obs[i].joint_velocities_left,
                joint_positions_left=obs[i].joint_positions_left,
                joint_forces_left=None,
                gripper_right_open=obs[i].gripper_right_open[0],
                gripper_right_pose=obs[i].gripper_right_pose,
                gripper_right_matrix=None,
                gripper_right_joint_positions=obs[i].gripper_right_joint_positions,
                gripper_right_touch_forces=None,
                gripper_left_open=obs[i].gripper_left_open[0],
                gripper_left_pose=obs[i].gripper_left_pose,
                gripper_left_matrix=None,
                gripper_left_joint_positions=obs[i].gripper_left_joint_positions,
                gripper_left_touch_forces=None,
                task_low_dim_state=None,
                ignore_collisions=np.array([obs[i].ignore_collisions]),
                misc=obs[i].misc,
                target_object_pos=None,
                auto_crop_radius=None)
        
            formated_obs_copy = copy.deepcopy(formated_obs)
            extracted_obs = env.extract_obs(formated_obs)
            obs_history = {k: [np.array(v, dtype=env._get_type(v))] * env._timesteps for k, v in extracted_obs.items()}

            dict_obj = {
                'obs_history': obs_history,
                'formated_obs': formated_obs_copy,
                'target_object_pos': obs[i].target_object_pos,
            }
            observations.append(dict_obj)
            observations_only.append(formated_obs_copy)

        demo = Demo(observations_only)


        if env.task_name == 'open_jar':
            acting_arm = obs[i].misc['descriptions'][0].split('lid of the jar with ')[-1].split(' hand to unscrew it')[0]
        else:
            raise NotImplementedError

        # acting_arm = env.get_dominant_arm()
        if acting_arm == 'right':
            stabilize_arm = 'left'
        else:
            stabilize_arm =  'right'
        print(f'Acting: {acting_arm}, stabilizing: {stabilize_arm}')

        # acting arms' keyframes
        episode_keypoints_acting, _ = demo_loading_utils.keypoint_discovery_no_duplicate(demo, which_arm='dominant', method='heuristic', saved_every_last_inserted=0, dominant_assistive_arm=acting_arm, use_default_stopped_buffer_timesteps=False, stopped_buffer_timesteps_overwrite=train_cfg.method.stopped_buffer_timesteps_overwrite)
        episode_keypoints_acting_lastremoved = episode_keypoints_acting[:-1]
        episode_keypoints_acting_input_ep = copy.deepcopy(np.array(observations)[episode_keypoints_acting_lastremoved])
        update_low_dim_time(env, episode_keypoints_acting_input_ep)
        episodes_acting_arm_input.append(episode_keypoints_acting_input_ep)

        episode_keypoints_acting_actions = episode_keypoints_acting[1:]
        episodes_acting_arm_gt_action_ep = copy.deepcopy(np.array(observations)[episode_keypoints_acting_actions])
        update_low_dim_time(env, episodes_acting_arm_gt_action_ep)
        episodes_acting_arm_gt_action.append(episodes_acting_arm_gt_action_ep)

        # stabilizing arms' keyframes
        episode_keypoints_stabilizing, _ = demo_loading_utils.keypoint_discovery_no_duplicate(demo, which_arm='assistive', method='heuristic', saved_every_last_inserted=0, dominant_assistive_arm=stabilize_arm, use_default_stopped_buffer_timesteps=False, stopped_buffer_timesteps_overwrite=train_cfg.method.stopped_buffer_timesteps_overwrite)
        episode_keypoints_stabilizing_lastremoved = episode_keypoints_stabilizing[:-1]
        episodes_stabilizing_arm_input_ep = copy.deepcopy(np.array(observations)[episode_keypoints_stabilizing_lastremoved])
        update_low_dim_time(env, episodes_stabilizing_arm_input_ep)
        episodes_stabilizing_arm_input.append(episodes_stabilizing_arm_input_ep)

        episode_keypoints_stabilizing_actions = episode_keypoints_stabilizing[1:]
        episode_keypoints_stabilizing_actions_ep = copy.deepcopy(np.array(observations)[episode_keypoints_stabilizing_actions])
        update_low_dim_time(env, episode_keypoints_stabilizing_actions_ep)
        episodes_stabilizing_arm_gt_action.append(episode_keypoints_stabilizing_actions_ep)

    return episodes_acting_arm_input, episodes_acting_arm_gt_action, episodes_stabilizing_arm_input, episodes_stabilizing_arm_gt_action


def compute_positional_angular_gripper_open_errors(episodes_input,
                                                    episodes_gt_action,
                                                    curr_agent,
                                                    act_which_arm,
                                                    crop_radius,
                                                    env,
                                                    device):
    positional_errors = []
    angular_errors = []
    gripper_open_errors = []
    for obs, acts in zip(episodes_input, episodes_gt_action):
        curr_agent.reset()

        new_scene_bounds = get_new_scene_bounds_based_on_crop(crop_radius, obs[0]['target_object_pos'])

        env._i = 0
        for step in range(obs.shape[0]):
            
            prepped_data = {k:torch.tensor([v], device=device) for k, v in obs[step]['obs_history'].items()}

            position, quat, gripper_close = curr_agent.act(step, prepped_data,
                                                deterministic=eval, which_arm=act_which_arm, new_scene_bounds=new_scene_bounds, dominant_assitive_policy=True, ep_number=0, is_real_robot=True)
            env._i += 1

            if act_which_arm == 'left':
                gt_gripper_position = acts[step]['formated_obs'].gripper_left_pose[:3]
                gt_gripper_orientation = acts[step]['formated_obs'].gripper_left_pose[3:]
                gt_gripper_open = acts[step]['formated_obs'].gripper_left_open
            else:
                # act_which_arm == 'right'
                gt_gripper_position = acts[step]['formated_obs'].gripper_right_pose[:3]
                gt_gripper_orientation = acts[step]['formated_obs'].gripper_right_pose[3:]
                gt_gripper_open = acts[step]['formated_obs'].gripper_right_open

            positional_errors.append(np.abs(position - gt_gripper_position))
            angular_errors.append(quaternion_angular_error(quat, gt_gripper_orientation))
            gripper_open_errors.append(np.abs(gripper_close[0] - gt_gripper_open))

    return np.mean(positional_errors, axis=0), np.mean(angular_errors), np.mean(gripper_open_errors)

def quaternion_angular_error(quat1, quat2):
    # Convert the quaternions to rotation objects
    rot1 = R.from_quat(quat1)
    rot2 = R.from_quat(quat2)

    # Compute the relative rotation (quat2 * quat1^-1)
    relative_rotation = rot2 * rot1.inv()

    # Convert the relative rotation to angle-axis representation
    angle_axis = relative_rotation.as_rotvec()

    # The angle is the norm of the angle-axis vector
    angle_rad = np.linalg.norm(angle_axis)

    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def eval_seed(train_cfg,
              eval_cfg,
              logdir,
              cams,
              env_device,
              multi_task,
              seed,
              env_config,
              left_arm_train_cfg,
              left_arm_ckpt) -> None:

    tasks = eval_cfg.rlbench.tasks
    rg = RolloutGenerator()

    if train_cfg.method.name == 'ARM':
        raise NotImplementedError('ARM not yet supported for eval.py')

    elif train_cfg.method.name == 'BC_LANG':
        acting_arm_agent = bc_lang.launch_utils.create_agent(
            cams[0],
            train_cfg.method.activation,
            train_cfg.method.lr,
            train_cfg.method.weight_decay,
            train_cfg.rlbench.camera_resolution,
            train_cfg.method.grad_clip)

    elif train_cfg.method.name == 'VIT_BC_LANG':
        acting_arm_agent = vit_bc_lang.launch_utils.create_agent(
            cams[0],
            train_cfg.method.activation,
            train_cfg.method.lr,
            train_cfg.method.weight_decay,
            train_cfg.rlbench.camera_resolution,
            train_cfg.method.grad_clip)

    elif train_cfg.method.name == 'C2FARM_LINGUNET_BC':
        acting_arm_agent = c2farm_lingunet_bc.launch_utils.create_agent(train_cfg)

    elif train_cfg.method.name == 'PERACT_BC':
        acting_arm_agent = peract_bc.launch_utils.create_agent(train_cfg)

    elif train_cfg.method.name == 'PERACT_RL':
        raise NotImplementedError("PERACT_RL not yet supported for eval.py")

    else:
        raise ValueError('Method %s does not exists.' % train_cfg.method.name)

    weightsdir = os.path.join(logdir, 'weights')

    if left_arm_train_cfg is not None:
        stabilizing_arm_agent = peract_bc.launch_utils.create_agent(left_arm_train_cfg)
    else:
        stabilizing_arm_agent = None

    if not isinstance(train_cfg.method.crop_radius, float):
        # this is a multi-task policy, get the appropriate crop_radius
        task_index = train_cfg.rlbench.tasks.index(tasks[0])
        train_cfg.method.crop_radius = train_cfg.method.crop_radius[task_index]

    # evaluate all checkpoints (0, 1000, ...) which don't have results, i.e. validation phase
    if eval_cfg.framework.eval_type == 'missing':
        weight_folders = os.listdir(weightsdir)
        weight_folders = sorted(map(int, weight_folders))

        env_data_csv_file = os.path.join(logdir, 'eval_data.csv')
        if os.path.exists(env_data_csv_file):
            env_dict = pd.read_csv(env_data_csv_file).to_dict()
            evaluated_weights = sorted(map(int, list(env_dict['step'].values())))
            weight_folders = [w for w in weight_folders if w not in evaluated_weights]

        print('Missing weights: ', weight_folders)

    # pick the best checkpoint from validation and evaluate, i.e. test phase
    elif eval_cfg.framework.eval_type == 'best':
        env_data_csv_file = os.path.join(logdir, 'eval_data.csv')
        if os.path.exists(env_data_csv_file):
            env_dict = pd.read_csv(env_data_csv_file).to_dict()
            existing_weights = list(map(int, sorted(os.listdir(os.path.join(logdir, 'weights')))))
            task_weights = {}
            for task in tasks:
                weights = list(env_dict['step'].values())

                if len(tasks) > 1:
                    task_score = list(env_dict['eval_envs/return/%s' % task].values())
                else:
                    task_score = list(env_dict['eval_envs/return'].values())

                avail_weights, avail_task_scores = [], []
                for step_idx, step in enumerate(weights):
                    if step in existing_weights:
                        avail_weights.append(step)
                        avail_task_scores.append(task_score[step_idx])

                assert(len(avail_weights) == len(avail_task_scores))
                best_weight = avail_weights[np.argwhere(avail_task_scores == np.amax(avail_task_scores)).flatten().tolist()[-1]]
                task_weights[task] = best_weight

            weight_folders = [task_weights]
            print("Best weights:", weight_folders)
        else:
            raise Exception('No existing eval_data.csv file found in %s' % logdir)

    # evaluate only the last checkpoint
    elif eval_cfg.framework.eval_type == 'last':
        weight_folders = os.listdir(weightsdir)
        weight_folders = sorted(map(int, weight_folders))
        weight_folders = [weight_folders[-1]]
        print("Last weight:", weight_folders)

    # evaluate a specific checkpoint
    elif type(eval_cfg.framework.eval_type) == int:
        weight_folders = [int(eval_cfg.framework.eval_type)]
        print("Weight:", weight_folders)

    else:
        raise Exception('Unknown eval type')

    # csv file
    csv_file_exist = False
    csv_filename = 'eval.csv'

    num_weights_to_eval = np.arange(len(weight_folders))
    if len(num_weights_to_eval) == 0:
        logging.info("No weights to evaluate. Results are already available in eval_data.csv")
        sys.exit(0)

    if env_device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    env = RobotEnv(train_cfg, eval_cfg)

    # NOTE: in multi-task settings, each task is evaluated serially, which makes everything slow!
    left_arm_ckpt = None
    if len(weight_folders) == 1 and '.pt' in eval_cfg.framework.left_arm_ckpt:
        # evaluate a single checkpoint 
        # here, we assume it's going to be robot execution
        # real-world robot execution
        env.initialize_robot('192.10.0.11', '192.10.0.12')
        key = input("Press c to move robots to starting states or anything else to exit:\n")
        if key == 'c':
            env.move_robot_to_starting_states()
        else:
            print('Exit')
            gc.collect()
            sys.exit(0)

        # initialize acting policy
        acting_arm_agent.build(training=False, device=device)
        acting_ckpt = os.path.join(weightsdir, str(weight_folders[0]))
        acting_arm_agent.load_weights(acting_ckpt)

        # initialize stabilizing policy
        stabilizing_arm_agent.build(training=False, device=device)
        stabilizing_ckpt = '/'.join(eval_cfg.framework.left_arm_ckpt.split('/')[:-1]) # removed '/QAttentionAgent_layer0.pt'
        stabilizing_arm_agent.load_weights(stabilizing_ckpt)

        acting_arm_agent.reset()
        stabilizing_arm_agent.reset()
        new_scene_bounds = env.get_new_scene_bounds_using_vlm()
        obs_history, info = env.get_observation()

        dominant_arm_agent = acting_arm_agent
        assistive_arm_agent = stabilizing_arm_agent

        dominant_arm = env.get_dominant_arm()
        if dominant_arm == 'right':
            assistive_arm = 'left'
        else:
            assistive_arm = 'right'
        print(f'Acting arm: {dominant_arm}; Stabilizing arm: {assistive_arm}')

        env._i = 0
        for step in range(eval_cfg.rlbench.episode_length):
            if step % 2 == 0:
                curr_agent = assistive_arm_agent
                act_which_arm = assistive_arm
            else:
                curr_agent = dominant_arm_agent
                act_which_arm = dominant_arm

            prepped_data = {k:torch.tensor([v], device=device) for k, v in obs_history.items()}

            position, quat, gripper_close = curr_agent.act(step, prepped_data,
                                deterministic=eval, which_arm=act_which_arm, new_scene_bounds=new_scene_bounds, dominant_assitive_policy=True, ep_number=0, is_real_robot=True)
            
            if act_which_arm == 'left':
                extrinsics = LEFT_ARM_EXTRINSICS
            elif act_which_arm == 'right':
                extrinsics = RIGHT_ARM_EXTRINSICS
            else:
                raise NotImplementedError

            position = [position[0], position[1], position[2], 1]
            tcp_point_in_base_frame = np.matmul(extrinsics, position)
            env._i += 1 

            env.move_robot(act_which_arm, tcp_point_in_base_frame, quat, gripper_close, info, extrinsics)
            obs_history, info = env.get_observation()
    else:

        eval_all_act_ckpts = True
        if eval_cfg.framework.left_arm_ckpt is not None and '.pt' not in eval_cfg.framework.left_arm_ckpt and type(eval_cfg.framework.eval_type) == int:
        # this happens when we want to find the best checkpoint for the stabilizing arm, and we've already found the best checkpoint for the acting arm
            stab_weight_folders = os.listdir(eval_cfg.framework.left_arm_ckpt)
            stab_weight_folders = sorted(map(int, stab_weight_folders))

            # skip checkpoints
            if eval_cfg.framework.left_arm_ckpt_skip is not None:
                # skip the every checkpoint before left_arm_ckpt_skip
                index_to_skip = 1
                for weight_num in stab_weight_folders:
                    if weight_num == eval_cfg.framework.left_arm_ckpt_skip:
                        break
                    index_to_skip += 1
                stab_weight_folders = stab_weight_folders[index_to_skip:]
                print('After skipping weights: ', stab_weight_folders)

            curr_weight_folders = stab_weight_folders
            eval_all_act_ckpts = False
        else:
            # skip checkpoints
            if eval_cfg.framework.act_arm_ckpt_skip is not None:
                # skip the every checkpoint before left_arm_ckpt_skip
                index_to_skip = 1
                for weight_num in weight_folders:
                    if weight_num == eval_cfg.framework.act_arm_ckpt_skip:
                        break
                    index_to_skip += 1
                weight_folders = weight_folders[index_to_skip:]
                print('After skipping weights: ', weight_folders)

            curr_weight_folders = weight_folders

        # evaluate all acting arm's checkpoints
        for weight_folder in curr_weight_folders:
            # compute positional and angular errors in validation data
            if eval_all_act_ckpts:
                # initialize acting policy
                acting_arm_agent.build(training=False, device=device)
                acting_ckpt = os.path.join(weightsdir, str(weight_folder))
                acting_arm_agent.load_weights(acting_ckpt)

                # initialize stabilizing policy
                stabilizing_arm_agent.build(training=False, device=device)
                stabilizing_ckpt = '/'.join(eval_cfg.framework.left_arm_ckpt.split('/')[:-1]) # removed '/QAttentionAgent_layer0.pt'
                stabilizing_arm_agent.load_weights(stabilizing_ckpt)
            else:
                # initialize acting policy
                acting_arm_agent.build(training=False, device=device)
                acting_ckpt = os.path.join(weightsdir, str(weight_folders[0]))
                acting_arm_agent.load_weights(acting_ckpt)

                # initialize stabilizing policy
                stabilizing_arm_agent.build(training=False, device=device)
                stabilizing_ckpt = os.path.join(eval_cfg.framework.left_arm_ckpt, str(weight_folder))
                stabilizing_arm_agent.load_weights(stabilizing_ckpt)

            episodes_acting_arm_input, episodes_acting_arm_gt_action, episodes_stabilizing_arm_input, episodes_stabilizing_arm_gt_action = get_observations_from_val(env, env_config[3], eval_cfg.rlbench.tasks[0], train_cfg)

            acting_pos_errors, acting_angular_errors, acting_gripper_errors = compute_positional_angular_gripper_open_errors(
                                                                episodes_acting_arm_input,
                                                                episodes_acting_arm_gt_action,
                                                                acting_arm_agent,
                                                                'left',
                                                                train_cfg.method.crop_radius,
                                                                env,
                                                                device
                                                            )
            stabilizing_pos_errors, stabilizing_angular_errors, stabilizing_gripper_errors = compute_positional_angular_gripper_open_errors(
                                                                episodes_stabilizing_arm_input,
                                                                episodes_stabilizing_arm_gt_action,
                                                                stabilizing_arm_agent,
                                                                'right',
                                                                train_cfg.method.crop_radius,
                                                                env,
                                                                device
                                                            )

            # save csv
            csv_dict = {
                'acting_ckpt': [acting_ckpt.split('/')[-1]],
                'stabilizing_ckpt': [stabilizing_ckpt.split('/')[-1]],
                'act_pos_x_err': [acting_pos_errors[0]],
                'act_pos_y_err': [acting_pos_errors[1]],
                'act_pos_z_err': [acting_pos_errors[2]],
                'act_angular_err': [acting_angular_errors],
                'act_grip_err': [acting_gripper_errors],
                'stab_pos_x_err': [stabilizing_pos_errors[0]],
                'stab_pos_y_err': [stabilizing_pos_errors[1]],
                'stab_pos_z_err': [stabilizing_pos_errors[2]],
                'stab_angular_err': [stabilizing_angular_errors],
                'stab_grip_err': [stabilizing_gripper_errors],
            }
            df = pd.DataFrame(csv_dict)
            if not csv_file_exist:
                df.to_csv(csv_filename, mode='w', header=True, index=False)
                csv_file_exist = True
            else:
                df.to_csv(csv_filename, mode='a', header=False, index=False)
            print(f'Acting {csv_dict["acting_ckpt"][0]} stabilizing {csv_dict["stabilizing_ckpt"][0]} eval.csv saved!')

            del acting_arm_agent
            del stabilizing_arm_agent
            acting_arm_agent = peract_bc.launch_utils.create_agent(train_cfg)
            stabilizing_arm_agent = peract_bc.launch_utils.create_agent(left_arm_train_cfg)
        
    del acting_arm_agent
    del stabilizing_arm_agent
    gc.collect()


@hydra.main(config_name='eval', config_path='conf')
def main(eval_cfg: DictConfig) -> None:
    logging.info('\n' + OmegaConf.to_yaml(eval_cfg))

    start_seed = eval_cfg.framework.start_seed
    logdir = os.path.join(eval_cfg.framework.logdir,
                                eval_cfg.rlbench.task_name,
                                eval_cfg.method.name,
                                'seed%d' % start_seed)

    train_config_path = os.path.join(logdir, 'config.yaml')
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            train_cfg = OmegaConf.load(f)
    else:
        raise Exception("Missing seed%d/config.yaml" % start_seed)

    env_device = eval_cfg.framework.gpu

    if eval_cfg.framework.left_arm_train_cfg is not None:
        with open(eval_cfg.framework.left_arm_train_cfg, 'r') as f:
            left_arm_train_cfg = OmegaConf.load(f)
        left_arm_ckpt = eval_cfg.framework.left_arm_ckpt
    else:
        left_arm_train_cfg = None
        left_arm_ckpt = None

    # gripper_mode = Discrete() # original code
    gripper_mode = Discrete2Robots()

    # arm_action_mode = EndEffectorPoseViaPlanning() # original code
    arm_action_mode = EndEffectorPoseViaPlanning2Robots()

    # action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode) # original code
    action_mode = MoveArmThenGripper2Robots(arm_action_mode, gripper_mode)

    task_files = [t.replace('.py', '') for t in os.listdir(rlbench_task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]
    eval_cfg.rlbench.cameras = eval_cfg.rlbench.cameras if isinstance(
        eval_cfg.rlbench.cameras, ListConfig) else [eval_cfg.rlbench.cameras]
    if train_cfg.method.crop_target_obj_voxel or eval_cfg.method.voxposer_only_eval or eval_cfg.method.which_arm == 'dominant_assistive':
        obs_config = utils.create_obs_config_voxposer(eval_cfg.rlbench.cameras,
                                            eval_cfg.rlbench.camera_resolution,
                                            train_cfg.method.name)
    else:
        obs_config = utils.create_obs_config(eval_cfg.rlbench.cameras,
                                            eval_cfg.rlbench.camera_resolution,
                                            train_cfg.method.name)

    if eval_cfg.cinematic_recorder.enabled:
        obs_config.record_gripper_closing = True

    # single-task or multi-task
    if len(eval_cfg.rlbench.tasks) > 1:
        tasks = eval_cfg.rlbench.tasks
        multi_task = True

        task_classes = []
        for task in tasks:
            if task not in task_files:
                raise ValueError('Task %s not recognised!.' % task)
            task_classes.append(task_file_to_task_class(task))

        env_config = (task_classes,
                      obs_config,
                      action_mode,
                      eval_cfg.rlbench.demo_path,
                      eval_cfg.rlbench.episode_length,
                      eval_cfg.rlbench.headless,
                      eval_cfg.framework.eval_episodes,
                      train_cfg.rlbench.include_lang_goal_in_obs,
                      eval_cfg.rlbench.time_in_state,
                      eval_cfg.framework.record_every_n)
    else:
        task = eval_cfg.rlbench.tasks[0]
        multi_task = False

        if task not in task_files:
            raise ValueError('Task %s not recognised!.' % task)
        task_class = task_file_to_task_class(task)

        env_config = (task_class,
                      obs_config,
                      action_mode,
                      eval_cfg.rlbench.demo_path,
                      eval_cfg.rlbench.episode_length,
                      eval_cfg.rlbench.headless,
                      train_cfg.rlbench.include_lang_goal_in_obs,
                      eval_cfg.rlbench.time_in_state,
                      eval_cfg.framework.record_every_n)

    logging.info('Evaluating seed %d.' % start_seed)
    eval_seed(train_cfg,
              eval_cfg,
              logdir,
              eval_cfg.rlbench.cameras,
              env_device,
              multi_task, start_seed,
              env_config,
              left_arm_train_cfg,
              left_arm_ckpt)

if __name__ == '__main__':
    main()
