import importlib
import pickle
from os import listdir
from os.path import join, exists
from typing import List

import numpy as np
from PIL import Image
from natsort import natsorted
from pyrep.objects import VisionSensor

from rlbench.backend.const import *
from rlbench.backend.utils import image_to_float_array, rgb_handles_to_mask
from rlbench.demo import Demo
from rlbench.observation_config import ObservationConfig


class InvalidTaskName(Exception):
    pass


def name_to_task_class(task_file: str):
    name = task_file.replace('.py', '')
    class_name = ''.join([w[0].upper() + w[1:] for w in name.split('_')])
    try:
        mod = importlib.import_module("rlbench.tasks.%s" % name)
        mod = importlib.reload(mod)
    except ModuleNotFoundError as e:
        raise InvalidTaskName(
            "The task file '%s' does not exist or cannot be compiled."
            % name) from e
    try:
        task_class = getattr(mod, class_name)
    except AttributeError as e:
        raise InvalidTaskName(
            "Cannot find the class name '%s' in the file '%s'."
            % (class_name, name)) from e
    return task_class


def get_stored_demos(amount: int, image_paths: bool, dataset_root: str,
                     variation_number: int, task_name: str,
                     obs_config: ObservationConfig,
                     random_selection: bool = True,
                     from_episode_number: int = 0,
                     which_arm: str = 'right') -> List[Demo]:

    task_root = join(dataset_root, task_name)
    if not exists(task_root):
        raise RuntimeError("Can't find the demos for %s at: %s" % (
            task_name, task_root))

    if variation_number == -1:
        # All variations
        examples_path = join(
            task_root, VARIATIONS_ALL_FOLDER,
            EPISODES_FOLDER)
        examples = listdir(examples_path)
    else:
        # Sample an amount of examples for the variation of this task
        examples_path = join(
            task_root, VARIATIONS_FOLDER % variation_number,
            EPISODES_FOLDER)
        examples = listdir(examples_path)

    # hack: ignore .DS_Store files from macOS zips
    examples = [e for e in examples if '.DS_Store' not in e]

    if amount == -1:
        amount = len(examples)
    if amount > len(examples):
        raise RuntimeError(
            'You asked for %d examples, but only %d were available.' % (
                amount, len(examples)))
    if random_selection:
        selected_examples = np.random.choice(examples, amount, replace=False)
    else:
        selected_examples = natsorted(
            examples)[from_episode_number:from_episode_number+amount]

    # Process these examples (e.g. loading observations)
    demos = []
    for example in selected_examples:
        example_path = join(examples_path, example)
        print('Example path: ', example_path) # for debugging
        with open(join(example_path, LOW_DIM_PICKLE), 'rb') as f:
            obs = pickle.load(f)

        if variation_number == -1:
            with open(join(example_path, VARIATION_NUMBER), 'rb') as f:
                obs.variation_number = pickle.load(f)
        else:
            obs.variation_number = variation_number

        # language description
        episode_descriptions_f = join(example_path, VARIATION_DESCRIPTIONS)
        if exists(episode_descriptions_f):
            with open(episode_descriptions_f, 'rb') as f:
                descriptions = pickle.load(f)
        else:
            descriptions = ["unknown task description"]

        wrist_rgb_f = join(example_path, WRIST_RGB_FOLDER)
        wrist_depth_f = join(example_path, WRIST_DEPTH_FOLDER)
        wrist_mask_f = join(example_path, WRIST_MASK_FOLDER)
        wrist2_rgb_f = join(example_path, WRIST2_RGB_FOLDER)
        wrist2_depth_f = join(example_path, WRIST2_DEPTH_FOLDER)
        wrist2_mask_f = join(example_path, WRIST2_MASK_FOLDER)
        front_rgb_f = join(example_path, FRONT_RGB_FOLDER)
        front_depth_f = join(example_path, FRONT_DEPTH_FOLDER)
        front_mask_f = join(example_path, FRONT_MASK_FOLDER)

        num_steps = len(obs)

        if not (num_steps == len(listdir(wrist_rgb_f)) == len(
                listdir(wrist_depth_f)) == len(listdir(wrist2_rgb_f)) == len(
                listdir(wrist2_depth_f)) == len(listdir(front_rgb_f)) == len(
                listdir(front_depth_f))):
            raise RuntimeError('Broken dataset assumption')

        for i in range(num_steps):
            # descriptions
            obs[i].misc['descriptions'] = descriptions

            si = IMAGE_FORMAT % i
            if obs_config.wrist_camera.rgb:
                obs[i].wrist_rgb = join(wrist_rgb_f, si)
            if obs_config.wrist_camera.depth or obs_config.wrist_camera.point_cloud:
                obs[i].wrist_depth = join(wrist_depth_f, si)
            if obs_config.wrist_camera.mask:
                obs[i].wrist_mask = join(wrist_mask_f, si)
            if obs_config.wrist2_camera.rgb:
                obs[i].wrist2_rgb = join(wrist2_rgb_f, si)
            if obs_config.wrist2_camera.depth or obs_config.wrist2_camera.point_cloud:
                obs[i].wrist2_depth = join(wrist2_depth_f, si)
            if obs_config.wrist2_camera.mask:
                obs[i].wrist2_mask = join(wrist2_mask_f, si)
            if obs_config.front_camera.rgb:
                obs[i].front_rgb = join(front_rgb_f, si)
            if obs_config.front_camera.depth or obs_config.front_camera.point_cloud:
                obs[i].front_depth = join(front_depth_f, si)
            if obs_config.front_camera.mask:
                obs[i].front_mask = join(front_mask_f, si)

            # Remove low dim info if necessary
            if not obs_config.joint_velocities_right:
                obs[i].joint_velocities_right = None
            if not obs_config.joint_positions_right:
                obs[i].joint_positions_right = None
            if not obs_config.joint_forces_right:
                obs[i].joint_forces_right = None

            if not obs_config.joint_velocities_left:
                obs[i].joint_velocities_left = None
            if not obs_config.joint_positions_left:
                obs[i].joint_positions_left = None
            if not obs_config.joint_forces_left:
                obs[i].joint_forces_left = None

            if not obs_config.gripper_right_open:
                obs[i].gripper_right_open = None
            if not obs_config.gripper_right_pose:
                obs[i].gripper_right_pose = None
            if not obs_config.gripper_right_joint_positions:
                obs[i].gripper_right_joint_positions = None
            if not obs_config.gripper_right_touch_forces:
                obs[i].gripper_right_touch_forces = None
            if not obs_config.gripper_left_open:
                obs[i].gripper_left_open = None
            if not obs_config.gripper_left_pose:
                obs[i].gripper_left_pose = None
            if not obs_config.gripper_left_joint_positions:
                obs[i].gripper_left_joint_positions = None
            if not obs_config.gripper_left_touch_forces:
                obs[i].gripper_left_touch_forces = None

            if not obs_config.task_low_dim_state:
                obs[i].task_low_dim_state = None

        if not image_paths:
            for i in range(num_steps):
                if obs_config.wrist_camera.rgb:
                    obs[i].wrist_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].wrist_rgb),
                            obs_config.wrist_camera.image_size))
                if obs_config.wrist2_camera.rgb:
                    obs[i].wrist2_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].wrist2_rgb),
                            obs_config.wrist2_camera.image_size))
                if obs_config.front_camera.rgb:
                    obs[i].front_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].front_rgb),
                            obs_config.front_camera.image_size))

                if obs_config.wrist_camera.depth or obs_config.wrist_camera.point_cloud:
                    wrist_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].wrist_depth),
                            obs_config.wrist_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['wrist_camera_near']
                    far = obs[i].misc['wrist_camera_far']
                    wrist_depth_m = near + wrist_depth * (far - near)
                    if obs_config.wrist_camera.depth:
                        d = wrist_depth_m if obs_config.wrist_camera.depth_in_meters else wrist_depth
                        obs[i].wrist_depth = obs_config.wrist_camera.depth_noise.apply(d)
                    else:
                        obs[i].wrist_depth = None

                if obs_config.wrist2_camera.depth or obs_config.wrist2_camera.point_cloud:
                    wrist2_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].wrist2_depth),
                            obs_config.wrist2_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['wrist2_camera_near']
                    far = obs[i].misc['wrist2_camera_far']
                    wrist2_depth_m = near + wrist2_depth * (far - near)
                    if obs_config.wrist2_camera.depth:
                        d = wrist2_depth_m if obs_config.wrist2_camera.depth_in_meters else wrist2_depth
                        obs[i].wrist2_depth = obs_config.wrist2_camera.depth_noise.apply(d)
                    else:
                        obs[i].wrist2_depth = None

                if obs_config.front_camera.depth or obs_config.front_camera.point_cloud:
                    front_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].front_depth),
                            obs_config.front_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['front_camera_near']
                    far = obs[i].misc['front_camera_far']
                    front_depth_m = near + front_depth * (far - near)
                    if obs_config.front_camera.depth:
                        d = front_depth_m if obs_config.front_camera.depth_in_meters else front_depth
                        obs[i].front_depth = obs_config.front_camera.depth_noise.apply(d)
                    else:
                        obs[i].front_depth = None

                if obs_config.wrist_camera.point_cloud:
                    obs[i].wrist_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        wrist_depth_m,
                        obs[i].misc['wrist_camera_extrinsics'],
                        obs[i].misc['wrist_camera_intrinsics'])
                if obs_config.front_camera.point_cloud:
                    obs[i].front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        front_depth_m,
                        obs[i].misc['front_camera_extrinsics'],
                        obs[i].misc['front_camera_intrinsics'])
                if obs_config.wrist2_camera.point_cloud:
                    obs[i].wrist2_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        wrist2_depth_m,
                        obs[i].misc['wrist2_camera_extrinsics'],
                        obs[i].misc['wrist2_camera_intrinsics'])

                # Masks are stored as coded RGB images.
                # Here we transform them into 1 channel handles.
                if obs_config.wrist_camera.mask:
                    obs[i].wrist_mask = rgb_handles_to_mask(np.array(
                        _resize_if_needed(Image.open(
                            obs[i].wrist_mask),
                            obs_config.wrist_camera.image_size)))
                if obs_config.front_camera.mask:
                    obs[i].front_mask = rgb_handles_to_mask(np.array(
                        _resize_if_needed(Image.open(
                            obs[i].front_mask),
                            obs_config.front_camera.image_size)))
                if obs_config.wrist2_camera.mask:
                    obs[i].wrist2_mask = rgb_handles_to_mask(np.array(
                        _resize_if_needed(Image.open(
                            obs[i].wrist2_mask),
                            obs_config.wrist2_camera.image_size)))

        demos.append(obs)
    return demos


def get_stored_real_world_demos(amount: int, image_paths: bool, dataset_root: str,
                     variation_number: int, task_name: str,
                     obs_config: ObservationConfig,
                     random_selection: bool = True,
                     from_episode_number: int = 0,
                     which_arm: str = 'right') -> List[Demo]:

    task_root = join(dataset_root, task_name)
    if not exists(task_root):
        raise RuntimeError("Can't find the demos for %s at: %s" % (
            task_name, task_root))

    if variation_number == -1:
        # All variations
        examples_path = join(
            task_root, VARIATIONS_ALL_FOLDER,
            EPISODES_FOLDER)
        examples = listdir(examples_path)
    else:
        # Sample an amount of examples for the variation of this task
        examples_path = join(
            task_root, VARIATIONS_FOLDER % variation_number,
            EPISODES_FOLDER)
        examples = listdir(examples_path)

    # hack: ignore .DS_Store files from macOS zips
    examples = [e for e in examples if '.DS_Store' not in e]

    if amount == -1:
        amount = len(examples)
    if amount > len(examples):
        raise RuntimeError(
            'You asked for %d examples, but only %d were available.' % (
                amount, len(examples)))
    if random_selection:
        selected_examples = np.random.choice(examples, amount, replace=False)
    else:
        selected_examples = natsorted(
            examples)[from_episode_number:from_episode_number+amount]

    # Process these examples (e.g. loading observations)
    demos = []
    for example in selected_examples:
        example_path = join(examples_path, example)
        print('Example path: ', example_path) # for debugging
        with open(join(example_path, LOW_DIM_PICKLE), 'rb') as f:
            obs = pickle.load(f)

        if variation_number == -1:
            with open(join(example_path, VARIATION_NUMBER), 'rb') as f:
                obs.variation_number = pickle.load(f)
        else:
            obs.variation_number = variation_number

        # language description
        episode_descriptions_f = join(example_path, VARIATION_DESCRIPTIONS)
        if exists(episode_descriptions_f):
            with open(episode_descriptions_f, 'rb') as f:
                descriptions = pickle.load(f)
        else:
            descriptions = ["unknown task description"]

        front_rgb_f = join(example_path, FRONT_RGB_FOLDER)
        front_depth_f = join(example_path, FRONT_DEPTH_FOLDER)

        num_steps = len(obs)

        if not (num_steps == len(listdir(front_rgb_f)) == len(
                listdir(front_depth_f))):
            raise RuntimeError('Broken dataset assumption')

        for i in range(num_steps):
            # descriptions
            obs[i].misc['descriptions'] = descriptions

            si = IMAGE_FORMAT % i
            if obs_config.front_camera.rgb:
                obs[i].front_rgb = join(front_rgb_f, si)
            if obs_config.front_camera.depth or obs_config.front_camera.point_cloud:
                obs[i].front_depth = join(front_depth_f, si)

            # Remove low dim info if necessary
            if not obs_config.joint_velocities_right:
                obs[i].joint_velocities_right = None
            if not obs_config.joint_positions_right:
                obs[i].joint_positions_right = None
            if not obs_config.joint_forces_right:
                obs[i].joint_forces_right = None

            if not obs_config.joint_velocities_left:
                obs[i].joint_velocities_left = None
            if not obs_config.joint_positions_left:
                obs[i].joint_positions_left = None
            if not obs_config.joint_forces_left:
                obs[i].joint_forces_left = None

            if not obs_config.gripper_right_open:
                obs[i].gripper_right_open = None
            if not obs_config.gripper_right_pose:
                obs[i].gripper_right_pose = None
            if not obs_config.gripper_right_joint_positions:
                obs[i].gripper_right_joint_positions = None
            if not obs_config.gripper_right_touch_forces:
                obs[i].gripper_right_touch_forces = None
            if not obs_config.gripper_left_open:
                obs[i].gripper_left_open = None
            if not obs_config.gripper_left_pose:
                obs[i].gripper_left_pose = None
            if not obs_config.gripper_left_joint_positions:
                obs[i].gripper_left_joint_positions = None
            if not obs_config.gripper_left_touch_forces:
                obs[i].gripper_left_touch_forces = None

            if not obs_config.task_low_dim_state:
                obs[i].task_low_dim_state = None

        if not image_paths:
            for i in range(num_steps):
                if obs_config.front_camera.rgb:
                    obs[i].front_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].front_rgb),
                            obs_config.front_camera.image_size))

                if obs_config.front_camera.depth or obs_config.front_camera.point_cloud:
                    front_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].front_depth),
                            obs_config.front_camera.image_size),
                        DEPTH_SCALE)
                    # near = obs[i].misc['front_camera_near']
                    # far = obs[i].misc['front_camera_far']
                    # front_depth_m = near + front_depth * (far - near)
                    front_depth_m = front_depth
                    if obs_config.front_camera.depth:
                        d = front_depth_m if obs_config.front_camera.depth_in_meters else front_depth
                        obs[i].front_depth = obs_config.front_camera.depth_noise.apply(d)
                    else:
                        obs[i].front_depth = None


                # original implementation: project point cloud to the world frame
                # if obs_config.front_camera.point_cloud:
                #     obs[i].front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                #         front_depth_m,
                #         obs[i].misc['front_camera_extrinsics'],
                #         obs[i].misc['front_camera_intrinsics'])

                # our implementation: project point cloud to the camera_color_optical_frame
                if obs_config.front_camera.point_cloud:
                    identity_extrinsics_matrix = np.eye(4)
                    obs[i].front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        front_depth_m,
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

                    # for debugging: make sure point cloud + RGB colors + gripper poses look correct
                    # import open3d as o3d
                    # # Reshape the data to a 2D array of points (N x 3)
                    # point_cloud_data = obs[i].front_point_cloud.reshape(-1, 3)
                    # # Create an Open3D PointCloud object
                    # pcd = o3d.geometry.PointCloud()
                    # # Assign the points to the PointCloud object
                    # pcd.points = o3d.utility.Vector3dVector(point_cloud_data)

                    # # here, we want to figure out the scene bounds
                    # print('Min x: ', np.min(point_cloud_data[:, 0]))
                    # print('Max x: ', np.max(point_cloud_data[:, 0]))
                    # print('Range x: ', np.max(point_cloud_data[:, 0]) - np.min(point_cloud_data[:, 0]))
                    # print('Min y: ', np.min(point_cloud_data[:, 1]))
                    # print('Max y: ', np.max(point_cloud_data[:, 1]))
                    # print('Range y: ', np.max(point_cloud_data[:, 1]) - np.min(point_cloud_data[:, 1]))
                    # print('Min z: ', np.min(point_cloud_data[:, 2]))
                    # print('Max z: ', np.max(point_cloud_data[:, 2]))
                    # print('Range z: ', np.max(point_cloud_data[:, 2]) - np.min(point_cloud_data[:, 2]))

                    # front_rgb = obs[i].front_rgb.reshape(-1, 3)
                    # front_rgb = front_rgb.astype(np.float) / 255.0
                    # pcd.colors = o3d.utility.Vector3dVector(front_rgb)

                    # # add gripper left position for visualization
                    # pcd_gripper_left = o3d.geometry.PointCloud()
                    # point_gripper_left = np.array([[obs[i].gripper_left_pose[0], obs[i].gripper_left_pose[1], obs[i].gripper_left_pose[2]]])
                    # pcd_gripper_left.points = o3d.utility.Vector3dVector(point_gripper_left)
                    # pcd_gripper_left.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0]])) # red color
                    # print('point_gripper_left: ', point_gripper_left)

                    # # add gripper right position for visualization
                    # pcd_gripper_right = o3d.geometry.PointCloud()
                    # point_gripper_right = np.array([[obs[i].gripper_right_pose[0], obs[i].gripper_right_pose[1], obs[i].gripper_right_pose[2]]])
                    # pcd_gripper_right.points = o3d.utility.Vector3dVector(point_gripper_right)
                    # pcd_gripper_right.colors = o3d.utility.Vector3dVector(np.array([[0.0, 1.0, 0.0]])) # green color
                    # print('point_gripper_right: ', point_gripper_right)
    
                    # # Visualize the point cloud
                    # o3d.visualization.draw_geometries([pcd, pcd_gripper_left, pcd_gripper_right])

                    # # scene bounds
                    # # range: 1.5
                    # # -0.2 to 1.3
                    # # -0.8 to 0.7
                    # # -0.4 to 1.1

                    # import sys
                    # import pdb

                    # class ForkedPdb(pdb.Pdb):
                    #     """A Pdb subclass that may be used
                    #     from a forked multiprocessing child

                    #     """
                    #     def interaction(self, *args, **kwargs):
                    #         _stdin = sys.stdin
                    #         try:
                    #             sys.stdin = open('/dev/stdin')
                    #             pdb.Pdb.interaction(self, *args, **kwargs)
                    #         finally:
                    #             sys.stdin = _stdin
                    # ForkedPdb().set_trace()
        demos.append(obs)
    return demos

def _resize_if_needed(image, size):
    if image.size[0] != size[0] or image.size[1] != size[1]:
        image = image.resize(size)
    return image

