import numpy as np
from typing import List


class Observation2Robots(object):
    """Storage for both visual and low-dimensional observations."""

    def __init__(self,
                 wrist_rgb: np.ndarray,
                 wrist_depth: np.ndarray,
                 wrist_mask: np.ndarray,
                 wrist_point_cloud: np.ndarray,
                 wrist2_rgb: np.ndarray,
                 wrist2_depth: np.ndarray,
                 wrist2_mask: np.ndarray,
                 wrist2_point_cloud: np.ndarray,
                 front_rgb: np.ndarray,
                 front_depth: np.ndarray,
                 front_mask: np.ndarray,
                 front_point_cloud: np.ndarray,
                 joint_velocities_right: np.ndarray,
                 joint_positions_right: np.ndarray,
                 joint_forces_right: np.ndarray,
                 joint_velocities_left: np.ndarray,
                 joint_positions_left: np.ndarray,
                 joint_forces_left: np.ndarray,
                 gripper_right_open: float,
                 gripper_right_pose: np.ndarray,
                 gripper_right_matrix: np.ndarray,
                 gripper_right_joint_positions: np.ndarray,
                 gripper_right_touch_forces: np.ndarray,
                 gripper_left_open: float,
                 gripper_left_pose: np.ndarray,
                 gripper_left_matrix: np.ndarray,
                 gripper_left_joint_positions: np.ndarray,
                 gripper_left_touch_forces: np.ndarray,
                 task_low_dim_state: np.ndarray,
                 ignore_collisions: np.ndarray,
                 misc: dict,
                 target_object_pos: np.ndarray,
                 auto_crop_radius: float):
        self.wrist_rgb = wrist_rgb
        self.wrist_depth = wrist_depth
        self.wrist_mask = wrist_mask
        self.wrist_point_cloud = wrist_point_cloud
        self.wrist2_rgb = wrist2_rgb
        self.wrist2_depth = wrist2_depth
        self.wrist2_mask = wrist2_mask
        self.wrist2_point_cloud = wrist2_point_cloud
        self.front_rgb = front_rgb
        self.front_depth = front_depth
        self.front_mask = front_mask
        self.front_point_cloud = front_point_cloud
        self.joint_velocities_right = joint_velocities_right
        self.joint_positions_right = joint_positions_right
        self.joint_forces_right = joint_forces_right
        self.joint_velocities_left = joint_velocities_left
        self.joint_positions_left = joint_positions_left
        self.joint_forces_left = joint_forces_left
        self.gripper_right_open = gripper_right_open
        self.gripper_right_pose = gripper_right_pose
        self.gripper_right_matrix = gripper_right_matrix
        self.gripper_right_joint_positions = gripper_right_joint_positions
        self.gripper_right_touch_forces = gripper_right_touch_forces
        self.gripper_left_open = gripper_left_open
        self.gripper_left_pose = gripper_left_pose
        self.gripper_left_matrix = gripper_left_matrix
        self.gripper_left_joint_positions = gripper_left_joint_positions
        self.gripper_left_touch_forces = gripper_left_touch_forces
        self.task_low_dim_state = task_low_dim_state
        self.ignore_collisions = ignore_collisions
        self.misc = misc
        self.target_object_pos = target_object_pos
        self.auto_crop_radius = auto_crop_radius

    def get_low_dim_data(self, which_arm=None) -> np.ndarray:
        """Gets a 1D array of all the low-dimensional obseervations.

        :return: 1D array of observations.
        """
        if which_arm == 'right':
            low_dim_data = [] if self.gripper_right_open is None else [[self.gripper_right_open]]
            for data in [self.joint_velocities_right, self.joint_positions_right,
                        self.joint_forces_right,
                        self.gripper_right_pose, self.gripper_right_joint_positions,
                        self.gripper_right_touch_forces, self.task_low_dim_state]:
                if data is not None:
                    low_dim_data.append(data)
            return np.concatenate(low_dim_data) if len(low_dim_data) > 0 else np.array([])
        elif which_arm == 'left':
            low_dim_data = [] if self.gripper_left_open is None else [[self.gripper_left_open]]
            for data in [self.joint_velocities_left, self.joint_positions_left,
                        self.joint_forces_left,
                        self.gripper_left_pose, self.gripper_left_joint_positions,
                        self.gripper_left_touch_forces, self.task_low_dim_state]:
                if data is not None:
                    low_dim_data.append(data)
            return np.concatenate(low_dim_data) if len(low_dim_data) > 0 else np.array([])
        else:
            raise NotImplementedError

    def print(self):
        print('front_rgb: ', self.front_rgb)
        print('front_depth: ', self.front_depth)
        print('front_mask: ', self.front_mask)
        print('front_point_cloud: ', self.front_point_cloud)

        print('joint_velocities_right: ', self.joint_velocities_right)
        print('joint_positions_right: ', self.joint_positions_right)
        print('joint_forces_right: ', self.joint_forces_right)
        print('joint_velocities_left: ', self.joint_velocities_left)
        print('joint_positions_left: ', self.joint_positions_left)
        print('joint_forces_left: ', self.joint_forces_left)

        print('gripper_right_open: ', self.gripper_right_open)
        print('gripper_right_pose: ', self.gripper_right_pose)
        print('gripper_right_matrix: ', self.gripper_right_matrix)
        print('gripper_right_joint_positions: ', self.gripper_right_joint_positions)
        print('gripper_right_touch_forces: ', self.gripper_right_touch_forces)

        print('gripper_left_open: ', self.gripper_left_open)
        print('gripper_left_pose: ', self.gripper_left_pose)
        print('gripper_left_matrix: ', self.gripper_left_matrix)
        print('gripper_left_joint_positions: ', self.gripper_left_joint_positions)
        print('gripper_left_touch_forces: ', self.gripper_left_touch_forces)

        print('task_low_dim_state: ', self.task_low_dim_state)
        print('ignore_collisions: ', self.ignore_collisions)
        print('misc: ', self.misc)
        print('target_object_pos: ', self.target_object_pos)
        print('auto_crop_radius: ', self.auto_crop_radius)
