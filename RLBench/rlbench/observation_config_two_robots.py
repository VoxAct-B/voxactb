from pyrep.const import RenderMode
from rlbench.noise_model import NoiseModel, Identity


class CameraConfig(object):
    def __init__(self,
                 rgb=True,
                 rgb_noise: NoiseModel=Identity(),
                 depth=True,
                 depth_noise: NoiseModel=Identity(),
                 point_cloud=True,
                 mask=True,
                 image_size=(128, 128),
                 render_mode=RenderMode.OPENGL3,
                 masks_as_one_channel=True,
                 depth_in_meters=False):
        self.rgb = rgb
        self.rgb_noise = rgb_noise
        self.depth = depth
        self.depth_noise = depth_noise
        self.point_cloud = point_cloud
        self.mask = mask
        self.image_size = image_size
        self.render_mode = render_mode
        self.masks_as_one_channel = masks_as_one_channel
        self.depth_in_meters = depth_in_meters

    def set_all(self, value: bool):
        self.rgb = value
        self.depth = value
        self.point_cloud = value
        self.mask = value


class ObservationConfig2Robots(object):
    def __init__(self,
                 wrist_camera: CameraConfig = None,
                 wrist2_camera: CameraConfig = None,
                 front_camera: CameraConfig = None,
                 joint_velocities_right=True,
                 joint_velocities_noise_right: NoiseModel=Identity(),
                 joint_positions_right=True,
                 joint_positions_noise_right: NoiseModel=Identity(),
                 joint_forces_right=True,
                 joint_forces_noise_right: NoiseModel=Identity(),
                 joint_velocities_left=True,
                 joint_velocities_noise_left: NoiseModel=Identity(),
                 joint_positions_left=True,
                 joint_positions_noise_left: NoiseModel=Identity(),
                 joint_forces_left=True,
                 joint_forces_noise_left: NoiseModel=Identity(),
                 gripper_right_open=True,
                 gripper_right_pose=True,
                 gripper_right_matrix=False,
                 gripper_right_joint_positions=False,
                 gripper_right_touch_forces=False,
                 gripper_left_open=True,
                 gripper_left_pose=True,
                 gripper_left_matrix=False,
                 gripper_left_joint_positions=False,
                 gripper_left_touch_forces=False,
                 wrist_camera_matrix=False,
                 wrist2_camera_matrix=False,
                 record_gripper_closing=False,
                 task_low_dim_state=False,
                 record_ignore_collisions=True,
                 ):
        self.wrist_camera = (
            CameraConfig() if wrist_camera is None
            else wrist_camera)
        self.wrist2_camera = (
            CameraConfig() if wrist2_camera is None
            else wrist2_camera)
        self.front_camera = (
            CameraConfig() if front_camera is None
            else front_camera)
        self.joint_velocities_right = joint_velocities_right
        self.joint_velocities_noise_right = joint_velocities_noise_right
        self.joint_positions_right = joint_positions_right
        self.joint_positions_noise_right = joint_positions_noise_right
        self.joint_forces_right = joint_forces_right
        self.joint_forces_noise_right = joint_forces_noise_right
        self.joint_velocities_left = joint_velocities_left
        self.joint_velocities_noise_left = joint_velocities_noise_left
        self.joint_positions_left = joint_positions_left
        self.joint_positions_noise_left = joint_positions_noise_left
        self.joint_forces_left = joint_forces_left
        self.joint_forces_noise_left = joint_forces_noise_left

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
        self.wrist_camera_matrix = wrist_camera_matrix
        self.wrist2_camera_matrix = wrist2_camera_matrix
        self.record_gripper_closing = record_gripper_closing
        self.task_low_dim_state = task_low_dim_state
        self.record_ignore_collisions = record_ignore_collisions

    def set_all(self, value: bool):
        self.set_all_high_dim(value)
        self.set_all_low_dim(value)

    def set_all_high_dim(self, value: bool):
        self.wrist_camera.set_all(value)
        self.wrist2_camera.set_all(value)
        self.front_camera.set_all(value)

    def set_all_low_dim(self, value: bool):
        self.joint_velocities_right = value
        self.joint_positions_right = value
        self.joint_forces_right = value
        self.joint_velocities_left = value
        self.joint_positions_left = value
        self.joint_forces_left = value

        self.gripper_right_open = value
        self.gripper_right_pose = value
        self.gripper_right_matrix = value
        self.gripper_right_joint_positions = value
        self.gripper_right_touch_forces = value
        self.gripper_left_open = value
        self.gripper_left_pose = value
        self.gripper_left_matrix = value
        self.gripper_left_joint_positions = value
        self.gripper_left_touch_forces = value
        self.wrist_camera_matrix = value
        self.wrist2_camera_matrix = value
        self.task_low_dim_state = value
