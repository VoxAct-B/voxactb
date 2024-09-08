from typing import List, Callable

import numpy as np
from pyrep import PyRep
from pyrep.const import ObjectType, RenderMode
from pyrep.errors import ConfigurationPathError
from pyrep.objects import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor

from rlbench.backend.exceptions import (
    WaypointError, BoundaryError, NoWaypointsError, DemoError)
from rlbench.backend.observation_two_robots import Observation2Robots
from rlbench.backend.robot import Robot
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task_two_robots import Task2Robots
from rlbench.backend.utils import rgb_handles_to_mask, get_quaternion_from_euler
from rlbench.demo import Demo
from rlbench.noise_model import NoiseModel
from rlbench.observation_config_two_robots import ObservationConfig2Robots, CameraConfig
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
import math
from rlbench.backend.vlm import VLM
from PIL import Image
import open3d as o3d
import gc

STEPS_BEFORE_EPISODE_START = 10


class Scene2Robots(object):
    """Controls what is currently in the vrep scene. This is used for making
    sure that the tasks are easily reachable. This may be just replaced by
    environment. Responsible for moving all the objects. """

    def __init__(self,
                 pyrep: PyRep,
                 robot_right_arm: Robot,
                 robot_left_arm: Robot,
                 obs_config: ObservationConfig2Robots = ObservationConfig2Robots(),
                 robot_setup: str = 'panda',
                 mode: str = 'default'):
        self.pyrep = pyrep
        self.robot_right_arm = robot_right_arm
        self.robot_left_arm = robot_left_arm
        self.robot_setup = robot_setup
        self.task = None
        self._obs_config = obs_config
        self._initial_task_state = None
        self._start_right_arm_joint_pos = robot_right_arm.arm.get_joint_positions()
        self._starting_right_gripper_joint_pos = robot_right_arm.gripper.get_joint_positions()
        self._start_left_arm_joint_pos = robot_left_arm.arm.get_joint_positions()
        self._starting_left_gripper_joint_pos = robot_left_arm.gripper.get_joint_positions()
        self._workspace = Shape('workspace')
        self._workspace_boundary = SpawnBoundary([self._workspace])
        self._cam_wrist = VisionSensor('cam_wrist')
        self._cam_wrist2 = VisionSensor('cam_wrist#0')
        self._cam_front = VisionSensor('cam_front')
        self._cam_wrist_mask = VisionSensor('cam_wrist_mask')
        self._cam_wrist2_mask = VisionSensor('cam_wrist_mask#0')
        self._cam_front_mask = VisionSensor('cam_front_mask')
        self._has_init_task = self._has_init_episode = False
        self._variation_index = 0
        self._mode = mode

        self._initial_robot_state_right_arm = (robot_right_arm.arm.get_configuration_tree(),
                                     robot_right_arm.gripper.get_configuration_tree())
        self._initial_robot_state_left_arm = (robot_left_arm.arm.get_configuration_tree(),
                                     robot_left_arm.gripper.get_configuration_tree())

        self._ignore_collisions_for_current_waypoint = False

        # Set camera properties from observation config
        self._set_camera_properties()

        x, y, z = self._workspace.get_position()
        minx, maxx, miny, maxy, _, _ = self._workspace.get_bounding_box()
        self._workspace_minx = x - np.fabs(minx) - 0.2
        self._workspace_maxx = x + maxx + 0.2
        self._workspace_miny = y - np.fabs(miny) - 0.2
        self._workspace_maxy = y + maxy + 0.2
        self._workspace_minz = z
        self._workspace_maxz = z + 1.0  # 1M above workspace

        self.target_workspace_check = Dummy.create()
        self._step_callback = None

        self._robot_shapes_right_arm = self.robot_right_arm.arm.get_objects_in_tree(
            object_type=ObjectType.SHAPE)
        self._robot_shapes_left_arm = self.robot_left_arm.arm.get_objects_in_tree(
            object_type=ObjectType.SHAPE)

        # HACK: this is used to prevent the gripper detached bug, documented on Notion
        self.unused_robot = Robot(Panda(2), PandaGripper(2))
        self._start_unused_arm_joint_pos = self.unused_robot.arm.get_joint_positions()
        self._starting_unused_gripper_joint_pos = self.unused_robot.gripper.get_joint_positions()

        self.target_object_pos = None
        self.auto_crop_radius = 0.0

    def load(self, task: Task2Robots) -> None:
        """Loads the task and positions at the centre of the workspace.

        :param task: The task to load in the scene.
        """
        task.load()  # Load the task in to the scene

        # Set at the centre of the workspace
        task.get_base().set_position(self._workspace.get_position())

        self._initial_task_state = task.get_state()
        self.task = task
        self._initial_task_pose = task.boundary_root().get_orientation()
        self._has_init_task = self._has_init_episode = False
        self._variation_index = 0
        self.vlm = VLM()

    def unload(self) -> None:
        """Clears the scene. i.e. removes all tasks. """
        if self.task is not None:
            self.robot_right_arm.gripper.release()
            self.robot_left_arm.gripper.release()
            if self._has_init_task:
                self.task.cleanup_()
            self.task.unload()
        self.task = None
        self._variation_index = 0
        self.vlm = None
        del self.vlm
        gc.collect()

    def get_highres_front_image_in_pil(self):
        self._highres_front_cam.handle_explicitly()
        highres_front_rgb = self._highres_front_cam.capture_rgb()
        highres_front_rgb = np.clip((highres_front_rgb * 255.).astype(np.uint8), 0, 255)
        front_rgb = Image.fromarray(highres_front_rgb)
        return front_rgb

    def init_task(self) -> None:
        self.task.init_task()
        self._initial_task_state = self.task.get_state()
        self._has_init_task = True
        self._variation_index = 0

    def init_task_dominant_assistive(self, dominant: str = 'right') -> None:
        self.task.init_task(dominant)
        self._initial_task_state = self.task.get_state()
        self._has_init_task = True
        self._variation_index = 0

    def init_episode(self, index: int, randomly_place: bool=True,
                     max_attempts: int = 100) -> List[str]:
        """Calls the task init_episode and puts randomly in the workspace.
        """

        self._variation_index = index

        if not self._has_init_task:
            self.init_task()

        # Try a few times to init and place in the workspace
        attempts = 0
        descriptions = None
        while attempts < max_attempts:
            descriptions = self.task.init_episode(index)
            try:
                if (randomly_place and
                        not self.task.is_static_workspace()):
                    self._place_task()
                    if self.robot_right_arm.arm.check_arm_collision():
                        raise BoundaryError()
                    # NOTE: somehow enabling this would always raise BoundaryError.
                    if self.robot_left_arm.arm.check_arm_collision():
                        raise BoundaryError()
                self.task.validate()
                break
            except (BoundaryError, WaypointError) as e:
                self.task.cleanup_()
                self.task.restore_state(self._initial_task_state)
                attempts += 1
                if attempts >= max_attempts:
                    raise e

        # Let objects come to rest
        [self.pyrep.step() for _ in range(STEPS_BEFORE_EPISODE_START)]
        self._has_init_episode = True
        return descriptions

    def init_episode_dominant_assistive(self, index: int, randomly_place: bool=True,
                     max_attempts: int = 100, dominant: str = 'right') -> List[str]:
        """Calls the task init_episode and puts randomly in the workspace.
        """

        self._variation_index = index

        if not self._has_init_task:
            self.init_task_dominant_assistive(dominant)

        # Try a few times to init and place in the workspace
        attempts = 0
        descriptions = None
        print('What is the dominant arm in init_episode_dominant_assistive: ', dominant)
        while attempts < max_attempts:
            self.task.resize_object_of_interest()
            descriptions = self.task.init_episode(index, dominant)
            try:
                if (randomly_place and
                        not self.task.is_static_workspace()):
                    self._place_task()
                    if self.robot_right_arm.arm.check_arm_collision():
                        raise BoundaryError()
                    # NOTE: somehow enabling this would always raise BoundaryError.
                    if self.robot_left_arm.arm.check_arm_collision():
                        raise BoundaryError()
                self.task.validate_dominant_assistive(dominant)
                print('Task in init_episode_dominant_assistive: ', self.task.__class__.__name__)
                if self.task.__class__.__name__ == 'OpenDrawer' or self.task.__class__.__name__ == 'PutItemInDrawer':
                    yaw_angle_radian = self.task._drawer_frame.get_orientation()[2]
                    yaw_angle_degrees = math.degrees(yaw_angle_radian)
                    print('Yaw degrees: ', yaw_angle_degrees)
                    break
                elif self.task.__class__.__name__ == 'OpenJar':
                    if dominant == 'left' and math.dist(self.task.jar.get_position(), self.robot_left_arm.arm.get_position()) > math.dist(self.task.jar.get_position(), self.robot_right_arm.arm.get_position()):
                        # find an open jar configuration that is closer to the left arm than to the right arm
                        self.task.cleanup_()
                        self.task.restore_state(self._initial_task_state)
                        attempts += 1
                        print(f'!!!! Resetting jar position for dominant {dominant} in init_episode_dominant_assistive')
                        if attempts >= max_attempts:
                            raise 'Max attempts in resetting jar position in init_episode_dominant_assistive'
                    elif dominant == 'right' and math.dist(self.task.jar.get_position(), self.robot_left_arm.arm.get_position()) < math.dist(self.task.jar.get_position(), self.robot_right_arm.arm.get_position()):
                        # find an open jar configuration that is closer to the right arm than to the left arm
                        self.task.cleanup_()
                        self.task.restore_state(self._initial_task_state)
                        attempts += 1
                        print(f'!!!! Resetting jar position for dominant {dominant} in init_episode_dominant_assistive')
                        if attempts >= max_attempts:
                            raise 'Max attempts in resetting jar position in init_episode_dominant_assistive'
                    else:
                        print('Found acceptable jar position in init_episode_dominant_assistive')
                        print('jar to right arm dist in init_episode_dominant_assistive: ', math.dist(self.task.jar.get_position(), self.robot_right_arm.arm.get_position()))
                        print('jar to left arm dist in init_episode_dominant_assistive: ', math.dist(self.task.jar.get_position(), self.robot_left_arm.arm.get_position()))
                        break
                elif self.task.__class__.__name__ == 'HandOverItem':
                    if dominant == 'right' and math.dist(self.task.cube.get_position(), self.robot_left_arm.arm.get_position()) > math.dist(self.task.cube.get_position(), self.robot_right_arm.arm.get_position()):
                        self.task.cleanup_()
                        self.task.restore_state(self._initial_task_state)
                        attempts += 1
                        print(f'!!!! Resetting cube position for dominant {dominant} in init_episode_dominant_assistive')
                        if attempts >= max_attempts:
                            raise 'Max attempts in resetting jar position in init_episode_dominant_assistive'
                    elif dominant == 'left' and math.dist(self.task.cube.get_position(), self.robot_left_arm.arm.get_position()) < math.dist(self.task.cube.get_position(), self.robot_right_arm.arm.get_position()):
                        self.task.cleanup_()
                        self.task.restore_state(self._initial_task_state)
                        attempts += 1
                        print(f'!!!! Resetting cube position for dominant {dominant} in init_episode_dominant_assistive')
                        if attempts >= max_attempts:
                            raise 'Max attempts in resetting jar position in init_episode_dominant_assistive'
                    else:
                        print('Found acceptable item position in init_episode_dominant_assistive')
                        print('cube to right arm dist in init_episode_dominant_assistive: ', math.dist(self.task.cube.get_position(), self.robot_right_arm.arm.get_position()))
                        print('cube to left arm dist in init_episode_dominant_assistive: ', math.dist(self.task.cube.get_position(), self.robot_left_arm.arm.get_position()))
                        break
                else:
                    raise NotImplementedError
            except (BoundaryError, WaypointError) as e:
                # print('Exception in init_episode_dominant_assistive: ', e)
                self.reset()
                self.task.cleanup_()
                self.task.restore_state(self._initial_task_state)
                attempts += 1
                if attempts >= max_attempts:
                    raise e

        # Let objects come to rest
        [self.pyrep.step() for _ in range(STEPS_BEFORE_EPISODE_START)]
        self._has_init_episode = True

        return descriptions

    def reset(self) -> None:
        """Resets the joint angles. """
        self.robot_right_arm.gripper.release()
        self.robot_left_arm.gripper.release()

        arm, gripper = self._initial_robot_state_right_arm
        self.pyrep.set_configuration_tree(arm)
        self.pyrep.set_configuration_tree(gripper)
        self.robot_right_arm.arm.set_joint_positions(self._start_right_arm_joint_pos, disable_dynamics=True)
        self.robot_right_arm.arm.set_joint_target_velocities(
            [0] * len(self.robot_right_arm.arm.joints))
        self.robot_right_arm.gripper.set_joint_positions(
            self._starting_right_gripper_joint_pos, disable_dynamics=True)
        self.robot_right_arm.gripper.set_joint_target_velocities(
            [0] * len(self.robot_right_arm.gripper.joints))

        left_arm, left_gripper = self._initial_robot_state_left_arm
        self.pyrep.set_configuration_tree(left_arm)
        self.pyrep.set_configuration_tree(left_gripper)
        self.robot_left_arm.arm.set_joint_positions(self._start_left_arm_joint_pos, disable_dynamics=True)
        self.robot_left_arm.arm.set_joint_target_velocities(
            [0] * len(self.robot_left_arm.arm.joints))
        self.robot_left_arm.gripper.set_joint_positions(
            self._starting_left_gripper_joint_pos, disable_dynamics=True)
        self.robot_left_arm.gripper.set_joint_target_velocities(
            [0] * len(self.robot_left_arm.gripper.joints))


        # HACK: this is used to prevent the gripper detached bug, documented on Notion
        self.unused_robot.arm.set_joint_positions(self._start_unused_arm_joint_pos, disable_dynamics=True)
        self.unused_robot.arm.set_joint_target_velocities(
            [0] * len(self.unused_robot.arm.joints))
        self.unused_robot.gripper.set_joint_positions(
            self._starting_unused_gripper_joint_pos, disable_dynamics=True)
        self.unused_robot.gripper.set_joint_target_velocities(
            [0] * len(self.unused_robot.gripper.joints))

        self.target_object_pos = None
        self.auto_crop_radius = 0.0

        if self.task is not None and self._has_init_task:
            self.task.cleanup_()
            self.task.restore_state(self._initial_task_state)
        self.task.set_initial_objects_in_scene()

    def get_observation(self) -> Observation2Robots:
        tip_right = self.robot_right_arm.arm.get_tip()
        tip_left = self.robot_left_arm.arm.get_tip()

        joint_forces_right = None
        if self._obs_config.joint_forces_right:
            fs = self.robot_right_arm.arm.get_joint_forces()
            vels = self.robot_right_arm.arm.get_joint_target_velocities()
            joint_forces_right = self._obs_config.joint_forces_noise_right.apply(
                np.array([-f if v < 0 else f for f, v in zip(fs, vels)]))

        joint_forces_left = None
        if self._obs_config.joint_forces_left:
            fs = self.robot_left_arm.arm.get_joint_forces()
            vels = self.robot_left_arm.arm.get_joint_target_velocities()
            joint_forces_left = self._obs_config.joint_forces_noise_left.apply(
                np.array([-f if v < 0 else f for f, v in zip(fs, vels)]))

        ee_forces_flat_right = None
        ee_forces_flat_left = None
        if self._obs_config.gripper_right_touch_forces and self._obs_config.gripper_left_touch_forces:
            ee_forces = self.robot_right_arm.gripper.get_touch_sensor_forces()
            ee_forces_flat_right = []
            for eef in ee_forces:
                ee_forces_flat_right.extend(eef)
            ee_forces_flat_right = np.array(ee_forces_flat_right)

            ee_forces = self.robot_left_arm.gripper.get_touch_sensor_forces()
            ee_forces_flat_left = []
            for eef in ee_forces:
                ee_forces_flat_left.extend(eef)
            ee_forces_flat_left = np.array(ee_forces_flat_left)

        wc_ob = self._obs_config.wrist_camera
        wc2_ob = self._obs_config.wrist2_camera
        fc_ob = self._obs_config.front_camera

        wc_mask_fn, wc2_mask_fn, fc_mask_fn = [
            (rgb_handles_to_mask if c.masks_as_one_channel else lambda x: x
             ) for c in [wc_ob, wc2_ob, fc_ob]]

        def get_rgb_depth(sensor: VisionSensor, get_rgb: bool, get_depth: bool,
                          get_pcd: bool, rgb_noise: NoiseModel,
                          depth_noise: NoiseModel, depth_in_meters: bool):
            rgb = depth = pcd = None
            if sensor is not None and (get_rgb or get_depth):
                sensor.handle_explicitly()
                if get_rgb:
                    rgb = sensor.capture_rgb()
                    if rgb_noise is not None:
                        rgb = rgb_noise.apply(rgb)
                    rgb = np.clip((rgb * 255.).astype(np.uint8), 0, 255)
                if get_depth or get_pcd:
                    depth = sensor.capture_depth(depth_in_meters)
                    if depth_noise is not None:
                        depth = depth_noise.apply(depth)
                if get_pcd:
                    depth_m = depth
                    if not depth_in_meters:
                        near = sensor.get_near_clipping_plane()
                        far = sensor.get_far_clipping_plane()
                        depth_m = near + depth * (far - near)
                    pcd = sensor.pointcloud_from_depth(depth_m)
                    if not get_depth:
                        depth = None
            return rgb, depth, pcd

        def get_mask(sensor: VisionSensor, mask_fn):
            mask = None
            if sensor is not None:
                sensor.handle_explicitly()
                mask = mask_fn(sensor.capture_rgb())
            return mask

        wrist_rgb, wrist_depth, wrist_pcd = get_rgb_depth(
            self._cam_wrist, wc_ob.rgb, wc_ob.depth, wc_ob.point_cloud,
            wc_ob.rgb_noise, wc_ob.depth_noise, wc_ob.depth_in_meters)
        wrist2_rgb, wrist2_depth, wrist2_pcd = get_rgb_depth(
            self._cam_wrist2, wc2_ob.rgb, wc2_ob.depth, wc2_ob.point_cloud,
            wc2_ob.rgb_noise, wc2_ob.depth_noise, wc2_ob.depth_in_meters)
        front_rgb, front_depth, front_pcd = get_rgb_depth(
            self._cam_front, fc_ob.rgb, fc_ob.depth, fc_ob.point_cloud,
            fc_ob.rgb_noise, fc_ob.depth_noise, fc_ob.depth_in_meters)

        wrist_mask = get_mask(self._cam_wrist_mask,
                              wc_mask_fn) if wc_ob.mask else None
        front_mask = get_mask(self._cam_front_mask,
                              fc_mask_fn) if fc_ob.mask else None
        wrist2_mask = get_mask(self._cam_wrist2_mask,
                              wc2_mask_fn) if wc2_ob.mask else None

        target_object_pos = None
        obs = Observation2Robots(
            wrist_rgb=wrist_rgb,
            wrist_depth=wrist_depth,
            wrist_point_cloud=wrist_pcd,
            wrist2_rgb=wrist2_rgb,
            wrist2_depth=wrist2_depth,
            wrist2_point_cloud=wrist2_pcd,
            front_rgb=front_rgb,
            front_depth=front_depth,
            front_point_cloud=front_pcd,
            wrist_mask=wrist_mask,
            wrist2_mask=wrist2_mask,
            front_mask=front_mask,
            joint_velocities_right=(
                self._obs_config.joint_velocities_noise_right.apply(
                    np.array(self.robot_right_arm.arm.get_joint_velocities()))
                if self._obs_config.joint_velocities_right else None),
            joint_positions_right=(
                self._obs_config.joint_positions_noise_right.apply(
                    np.array(self.robot_right_arm.arm.get_joint_positions()))
                if self._obs_config.joint_positions_right else None),
            joint_forces_right=(joint_forces_right
                          if self._obs_config.joint_forces_right else None),
            joint_velocities_left=(
                self._obs_config.joint_velocities_noise_left.apply(
                    np.array(self.robot_left_arm.arm.get_joint_velocities()))
                if self._obs_config.joint_velocities_left else None),
            joint_positions_left=(
                self._obs_config.joint_positions_noise_left.apply(
                    np.array(self.robot_left_arm.arm.get_joint_positions()))
                if self._obs_config.joint_positions_left else None),
            joint_forces_left=(joint_forces_left
                          if self._obs_config.joint_forces_left else None),
            gripper_right_open=(
                (1.0 if self.robot_right_arm.gripper.get_open_amount()[0] > 0.95 else 0.0) # Changed from 0.9 to 0.95 because objects, the gripper does not close completely
                if self._obs_config.gripper_right_open else None),
            gripper_right_pose=(
                np.array(tip_right.get_pose())
                if self._obs_config.gripper_right_pose else None),
            gripper_right_matrix=(
                tip_right.get_matrix()
                if self._obs_config.gripper_right_matrix else None),
            gripper_right_touch_forces=(
                ee_forces_flat_right
                if self._obs_config.gripper_right_touch_forces else None),
            gripper_right_joint_positions=(
                np.array(self.robot_right_arm.gripper.get_joint_positions())
                if self._obs_config.gripper_right_joint_positions else None),
            gripper_left_open=(
                (1.0 if self.robot_left_arm.gripper.get_open_amount()[0] > 0.95 else 0.0) # Changed from 0.9 to 0.95 because objects, the gripper does not close completely
                if self._obs_config.gripper_left_open else None),
            gripper_left_pose=(
                np.array(tip_left.get_pose())
                if self._obs_config.gripper_left_pose else None),
            gripper_left_matrix=(
                tip_left.get_matrix()
                if self._obs_config.gripper_left_matrix else None),
            gripper_left_touch_forces=(
                ee_forces_flat_left
                if self._obs_config.gripper_left_touch_forces else None),
            gripper_left_joint_positions=(
                np.array(self.robot_left_arm.gripper.get_joint_positions())
                if self._obs_config.gripper_left_joint_positions else None),
            task_low_dim_state=(
                self.task.get_low_dim_state() if
                self._obs_config.task_low_dim_state else None),
            ignore_collisions=(
                np.array((1.0 if self._ignore_collisions_for_current_waypoint else 0.0))
                if self._obs_config.record_ignore_collisions else None),
            misc=self._get_misc(),
            target_object_pos=target_object_pos,
            auto_crop_radius=None)
        obs = self.task.decorate_observation(obs)
        return obs

    def get_observation_vlm(self) -> Observation2Robots:
        tip_right = self.robot_right_arm.arm.get_tip()
        tip_left = self.robot_left_arm.arm.get_tip()

        joint_forces_right = None
        if self._obs_config.joint_forces_right:
            fs = self.robot_right_arm.arm.get_joint_forces()
            vels = self.robot_right_arm.arm.get_joint_target_velocities()
            joint_forces_right = self._obs_config.joint_forces_noise_right.apply(
                np.array([-f if v < 0 else f for f, v in zip(fs, vels)]))

        joint_forces_left = None
        if self._obs_config.joint_forces_left:
            fs = self.robot_left_arm.arm.get_joint_forces()
            vels = self.robot_left_arm.arm.get_joint_target_velocities()
            joint_forces_left = self._obs_config.joint_forces_noise_left.apply(
                np.array([-f if v < 0 else f for f, v in zip(fs, vels)]))

        ee_forces_flat_right = None
        ee_forces_flat_left = None
        if self._obs_config.gripper_right_touch_forces and self._obs_config.gripper_left_touch_forces:
            ee_forces = self.robot_right_arm.gripper.get_touch_sensor_forces()
            ee_forces_flat_right = []
            for eef in ee_forces:
                ee_forces_flat_right.extend(eef)
            ee_forces_flat_right = np.array(ee_forces_flat_right)

            ee_forces = self.robot_left_arm.gripper.get_touch_sensor_forces()
            ee_forces_flat_left = []
            for eef in ee_forces:
                ee_forces_flat_left.extend(eef)
            ee_forces_flat_left = np.array(ee_forces_flat_left)

        wc_ob = self._obs_config.wrist_camera
        wc2_ob = self._obs_config.wrist2_camera
        fc_ob = self._obs_config.front_camera

        wc_mask_fn, wc2_mask_fn, fc_mask_fn = [
            (rgb_handles_to_mask if c.masks_as_one_channel else lambda x: x
             ) for c in [wc_ob, wc2_ob, fc_ob]]

        def get_rgb_depth(sensor: VisionSensor, get_rgb: bool, get_depth: bool,
                          get_pcd: bool, rgb_noise: NoiseModel,
                          depth_noise: NoiseModel, depth_in_meters: bool):
            rgb = depth = pcd = None
            if sensor is not None and (get_rgb or get_depth):
                sensor.handle_explicitly()
                if get_rgb:
                    rgb = sensor.capture_rgb()
                    if rgb_noise is not None:
                        rgb = rgb_noise.apply(rgb)
                    rgb = np.clip((rgb * 255.).astype(np.uint8), 0, 255)
                if get_depth or get_pcd:
                    depth = sensor.capture_depth(depth_in_meters)
                    if depth_noise is not None:
                        depth = depth_noise.apply(depth)
                if get_pcd:
                    depth_m = depth
                    if not depth_in_meters:
                        near = sensor.get_near_clipping_plane()
                        far = sensor.get_far_clipping_plane()
                        depth_m = near + depth * (far - near)
                    pcd = sensor.pointcloud_from_depth(depth_m)
                    if not get_depth:
                        depth = None
            return rgb, depth, pcd

        def get_mask(sensor: VisionSensor, mask_fn):
            mask = None
            if sensor is not None:
                sensor.handle_explicitly()
                mask = mask_fn(sensor.capture_rgb())
            return mask

        wrist_rgb, wrist_depth, wrist_pcd = get_rgb_depth(
            self._cam_wrist, wc_ob.rgb, wc_ob.depth, wc_ob.point_cloud,
            wc_ob.rgb_noise, wc_ob.depth_noise, wc_ob.depth_in_meters)
        wrist2_rgb, wrist2_depth, wrist2_pcd = get_rgb_depth(
            self._cam_wrist2, wc2_ob.rgb, wc2_ob.depth, wc2_ob.point_cloud,
            wc2_ob.rgb_noise, wc2_ob.depth_noise, wc2_ob.depth_in_meters)
        front_rgb, front_depth, front_pcd = get_rgb_depth(
            self._cam_front, fc_ob.rgb, fc_ob.depth, fc_ob.point_cloud,
            fc_ob.rgb_noise, fc_ob.depth_noise, fc_ob.depth_in_meters)

        wrist_mask = get_mask(self._cam_wrist_mask,
                              wc_mask_fn) if wc_ob.mask else None
        front_mask = get_mask(self._cam_front_mask,
                              fc_mask_fn) if fc_ob.mask else None
        wrist2_mask = get_mask(self._cam_wrist2_mask,
                              wc2_mask_fn) if wc2_ob.mask else None

        if self.target_object_pos is None:
            # method 1: use VLM to get target_object_pos
            # # only get self.target_object_pos at timestep = 0
            # front_rgb_pil = Image.fromarray(front_rgb)
            # front_rgb_pil = front_rgb_pil.resize((1024, 1024))
            # points = front_pcd
            # self.target_object_pos, _ = self.vlm.get_target_object_world_coords(front_rgb_pil, points, self.task.name) # for debugging...
            # # self.target_object_pos, _ = self.vlm.get_target_object_world_coords(front_rgb_pil, points, self.task.name, debug=False)
            # print('!!! VLM target_object_pos: ', self.target_object_pos)

            # method 2: use the simulator to get ground truth target_object_pos
            points = front_pcd
            mask = get_mask(self._cam_front_mask, rgb_handles_to_mask)
            if self.task.name in ['OpenDrawer', 'open_drawer', 'PutItemInDrawer', 'put_item_in_drawer']:
                object_handle = Shape('drawer_middle').get_handle()
            elif self.task.name in ['OpenJar', 'open_jar']:
                object_handle = [Shape('jar_lid0').get_handle(), Shape('jar0').get_handle()]
            elif self.task.name in ['HandOverItem', 'hand_over_item']:
                object_handle = Shape('cube').get_handle()
            else:
                print('!!!!!!!!!! NotImplementedError in get_observation_vlm() !!!!!!!!!!')
                raise NotImplementedError
            obj_points = points[np.isin(mask, object_handle)]
            if len(obj_points) == 0:
                raise ValueError(f"Object {object_handle} not found in the scene")

            # for debugging method 2....
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(10, 10))
            # plt.imshow(front_rgb)
            # object_mask = np.isin(mask, object_handle)
            # plt.imshow(object_mask, cmap='gray', alpha=0.7)
            # y_indices, x_indices = np.nonzero(object_mask)
            # cX = np.mean(x_indices).astype(int)
            # cY = np.mean(y_indices).astype(int)
            # plt.plot(cX, cY, marker='v', color="red")
            # plt.savefig('debug_gt_mask.png')
            # plt.close()

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj_points)
            pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
            obj_points = np.asarray(pcd_downsampled.points)
            self.target_object_pos = np.mean(obj_points, axis=0)
            print('!!! Ground truth target_object_pos: ', self.target_object_pos)

        # TODO: we only saved auto_crop_radius for drawer tasks
        if self.auto_crop_radius == 0.0 and self.task.name in ['OpenDrawer', 'open_drawer', 'PutItemInDrawer', 'put_item_in_drawer']:
            auto_crop_padding = 0.05
            obj_frame_handle = Shape('drawer_frame').get_handle()
            entire_obj_points = points[np.isin(mask, obj_frame_handle)]
            obj_x_min = np.min(entire_obj_points[:, 0])
            obj_x_max = np.max(entire_obj_points[:, 0])
            obj_y_min = np.min(entire_obj_points[:, 1])
            obj_y_max = np.max(entire_obj_points[:, 1])
            obj_z_min = np.min(entire_obj_points[:, 2])
            obj_z_max = np.max(entire_obj_points[:, 2])
            obj_max_dim = np.max([obj_x_max - obj_x_min, obj_y_max - obj_y_min, obj_z_max - obj_z_min])
            self.auto_crop_radius = obj_max_dim + auto_crop_padding
            print('!!! Auto crop radius: ', self.auto_crop_radius)

        obs = Observation2Robots(
            wrist_rgb=wrist_rgb,
            wrist_depth=wrist_depth,
            wrist_point_cloud=wrist_pcd,
            wrist2_rgb=wrist2_rgb,
            wrist2_depth=wrist2_depth,
            wrist2_point_cloud=wrist2_pcd,
            front_rgb=front_rgb,
            front_depth=front_depth,
            front_point_cloud=front_pcd,
            wrist_mask=wrist_mask,
            wrist2_mask=wrist2_mask,
            front_mask=front_mask,
            joint_velocities_right=(
                self._obs_config.joint_velocities_noise_right.apply(
                    np.array(self.robot_right_arm.arm.get_joint_velocities()))
                if self._obs_config.joint_velocities_right else None),
            joint_positions_right=(
                self._obs_config.joint_positions_noise_right.apply(
                    np.array(self.robot_right_arm.arm.get_joint_positions()))
                if self._obs_config.joint_positions_right else None),
            joint_forces_right=(joint_forces_right
                          if self._obs_config.joint_forces_right else None),
            joint_velocities_left=(
                self._obs_config.joint_velocities_noise_left.apply(
                    np.array(self.robot_left_arm.arm.get_joint_velocities()))
                if self._obs_config.joint_velocities_left else None),
            joint_positions_left=(
                self._obs_config.joint_positions_noise_left.apply(
                    np.array(self.robot_left_arm.arm.get_joint_positions()))
                if self._obs_config.joint_positions_left else None),
            joint_forces_left=(joint_forces_left
                          if self._obs_config.joint_forces_left else None),
            gripper_right_open=(
                (1.0 if self.robot_right_arm.gripper.get_open_amount()[0] > 0.95 else 0.0) # Changed from 0.9 to 0.95 because objects, the gripper does not close completely
                if self._obs_config.gripper_right_open else None),
            gripper_right_pose=(
                np.array(tip_right.get_pose())
                if self._obs_config.gripper_right_pose else None),
            gripper_right_matrix=(
                tip_right.get_matrix()
                if self._obs_config.gripper_right_matrix else None),
            gripper_right_touch_forces=(
                ee_forces_flat_right
                if self._obs_config.gripper_right_touch_forces else None),
            gripper_right_joint_positions=(
                np.array(self.robot_right_arm.gripper.get_joint_positions())
                if self._obs_config.gripper_right_joint_positions else None),
            gripper_left_open=(
                (1.0 if self.robot_left_arm.gripper.get_open_amount()[0] > 0.95 else 0.0) # Changed from 0.9 to 0.95 because objects, the gripper does not close completely
                if self._obs_config.gripper_left_open else None),
            gripper_left_pose=(
                np.array(tip_left.get_pose())
                if self._obs_config.gripper_left_pose else None),
            gripper_left_matrix=(
                tip_left.get_matrix()
                if self._obs_config.gripper_left_matrix else None),
            gripper_left_touch_forces=(
                ee_forces_flat_left
                if self._obs_config.gripper_left_touch_forces else None),
            gripper_left_joint_positions=(
                np.array(self.robot_left_arm.gripper.get_joint_positions())
                if self._obs_config.gripper_left_joint_positions else None),
            task_low_dim_state=(
                self.task.get_low_dim_state() if
                self._obs_config.task_low_dim_state else None),
            ignore_collisions=(
                np.array((1.0 if self._ignore_collisions_for_current_waypoint else 0.0))
                if self._obs_config.record_ignore_collisions else None),
            misc=self._get_misc(),
            target_object_pos=self.target_object_pos,
            auto_crop_radius=self.auto_crop_radius)
        obs = self.task.decorate_observation(obs)
        return obs

    def step(self):
        self.pyrep.step()
        self.task.step()
        if self._step_callback is not None:
            self._step_callback()

    def register_step_callback(self, func):
        self._step_callback = func

    def use_existing_waypoint_to_set_a_new_waypoint(self, demo, point, waypoints_labels, i, record, callable_each_step, xyz_indices=None, offsets=None, gripper=None):
        old_pos = point._waypoint.get_position().copy()
        old_orientation = point._waypoint.get_orientation().copy()
        # print('Original pose: ', old_pos) # for debugging

        if gripper is None:
            # use point's position as the reference for applying offsets
            new_pos = point._waypoint.get_position()
        else:
            # use gripper's position as the reference for applying offsets
            new_pos = gripper.get_position()
            point._waypoint.set_orientation(gripper.get_orientation())

        for xyz_index, offset in zip(xyz_indices, offsets):
            new_pos[xyz_index] += offset
        point._waypoint.set_position(new_pos)
        # print('New pose: ', new_pos) # for debugging

        self._ignore_collisions_for_current_waypoint = point._ignore_collisions
        point.start_of_path()
        if waypoints_labels[i] == 'left':
            grasped_objects = self.robot_left_arm.gripper.get_grasped_objects()
            colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                object_type=ObjectType.SHAPE) if s not in grasped_objects
                                and s not in self._robot_shapes_left_arm and s.is_collidable()
                                and self.robot_left_arm.arm.check_arm_collision(s)]
        elif waypoints_labels[i] == 'right':
            grasped_objects = self.robot_right_arm.gripper.get_grasped_objects()
            colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                object_type=ObjectType.SHAPE) if s not in grasped_objects
                                and s not in self._robot_shapes_right_arm and s.is_collidable()
                                and self.robot_right_arm.arm.check_arm_collision(s)]
        [s.set_collidable(False) for s in colliding_shapes]
        try:
            path = point.get_path()
            [s.set_collidable(True) for s in colliding_shapes]
        except ConfigurationPathError as e:
            [s.set_collidable(True) for s in colliding_shapes]
            raise DemoError(
                'Could not get a path for waypoint %d.' % i,
                self.task) from e
        ext = point.get_ext()
        path.visualize()

        done = False
        success = False
        while not done:
            done = path.step()
            self.step()
            # self._demo_record_step(demo, record, callable_each_step)
            success, term = self.task.success()

        point.end_of_path()

        path.clear_visualization()

        # if len(ext) > 0:
        #     self._demo_record_step(demo, record, callable_each_step)

        point._waypoint.set_position(old_pos)
        point._waypoint.set_orientation(old_orientation)
        # print('Restore to original pose: ', point._waypoint.get_position()) # for debugging

    def get_demo_hand_over_item_noises_starting_states_dominant_assistive(self, record: bool = True,
                 callable_each_step: Callable[[Observation2Robots], None] = None,
                 randomly_place: bool = True, dominant: str = 'right') -> Demo:
        if not self._has_init_task:
            self.init_task_dominant_assistive(dominant)
        if not self._has_init_episode:
            self.init_episode_dominant_assistive(self._variation_index,
                                randomly_place=randomly_place, dominant=dominant)
        self._has_init_episode = False

        waypoints, waypoints_labels = self.task.get_waypoints_dominant_assistive(dominant=dominant)
        if len(waypoints) == 0:
            raise NoWaypointsError(
                'No waypoints were found.', self.task)

        demo = []
        if record:
            self.pyrep.step()  # Need this here or get_force doesn't work...
            # demo.append(self.get_observation())

        print('waypoints_labels: ', waypoints_labels)
        while True:
            success = False
            self._ignore_collisions_for_current_waypoint = False
            for i, point in enumerate(waypoints):
                print('waypoint i: ', i)
                self._ignore_collisions_for_current_waypoint = point._ignore_collisions
                point.start_of_path()
                if point.skip:
                    continue
                if waypoints_labels[i] == 'left':
                    grasped_objects = self.robot_left_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_left_arm and s.is_collidable()
                                        and self.robot_left_arm.arm.check_arm_collision(s)]
                elif waypoints_labels[i] == 'right':
                    grasped_objects = self.robot_right_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_right_arm and s.is_collidable()
                                        and self.robot_right_arm.arm.check_arm_collision(s)]
                [s.set_collidable(False) for s in colliding_shapes]
                try:
                    path = point.get_path()
                    [s.set_collidable(True) for s in colliding_shapes]
                except ConfigurationPathError as e:
                    [s.set_collidable(True) for s in colliding_shapes]
                    raise DemoError(
                        'Could not get a path for waypoint %d.' % i,
                        self.task) from e
                ext = point.get_ext()
                path.visualize()

                done = False
                success = False
                while not done:
                    done = path.step()
                    self.step()
                    self._demo_record_step(demo, record, callable_each_step)
                    success, term = self.task.success()

                point.end_of_path()

                path.clear_visualization()

                if len(ext) > 0:
                    contains_param = False
                    start_of_bracket = -1
                    if waypoints_labels[i] == 'left':
                        gripper = self.robot_left_arm.gripper
                    elif waypoints_labels[i] == 'right':
                        gripper = self.robot_right_arm.gripper
                    if 'open_gripper(' in ext:
                        gripper.release()
                        start_of_bracket = ext.index('open_gripper(') + 13
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(1.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)
                    elif 'close_gripper(' in ext:
                        start_of_bracket = ext.index('close_gripper(') + 14
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(0.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)

                    if contains_param:
                        rest = ext[start_of_bracket:]
                        num = float(rest[:rest.index(')')])
                        done = False
                        while not done:
                            done = gripper.actuate(num, 0.04)
                            self.pyrep.step()
                            self.task.step()
                            if self._obs_config.record_gripper_closing:
                                self._demo_record_step(
                                    demo, record, callable_each_step)

                    if 'close_gripper(' in ext:
                        for g_obj in self.task.get_graspable_objects():
                            gripper.grasp(g_obj)

                    self._demo_record_step(demo, record, callable_each_step)
                
                if i == 4:
                    if dominant == 'right':
                        print('opening left gripper...') # for debugging...
                        gripper = self.robot_left_arm.gripper
                    else:
                        print('opening right gripper...') # for debugging...
                        gripper = self.robot_right_arm.gripper
                    gripper.release()
                    done = False
                    while not done:
                        done = gripper.actuate(1.0, 0.04)
                        self.pyrep.step()
                        self.task.step()
                        if self._obs_config.record_gripper_closing:
                            self._demo_record_step(
                                demo, record, callable_each_step)
                    print('opening left gripper done!!!') # for debugging...
                    self._demo_record_step(demo, record, callable_each_step)

            if not self.task.should_repeat_waypoints() or success:
                break

        # Some tasks may need additional physics steps
        # (e.g. ball rowling to goal)
        if not success:
            for _ in range(10):
                self.pyrep.step()
                self.task.step()
                self._demo_record_step(demo, record, callable_each_step)
                success, term = self.task.success()
                if success:
                    break

        success, term = self.task.success()
        if not success:
            raise DemoError('Demo was completed, but was not successful.',
                            self.task)
        return Demo(demo)

    def get_demo_put_item_in_drawer_close_to_drawer_dominant_assistive(self, record: bool = True,
                 callable_each_step: Callable[[Observation2Robots], None] = None,
                 randomly_place: bool = True, dominant: str = 'right') -> Demo:
        """
        get_demo function for put_item_in_drawer. Specifically, we use waypoints to
        navigate the left and right arms to areas close to the drawer, without recording any demos.
        Once we reach those areas, we start recording demos. The purpose of this function is to
        record demonstrations that start with the left and right arms close to the drawer for
        learning contact-rich manipulation.
        """
        if not self._has_init_task:
            self.init_task_dominant_assistive(dominant)
        if not self._has_init_episode:
            self.init_episode_dominant_assistive(self._variation_index,
                                randomly_place=randomly_place, dominant=dominant)
        self._has_init_episode = False

        waypoints, waypoints_labels = self.task.get_waypoints_dominant_assistive(dominant=dominant)
        if len(waypoints) == 0:
            raise NoWaypointsError(
                'No waypoints were found.', self.task)

        demo = []
        if record:
            self.pyrep.step()  # Need this here or get_force doesn't work...
            # demo.append(self.get_observation())

        # do pre-grasp for the left and right arms first, so we want to skip (not record) them
        waypoints = self.swap_positions(waypoints, 3, 4)
        waypoints_labels = self.swap_positions(waypoints_labels, 3, 4)
        waypoints = self.swap_positions(waypoints, 2, 3)
        waypoints_labels = self.swap_positions(waypoints_labels, 2, 3)
        indicies_to_skip_recording = 2

        while True:
            success = False
            self._ignore_collisions_for_current_waypoint = False
            for i, point in enumerate(waypoints):
                if i == 1:
                    print(f'Adding randomizations to {waypoints_labels[i]} {i}')
                    x_offset = np.random.uniform(low=-0.0175, high=0.0175)      # left and right
                    y_offset = np.random.uniform(low=-0.01, high=0.02)          # distance away from the drawer
                    z_offset = np.random.uniform(low=-0.0175, high=0.0175)      # up and down
                    offsets = [x_offset, y_offset, z_offset]

                    # old_pos = point._waypoint.get_position().copy() # for debugging
                    new_pos = point._waypoint.get_position()
                    for xyz_index, offset in zip([0, 1, 2], offsets):
                        new_pos[xyz_index] += offset
                    point._waypoint.set_position(new_pos)
                    # print(f'old_pos: {old_pos}, new_pos: {new_pos}') # for debugging
                    # print(f'new_pos: {new_pos}') # for debugging
                elif i == 2:
                    print(f'Adding randomizations to {waypoints_labels[i]} {i}')
                    x_offset = np.random.uniform(low=-0.0125, high=0.0125)     # left and right
                    y_offset = np.random.uniform(low=-0.0125, high=0.0125)     # distance away from the drawer
                    z_offset = np.random.uniform(low=-0.01, high=0.005)       # up and down
                    offsets = [x_offset, y_offset, z_offset]

                    # old_pos = point._waypoint.get_position().copy() # for debugging
                    new_pos = point._waypoint.get_position()
                    for xyz_index, offset in zip([0, 1, 2], offsets):
                        new_pos[xyz_index] += offset
                    point._waypoint.set_position(new_pos)
                    # print(f'old_pos: {old_pos}, new_pos: {new_pos}') # for debugging
                    # print(f'new_pos: {new_pos}') # for debugging

                self._ignore_collisions_for_current_waypoint = point._ignore_collisions
                point.start_of_path()
                if point.skip:
                    continue
                if waypoints_labels[i] == 'left':
                    grasped_objects = self.robot_left_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_left_arm and s.is_collidable()
                                        and self.robot_left_arm.arm.check_arm_collision(s)]
                elif waypoints_labels[i] == 'right':
                    grasped_objects = self.robot_right_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_right_arm and s.is_collidable()
                                        and self.robot_right_arm.arm.check_arm_collision(s)]
                [s.set_collidable(False) for s in colliding_shapes]
                try:
                    path = point.get_path()
                    [s.set_collidable(True) for s in colliding_shapes]
                except ConfigurationPathError as e:
                    [s.set_collidable(True) for s in colliding_shapes]
                    raise DemoError(
                        'Could not get a path for waypoint %d.' % i,
                        self.task) from e
                ext = point.get_ext()
                path.visualize()

                done = False
                success = False
                while not done:
                    done = path.step()
                    self.step()
                    if i > indicies_to_skip_recording:
                        self._demo_record_step(demo, record, callable_each_step)
                    success, term = self.task.success()

                point.end_of_path()

                path.clear_visualization()

                if len(ext) > 0:
                    contains_param = False
                    start_of_bracket = -1
                    if waypoints_labels[i] == 'left':
                        gripper = self.robot_left_arm.gripper
                    elif waypoints_labels[i] == 'right':
                        gripper = self.robot_right_arm.gripper
                    if 'open_gripper(' in ext:
                        gripper.release()
                        start_of_bracket = ext.index('open_gripper(') + 13
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(1.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing and i > indicies_to_skip_recording:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)
                    elif 'close_gripper(' in ext:
                        start_of_bracket = ext.index('close_gripper(') + 14
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(0.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing and i > indicies_to_skip_recording:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)

                    if contains_param:
                        rest = ext[start_of_bracket:]
                        num = float(rest[:rest.index(')')])
                        done = False
                        while not done:
                            done = gripper.actuate(num, 0.04)
                            self.pyrep.step()
                            self.task.step()
                            if self._obs_config.record_gripper_closing and i > indicies_to_skip_recording:
                                self._demo_record_step(
                                    demo, record, callable_each_step)

                    if 'close_gripper(' in ext:
                        for g_obj in self.task.get_graspable_objects():
                            gripper.grasp(g_obj)

                    if i > indicies_to_skip_recording:
                        self._demo_record_step(demo, record, callable_each_step)

            if not self.task.should_repeat_waypoints() or success:
                break

        # Some tasks may need additional physics steps
        # (e.g. ball rowling to goal)
        if not success:
            for _ in range(10):
                self.pyrep.step()
                self.task.step()
                self._demo_record_step(demo, record, callable_each_step)
                success, term = self.task.success()
                if success:
                    break

        success, term = self.task.success()
        if not success:
            raise DemoError('Demo was completed, but was not successful.',
                            self.task)
        return Demo(demo)

    def get_demo_put_item_in_drawer_noises_starting_states_dominant_assistive(self, record: bool = True,
                 callable_each_step: Callable[[Observation2Robots], None] = None,
                 randomly_place: bool = True, dominant: str = 'right') -> Demo:
        if not self._has_init_task:
            self.init_task_dominant_assistive(dominant)
        if not self._has_init_episode:
            self.init_episode_dominant_assistive(self._variation_index,
                                randomly_place=randomly_place, dominant=dominant)
        self._has_init_episode = False

        waypoints, waypoints_labels = self.task.get_waypoints_dominant_assistive(dominant=dominant)
        if len(waypoints) == 0:
            raise NoWaypointsError(
                'No waypoints were found.', self.task)

        demo = []
        if record:
            self.pyrep.step()  # Need this here or get_force doesn't work...
            # demo.append(self.get_observation())

        while True:
            success = False
            self._ignore_collisions_for_current_waypoint = False
            for i, point in enumerate(waypoints):
                # get both arms close to the jar (pre-grasp poses)
                if i == 0:
                    tip_right = self.robot_right_arm.arm.get_tip()
                    tip_left = self.robot_left_arm.arm.get_tip()

                    # x_offset = np.random.uniform(low=-0.0175, high=0.0175)      # our method's randomization values
                    # y_offset = np.random.uniform(low=-0.01, high=0.02)          # our method's randomization values
                    # z_offset = np.random.uniform(low=-0.0175, high=0.0175)      # our method's randomization values
                    xy_offset = np.random.uniform(low=-0.02, high=0.02)
                    x_offset = xy_offset
                    y_offset = xy_offset
                    z_offset = np.random.uniform(low=-0.0175, high=0.0175)
                    # stabilizing action. Here we insert an intermediate waypoint for the stabilizing arm so that it reaches
                    # a different starting location
                    if waypoints_labels[0] == 'right':
                        cur_gripper = tip_right
                    else:
                        cur_gripper = tip_left
                    self.use_existing_waypoint_to_set_a_new_waypoint(demo, point, waypoints_labels, 0, record, callable_each_step, xyz_indices=[0, 1, 2], offsets=[x_offset, y_offset, z_offset], gripper=cur_gripper)
                    print('############# Finished moving to a different starting location for left arm')


                    # x_offset = np.random.uniform(low=-0.0125, high=0.0125)      # our method's randomization values
                    # y_offset = np.random.uniform(low=-0.0125, high=0.0125)      # our method's randomization values
                    # z_offset = np.random.uniform(low=-0.01, high=0.005)         # our method's randomization values
                    xy_offset = np.random.uniform(low=-0.0125, high=0.0125)
                    x_offset = xy_offset
                    y_offset = xy_offset
                    z_offset = np.random.uniform(low=-0.01, high=0.005)
                    # acting action. Here we insert an intermediate waypoint for the acting arm so that it reaches
                    # a different starting location
                    if waypoints_labels[4] == 'right':
                        cur_gripper = tip_right
                    else:
                        cur_gripper = tip_left
                    self.use_existing_waypoint_to_set_a_new_waypoint(demo, waypoints[4], waypoints_labels, 4, record, callable_each_step, xyz_indices=[0, 1, 2], offsets=[x_offset, y_offset, z_offset], gripper=cur_gripper)
                    print('############# Finished moving to a different starting location for right arm')

                self._ignore_collisions_for_current_waypoint = point._ignore_collisions
                point.start_of_path()
                if point.skip:
                    continue
                if waypoints_labels[i] == 'left':
                    grasped_objects = self.robot_left_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_left_arm and s.is_collidable()
                                        and self.robot_left_arm.arm.check_arm_collision(s)]
                elif waypoints_labels[i] == 'right':
                    grasped_objects = self.robot_right_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_right_arm and s.is_collidable()
                                        and self.robot_right_arm.arm.check_arm_collision(s)]
                [s.set_collidable(False) for s in colliding_shapes]
                try:
                    path = point.get_path()
                    [s.set_collidable(True) for s in colliding_shapes]
                except ConfigurationPathError as e:
                    [s.set_collidable(True) for s in colliding_shapes]
                    raise DemoError(
                        'Could not get a path for waypoint %d.' % i,
                        self.task) from e
                ext = point.get_ext()
                path.visualize()

                done = False
                success = False
                while not done:
                    done = path.step()
                    self.step()
                    self._demo_record_step(demo, record, callable_each_step)
                    success, term = self.task.success()

                point.end_of_path()

                path.clear_visualization()

                if len(ext) > 0:
                    contains_param = False
                    start_of_bracket = -1
                    if waypoints_labels[i] == 'left':
                        gripper = self.robot_left_arm.gripper
                    elif waypoints_labels[i] == 'right':
                        gripper = self.robot_right_arm.gripper
                    if 'open_gripper(' in ext:
                        gripper.release()
                        start_of_bracket = ext.index('open_gripper(') + 13
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(1.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)
                    elif 'close_gripper(' in ext:
                        start_of_bracket = ext.index('close_gripper(') + 14
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(0.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)

                    if contains_param:
                        rest = ext[start_of_bracket:]
                        num = float(rest[:rest.index(')')])
                        done = False
                        while not done:
                            done = gripper.actuate(num, 0.04)
                            self.pyrep.step()
                            self.task.step()
                            if self._obs_config.record_gripper_closing:
                                self._demo_record_step(
                                    demo, record, callable_each_step)

                    if 'close_gripper(' in ext:
                        for g_obj in self.task.get_graspable_objects():
                            gripper.grasp(g_obj)

                    self._demo_record_step(demo, record, callable_each_step)

            if not self.task.should_repeat_waypoints() or success:
                break

        # Some tasks may need additional physics steps
        # (e.g. ball rowling to goal)
        if not success:
            for _ in range(10):
                self.pyrep.step()
                self.task.step()
                self._demo_record_step(demo, record, callable_each_step)
                success, term = self.task.success()
                if success:
                    break

        success, term = self.task.success()
        if not success:
            raise DemoError('Demo was completed, but was not successful.',
                            self.task)
        return Demo(demo)

    def get_demo_sweep_to_dustpan_close_to_broom_and_dustpand(self, record: bool = True,
                 callable_each_step: Callable[[Observation2Robots], None] = None,
                 randomly_place: bool = True) -> Demo:
        """
        get_demo function for open_drawer. Specifically, we use waypoints to
        navigate the left and right arms to areas close to the drawer, without recording any demos.
        Once we reach those areas, we start recording demos. The purpose of this function is to
        record demonstrations that start with the left and right arms close to the drawer for
        learning contact-rich manipulation.
        """
        if not self._has_init_task:
            self.init_task()
        if not self._has_init_episode:
            self.init_episode(self._variation_index,
                              randomly_place=randomly_place)
        self._has_init_episode = False

        waypoints, waypoints_labels = self.task.get_waypoints()
        if len(waypoints) == 0:
            raise NoWaypointsError(
                'No waypoints were found.', self.task)

        demo = []
        if record:
            self.pyrep.step()  # Need this here or get_force doesn't work...
            # demo.append(self.get_observation())

        waypoints = self.swap_positions(waypoints, 1, 2)
        waypoints_labels = self.swap_positions(waypoints_labels, 1, 2)
        indicies_to_skip_recording = 1

        while True:
            success = False
            self._ignore_collisions_for_current_waypoint = False
            for i, point in enumerate(waypoints):
                if i == 1:
                    x_offset = np.random.uniform(low=-0.03, high=0.03)     # left and right
                    y_offset = np.random.uniform(low=-0.1, high=0.03)      # distance away from broom
                    z_offset = np.random.uniform(low=-0.05, high=0.1)      # up and down
                    offsets = [x_offset, y_offset, z_offset]

                    old_pos = point._waypoint.get_position().copy() # for debugging
                    new_pos = point._waypoint.get_position()
                    for xyz_index, offset in zip([0, 1, 2], offsets):
                        new_pos[xyz_index] += offset
                    point._waypoint.set_position(new_pos)
                    print(f'First waypoint: old_pos: {old_pos}, new_pos: {new_pos}') # for debugging
                elif i >= 3:
                    # i >= 3 because the height must be consistent throughout the sweeping process; otherwise, dirts would not be sweeped up
                    x_offset = 0                                            # left and right
                    if i == 3:
                        # only gets one random z offset
                        z_offset = np.random.uniform(low=-0.08, high=0.08)  # up and down
                        # only y is adjusted when broom is grasped; otherwise, program would miss grasp the broom
                        y_offset = z_offset / 2                             # distance away from broom
                    else:
                        y_offset = 0                                        # distance away from broom
                    offsets = [x_offset, y_offset, z_offset]

                    old_pos = point._waypoint.get_position().copy() # for debugging
                    new_pos = point._waypoint.get_position()
                    for xyz_index, offset in zip([0, 1, 2], offsets):
                        new_pos[xyz_index] += offset
                    point._waypoint.set_position(new_pos)
                    print(f'Broom grasping position: old_pos: {old_pos}, new_pos: {new_pos}') # for debugging

                self._ignore_collisions_for_current_waypoint = point._ignore_collisions
                point.start_of_path()
                if point.skip:
                    continue
                if waypoints_labels[i] == 'left':
                    grasped_objects = self.robot_left_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_left_arm and s.is_collidable()
                                        and self.robot_left_arm.arm.check_arm_collision(s)]
                elif waypoints_labels[i] == 'right':
                    grasped_objects = self.robot_right_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_right_arm and s.is_collidable()
                                        and self.robot_right_arm.arm.check_arm_collision(s)]
                [s.set_collidable(False) for s in colliding_shapes]
                try:
                    path = point.get_path()
                    [s.set_collidable(True) for s in colliding_shapes]
                except ConfigurationPathError as e:
                    [s.set_collidable(True) for s in colliding_shapes]
                    raise DemoError(
                        'Could not get a path for waypoint %d.' % i,
                        self.task) from e
                ext = point.get_ext()
                path.visualize()

                done = False
                success = False
                while not done:
                    done = path.step()
                    self.step()
                    if i > indicies_to_skip_recording:
                        self._demo_record_step(demo, record, callable_each_step)
                    success, term = self.task.success()

                point.end_of_path()

                path.clear_visualization()

                if len(ext) > 0:
                    contains_param = False
                    start_of_bracket = -1
                    if waypoints_labels[i] == 'left':
                        gripper = self.robot_left_arm.gripper
                    elif waypoints_labels[i] == 'right':
                        gripper = self.robot_right_arm.gripper
                    if 'open_gripper(' in ext:
                        gripper.release()
                        start_of_bracket = ext.index('open_gripper(') + 13
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(1.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing and i > indicies_to_skip_recording:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)
                    elif 'close_gripper(' in ext:
                        start_of_bracket = ext.index('close_gripper(') + 14
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(0.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing and i > indicies_to_skip_recording:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)

                    if contains_param:
                        rest = ext[start_of_bracket:]
                        num = float(rest[:rest.index(')')])
                        done = False
                        while not done:
                            done = gripper.actuate(num, 0.04)
                            self.pyrep.step()
                            self.task.step()
                            if self._obs_config.record_gripper_closing and i > indicies_to_skip_recording:
                                self._demo_record_step(
                                    demo, record, callable_each_step)

                    if 'close_gripper(' in ext:
                        for g_obj in self.task.get_graspable_objects():
                            gripper.grasp(g_obj)

                    if i > indicies_to_skip_recording:
                        self._demo_record_step(demo, record, callable_each_step)

            if not self.task.should_repeat_waypoints() or success:
                break

        # Some tasks may need additional physics steps
        # (e.g. ball rowling to goal)
        if not success:
            for _ in range(10):
                self.pyrep.step()
                self.task.step()
                self._demo_record_step(demo, record, callable_each_step)
                success, term = self.task.success()
                if success:
                    break

        success, term = self.task.success()
        if not success:
            raise DemoError('Demo was completed, but was not successful.',
                            self.task)
        return Demo(demo)

    def get_demo_open_drawer_close_to_drawer_dominant_assistive(self, record: bool = True,
                 callable_each_step: Callable[[Observation2Robots], None] = None,
                 randomly_place: bool = True, dominant: str = 'right') -> Demo:
        """
        get_demo function for open_drawer. Specifically, we use waypoints to
        navigate the left and right arms to areas close to the drawer, without recording any demos.
        Once we reach those areas, we start recording demos. The purpose of this function is to
        record demonstrations that start with the left and right arms close to the drawer for
        learning contact-rich manipulation.
        """
        if not self._has_init_task:
            self.init_task_dominant_assistive(dominant)
        if not self._has_init_episode:
            self.init_episode_dominant_assistive(self._variation_index,
                                randomly_place=randomly_place, dominant=dominant)
        self._has_init_episode = False

        waypoints, waypoints_labels = self.task.get_waypoints_dominant_assistive(dominant=dominant)
        if len(waypoints) == 0:
            raise NoWaypointsError(
                'No waypoints were found.', self.task)

        demo = []
        if record:
            self.pyrep.step()  # Need this here or get_force doesn't work...
            # demo.append(self.get_observation())

        # the first 'left' and 'right' are pre-grasp poses for the left and right arms, so we want to skip (not record) them
        waypoints = self.swap_positions(waypoints, 1, 2)
        waypoints_labels = self.swap_positions(waypoints_labels, 1, 2)
        indicies_to_skip_recording = 1

        while True:
            success = False
            self._ignore_collisions_for_current_waypoint = False
            for i, point in enumerate(waypoints):
                if i == 1:
                    print('Adding randomizations to ', waypoints_labels[i])
                    x_offset = np.random.uniform(low=-0.0175, high=0.0175)      # left and right
                    y_offset = np.random.uniform(low=-0.02, high=0.02)          # distance away from the drawer
                    z_offset = np.random.uniform(low=-0.0175, high=0.0175)      # up and down
                    offsets = [x_offset, y_offset, z_offset]

                    # old_pos = point._waypoint.get_position().copy() # for debugging
                    new_pos = point._waypoint.get_position()
                    for xyz_index, offset in zip([0, 1, 2], offsets):
                        new_pos[xyz_index] += offset
                    point._waypoint.set_position(new_pos)
                    # print(f'old_pos: {old_pos}, new_pos: {new_pos}') # for debugging
                    # print(f'new_pos: {new_pos}') # for debugging

                self._ignore_collisions_for_current_waypoint = point._ignore_collisions
                point.start_of_path()
                if point.skip:
                    continue
                if waypoints_labels[i] == 'left':
                    grasped_objects = self.robot_left_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_left_arm and s.is_collidable()
                                        and self.robot_left_arm.arm.check_arm_collision(s)]
                elif waypoints_labels[i] == 'right':
                    grasped_objects = self.robot_right_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_right_arm and s.is_collidable()
                                        and self.robot_right_arm.arm.check_arm_collision(s)]
                [s.set_collidable(False) for s in colliding_shapes]
                try:
                    path = point.get_path()
                    [s.set_collidable(True) for s in colliding_shapes]
                except ConfigurationPathError as e:
                    [s.set_collidable(True) for s in colliding_shapes]
                    raise DemoError(
                        'Could not get a path for waypoint %d.' % i,
                        self.task) from e
                ext = point.get_ext()
                path.visualize()

                done = False
                success = False
                while not done:
                    done = path.step()
                    self.step()
                    if i > indicies_to_skip_recording:
                        self._demo_record_step(demo, record, callable_each_step)
                    success, term = self.task.success()

                point.end_of_path()

                path.clear_visualization()

                if len(ext) > 0:
                    contains_param = False
                    start_of_bracket = -1
                    if waypoints_labels[i] == 'left':
                        gripper = self.robot_left_arm.gripper
                    elif waypoints_labels[i] == 'right':
                        gripper = self.robot_right_arm.gripper
                    if 'open_gripper(' in ext:
                        gripper.release()
                        start_of_bracket = ext.index('open_gripper(') + 13
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(1.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing and i > indicies_to_skip_recording:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)
                    elif 'close_gripper(' in ext:
                        start_of_bracket = ext.index('close_gripper(') + 14
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(0.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing and i > indicies_to_skip_recording:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)

                    if contains_param:
                        rest = ext[start_of_bracket:]
                        num = float(rest[:rest.index(')')])
                        done = False
                        while not done:
                            done = gripper.actuate(num, 0.04)
                            self.pyrep.step()
                            self.task.step()
                            if self._obs_config.record_gripper_closing and i > indicies_to_skip_recording:
                                self._demo_record_step(
                                    demo, record, callable_each_step)

                    if 'close_gripper(' in ext:
                        for g_obj in self.task.get_graspable_objects():
                            gripper.grasp(g_obj)

                    if i > indicies_to_skip_recording:
                        self._demo_record_step(demo, record, callable_each_step)

            if not self.task.should_repeat_waypoints() or success:
                break

        # Some tasks may need additional physics steps
        # (e.g. ball rowling to goal)
        if not success:
            for _ in range(10):
                self.pyrep.step()
                self.task.step()
                self._demo_record_step(demo, record, callable_each_step)
                success, term = self.task.success()
                if success:
                    break

        success, term = self.task.success()
        if not success:
            raise DemoError('Demo was completed, but was not successful.',
                            self.task)
        return Demo(demo)

    def get_demo_open_drawer_noises_starting_states_dominant_assistive(self, record: bool = True,
                 callable_each_step: Callable[[Observation2Robots], None] = None,
                 randomly_place: bool = True, dominant: str = 'right') -> Demo:
        if not self._has_init_task:
            self.init_task_dominant_assistive(dominant)
        if not self._has_init_episode:
            self.init_episode_dominant_assistive(self._variation_index,
                                randomly_place=randomly_place, dominant=dominant)
        self._has_init_episode = False

        waypoints, waypoints_labels = self.task.get_waypoints_dominant_assistive(dominant=dominant)
        if len(waypoints) == 0:
            raise NoWaypointsError(
                'No waypoints were found.', self.task)

        demo = []
        if record:
            self.pyrep.step()  # Need this here or get_force doesn't work...
            # demo.append(self.get_observation())

        while True:
            success = False
            self._ignore_collisions_for_current_waypoint = False
            for i, point in enumerate(waypoints):
                # get both arms close to the jar (pre-grasp poses)
                if i == 0:
                    tip_right = self.robot_right_arm.arm.get_tip()
                    tip_left = self.robot_left_arm.arm.get_tip()

                    xy_offset = np.random.uniform(low=-0.02, high=0.02)
                    x_offset = xy_offset
                    y_offset = xy_offset
                    z_offset = np.random.uniform(low=-0.0175, high=0.0175)
                    # stabilizing action. Here we insert an intermediate waypoint for the stabilizing arm so that it reaches
                    # a different starting location
                    if waypoints_labels[0] == 'right':
                        cur_gripper = tip_right
                    else:
                        cur_gripper = tip_left
                    self.use_existing_waypoint_to_set_a_new_waypoint(demo, point, waypoints_labels, 0, record, callable_each_step, xyz_indices=[0, 1, 2], offsets=[x_offset, y_offset, z_offset], gripper=cur_gripper)
                    print('############# Finished moving to a different starting location for left arm')

                    # x_offset = np.random.uniform(low=-0.0175, high=0.0175)      # our method's randomization values
                    # y_offset = np.random.uniform(low=-0.02, high=0.02)          # our method's randomization values
                    # z_offset = np.random.uniform(low=-0.0175, high=0.0175)      # our method's randomization values

                    xy_offset = np.random.uniform(low=-0.02, high=0.02)
                    x_offset = xy_offset
                    y_offset = xy_offset
                    z_offset = np.random.uniform(low=-0.0175, high=0.0175)
                    # acting action. Here we insert an intermediate waypoint for the acting arm so that it reaches
                    # a different starting location
                    if waypoints_labels[2] == 'right':
                        cur_gripper = tip_right
                    else:
                        cur_gripper = tip_left
                    self.use_existing_waypoint_to_set_a_new_waypoint(demo, waypoints[2], waypoints_labels, 2, record, callable_each_step, xyz_indices=[0, 1, 2], offsets=[x_offset, y_offset, z_offset], gripper=cur_gripper)
                    print('############# Finished moving to a different starting location for right arm')

                self._ignore_collisions_for_current_waypoint = point._ignore_collisions
                point.start_of_path()
                if point.skip:
                    continue
                if waypoints_labels[i] == 'left':
                    grasped_objects = self.robot_left_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_left_arm and s.is_collidable()
                                        and self.robot_left_arm.arm.check_arm_collision(s)]
                elif waypoints_labels[i] == 'right':
                    grasped_objects = self.robot_right_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_right_arm and s.is_collidable()
                                        and self.robot_right_arm.arm.check_arm_collision(s)]
                [s.set_collidable(False) for s in colliding_shapes]
                try:
                    path = point.get_path()
                    [s.set_collidable(True) for s in colliding_shapes]
                except ConfigurationPathError as e:
                    [s.set_collidable(True) for s in colliding_shapes]
                    raise DemoError(
                        'Could not get a path for waypoint %d.' % i,
                        self.task) from e
                ext = point.get_ext()
                path.visualize()

                done = False
                success = False
                while not done:
                    done = path.step()
                    self.step()
                    self._demo_record_step(demo, record, callable_each_step)
                    success, term = self.task.success()

                point.end_of_path()

                path.clear_visualization()

                if len(ext) > 0:
                    contains_param = False
                    start_of_bracket = -1
                    if waypoints_labels[i] == 'left':
                        gripper = self.robot_left_arm.gripper
                    elif waypoints_labels[i] == 'right':
                        gripper = self.robot_right_arm.gripper
                    if 'open_gripper(' in ext:
                        gripper.release()
                        start_of_bracket = ext.index('open_gripper(') + 13
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(1.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)
                    elif 'close_gripper(' in ext:
                        start_of_bracket = ext.index('close_gripper(') + 14
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(0.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)

                    if contains_param:
                        rest = ext[start_of_bracket:]
                        num = float(rest[:rest.index(')')])
                        done = False
                        while not done:
                            done = gripper.actuate(num, 0.04)
                            self.pyrep.step()
                            self.task.step()
                            if self._obs_config.record_gripper_closing:
                                self._demo_record_step(
                                    demo, record, callable_each_step)

                    if 'close_gripper(' in ext:
                        for g_obj in self.task.get_graspable_objects():
                            gripper.grasp(g_obj)

                    self._demo_record_step(demo, record, callable_each_step)

            if not self.task.should_repeat_waypoints() or success:
                break

        # Some tasks may need additional physics steps
        # (e.g. ball rowling to goal)
        if not success:
            for _ in range(10):
                self.pyrep.step()
                self.task.step()
                self._demo_record_step(demo, record, callable_each_step)
                success, term = self.task.success()
                if success:
                    break

        success, term = self.task.success()
        if not success:
            raise DemoError('Demo was completed, but was not successful.',
                            self.task)
        return Demo(demo)

    def get_demo_open_drawer_close_to_drawer(self, record: bool = True,
                 callable_each_step: Callable[[Observation2Robots], None] = None,
                 randomly_place: bool = True) -> Demo:
        """
        get_demo function for open_drawer. Specifically, we use waypoints to
        navigate the left and right arms to areas close to the drawer, without recording any demos.
        Once we reach those areas, we start recording demos. The purpose of this function is to
        record demonstrations that start with the left and right arms close to the drawer for
        learning contact-rich manipulation.
        """
        if not self._has_init_task:
            self.init_task()
        if not self._has_init_episode:
            self.init_episode(self._variation_index,
                              randomly_place=randomly_place)
        self._has_init_episode = False

        waypoints, waypoints_labels = self.task.get_waypoints()
        if len(waypoints) == 0:
            raise NoWaypointsError(
                'No waypoints were found.', self.task)

        demo = []
        if record:
            self.pyrep.step()  # Need this here or get_force doesn't work...
            # demo.append(self.get_observation())

        # ['left', 'left', 'right', 'right', 'right'] to ['left', 'right', 'left', 'right', 'right']
        # the first 'left' and 'right' are pre-grasp poses for the left and right arms, so we want to skip (not record) them
        waypoints = self.swap_positions(waypoints, 1, 2)
        waypoints_labels = self.swap_positions(waypoints_labels, 1, 2)
        indicies_to_skip_recording = 1

        while True:
            success = False
            self._ignore_collisions_for_current_waypoint = False
            for i, point in enumerate(waypoints):
                if i == 1:
                    x_offset = np.random.uniform(low=-0.0175, high=0.0175)      # left and right
                    y_offset = np.random.uniform(low=-0.02, high=0.02)          # distance away from the drawer
                    z_offset = np.random.uniform(low=-0.0175, high=0.0175)      # up and down
                    offsets = [x_offset, y_offset, z_offset]

                    # old_pos = point._waypoint.get_position().copy() # for debugging
                    new_pos = point._waypoint.get_position()
                    for xyz_index, offset in zip([0, 1, 2], offsets):
                        new_pos[xyz_index] += offset
                    point._waypoint.set_position(new_pos)
                    # print(f'old_pos: {old_pos}, new_pos: {new_pos}') # for debugging
                    # print(f'new_pos: {new_pos}') # for debugging

                self._ignore_collisions_for_current_waypoint = point._ignore_collisions
                point.start_of_path()
                if point.skip:
                    continue
                if waypoints_labels[i] == 'left':
                    grasped_objects = self.robot_left_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_left_arm and s.is_collidable()
                                        and self.robot_left_arm.arm.check_arm_collision(s)]
                elif waypoints_labels[i] == 'right':
                    grasped_objects = self.robot_right_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_right_arm and s.is_collidable()
                                        and self.robot_right_arm.arm.check_arm_collision(s)]
                [s.set_collidable(False) for s in colliding_shapes]
                try:
                    path = point.get_path()
                    [s.set_collidable(True) for s in colliding_shapes]
                except ConfigurationPathError as e:
                    [s.set_collidable(True) for s in colliding_shapes]
                    raise DemoError(
                        'Could not get a path for waypoint %d.' % i,
                        self.task) from e
                ext = point.get_ext()
                path.visualize()

                done = False
                success = False
                while not done:
                    done = path.step()
                    self.step()
                    if i > indicies_to_skip_recording:
                        self._demo_record_step(demo, record, callable_each_step)
                    success, term = self.task.success()

                point.end_of_path()

                path.clear_visualization()

                if len(ext) > 0:
                    contains_param = False
                    start_of_bracket = -1
                    if waypoints_labels[i] == 'left':
                        gripper = self.robot_left_arm.gripper
                    elif waypoints_labels[i] == 'right':
                        gripper = self.robot_right_arm.gripper
                    if 'open_gripper(' in ext:
                        gripper.release()
                        start_of_bracket = ext.index('open_gripper(') + 13
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(1.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing and i > indicies_to_skip_recording:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)
                    elif 'close_gripper(' in ext:
                        start_of_bracket = ext.index('close_gripper(') + 14
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(0.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing and i > indicies_to_skip_recording:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)

                    if contains_param:
                        rest = ext[start_of_bracket:]
                        num = float(rest[:rest.index(')')])
                        done = False
                        while not done:
                            done = gripper.actuate(num, 0.04)
                            self.pyrep.step()
                            self.task.step()
                            if self._obs_config.record_gripper_closing and i > indicies_to_skip_recording:
                                self._demo_record_step(
                                    demo, record, callable_each_step)

                    if 'close_gripper(' in ext:
                        for g_obj in self.task.get_graspable_objects():
                            gripper.grasp(g_obj)

                    if i > indicies_to_skip_recording:
                        self._demo_record_step(demo, record, callable_each_step)

            if not self.task.should_repeat_waypoints() or success:
                break

        # Some tasks may need additional physics steps
        # (e.g. ball rowling to goal)
        if not success:
            for _ in range(10):
                self.pyrep.step()
                self.task.step()
                self._demo_record_step(demo, record, callable_each_step)
                success, term = self.task.success()
                if success:
                    break

        success, term = self.task.success()
        if not success:
            raise DemoError('Demo was completed, but was not successful.',
                            self.task)
        return Demo(demo)

    def get_demo_open_jar_close_to_jar(self, record: bool = True,
                 callable_each_step: Callable[[Observation2Robots], None] = None,
                 randomly_place: bool = True) -> Demo:
        """
        get_demo function for open_jar. Specifically, we use waypoints to
        navigate the left and right arms to areas close to the jar, without recording any demos.
        Once we reach those areas, we start recording demos. The purpose of this function is to
        record demonstrations that start with the left and right arms close to the jar for
        learning contact-rich manipulation.
        """
        if not self._has_init_task:
            self.init_task()
        if not self._has_init_episode:
            self.init_episode(self._variation_index,
                              randomly_place=randomly_place)
        self._has_init_episode = False

        waypoints, waypoints_labels = self.task.get_waypoints()
        if len(waypoints) == 0:
            raise NoWaypointsError(
                'No waypoints were found.', self.task)

        demo = []
        if record:
            self.pyrep.step()  # Need this here or get_force doesn't work...
            # demo.append(self.get_observation())

        # the first 'left' and 'right' are pre-grasp poses for the left and right arms, so we want to skip (not record) them
        waypoints = self.swap_positions(waypoints, 1, 2)
        waypoints_labels = self.swap_positions(waypoints_labels, 1, 2)
        indicies_to_skip_recording = 1

        while True:
            success = False
            self._ignore_collisions_for_current_waypoint = False
            for i, point in enumerate(waypoints):
                # get both arms close to the jar (pre-grasp poses)
                if i == 0:
                    x_offset = np.random.uniform(low=-0.03, high=0.07)      # left and right
                    y_offset = np.random.uniform(low=-0.03, high=0.07)      # distance away from the drawer
                    z_offset = 0.0                                          # up and down
                    offsets = [x_offset, y_offset, z_offset]

                    # old_pos = point._waypoint.get_position().copy() # for debugging
                    new_pos = point._waypoint.get_position()
                    for xyz_index, offset in zip([0, 1, 2], offsets):
                        new_pos[xyz_index] += offset
                    point._waypoint.set_position(new_pos)
                    # print(f'i == 0, old_pos: {old_pos}, new_pos: {new_pos}') # for debugging
                elif i == 1:
                    x_offset = np.random.uniform(low=0.0125, high=0.0175)     # left and right
                    y_offset = np.random.uniform(low=0.0125, high=0.0175)     # distance away from the drawer
                    z_offset =  np.random.uniform(low=-0.03, high=0.10)       # up and down
                    offsets = [x_offset, y_offset, z_offset]

                    # old_pos = point._waypoint.get_position().copy() # for debugging
                    new_pos = point._waypoint.get_position()
                    for xyz_index, offset in zip([0, 1, 2], offsets):
                        new_pos[xyz_index] += offset
                    point._waypoint.set_position(new_pos)
                    # print(f'i == 1, old_pos: {old_pos}, new_pos: {new_pos}') # for debugging

                self._ignore_collisions_for_current_waypoint = point._ignore_collisions
                point.start_of_path()
                if point.skip:
                    continue
                if waypoints_labels[i] == 'left':
                    grasped_objects = self.robot_left_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_left_arm and s.is_collidable()
                                        and self.robot_left_arm.arm.check_arm_collision(s)]
                elif waypoints_labels[i] == 'right':
                    grasped_objects = self.robot_right_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_right_arm and s.is_collidable()
                                        and self.robot_right_arm.arm.check_arm_collision(s)]
                [s.set_collidable(False) for s in colliding_shapes]
                try:
                    path = point.get_path()
                    [s.set_collidable(True) for s in colliding_shapes]
                except ConfigurationPathError as e:
                    [s.set_collidable(True) for s in colliding_shapes]
                    raise DemoError(
                        'Could not get a path for waypoint %d.' % i,
                        self.task) from e
                ext = point.get_ext()
                path.visualize()

                done = False
                success = False
                while not done:
                    done = path.step()
                    self.step()
                    if i > indicies_to_skip_recording:
                        self._demo_record_step(demo, record, callable_each_step)
                    success, term = self.task.success()

                point.end_of_path()

                path.clear_visualization()

                if len(ext) > 0:
                    contains_param = False
                    start_of_bracket = -1
                    if waypoints_labels[i] == 'left':
                        gripper = self.robot_left_arm.gripper
                    elif waypoints_labels[i] == 'right':
                        gripper = self.robot_right_arm.gripper
                    if 'open_gripper(' in ext:
                        gripper.release()
                        start_of_bracket = ext.index('open_gripper(') + 13
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(1.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing and i > indicies_to_skip_recording:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)
                    elif 'close_gripper(' in ext:
                        start_of_bracket = ext.index('close_gripper(') + 14
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(0.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing and i > indicies_to_skip_recording:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)

                    if contains_param:
                        rest = ext[start_of_bracket:]
                        num = float(rest[:rest.index(')')])
                        done = False
                        while not done:
                            done = gripper.actuate(num, 0.04)
                            self.pyrep.step()
                            self.task.step()
                            if self._obs_config.record_gripper_closing and i > indicies_to_skip_recording:
                                self._demo_record_step(
                                    demo, record, callable_each_step)

                    if 'close_gripper(' in ext:
                        for g_obj in self.task.get_graspable_objects():
                            gripper.grasp(g_obj)

                    if i > indicies_to_skip_recording:
                        self._demo_record_step(demo, record, callable_each_step)

            if not self.task.should_repeat_waypoints() or success:
                break

        # Some tasks may need additional physics steps
        # (e.g. ball rowling to goal)
        if not success:
            for _ in range(10):
                self.pyrep.step()
                self.task.step()
                self._demo_record_step(demo, record, callable_each_step)
                success, term = self.task.success()
                if success:
                    break

        success, term = self.task.success()
        if not success:
            raise DemoError('Demo was completed, but was not successful.',
                            self.task)
        return Demo(demo)

    def get_demo_open_jar_close_to_jar_dominant_assistive(self, record: bool = True,
                 callable_each_step: Callable[[Observation2Robots], None] = None,
                 randomly_place: bool = True, dominant: str = 'right') -> Demo:
        """
        get_demo function for open_jar. Specifically, we use waypoints to
        navigate the left and right arms to areas close to the jar, without recording any demos.
        Once we reach those areas, we start recording demos. The purpose of this function is to
        record demonstrations that start with the left and right arms close to the jar for
        learning contact-rich manipulation.
        """
        if not self._has_init_task:
            self.init_task_dominant_assistive(dominant)
        if not self._has_init_episode:
            self.init_episode_dominant_assistive(self._variation_index,
                                randomly_place=randomly_place, dominant=dominant)
        self._has_init_episode = False

        waypoints, waypoints_labels = self.task.get_waypoints_dominant_assistive(dominant=dominant)
        if len(waypoints) == 0:
            raise NoWaypointsError(
                'No waypoints were found.', self.task)

        demo = []
        if record:
            self.pyrep.step()  # Need this here or get_force doesn't work...
            # demo.append(self.get_observation())

        # the first 'left' and 'right' are pre-grasp poses for the left and right arms, so we want to skip (not record) them
        waypoints = self.swap_positions(waypoints, 1, 2)
        waypoints_labels = self.swap_positions(waypoints_labels, 1, 2)
        indicies_to_skip_recording = 1

        while True:
            success = False
            self._ignore_collisions_for_current_waypoint = False
            for i, point in enumerate(waypoints):
                # get both arms close to the jar (pre-grasp poses)
                if i == 0:
                    x_offset = np.random.uniform(low=-0.03, high=0.07)      # left and right
                    y_offset = np.random.uniform(low=-0.03, high=0.07)      # distance away from the drawer
                    z_offset = 0.0                                          # up and down
                    offsets = [x_offset, y_offset, z_offset]

                    # old_pos = point._waypoint.get_position().copy() # for debugging
                    new_pos = point._waypoint.get_position()
                    for xyz_index, offset in zip([0, 1, 2], offsets):
                        new_pos[xyz_index] += offset
                    point._waypoint.set_position(new_pos)
                    # print(f'i == 0, old_pos: {old_pos}, new_pos: {new_pos}') # for debugging
                elif i == 1:
                    x_offset = np.random.uniform(low=0.0125, high=0.0175)     # left and right
                    y_offset = np.random.uniform(low=0.0125, high=0.0175)     # distance away from the drawer
                    z_offset =  np.random.uniform(low=-0.03, high=0.10)       # up and down
                    offsets = [x_offset, y_offset, z_offset]

                    # old_pos = point._waypoint.get_position().copy() # for debugging
                    new_pos = point._waypoint.get_position()
                    for xyz_index, offset in zip([0, 1, 2], offsets):
                        new_pos[xyz_index] += offset
                    point._waypoint.set_position(new_pos)
                    # print(f'i == 1, old_pos: {old_pos}, new_pos: {new_pos}') # for debugging

                self._ignore_collisions_for_current_waypoint = point._ignore_collisions
                point.start_of_path()
                if point.skip:
                    continue
                if waypoints_labels[i] == 'left':
                    grasped_objects = self.robot_left_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_left_arm and s.is_collidable()
                                        and self.robot_left_arm.arm.check_arm_collision(s)]
                elif waypoints_labels[i] == 'right':
                    grasped_objects = self.robot_right_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_right_arm and s.is_collidable()
                                        and self.robot_right_arm.arm.check_arm_collision(s)]
                [s.set_collidable(False) for s in colliding_shapes]
                try:
                    path = point.get_path()
                    [s.set_collidable(True) for s in colliding_shapes]
                except ConfigurationPathError as e:
                    [s.set_collidable(True) for s in colliding_shapes]
                    raise DemoError(
                        'Could not get a path for waypoint %d.' % i,
                        self.task) from e
                ext = point.get_ext()
                path.visualize()

                done = False
                success = False
                while not done:
                    done = path.step()
                    self.step()
                    if i > indicies_to_skip_recording:
                        self._demo_record_step(demo, record, callable_each_step)
                    success, term = self.task.success()

                point.end_of_path()

                path.clear_visualization()

                if len(ext) > 0:
                    contains_param = False
                    start_of_bracket = -1
                    if waypoints_labels[i] == 'left':
                        gripper = self.robot_left_arm.gripper
                    elif waypoints_labels[i] == 'right':
                        gripper = self.robot_right_arm.gripper
                    if 'open_gripper(' in ext:
                        gripper.release()
                        start_of_bracket = ext.index('open_gripper(') + 13
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(1.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing and i > indicies_to_skip_recording:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)
                    elif 'close_gripper(' in ext:
                        start_of_bracket = ext.index('close_gripper(') + 14
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(0.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing and i > indicies_to_skip_recording:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)

                    if contains_param:
                        rest = ext[start_of_bracket:]
                        num = float(rest[:rest.index(')')])
                        done = False
                        while not done:
                            done = gripper.actuate(num, 0.04)
                            self.pyrep.step()
                            self.task.step()
                            if self._obs_config.record_gripper_closing and i > indicies_to_skip_recording:
                                self._demo_record_step(
                                    demo, record, callable_each_step)

                    if 'close_gripper(' in ext:
                        for g_obj in self.task.get_graspable_objects():
                            gripper.grasp(g_obj)

                    if i > indicies_to_skip_recording:
                        self._demo_record_step(demo, record, callable_each_step)

            if not self.task.should_repeat_waypoints() or success:
                break

        # Some tasks may need additional physics steps
        # (e.g. ball rowling to goal)
        if not success:
            for _ in range(10):
                self.pyrep.step()
                self.task.step()
                self._demo_record_step(demo, record, callable_each_step)
                success, term = self.task.success()
                if success:
                    break

        success, term = self.task.success()
        if not success:
            raise DemoError('Demo was completed, but was not successful.',
                            self.task)
        return Demo(demo)

    def get_demo_open_jar_noises_starting_states(self, record: bool = True,
                 callable_each_step: Callable[[Observation2Robots], None] = None,
                 randomly_place: bool = True) -> Demo:
        if not self._has_init_task:
            self.init_task()
        if not self._has_init_episode:
            self.init_episode(self._variation_index,
                              randomly_place=randomly_place)
        self._has_init_episode = False

        waypoints, waypoints_labels = self.task.get_waypoints()
        if len(waypoints) == 0:
            raise NoWaypointsError(
                'No waypoints were found.', self.task)

        demo = []
        if record:
            self.pyrep.step()  # Need this here or get_force doesn't work...
            # demo.append(self.get_observation())

        while True:
            success = False
            self._ignore_collisions_for_current_waypoint = False
            for i, point in enumerate(waypoints):
                # get both arms close to the jar (pre-grasp poses)
                if i == 0:
                    randomizations_prob = np.random.uniform(low=-0, high=1.0)
                    if randomizations_prob <= 0.4:
                        tip_right = self.robot_right_arm.arm.get_tip()
                        tip_left = self.robot_left_arm.arm.get_tip()

                        # add randomizations to x and y offsets
                        # x_offset = np.random.uniform(low=0.05, high=0.14)  # our method's randomization values
                        # y_offset = np.random.uniform(low=0.05, high=0.14)  # our method's randomization values
                        offset = np.random.uniform(low=-0.03, high=0)
                        x_offset = offset
                        y_offset = offset
                        # left-armed action. Here we insert an intermediate waypoint for the left arm so that it reaches
                        # a different starting location
                        self.use_existing_waypoint_to_set_a_new_waypoint(demo, point, waypoints_labels, 0, record, callable_each_step, xyz_indices=[0, 1], offsets=[x_offset, y_offset], gripper=tip_left)

                        # x_offset = np.random.uniform(low=0.0125, high=0.0175)  # our method's randomization values
                        # y_offset = np.random.uniform(low=0.0125, high=0.0175)  # our method's randomization values
                        # z_offset = np.random.uniform(low=0.09, high=0.14)      # our method's randomization values
                        xy_offset = np.random.uniform(low=0, high=0.0175)
                        x_offset = xy_offset
                        y_offset = xy_offset
                        z_offset = np.random.uniform(low=-0.1, high=0)
                        # right-armed action. Here we insert an intermediate waypoint for the right arm so that it reaches
                        # a different starting location
                        self.use_existing_waypoint_to_set_a_new_waypoint(demo, waypoints[1], waypoints_labels, 1, record, callable_each_step, xyz_indices=[0, 1, 2], offsets=[x_offset, y_offset, z_offset], gripper=tip_right)
                        demo.append(self.get_observation())
                        print('Finished moving to a different starting location')

                self._ignore_collisions_for_current_waypoint = point._ignore_collisions
                point.start_of_path()
                if point.skip:
                    continue
                if waypoints_labels[i] == 'left':
                    grasped_objects = self.robot_left_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_left_arm and s.is_collidable()
                                        and self.robot_left_arm.arm.check_arm_collision(s)]
                elif waypoints_labels[i] == 'right':
                    grasped_objects = self.robot_right_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_right_arm and s.is_collidable()
                                        and self.robot_right_arm.arm.check_arm_collision(s)]
                [s.set_collidable(False) for s in colliding_shapes]
                try:
                    path = point.get_path()
                    [s.set_collidable(True) for s in colliding_shapes]
                except ConfigurationPathError as e:
                    [s.set_collidable(True) for s in colliding_shapes]
                    raise DemoError(
                        'Could not get a path for waypoint %d.' % i,
                        self.task) from e
                ext = point.get_ext()
                path.visualize()

                done = False
                success = False
                while not done:
                    done = path.step()
                    self.step()
                    self._demo_record_step(demo, record, callable_each_step)
                    success, term = self.task.success()

                point.end_of_path()

                path.clear_visualization()

                if len(ext) > 0:
                    contains_param = False
                    start_of_bracket = -1
                    if waypoints_labels[i] == 'left':
                        gripper = self.robot_left_arm.gripper
                    elif waypoints_labels[i] == 'right':
                        gripper = self.robot_right_arm.gripper
                    if 'open_gripper(' in ext:
                        gripper.release()
                        start_of_bracket = ext.index('open_gripper(') + 13
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(1.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)
                    elif 'close_gripper(' in ext:
                        start_of_bracket = ext.index('close_gripper(') + 14
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(0.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)

                    if contains_param:
                        rest = ext[start_of_bracket:]
                        num = float(rest[:rest.index(')')])
                        done = False
                        while not done:
                            done = gripper.actuate(num, 0.04)
                            self.pyrep.step()
                            self.task.step()
                            if self._obs_config.record_gripper_closing:
                                self._demo_record_step(
                                    demo, record, callable_each_step)

                    if 'close_gripper(' in ext:
                        for g_obj in self.task.get_graspable_objects():
                            gripper.grasp(g_obj)

                    self._demo_record_step(demo, record, callable_each_step)

            if not self.task.should_repeat_waypoints() or success:
                break

        # Some tasks may need additional physics steps
        # (e.g. ball rowling to goal)
        if not success:
            for _ in range(10):
                self.pyrep.step()
                self.task.step()
                self._demo_record_step(demo, record, callable_each_step)
                success, term = self.task.success()
                if success:
                    break

        success, term = self.task.success()
        if not success:
            raise DemoError('Demo was completed, but was not successful.',
                            self.task)
        return Demo(demo)

    def get_demo_open_jar_noises_starting_states_dominant_assistive(self, record: bool = True,
                 callable_each_step: Callable[[Observation2Robots], None] = None,
                 randomly_place: bool = True, dominant: str = 'right') -> Demo:
        if not self._has_init_task:
            self.init_task_dominant_assistive(dominant)
        if not self._has_init_episode:
            self.init_episode_dominant_assistive(self._variation_index,
                                randomly_place=randomly_place, dominant=dominant)
        self._has_init_episode = False

        waypoints, waypoints_labels = self.task.get_waypoints_dominant_assistive(dominant=dominant)
        if len(waypoints) == 0:
            raise NoWaypointsError(
                'No waypoints were found.', self.task)

        demo = []
        if record:
            self.pyrep.step()  # Need this here or get_force doesn't work...
            # demo.append(self.get_observation())

        while True:
            success = False
            self._ignore_collisions_for_current_waypoint = False
            for i, point in enumerate(waypoints):
                # get both arms close to the jar (pre-grasp poses)
                if i == 0:
                    tip_right = self.robot_right_arm.arm.get_tip()
                    tip_left = self.robot_left_arm.arm.get_tip()

                    # x_offset = np.random.uniform(low=-0.03, high=0.07)      # our method's randomization values
                    # y_offset = np.random.uniform(low=-0.03, high=0.07)      # our method's randomization values
                    # z_offset = 0.0                                          # our method's randomization values
                    xy_offset = np.random.uniform(low=-0.05, high=0.05)
                    x_offset = xy_offset
                    y_offset = xy_offset
                    z_offset = np.random.uniform(low=-0.03, high=0.01)
                    # stabilizing action. Here we insert an intermediate waypoint for the stabilizing arm so that it reaches
                    # a different starting location
                    if waypoints_labels[0] == 'right':
                        cur_gripper = tip_right
                    else:
                        cur_gripper = tip_left
                    self.use_existing_waypoint_to_set_a_new_waypoint(demo, point, waypoints_labels, 0, record, callable_each_step, xyz_indices=[0, 1, 2], offsets=[x_offset, y_offset, z_offset], gripper=cur_gripper)
                    print('############# Finished moving to a different starting location for left arm')

                    # x_offset = np.random.uniform(low=0.0125, high=0.0175)     # our method's randomization values
                    # y_offset = np.random.uniform(low=0.0125, high=0.0175)     # our method's randomization values
                    # z_offset =  np.random.uniform(low=-0.03, high=0.10)       # our method's randomization values
                    xy_offset = np.random.uniform(low=-0.05, high=0.05)
                    x_offset = xy_offset
                    y_offset = xy_offset
                    z_offset = np.random.uniform(low=-0.03, high=0.01)
                    # acting action. Here we insert an intermediate waypoint for the acting arm so that it reaches
                    # a different starting location
                    if waypoints_labels[2] == 'right':
                        cur_gripper = tip_right
                    else:
                        cur_gripper = tip_left
                    self.use_existing_waypoint_to_set_a_new_waypoint(demo, waypoints[2], waypoints_labels, 2, record, callable_each_step, xyz_indices=[0, 1, 2], offsets=[x_offset, y_offset, z_offset], gripper=cur_gripper)
                    print('############# Finished moving to a different starting location for right arm')

                self._ignore_collisions_for_current_waypoint = point._ignore_collisions
                point.start_of_path()
                if point.skip:
                    continue
                if waypoints_labels[i] == 'left':
                    grasped_objects = self.robot_left_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_left_arm and s.is_collidable()
                                        and self.robot_left_arm.arm.check_arm_collision(s)]
                elif waypoints_labels[i] == 'right':
                    grasped_objects = self.robot_right_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_right_arm and s.is_collidable()
                                        and self.robot_right_arm.arm.check_arm_collision(s)]
                [s.set_collidable(False) for s in colliding_shapes]
                try:
                    path = point.get_path()
                    [s.set_collidable(True) for s in colliding_shapes]
                except ConfigurationPathError as e:
                    [s.set_collidable(True) for s in colliding_shapes]
                    raise DemoError(
                        'Could not get a path for waypoint %d.' % i,
                        self.task) from e
                ext = point.get_ext()
                path.visualize()

                done = False
                success = False
                while not done:
                    done = path.step()
                    self.step()
                    self._demo_record_step(demo, record, callable_each_step)
                    success, term = self.task.success()

                point.end_of_path()

                path.clear_visualization()

                if len(ext) > 0:
                    contains_param = False
                    start_of_bracket = -1
                    if waypoints_labels[i] == 'left':
                        gripper = self.robot_left_arm.gripper
                    elif waypoints_labels[i] == 'right':
                        gripper = self.robot_right_arm.gripper
                    if 'open_gripper(' in ext:
                        gripper.release()
                        start_of_bracket = ext.index('open_gripper(') + 13
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(1.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)
                    elif 'close_gripper(' in ext:
                        start_of_bracket = ext.index('close_gripper(') + 14
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(0.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)

                    if contains_param:
                        rest = ext[start_of_bracket:]
                        num = float(rest[:rest.index(')')])
                        done = False
                        while not done:
                            done = gripper.actuate(num, 0.04)
                            self.pyrep.step()
                            self.task.step()
                            if self._obs_config.record_gripper_closing:
                                self._demo_record_step(
                                    demo, record, callable_each_step)

                    if 'close_gripper(' in ext:
                        for g_obj in self.task.get_graspable_objects():
                            gripper.grasp(g_obj)

                    self._demo_record_step(demo, record, callable_each_step)

            if not self.task.should_repeat_waypoints() or success:
                break

        # Some tasks may need additional physics steps
        # (e.g. ball rowling to goal)
        if not success:
            for _ in range(10):
                self.pyrep.step()
                self.task.step()
                self._demo_record_step(demo, record, callable_each_step)
                success, term = self.task.success()
                if success:
                    break

        success, term = self.task.success()
        if not success:
            raise DemoError('Demo was completed, but was not successful.',
                            self.task)
        return Demo(demo)

    def get_demo_original(self, record: bool = True,
                 callable_each_step: Callable[[Observation2Robots], None] = None,
                 randomly_place: bool = True) -> Demo:
        """Original get_demo function from PerAct"""
        if not self._has_init_task:
            self.init_task()
        if not self._has_init_episode:
            self.init_episode(self._variation_index,
                              randomly_place=randomly_place)
        self._has_init_episode = False

        waypoints, waypoints_labels = self.task.get_waypoints()
        if len(waypoints) == 0:
            raise NoWaypointsError(
                'No waypoints were found.', self.task)

        demo = []
        if record:
            self.pyrep.step()  # Need this here or get_force doesn't work...
            demo.append(self.get_observation())

        while True:
            success = False
            self._ignore_collisions_for_current_waypoint = False
            for i, point in enumerate(waypoints):
                self._ignore_collisions_for_current_waypoint = point._ignore_collisions
                point.start_of_path()
                if point.skip:
                    continue
                if waypoints_labels[i] == 'left':
                    grasped_objects = self.robot_left_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_left_arm and s.is_collidable()
                                        and self.robot_left_arm.arm.check_arm_collision(s)]
                elif waypoints_labels[i] == 'right':
                    grasped_objects = self.robot_right_arm.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if s not in grasped_objects
                                        and s not in self._robot_shapes_right_arm and s.is_collidable()
                                        and self.robot_right_arm.arm.check_arm_collision(s)]
                [s.set_collidable(False) for s in colliding_shapes]
                try:
                    path = point.get_path()
                    [s.set_collidable(True) for s in colliding_shapes]
                except ConfigurationPathError as e:
                    [s.set_collidable(True) for s in colliding_shapes]
                    raise DemoError(
                        'Could not get a path for waypoint %d.' % i,
                        self.task) from e
                ext = point.get_ext()
                path.visualize()

                done = False
                success = False
                while not done:
                    done = path.step()
                    self.step()
                    self._demo_record_step(demo, record, callable_each_step)
                    success, term = self.task.success()

                point.end_of_path()

                path.clear_visualization()

                if len(ext) > 0:
                    contains_param = False
                    start_of_bracket = -1
                    if waypoints_labels[i] == 'left':
                        gripper = self.robot_left_arm.gripper
                    elif waypoints_labels[i] == 'right':
                        gripper = self.robot_right_arm.gripper
                    if 'open_gripper(' in ext:
                        gripper.release()
                        start_of_bracket = ext.index('open_gripper(') + 13
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(1.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)
                    elif 'close_gripper(' in ext:
                        start_of_bracket = ext.index('close_gripper(') + 14
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(0.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)

                    if contains_param:
                        rest = ext[start_of_bracket:]
                        num = float(rest[:rest.index(')')])
                        done = False
                        while not done:
                            done = gripper.actuate(num, 0.04)
                            self.pyrep.step()
                            self.task.step()
                            if self._obs_config.record_gripper_closing:
                                self._demo_record_step(
                                    demo, record, callable_each_step)

                    if 'close_gripper(' in ext:
                        for g_obj in self.task.get_graspable_objects():
                            gripper.grasp(g_obj)

                    self._demo_record_step(demo, record, callable_each_step)

            if not self.task.should_repeat_waypoints() or success:
                break

        # Some tasks may need additional physics steps
        # (e.g. ball rowling to goal)
        if not success:
            for _ in range(10):
                self.pyrep.step()
                self.task.step()
                self._demo_record_step(demo, record, callable_each_step)
                success, term = self.task.success()
                if success:
                    break

        success, term = self.task.success()
        if not success:
            raise DemoError('Demo was completed, but was not successful.',
                            self.task)
        return Demo(demo)

    def get_demo(self, record: bool = True,
                 callable_each_step: Callable[[Observation2Robots], None] = None,
                 randomly_place: bool = True, dominant: str = 'right') -> Demo:
        """Returns a demo (list of observations)"""
        if self._mode == 'default':
            output = self.get_demo_original(record, callable_each_step, randomly_place)
        elif self._mode == 'open_jar_noises_starting_states':
            output = self.get_demo_open_jar_noises_starting_states(record, callable_each_step, randomly_place)
        elif self._mode == 'open_jar_noises_starting_states_dominant_assistive':
            output = self.get_demo_open_jar_noises_starting_states_dominant_assistive(record, callable_each_step, randomly_place, dominant)
        elif self._mode == 'open_jar_close_to_jar':
            output = self.get_demo_open_jar_close_to_jar(record, callable_each_step, randomly_place)
        elif self._mode == 'open_jar_close_to_jar_dominant_assistive':
            output = self.get_demo_open_jar_close_to_jar_dominant_assistive(record, callable_each_step, randomly_place, dominant)
        elif self._mode == 'open_drawer_close_to_drawer':
            output = self.get_demo_open_drawer_close_to_drawer(record, callable_each_step, randomly_place)
        elif self._mode == 'open_drawer_close_to_drawer_dominant_assistive':
            output = self.get_demo_open_drawer_close_to_drawer_dominant_assistive(record, callable_each_step, randomly_place, dominant)
        elif self._mode == 'open_drawer_noises_starting_states_dominant_assistive':
            output = self.get_demo_open_drawer_noises_starting_states_dominant_assistive(record, callable_each_step, randomly_place, dominant)
        elif self._mode == 'put_item_in_drawer_close_to_drawer_dominant_assistive':
            output = self.get_demo_put_item_in_drawer_close_to_drawer_dominant_assistive(record, callable_each_step, randomly_place, dominant)
        elif self._mode == 'put_item_in_drawer_noises_starting_states_dominant_assistive':
            output = self.get_demo_put_item_in_drawer_noises_starting_states_dominant_assistive(record, callable_each_step, randomly_place, dominant)
        elif self._mode == 'hand_over_item_noises_starting_states_dominant_assistive':
            output = self.get_demo_hand_over_item_noises_starting_states_dominant_assistive(record, callable_each_step, randomly_place, dominant)
        elif self._mode == 'sweep_to_dustpan_close_to_broom_and_dustpand':
            output = self.get_demo_sweep_to_dustpan_close_to_broom_and_dustpand(record, callable_each_step, randomly_place)
        else:
            raise NotImplementedError
        return output

    def get_observation_config(self) -> ObservationConfig2Robots:
        return self._obs_config

    def check_target_in_workspace(self, target_pos: np.ndarray) -> bool:
        x, y, z = target_pos
        return (self._workspace_maxx > x > self._workspace_minx and
                self._workspace_maxy > y > self._workspace_miny and
                self._workspace_maxz > z > self._workspace_minz)

    def _demo_record_step(self, demo_list, record, func):
        if record:
            # demo_list.append(self.get_observation())
            demo_list.append(self.get_observation_vlm())
        if func is not None:
            # func(self.get_observation())
            func(self.get_observation_vlm())

    def _set_camera_properties(self) -> None:
        def _set_rgb_props(rgb_cam: VisionSensor,
                           rgb: bool, depth: bool, conf: CameraConfig):
            if not (rgb or depth or conf.point_cloud):
                rgb_cam.remove()
            else:
                rgb_cam.set_explicit_handling(1)
                rgb_cam.set_resolution(conf.image_size)
                rgb_cam.set_render_mode(conf.render_mode)

        def _set_mask_props(mask_cam: VisionSensor, mask: bool,
                            conf: CameraConfig):
                if not mask:
                    mask_cam.remove()
                else:
                    mask_cam.set_explicit_handling(1)
                    mask_cam.set_resolution(conf.image_size)
        _set_rgb_props(
            self._cam_wrist, self._obs_config.wrist_camera.rgb,
            self._obs_config.wrist_camera.depth,
            self._obs_config.wrist_camera)
        _set_rgb_props(
            self._cam_wrist2, self._obs_config.wrist2_camera.rgb,
            self._obs_config.wrist2_camera.depth,
            self._obs_config.wrist2_camera)
        _set_rgb_props(
            self._cam_front, self._obs_config.front_camera.rgb,
            self._obs_config.front_camera.depth,
            self._obs_config.front_camera)
        _set_mask_props(
            self._cam_wrist_mask, self._obs_config.wrist_camera.mask,
            self._obs_config.wrist_camera)
        _set_mask_props(
            self._cam_front_mask, self._obs_config.front_camera.mask,
            self._obs_config.front_camera)
        _set_mask_props(
            self._cam_wrist2_mask, self._obs_config.wrist2_camera.mask,
            self._obs_config.wrist2_camera)

    def _place_task(self) -> None:
        self._workspace_boundary.clear()
        # Find a place in the robot workspace for task
        self.task.boundary_root().set_orientation(
            self._initial_task_pose)
        min_rot, max_rot = self.task.base_rotation_bounds()
        self._workspace_boundary.sample(
            self.task.boundary_root(),
            min_rotation=min_rot, max_rotation=max_rot)

    def _get_misc(self):
        def _get_cam_data(cam: VisionSensor, name: str):
            d = {}
            if cam.still_exists():
                d = {
                    '%s_extrinsics' % name: cam.get_matrix(),
                    '%s_intrinsics' % name: cam.get_intrinsic_matrix(),
                    '%s_near' % name: cam.get_near_clipping_plane(),
                    '%s_far' % name: cam.get_far_clipping_plane(),
                }
            return d
        misc = _get_cam_data(self._cam_front, 'front_camera')
        misc.update(_get_cam_data(self._cam_wrist, 'wrist_camera'))
        misc.update(_get_cam_data(self._cam_wrist2, 'wrist2_camera'))
        return misc

    def swap_positions(self, cur_list, pos1, pos2):
        cur_list[pos1], cur_list[pos2] = cur_list[pos2], cur_list[pos1]
        return cur_list