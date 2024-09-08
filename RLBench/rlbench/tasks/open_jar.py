from typing import List, Tuple
from rlbench.backend.task_two_robots import Task2Robots
from typing import List
from rlbench.const import colors
from rlbench.backend.conditions import NothingGrasped, DetectedCondition, DetectedGrasped, OrConditions
from rlbench.backend.spawn_boundary import SpawnBoundary
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.dummy import Dummy
from pyrep.backend import sim
from pyrep.objects.object import Object
import math

class OpenJar(Task2Robots):

    def init_task(self, dominant: str = '') -> None:
        self.lid = Shape('jar_lid0')
        self.jar = Shape('jar0')
        self.register_graspable_objects([self.lid])
        self.boundary = Shape('spawn_boundary')
        self.conditions = []
        self._dominant = dominant

        self.w0 = Dummy('waypoint0')
        self.w1 = Dummy('waypoint1')
        self.w2 = Dummy('waypoint2')
        self.w3 = Dummy('waypoint3')

        self.jar_lid_waypoints_01 = [self.w0, self.w1]
        self.jar_lid_waypoints_23 = [self.w2, self.w3]

        self.jar_body_w0 = Dummy('waypoint#0')
        self.jar_body_w1 = Dummy('waypoint#1')

        # size variations
        self.has_scaled = False
        self.save_model_path = './open_jar.ttm'

    def init_episode(self, index: int, dominant: str = '') -> List[str]:
        b = SpawnBoundary([self.boundary])
        b.sample(self.jar, min_distance=0.01)

        success = ProximitySensor('success')
        success_bottle = ProximitySensor('success_bottle')

        if dominant == 'left':
            # NOTE: make sure the orientations and positions we set here must also show up in the else statement;
            # otherwise, they will not be set properly during data collection
            for waypoint in self.jar_lid_waypoints_01:
                waypoint.set_orientation([-np.pi, 0, 0], reset_dynamics=False)

            for waypoint in self.jar_lid_waypoints_23:
                # 4.17 = 270 degrees
                waypoint.set_orientation([-np.pi, 0, 4.17], reset_dynamics=False)

            self.w0.set_position([0,0,0.1], relative_to=self.lid, reset_dynamics=False)
            self.w1.set_position([0,0,-0.01], relative_to=self.lid, reset_dynamics=False)

            self.jar_body_w0.set_position([0, -0.2, 0.03], relative_to=self.w1, reset_dynamics=False)
            self.jar_body_w0.set_orientation([np.pi/2, 0, np.pi/2], reset_dynamics=False)
            self.jar_body_w1.set_orientation([np.pi/2, 0, np.pi/2], reset_dynamics=False)

        else:
            # dominant == 'right' or dominant == ''
            # NOTE: make sure the orientations and positions we set here must also show up in the if statement;
            # otherwise, they will not be set properly during data collection
            for waypoint in self.jar_lid_waypoints_01:
                waypoint.set_orientation([-np.pi, 0, -np.pi], reset_dynamics=False)

            for waypoint in self.jar_lid_waypoints_23:
                waypoint.set_orientation([-np.pi, 0, np.pi/2], reset_dynamics=False)

            self.w0.set_position([0,0,0.1], relative_to=self.lid, reset_dynamics=False)
            self.w1.set_position([0,0,-0.01], relative_to=self.lid, reset_dynamics=False)

            self.jar_body_w0.set_position([0, 0.2, 0.03], relative_to=self.w1, reset_dynamics=False)
            self.jar_body_w0.set_orientation([np.pi/2, 0, -np.pi/2], reset_dynamics=False)
            self.jar_body_w1.set_orientation([np.pi/2, 0, -np.pi/2], reset_dynamics=False)

        target_color_name, target_color_rgb = colors[index]
        self.jar.set_color(target_color_rgb)
        self._dominant = dominant

        print(f'dominant arm in init_episode: {dominant}') # for debugging
        success_conditions = [DetectedCondition(self.lid, success), OrConditions([DetectedGrasped(
                self.robot_right_arm.gripper, success_bottle), DetectedGrasped(
                self.robot_left_arm.gripper, success_bottle)])]

        self.register_success_conditions(success_conditions)
        if dominant == 'left':
            return ['grasp the jar with right hand and '
                    'grasp the lid of the jar with left hand to unscrew it in an anti_clockwise '
                    'direction until it is removed from the jar']
        else:
            # dominant == right or dominant == ''
            return ['grasp the jar with left hand and '
                    'grasp the lid of the jar with right hand to unscrew it in an anti_clockwise '
                    'direction until it is removed from the jar']

    def resize_object_of_interest(self):
        print('Lid scale factor: ', sim.simGetObjectSizeFactor(self.lid.get_handle()))
        print('Jar scale factor: ', sim.simGetObjectSizeFactor(self.jar.get_handle()))
        assert sim.simGetObjectSizeFactor(self.lid.get_handle()) == sim.simGetObjectSizeFactor(self.jar.get_handle())
        if not self.has_scaled:
            # self.scale_factor = np.random.uniform(0.8, 1.01) # v4
            self.scale_factor = np.random.uniform(0.9, 1.01) # v5 CoRL experiments
            print('Rescaling factor: ', self.scale_factor) # for debugging....
            print('Saving object model and rescaling object!!!') # for debugging...
            # save object model before modifying object's scale
            sim.simSaveModel(sim.simGetObjectHandle('open_jar'), self.save_model_path)
            sim.simScaleObject(self.jar.get_handle(), self.scale_factor, self.scale_factor, self.scale_factor)
            sim.simScaleObject(self.lid.get_handle(), self.scale_factor, self.scale_factor, self.scale_factor)
            self.has_scaled = True

    def variation_count(self) -> int:
        return len(colors)

    def cleanup(self) -> None:
        if self._dominant == 'right':
            self.conditions = [NothingGrasped(self.robot_right_arm.gripper)]
        elif self._dominant == 'left':
            self.conditions = [NothingGrasped(self.robot_left_arm.gripper)]
        else:
            self.conditions = [NothingGrasped(self.robot_right_arm.gripper)]

        if self.has_scaled:
            print('Removing model and loading it again!!!!') # for debugging...
            sim.simRemoveModel(sim.simGetObjectHandle('open_jar'))
            sim.simLoadModel(self.save_model_path)
            self.has_scaled = False
            # update variables since their handle may be different after loading from model
            self.lid = Shape('jar_lid0')
            self.jar = Shape('jar0')
            self.register_graspable_objects([self.lid])
            self.boundary = Shape('spawn_boundary')

    def base_rotation_bounds(self) -> Tuple[Tuple[float, float, float],
                                            Tuple[float, float, float]]:
        # This is here to stop the issue of gripper rotation joint reaching its
        # limit and not being able to go through the full range of rotation to
        # unscrew, leading to a weird jitery and tilted cap while unscrewing.
        # Issue occured rarely so is only minor
        return (0.0, 0.0, -0.2*np.pi), (0.0, 0.0, +0.2*np.pi)

    def is_static_workspace(self) -> bool:
        return True

    def get_conditions(self):
        return self.conditions
