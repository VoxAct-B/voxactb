from typing import List, Tuple
from rlbench.backend.task_two_robots import Task2Robots
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.conditions import DetectedCondition
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
import numpy as np
from pyrep.backend import sim
from rlbench.const import colors

class HandOverItem(Task2Robots):

    def init_task(self, dominant: str = '') -> None:
        self.workspace = Dummy('hand_over_item')
        self.cube = Shape('cube')
        self.waypoint_l0 = Dummy('waypoint#0')
        self.waypoint_l1 = Dummy('waypoint#1')
        self.waypoint_l2 = Dummy('waypoint_hash2')
        self.waypoint_r0 = Dummy('waypoint0')
        self.waypoint_r1 = Dummy('waypoint1')
        self.waypoint_r2 = Dummy('waypoint2')
        self.left_arm_waypoints = [self.waypoint_l0, self.waypoint_l1, self.waypoint_l2]
        self.right_arm_waypoints = [self.waypoint_r0, self.waypoint_r1, self.waypoint_r2]
        self._dominant = dominant
        # size variations
        self.has_scaled = False
        self.save_model_path = './tmp.ttm'

    def init_episode(self, index: int, dominant: str = '') -> List[str]:
        success = ProximitySensor('success')
        if dominant == 'left':
            self.waypoint_l0.set_orientation([-np.pi, 0, 0], reset_dynamics=False)
            self.waypoint_l1.set_orientation([-np.pi, 0, 0], reset_dynamics=False)
            self.waypoint_l2.set_orientation([0, np.pi/2, np.pi/2], reset_dynamics=False)
            self.waypoint_l1.set_position([0, 0, -1.3298e-02], relative_to=self.cube, reset_dynamics=False)

            for waypoint in self.right_arm_waypoints:
                waypoint.set_orientation([0, -np.pi/2, 0], reset_dynamics=False)

            self.waypoint_r0.set_position([+5.0000e-02, 0, +2.7160e-01], relative_to=self.workspace, reset_dynamics=False)
            self.waypoint_r1.set_position([-1.1000e-01, 0, +2.7160e-01], relative_to=self.workspace, reset_dynamics=False)
            self.waypoint_r2.set_position([+5.0000e-02, 0, +2.7160e-01], relative_to=self.workspace, reset_dynamics=False)
            success.set_position([+5.0000e-02, 0, +2.9160e-01], relative_to=self.workspace, reset_dynamics=False)

            self.cube.set_position([-8.0020e-02, 0, +4.1648e-02], relative_to=self.workspace, reset_dynamics=False)
            self.cube.set_orientation([0, 0, 0], reset_dynamics=False)
        else:
            self.waypoint_l0.set_orientation([-np.pi, 0, -np.pi], reset_dynamics=False)
            self.waypoint_l1.set_orientation([-np.pi, 0, -np.pi], reset_dynamics=False)
            self.waypoint_l2.set_orientation([0, -np.pi/2, -np.pi/2], reset_dynamics=False)

            for waypoint in self.right_arm_waypoints:
                waypoint.set_orientation([0, np.pi/2, np.pi], reset_dynamics=False)
            
            self.waypoint_r0.set_position([0, 0, +2.7160e-01], relative_to=self.workspace, reset_dynamics=False)
            self.waypoint_r1.set_position([+1.6900e-01, 0, +2.7160e-01], relative_to=self.workspace, reset_dynamics=False)
            self.waypoint_r2.set_position([0, 0, +2.7160e-01], relative_to=self.workspace, reset_dynamics=False)
            success.set_position([0, 0, +2.9160e-01], relative_to=self.workspace, reset_dynamics=False)

            self.cube.set_position([+1.4002e-01, 0, +4.1648e-02], relative_to=self.workspace, reset_dynamics=False)
            self.cube.set_orientation([0, 0, 0], reset_dynamics=False)

        target_color_name, target_color_rgb = colors[index]
        self.cube.set_color(target_color_rgb)
        self._dominant = dominant
        print(f'dominant arm in init_episode: {dominant}') # for debugging
        self.register_success_conditions([DetectedCondition(self.cube, success)])
        if dominant == 'left':
            return ['grasp the block with right hand and hand it over to the left hand']
        else:
            return ['grasp the block with left hand and hand it over to the right hand']

    def resize_object_of_interest(self):
        # print('Cube scale factor: ', sim.simGetObjectSizeFactor(self.cube.get_handle()))
        # if not self.has_scaled:
        #     # self.scale_factor = np.random.uniform(0.9, 1.01)
        #     self.scale_factor = 1.0
        #     print('Rescaling factor: ', self.scale_factor) # for debugging....
        #     print('Saving object model and rescaling object!!!') # for debugging...
        #     # save object model before modifying object's scale
        #     sim.simSaveModel(sim.simGetObjectHandle('hand_over_item'), self.save_model_path)
        #     sim.simScaleObjects([self.cube.get_handle()], self.scale_factor, True)
        #     self.has_scaled = True
        pass

    def variation_count(self) -> int:
        return len(colors)

    def cleanup(self) -> None:
        # Called during at the end of each episode. Remove this if not using.
        # if self.has_scaled:
        #     print('Removing model and loading it again!!!!') # for debugging...
        #     sim.simRemoveModel(sim.simGetObjectHandle('hand_over_item'))
        #     sim.simLoadModel(self.save_model_path)
        #     self.has_scaled = False
        #     # update variables since their handle may be different after loading from model
        #     self.workspace = Dummy('hand_over_item')
        #     self.cube = Shape('cube')
        #     self.waypoint_l0 = Dummy('waypoint#0')
        #     self.waypoint_l1 = Dummy('waypoint#1')
        #     self.waypoint_l2 = Dummy('waypoint_hash2')
        #     self.waypoint_r0 = Dummy('waypoint0')
        #     self.waypoint_r1 = Dummy('waypoint1')
        #     self.waypoint_r2 = Dummy('waypoint2')
        #     self.left_arm_waypoints = [self.waypoint_l0, self.waypoint_l1, self.waypoint_l2]
        #     self.right_arm_waypoints = [self.waypoint_r0, self.waypoint_r1, self.waypoint_r2]
        pass

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, 0], [0, 0, 0]
