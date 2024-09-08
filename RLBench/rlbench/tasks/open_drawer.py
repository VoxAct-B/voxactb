from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.conditions import JointCondition, DetectedCondition, OrConditions
from rlbench.backend.task_two_robots import Task2Robots
from rlbench.backend.task import Task
from pyrep.backend import sim


class OpenDrawer(Task2Robots):

    def init_task(self, dominant: str = '') -> None:
        self._drawer_frame = Shape('drawer_frame')
        self._anchor = Dummy('waypoint_anchor_bottom')
        self._joint = Joint('drawer_joint_bottom')
        self._waypoint0 = Dummy('waypoint0')
        self._waypoint1 = Dummy('waypoint1')
        self._waypoint2 = Dummy('waypoint2')
        self._left_gripper = Dummy('Panda_tip#0')
        self._right_gripper = Dummy('Panda_tip')
        self._extra_dist_away_from_drawer = 0.07
        self._dominant = dominant
        self.conditions = []

        # size variations
        self.has_scaled = False
        self.save_model_path = './open_drawer.ttm'

    def init_episode(self, index: int, dominant: str = '') -> List[str]:
        self._waypoint1.set_position(self._anchor.get_position())
        waypoint0_pose = self._waypoint0.get_pose()
        # x (left and right), z (forward and backward), y (up and down)
        self._waypoint0.set_position([waypoint0_pose[0], waypoint0_pose[1] + self._extra_dist_away_from_drawer, waypoint0_pose[2]], reset_dynamics=False)
        self._waypoint2.set_position([0.0,0.0,0.0], relative_to=self._waypoint0, reset_dynamics=False)

        # check if left gripper reaches the hold sensor
        hold_sensor = ProximitySensor('hold_sensor')

        print('dominant hand in open_drawer init_episode: ', dominant) # for debugging
        self._dominant = dominant
        self.conditions = [JointCondition(self._joint, 0.07), OrConditions([DetectedCondition(self._left_gripper, hold_sensor), DetectedCondition(self._right_gripper, hold_sensor)])]
        self.register_success_conditions(self.conditions)
        if dominant == 'right':
            return ['hold the drawer with left hand and open the bottom drawer with right hand']
        elif dominant == 'left':
            return ['hold the drawer with right hand and open the bottom drawer with left hand']
        else:
            return ['hold the drawer with left hand and open the bottom drawer with right hand']

    def resize_object_of_interest(self):
        print('Drawer original scale factor: ', sim.simGetObjectSizeFactor(self._drawer_frame.get_handle()))
        if not self.has_scaled:
            # self.scale_factor = np.random.uniform(0.8, 1.01) # v4
            self.scale_factor = np.random.uniform(0.9, 1.01) # v5 CoRL experiments
            print('Rescaling factor: ', self.scale_factor) # for debugging....
            print('Saving object model!!!') # for debugging...
            # save object model before modifying object's scale
            sim.simSaveModel(sim.simGetObjectHandle('open_drawer'), self.save_model_path)
            print('Rescaling object!!!') # for debugging...
            # note that this object has more complex components (joints + shapes); therefore, we need to scale the object using object common properties' scaling function.
            sim.simScaleObjects([self._drawer_frame.get_handle()], self.scale_factor, True)
            self.has_scaled = True

    def variation_count(self) -> int:
        return 1

    def cleanup(self) -> None:
        if self.has_scaled:
            print('Removing model and loading it again!!!!') # for debugging...
            sim.simRemoveModel(sim.simGetObjectHandle('open_drawer'))
            sim.simLoadModel(self.save_model_path)
            self.has_scaled = False
            # update variables since their handle may be different after loading from model
            self._drawer_frame = Shape('drawer_frame')
            self._anchor = Dummy('waypoint_anchor_bottom')
            self._joint = Joint('drawer_joint_bottom')
            self._waypoint0 = Dummy('waypoint0')
            self._waypoint1 = Dummy('waypoint1')
            self._waypoint2 = Dummy('waypoint2')
            self._left_gripper = Dummy('Panda_tip#0')
            self._right_gripper = Dummy('Panda_tip')

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        print('base_rotation_bounds: ', self._dominant)
        if self._dominant == 'left':
            return [0, 0, - np.pi / 8], [0, 0, 0]
        elif self._dominant == 'right':
            return [0, 0, 0], [0, 0, np.pi / 8]
        else:
            return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]

    def get_conditions(self):
        return self.conditions