from typing import List, Tuple

import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task_two_robots import Task2Robots
from pyrep.backend import sim


class PutItemInDrawer(Task2Robots):

    def init_task(self, dominant: str = '') -> None:
        self._drawer_frame = Shape('drawer_frame')
        self._joint = Joint('drawer_joint_top')
        self._anchor = Dummy('waypoint_anchor_top')
        self._waypoint_hash_1 = Dummy('waypoint#1')
        self._waypoint_hash_2 = Dummy('waypoint_hash2')
        self._waypoint_hash_3 = Dummy('waypoint_hash3') # this is only used to ensure waypoint_hash3 exists
        self._item = Shape('item')
        self.register_graspable_objects([self._item])
        self._dominant = dominant

        # size variations
        self.has_scaled = False
        self.save_model_path = './put_item_in_drawer.ttm'

    def init_episode(self, index, dominant: str = '') -> List[str]:
        print('dominant hand in put_item_in_drawer init_episode: ', dominant) # for debugging
        self._dominant = dominant
        self._waypoint_hash_2.set_position(self._anchor.get_position())

        success_sensor = ProximitySensor('success_top')
        self.register_success_conditions(
            [DetectedCondition(self._item, success_sensor)])
        if dominant == 'left':
            return ['open the top drawer with right hand and put the item in the top drawer with left hand']
        else:
            return ['open the top drawer with left hand and put the item in the top drawer with right hand']

    def resize_object_of_interest(self):
        if not self.has_scaled:
            # self.drawer_scale_factor = np.random.uniform(0.8, 1.01) # v3
            self.drawer_scale_factor = np.random.uniform(0.9, 1.01) # v4 CoRL experiments
            # self._item_scale_factor = np.random.uniform(self.drawer_scale_factor, 1.01) # item scale must be bigger than or equal to drawer scale; otherwise, item will disappear
            print('Drawer rescaling factor: ', self.drawer_scale_factor) # for debugging....
            # print('Item rescaling factor: ', self._item_scale_factor) # for debugging....
            print('Saving object model!!!') # for debugging...
            # save object model before modifying object's scale
            sim.simSaveModel(sim.simGetObjectHandle('put_item_in_drawer'), self.save_model_path)
            print('Rescaling object!!!') # for debugging...
            # note that this object has more complex components (joints + shapes); therefore, we need to scale the object using object common properties' scaling function.
            sim.simScaleObjects([self._drawer_frame.get_handle()], self.drawer_scale_factor, True)
            # sim.simScaleObjects([self._item.get_handle()], self._item_scale_factor, True)
            self.has_scaled = True

    def variation_count(self) -> int:
        return 1

    def cleanup(self) -> None:
        if self.has_scaled:
            print('Removing model and loading it again!!!!') # for debugging...
            sim.simRemoveModel(sim.simGetObjectHandle('put_item_in_drawer'))
            sim.simLoadModel(self.save_model_path)
            self.has_scaled = False
            # update variables since their handle may be different after loading from model
            self._drawer_frame = Shape('drawer_frame')
            self._joint = Joint('drawer_joint_top')
            self._anchor = Dummy('waypoint_anchor_top')
            self._waypoint_hash_1 = Dummy('waypoint#1')
            self._waypoint_hash_2 = Dummy('waypoint_hash2')
            self._waypoint_hash_3 = Dummy('waypoint_hash3') # this is only used to ensure waypoint_hash3 exists
            self._item = Shape('item')
            self.register_graspable_objects([self._item])

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        print('base_rotation_bounds: ', self._dominant)
        if self._dominant == 'left':
            return [0, 0, 0], [0, 0, np.pi / 8]
        elif self._dominant == 'right':
            return [0, 0, - np.pi / 8], [0, 0, 0]
        else:
            # very difficult for the left arm to open the drawer if the drawer is right rotated; therefore, we only use left rotations.
            return [0, 0, - np.pi / 8], [0, 0, 0]
