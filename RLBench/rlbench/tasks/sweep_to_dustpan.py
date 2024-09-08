from typing import List
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task_two_robots import Task2Robots
from rlbench.backend.conditions import DetectedCondition
from pyrep.objects.dummy import Dummy
from rlbench.backend.conditions import DetectedCondition

DIRT_NUM = 5


class SweepToDustpan(Task2Robots):

    def init_task(self) -> None:
        broom = Shape('broom')
        success_sensor = ProximitySensor('success')
        dirts = [Shape('dirt' + str(i)) for i in range(DIRT_NUM)]
        conditions = [DetectedCondition(dirt, success_sensor) for dirt in dirts]
        left_success_sensor = ProximitySensor('left_success')
        left_gripper = Dummy('Panda_tip#0')
        conditions.append(DetectedCondition(left_gripper, left_success_sensor))

        self.dirts = dirts
        self.broom = broom
        self.register_graspable_objects([broom])
        self.register_success_conditions(conditions)

    def init_episode(self, index: int) -> List[str]:
        return ['hold the dustpan with left hand and grasp the broom with right hand to brush the dirt into the dustpan']

    def variation_count(self) -> int:
        return 1
