import os
import sys
from os.path import join, dirname, abspath, isfile

CURRENT_DIR = dirname(abspath(__file__))
sys.path.insert(0, join(CURRENT_DIR, '..'))  # Use local RLBench rather than installed

import traceback
import readline

from pyrep.const import RenderMode

from rlbench.backend import task
from rlbench.backend.const import TTT_FILE
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from rlbench.backend.scene_two_robots import Scene2Robots
from rlbench.backend.exceptions import *
from rlbench.backend.const import *
from rlbench.observation_config_two_robots import ObservationConfig2Robots, CameraConfig
from rlbench.backend.robot import Robot
from rlbench.utils import name_to_task_class
from task_validator import task_smoke, TaskValidationError
import shutil
import argparse
import numpy as np

CURRENT_DIR = dirname(abspath(__file__))


def print_fail(message, end='\n'):
    message = str(message)
    sys.stderr.write('\x1b[1;31m' + message.strip() + '\x1b[0m' + end)


def setup_list_completer():
    task_files = [t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]

    def list_completer(_, state):
        line = readline.get_line_buffer()
        if not line:
            return [c + " " for c in task_files][state]

        else:
            return [c + " " for c in task_files if c.startswith(line)][state]

    readline.parse_and_bind("tab: complete")
    readline.set_completer(list_completer)


class LoadedTask(object):

    def __init__(self, pr: PyRep, scene: Scene2Robots, robot_right_arm: Robot, robot_left_arm: Robot):
        self.pr = pr
        self.scene = scene
        self.robot_right_arm = robot_right_arm
        self.robot_left_arm = robot_left_arm
        self.task = self.task_class = self.task_file = None
        self._variation_index = 0

    def _load_task_to_scene(self):
        self.scene.unload()
        self.task = self.task_class(
            self.pr, self.robot_right_arm, self.robot_left_arm, self.task_file.replace('.py', ''))
        try:
            # Try and load the task
            scene.load(self.task)
        except FileNotFoundError as e:
            # The .ttt file must not exist
            handle = Dummy.create()
            handle.set_name(self.task_file.replace('.py', ''))
            handle.set_model(True)
            # Put the dummy at the centre of the workspace
            self.task.get_base().set_position(Shape('workspace').get_position())

    def _edit_new_task(self):
        task_file = input('What task would you like to edit?\n')
        task_file = task_file.strip(' ')
        if len(task_file) > 3 and task_file[-3:] != '.py':
            task_file += '.py'
        try:
            task_class = name_to_task_class(task_file)
        except:
            print('There was no task named: %s. '
                  'Would you like to create it?' % task_file)
            inp = input()
            if inp == 'y':
                self._create_python_file(task_file)
                task_class = name_to_task_class(task_file)
            else:
                print('Please pick a defined task in that case.')
                task_class, task_file = self._edit_new_task()
        return task_class, task_file

    def _create_python_file(self, task_file: str):
        with open(join(CURRENT_DIR, 'assets', 'task_template.txt'), 'r') as f:
            file_content = f.read()
        class_name = self._file_to_class_name(task_file)
        file_content = file_content % (class_name,)
        new_file_path = join(CURRENT_DIR, '../rlbench/tasks', task_file)
        if isfile(new_file_path):
            raise RuntimeError('File already exists. Will not override this.')
        with open(new_file_path, 'w+') as f:
            f.write(file_content)

    def _file_to_class_name(self, name):
        name = name.replace('.py', '')
        return ''.join([w[0].upper() + w[1:] for w in name.split('_')])

    def reload_python(self):
        try:
            task_class = name_to_task_class(self.task_file)
        except Exception as e:
            print_fail('The python file could not be loaded!')
            traceback.print_exc()
            return None, None
        self.task = task_class(
            self.pr, self.robot_right_arm, self.robot_left_arm, self.task_file.replace('.py', ''))
        self.scene.load(self.task)

    def new_task(self):
        self._variation_index = 0
        self.task_class, self.task_file = self._edit_new_task()
        self._load_task_to_scene()
        self.pr.step_ui()
        print('You are now editing: %s' % str(self.task_class))

    def reset_variation(self):
        self._variation_index = 0

    def new_variation(self, args, dominant):
        try:
            self._variation_index += 1
            if args.mode in DOMINANT_ASSISTIVE_MODES:
                descriptions = self.scene.init_episode_dominant_assistive(
                    self._variation_index % self.task.variation_count(),
                    max_attempts=10, dominant=dominant)
            else:
                descriptions = self.scene.init_episode(
                    self._variation_index % self.task.variation_count(),
                    max_attempts=10)
            print('Task descriptions: ', descriptions)
        except (WaypointError, BoundaryError, Exception) as ex:
            traceback.print_exc()
        self.pr.step_ui()

    def new_episode(self, args, dominant):
        try:
            if args.mode in DOMINANT_ASSISTIVE_MODES:
                descriptions = self.scene.init_episode_dominant_assistive(
                                self._variation_index % self.task.variation_count(),
                                max_attempts=10, dominant=dominant)
            else:
                descriptions = self.scene.init_episode(
                    self._variation_index % self.task.variation_count(),
                    max_attempts=10)
            print('Task descriptions: ', descriptions)
        except (WaypointError, BoundaryError, Exception) as ex:
            traceback.print_exc()
            self.scene.reset()
        self.pr.step_ui()

    def new_demo(self, dominant):
        try:
            # NOTE: somehow we need to reset the environment to avoid the weird motion
            # from the right arm (only happens before the first demo). This doesn't
            # seem to happen when we generate the demonstrations, so I commented this out
            # self.scene.reset()
            self.pr.step_ui()
            self.pr.step_ui()
            self.scene.get_demo(False, randomly_place=False, dominant=dominant)
        except (WaypointError, NoWaypointsError, DemoError, Exception) as e:
            traceback.print_exc()
        success, terminate = self.task.success()
        if success:
            print("Demo was a success!")
        self.scene.reset()
        self.pr.step_ui()
        self.pr.step_ui()

    def save_task(self):
        ttm_path = join(CURRENT_DIR, '../rlbench/task_ttms',
                        self.task_file.replace('.py', '.ttm'))
        self.task.get_base().save_model(ttm_path)
        print('Task saved to:', ttm_path)

    def run_task_validator(self):
        print('About to perform task validation.')
        print("What variation to test? Pick int in range: 0 to %d, or -1 to "
              "test all. Or press 'e' to exit."
              % self.task.variation_count())
        inp = input()
        if inp == 'e':
            return
        self.pr.start()
        try:
            v = int(inp)
            v = v if v < 0 else v % self.task.variation_count()
            task_smoke(self.task, self.scene, variation=v)
        except TaskValidationError as e:
            traceback.print_exc()
        self.pr.stop()

    def rename(self):
        print('Enter new name (or q to abort).')
        inp = input()
        if inp == 'q':
            return

        name = inp.replace('.py', '')
        python_file = name + '.py'

        # Change name of base
        handle = Dummy(self.task_file.replace('.py', ''))
        handle.set_name(name)

        # Change the class name
        old_file_path = join(CURRENT_DIR, '../rlbench/tasks', self.task_file)
        old_class_name = self._file_to_class_name(self.task_file)
        new_class_name = self._file_to_class_name(name)
        with open(old_file_path, 'r') as f:
            content = f.read()
        content = content.replace(old_class_name, new_class_name)
        with open(old_file_path, 'w') as f:
            f.write(content)

        # Rename python task file
        new_file_path = join(CURRENT_DIR, '../rlbench/tasks', python_file)
        os.rename(old_file_path, new_file_path)

        # Rename .ttt
        old_ttm_path = join(CURRENT_DIR, '../rlbench/task_ttms',
                        self.task_file.replace('.py', '.ttm'))
        new_ttm_path = join(CURRENT_DIR, '../rlbench/task_ttms',
                            python_file.replace('.py', '.ttm'))
        os.rename(old_ttm_path, new_ttm_path)

        self.task_file = python_file
        self.reload_python()
        self.save_task()
        print('Rename complete!')

    def duplicate_task(self):
        print('Enter new name for duplicate (or q to abort).')
        inp = input()
        if inp == 'q':
            return

        name = inp.replace('.py', '')
        new_python_file = name + '.py'

        # Change the class name
        old_file_path = join(CURRENT_DIR, '../rlbench/tasks', self.task_file)
        old_class_name = self._file_to_class_name(self.task_file)
        new_file_path = join(CURRENT_DIR, '../rlbench/tasks', new_python_file)
        new_class_name = self._file_to_class_name(name)

        if os.path.isfile(new_file_path):
            print('File: %s already exists!' % new_file_path)
            return

        # Change name of base
        handle = Dummy(self.task_file.replace('.py', ''))
        handle.set_name(name)

        with open(old_file_path, 'r') as f:
            content = f.read()
        content = content.replace(old_class_name, new_class_name)
        with open(new_file_path, 'w') as f:
            f.write(content)

        # Rename .ttt
        old_ttm_path = join(CURRENT_DIR, '../rlbench/task_ttms',
                        self.task_file.replace('.py', '.ttm'))
        new_ttm_path = join(CURRENT_DIR, '../rlbench/task_ttms',
                            new_python_file.replace('.py', '.ttm'))
        shutil.copy(old_ttm_path, new_ttm_path)

        self.task_file = new_python_file
        self.reload_python()
        self.save_task()
        print('Duplicate complete!')


def get_new_acting_arm():
    rand = np.random.uniform(low=0, high=1.0)
    if rand >= 0.5:
        dominant = 'right'
    else:
        dominant = 'left'
    print('New episode dominant arm: ', dominant)
    return dominant

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='default', type=str, choices=['default', 'open_jar_close_to_jar', 'open_drawer_close_to_drawer', 'open_drawer_close_to_drawer_dominant_assistive', 'sweep_to_dustpan_close_to_broom_and_dustpand', 'open_jar_noises_starting_states', 'open_jar_close_to_jar_dominant_assistive', 'open_jar_noises_starting_states_dominant_assistive', 'put_item_in_drawer_close_to_drawer_dominant_assistive', 'open_drawer_noises_starting_states_dominant_assistive', 'put_item_in_drawer_noises_starting_states_dominant_assistive', 'hand_over_item_noises_starting_states_dominant_assistive'], help="Task-specific mode. For example, 'open_jar_close_to_jar' means we only record demonstrations starting with the left and right arms close to the jar.")
    parser.add_argument('--ttt_file', default='task_design_open_jar.ttt', type=str, choices=['task_design_open_jar.ttt', 'task_design_open_drawer.ttt', 'task_design_sweep_to_dustpan.ttt', 'task_design_put_item_in_drawer.ttt', 'task_design_hand_over_item.ttt'], help="overwrite TTT_FILE")
    args = parser.parse_args()

    TTT_FILE = args.ttt_file

    setup_list_completer()

    pr = PyRep()
    ttt_file = join(CURRENT_DIR, '..', 'rlbench', TTT_FILE)
    pr.launch(ttt_file, responsive_ui=True)
    pr.step_ui()

    robot_right_arm = Robot(Panda(), PandaGripper())
    robot_left_arm = Robot(Panda(1), PandaGripper(1))

    cam_config = CameraConfig(rgb=True, depth=False, mask=False,
                              render_mode=RenderMode.OPENGL)
    obs_config = ObservationConfig2Robots()
    obs_config.set_all(False)
    obs_config.wrist_camera = cam_config
    obs_config.wrist2_camera = cam_config
    obs_config.front_camera = cam_config

    scene = Scene2Robots(pr, robot_right_arm, robot_left_arm, obs_config, mode=args.mode)
    loaded_task = LoadedTask(pr, scene, robot_right_arm, robot_left_arm)

    print('  ,')
    print(' /(  ___________')
    print('|  >:===========`  Welcome to task builder!')
    print(' )(')
    print(' ""')

    loaded_task.new_task()
    dominant = get_new_acting_arm()

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print('\n-----------------\n')
        print('The python file will be reloaded when simulation is restarted.')
        print('(q) to quit.')
        if pr.running:
            print('(+) stop the simulator')
            print('(v) for task variation.')
            print('(e) for episode of same variation.')
            print('(d) for demo.')
            print('(p) for running the sim for 100 steps (with rendering).')
        else:
            print('(!) to run task validator.')
            print('(+) run the simulator')
            print('(n) for new task.')
            print('(s) to save the .ttm')
            print('(r) to rename the task')
            print('(u) to duplicate/copy the task')

        inp = input()

        if inp == 'q':
            break

        if pr.running:

            if inp == '+':
                pr.stop()
                pr.step_ui()
            elif inp == 'p':
                [(pr.step(), scene.get_observation()) for _ in range(100)]
            elif inp == 'd':
                loaded_task.new_demo(dominant)
                dominant = get_new_acting_arm()
            elif inp == 'v':
                loaded_task.new_variation(args, dominant)
            elif inp == 'e':
                loaded_task.new_episode(args, dominant)
        else:
            if inp == '+':
                loaded_task.reload_python()
                loaded_task.reset_variation()
                pr.start()
                pr.step_ui()
            elif inp == 'n':
                inp = input('Do you want to save the current task first?\n')
                if inp == 'y':
                    loaded_task.save_task()
                loaded_task.new_task()
            elif inp == 's':
                loaded_task.save_task()
            elif inp == '!':
                loaded_task.run_task_validator()
                dominant = get_new_acting_arm()
            elif inp == 'r':
                loaded_task.rename()
            elif inp == 'u':
                loaded_task.duplicate_task()

    pr.stop()
    pr.shutdown()
    print('Done. Goodbye!')
