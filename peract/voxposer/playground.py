import openai
from arguments import get_config
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
# from envs.rlbench_env import VoxPoserRLBench
from envs.rlbench_env import VoxPoserRLBench2Robots
from utils import set_lmp_objects
import numpy as np
from rlbench import tasks

openai.api_key = 'REPLACE-ME'  # set your API key here

######### setup #########
config = get_config('rlbench')
# uncomment this if you'd like to change the language model (e.g., for faster speed or lower cost)
for lmp_name, cfg in config['lmp_config']['lmps'].items():
    cfg['model'] = 'gpt-3.5-turbo'

# initialize env and voxposer ui
visualizer = ValueMapVisualizer(config['visualizer'])
env = VoxPoserRLBench2Robots(visualizer=visualizer)
lmps, lmp_env = setup_LMP(env, config, debug=False)
voxposer_ui = lmps['plan_ui']


######### playground #########

# # uncomment this to show all available tasks in rlbench
# # NOTE: in order to run a new task, you need to add the list of objects (and their corresponding env names) to the "task_object_names.json" file. See README for more details.
# print([task for task in dir(tasks) if task[0].isupper() and not '_' in task])

# below are the tasks that have object names added to the "task_object_names.json" file
# uncomment one to use
env.load_task(tasks.OpenJar)
# env.load_task(tasks.PutRubbishInBin)
# env.load_task(tasks.LampOff)
# env.load_task(tasks.OpenWineBottle)
# env.load_task(tasks.PushButton)
# env.load_task(tasks.TakeOffWeighingScales)
# env.load_task(tasks.MeatOffGrill)
# env.load_task(tasks.SlideBlockToTarget)
# env.load_task(tasks.TakeLidOffSaucepan)
# env.load_task(tasks.TakeUmbrellaOutOfUmbrellaStand)

descriptions, obs = env.reset()
set_lmp_objects(lmps, env.get_object_names())  # set the object names to be used by voxposer


instruction = np.random.choice(descriptions)
voxposer_ui(instruction)
print('Is task successful ', env.task._scene.task.success()[0])

