import numpy as np
import torch
from typing import List
from typing import Union

from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.replay_buffer.replay_buffer import ReplayBuffer
from yarr.runners._independent_env_runner import _IndependentEnvRunner
from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.stat_accumulator import StatAccumulator, SimpleAccumulator
from yarr.agents.agent import Summary
# from helpers.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv
from helpers.custom_rlbench_env_two_robots import CustomRLBenchEnv2Robots, CustomMultiTaskRLBenchEnv

from yarr.runners.env_runner import EnvRunner


class IndependentEnvRunner(EnvRunner):

    def __init__(self,
                 train_env: Env,
                 agent: Agent,
                 train_replay_buffer: Union[ReplayBuffer, List[ReplayBuffer]],
                 num_train_envs: int,
                 num_eval_envs: int,
                 rollout_episodes: int,
                 eval_episodes: int,
                 training_iterations: int,
                 eval_from_eps_number: int,
                 episode_length: int,
                 eval_env: Union[Env, None] = None,
                 eval_replay_buffer: Union[ReplayBuffer, List[ReplayBuffer], None] = None,
                 stat_accumulator: Union[StatAccumulator, None] = None,
                 rollout_generator: RolloutGenerator = None,
                 weightsdir: str = None,
                 logdir: str = None,
                 max_fails: int = 10,
                 num_eval_runs: int = 1,
                 env_device: torch.device = None,
                 multi_task: bool = False,
                 train_cfg = None,
                 left_arm_agent = None,
                 left_arm_ckpt = None,
                 which_arm = None,
                 voxposer_only_eval = False,
                 no_voxposer = False,
                 no_acting_stabilizing = False,
                 baseline_name = '',
                 gt_target_object_world_coords = False,
                 kwargs:dict = None):
            super().__init__(train_env, agent, train_replay_buffer, num_train_envs, num_eval_envs,
                            rollout_episodes, eval_episodes, training_iterations, eval_from_eps_number,
                            episode_length, eval_env, eval_replay_buffer, stat_accumulator,
                            rollout_generator, weightsdir, logdir, max_fails, num_eval_runs,
                            env_device, multi_task, train_cfg, left_arm_agent, left_arm_ckpt, which_arm, voxposer_only_eval, no_voxposer, no_acting_stabilizing, baseline_name, gt_target_object_world_coords, kwargs)

    def summaries(self) -> List[Summary]:
        summaries = []
        if self._stat_accumulator is not None:
            summaries.extend(self._stat_accumulator.pop())
        self._new_transitions = {'train_envs': 0, 'eval_envs': 0}
        summaries.extend(self._agent_summaries)

        # add current task_name to eval summaries .... argh this should be inside a helper function
        if hasattr(self._eval_env, '_task_class'):
            eval_task_name = change_case(self._eval_env._task_class.__name__)
        elif hasattr(self._eval_env, '_task_classes'):
            if self._current_task_id != -1:
                task_id = (self._current_task_id) % len(self._eval_env._task_classes)
                eval_task_name = change_case(self._eval_env._task_classes[task_id].__name__)
            else:
                eval_task_name = ''
        else:
            raise Exception('Neither task_class nor task_classes found in eval env')

        # multi-task summaries
        if eval_task_name and self._multi_task:
            for s in summaries:
                if 'eval' in s.name:
                    s.name = '%s/%s' % (s.name, eval_task_name)

        return summaries

    # serialized evaluator for individual tasks
    def start(self, weight,
              save_load_lock, writer_lock,
              env_config,
              device_idx,
              save_metrics,
              cinematic_recorder_cfg,
              left_arm_ckpt):
        if left_arm_ckpt is not None:
            # overwrite _left_arm_ckpt with current left_arm_ckpt
            self._left_arm_ckpt = left_arm_ckpt

        multi_task = isinstance(env_config[0], list)
        if multi_task:
            eval_env = CustomMultiTaskRLBenchEnv(
                task_classes=env_config[0],
                observation_config=env_config[1],
                action_mode=env_config[2],
                dataset_root=env_config[3],
                episode_length=env_config[4],
                headless=env_config[5],
                swap_task_every=env_config[6],
                include_lang_goal_in_obs=env_config[7],
                time_in_state=env_config[8],
                record_every_n=env_config[9])
        else:
            eval_env = CustomRLBenchEnv2Robots(
                task_class=env_config[0],
                observation_config=env_config[1],
                action_mode=env_config[2],
                dataset_root=env_config[3],
                episode_length=env_config[4],
                headless=env_config[5],
                include_lang_goal_in_obs=env_config[6],
                time_in_state=env_config[7],
                record_every_n=env_config[8],
                train_cfg=self._train_cfg,
                voxposer_only_eval=self._voxposer_only_eval,
                eval_which_arm=self._which_arm)

        self._internal_env_runner = _IndependentEnvRunner(
            self._train_env, eval_env, self._agent, self._timesteps, self._train_envs,
            self._eval_envs, self._rollout_episodes, self._eval_episodes,
            self._training_iterations, self._eval_from_eps_number, self._episode_length, self._kill_signal,
            self._step_signal, self._num_eval_episodes_signal,
            self._eval_epochs_signal, self._eval_report_signal,
            self.log_freq, self._rollout_generator, None,
            self.current_replay_ratio, self.target_replay_ratio,
            self._weightsdir, self._logdir,
            self._env_device, self._previous_loaded_weight_folder,
            num_eval_runs=self._num_eval_runs, left_arm_agent=self._left_arm_agent, left_arm_ckpt=self._left_arm_ckpt,
            which_arm=self._which_arm, crop_target_obj_voxel=self._crop_target_obj_voxel,
            crop_radius=self._crop_radius, voxposer_only_eval=self._voxposer_only_eval, no_voxposer=self._no_voxposer, no_acting_stabilizing=self._no_acting_stabilizing, baseline_name=self._baseline_name, gt_target_object_world_coords=self._gt_target_object_world_coords, kwargs=self._kwargs)

        stat_accumulator = SimpleAccumulator(eval_video_fps=30)
        self._internal_env_runner._run_eval_independent('eval_env',
                                                        stat_accumulator,
                                                        weight,
                                                        writer_lock,
                                                        True,
                                                        device_idx,
                                                        save_metrics,
                                                        cinematic_recorder_cfg)