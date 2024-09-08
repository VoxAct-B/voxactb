import copy
import logging
import os
import time
import pandas as pd

from multiprocessing import Process, Manager
from multiprocessing import get_start_method, set_start_method
from typing import Any

import numpy as np
import torch
from yarr.agents.agent import Agent
from yarr.agents.agent import ScalarSummary
from yarr.agents.agent import Summary
from yarr.envs.env import Env
from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.log_writer import LogWriter
from yarr.utils.process_str import change_case
from yarr.utils.video_utils import CircleCameraMotion, TaskRecorder

from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from yarr.runners._env_runner import _EnvRunner

class _IndependentEnvRunner(_EnvRunner):

    def __init__(self,
                 train_env: Env,
                 eval_env: Env,
                 agent: Agent,
                 timesteps: int,
                 train_envs: int,
                 eval_envs: int,
                 rollout_episodes: int,
                 eval_episodes: int,
                 training_iterations: int,
                 eval_from_eps_number: int,
                 episode_length: int,
                 kill_signal: Any,
                 step_signal: Any,
                 num_eval_episodes_signal: Any,
                 eval_epochs_signal: Any,
                 eval_report_signal: Any,
                 log_freq: int,
                 rollout_generator: RolloutGenerator,
                 save_load_lock,
                 current_replay_ratio,
                 target_replay_ratio,
                 weightsdir: str = None,
                 logdir: str = None,
                 env_device: torch.device = None,
                 previous_loaded_weight_folder: str = '',
                 num_eval_runs: int = 1,
                 left_arm_agent = None,
                 left_arm_ckpt = None,
                 which_arm = None,
                 crop_target_obj_voxel = None,
                 crop_radius = None,
                 voxposer_only_eval = False,
                 no_voxposer = False,
                 no_acting_stabilizing = False,
                 baseline_name = '',
                 gt_target_object_world_coords = False,
                 kwargs = None,
                 ):

            super().__init__(train_env, eval_env, agent, timesteps,
                             train_envs, eval_envs, rollout_episodes, eval_episodes,
                             training_iterations, eval_from_eps_number, episode_length,
                             kill_signal, step_signal, num_eval_episodes_signal,
                             eval_epochs_signal, eval_report_signal, log_freq,
                             rollout_generator, save_load_lock, current_replay_ratio,
                             target_replay_ratio, weightsdir, logdir, env_device,
                             previous_loaded_weight_folder, num_eval_runs, left_arm_agent, left_arm_ckpt, which_arm, crop_target_obj_voxel, crop_radius, voxposer_only_eval, no_voxposer, no_acting_stabilizing, baseline_name, gt_target_object_world_coords, kwargs)

    def _load_save(self):
        if self._weightsdir is None:
            logging.info("'weightsdir' was None, so not loading weights.")
            return
        while True:
            weight_folders = []
            with self._save_load_lock:
                if os.path.exists(self._weightsdir):
                    weight_folders = os.listdir(self._weightsdir)
                if len(weight_folders) > 0:
                    weight_folders = sorted(map(int, weight_folders))
                    # only load if there has been a new weight saving
                    if self._previous_loaded_weight_folder != weight_folders[-1]:
                        self._previous_loaded_weight_folder = weight_folders[-1]
                        d = os.path.join(self._weightsdir, str(weight_folders[-1]))
                        try:
                            self._agent.load_weights(d)
                        except FileNotFoundError:
                            # rare case when agent hasn't finished writing.
                            time.sleep(1)
                            self._agent.load_weights(d)
                        logging.info('Agent %s: Loaded weights: %s' % (self._name, d))

                        if self._left_arm_agent is not None:
                            try:
                                self._left_arm_agent.load_weight(self._left_arm_ckpt)
                            except FileNotFoundError:
                                # rare case when agent hasn't finished writing.
                                time.sleep(1)
                                self._left_arm_agent.load_weight(self._left_arm_ckpt)
                            logging.info(f'Left Arm Agent: Loaded weights: {self._left_arm_ckpt}')

                        self._new_weights = True
                    else:
                        self._new_weights = False
                    break
            logging.info('Waiting for weights to become available.')
            time.sleep(1)

    def _get_task_name(self):
        if hasattr(self._eval_env, '_task_class'):
            eval_task_name = change_case(self._eval_env._task_class.__name__)
            multi_task = False
        elif hasattr(self._eval_env, '_task_classes'):
            if self._eval_env.active_task_id != -1:
                task_id = (self._eval_env.active_task_id) % len(self._eval_env._task_classes)
                eval_task_name = change_case(self._eval_env._task_classes[task_id].__name__)
            else:
                eval_task_name = ''
            multi_task = True
        else:
            raise Exception('Neither task_class nor task_classes found in eval env')
        return eval_task_name, multi_task

    def tr_init_func(self, _env):
        if self.rec_cfg.enabled:
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam = VisionSensor.create(self.rec_cfg.camera_resolution)
            cam_placeholder_pos = np.array([-4.9599e-01, -1.6324e+00, +2.4985e+00, -0.2325498, -0.868057, -0.4236882, -0.1135163])
            cam.set_pose(cam_placeholder_pos)
            cam.set_parent(cam_placeholder)

            cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), self.rec_cfg.rotate_speed)
            self.tr = TaskRecorder(_env, cam_motion, fps=self.rec_cfg.fps)

            if self._which_arm == 'dominant_assistive':
                _env.env.rlbench_env._action_mode.arm_action_mode.set_callable_each_step(self.tr.take_snap)
            else:
                _env.env._action_mode.arm_action_mode.set_callable_each_step(self.tr.take_snap)
            self.tr._cam_motion.save_pose()

    def _run_eval_independent(self, name: str,
                              stats_accumulator,
                              weight,
                              writer_lock,
                              eval=True,
                              device_idx=0,
                              save_metrics=True,
                              cinematic_recorder_cfg=None):

        self._name = name
        self._save_metrics = save_metrics
        self._is_test_set = type(weight) == dict

        self._agent = copy.deepcopy(self._agent)

        device = torch.device('cuda:%d' % device_idx) if torch.cuda.device_count() > 1 else torch.device('cuda:0')
        with writer_lock: # hack to prevent multiple CLIP downloads ... argh should use a separate lock
            self._agent.build(training=False, device=device)

        logging.info('%s: Launching env.' % name)
        np.random.seed()

        logging.info('Agent information:')
        logging.info(self._agent)

        env = self._eval_env
        env.eval = eval
        env.launch()

        if self._left_arm_agent is not None:
            self._left_arm_agent = copy.deepcopy(self._left_arm_agent)
            with writer_lock: # hack to prevent multiple CLIP downloads ... argh should use a separate lock
                self._left_arm_agent.build(training=False, device=device)
            self._left_arm_agent.load_weight(self._left_arm_ckpt)

        # initialize cinematic recorder if specified
        self.rec_cfg = cinematic_recorder_cfg
        if self.rec_cfg.enabled:
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam = VisionSensor.create(self.rec_cfg.camera_resolution)
            cam_placeholder_pos = np.array([-4.9599e-01, -1.6324e+00, +2.4985e+00, -0.2325498, -0.868057, -0.4236882, -0.1135163])
            cam.set_pose(cam_placeholder_pos)
            cam.set_parent(cam_placeholder)

            cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), self.rec_cfg.rotate_speed)
            self.tr = TaskRecorder(env, cam_motion, fps=self.rec_cfg.fps)

            if self._which_arm == 'dominant_assistive':
                env.env.rlbench_env._action_mode.arm_action_mode.set_callable_each_step(self.tr.take_snap)
            else:
                env.env._action_mode.arm_action_mode.set_callable_each_step(self.tr.take_snap)

        if not os.path.exists(self._weightsdir):
            raise Exception('No weights directory found.')

        # to save or not to save evaluation metrics (set as False for recording videos)
        if self._save_metrics:
            csv_file = f'eval_data_{env.env.task_name}.csv' if not self._is_test_set else f'test_data_{env.env.task_name}.csv'
            writer = LogWriter(self._logdir, True, True,
                               env_csv=csv_file)

        # one weight for all tasks (used for validation)
        if type(weight) == int:
            logging.info('Evaluating weight %s' % weight)
            weight_path = os.path.join(self._weightsdir, str(weight))
            seed_path = self._weightsdir.replace('/weights', '')
            self._agent.load_weights(weight_path)
            weight_name = str(weight)

        new_transitions = {'train_envs': 0, 'eval_envs': 0}
        total_transitions = {'train_envs': 0, 'eval_envs': 0}

        lid_open_and_reach_goal = 0
        jar_grasp_successfully = 0
        current_task_id = -1
        self._kwargs['_IndependentEnvRunner'] = self
        for n_eval in range(self._num_eval_runs):
            if self.rec_cfg.enabled:
                self.tr._cam_motion.save_pose()

            # best weight for each task (used for test evaluation)
            if type(weight) == dict:
                task_name = list(weight.keys())[n_eval]
                task_weight = weight[task_name]
                weight_path = os.path.join(self._weightsdir, str(task_weight))
                seed_path = self._weightsdir.replace('/weights', '')
                self._agent.load_weights(weight_path)
                weight_name = str(task_weight)
                print('Evaluating weight %s for %s' % (weight_name, task_name))

            # evaluate on N tasks * M episodes per task = total eval episodes
            for ep in range(self._eval_episodes):
                eval_demo_seed = ep + self._eval_from_eps_number
                logging.info('%s: Starting episode %d, seed %d.' % (name, ep, eval_demo_seed))

                # the current task gets reset after every M episodes
                episode_rollout = []
                generator = self._rollout_generator.generator(
                    self._step_signal, env, self._agent,
                    self._episode_length, self._timesteps,
                    eval, eval_demo_seed=eval_demo_seed,
                    record_enabled=self.rec_cfg.enabled, left_arm_agent=self._left_arm_agent,
                    which_arm=self._which_arm, crop_target_obj_voxel=self._crop_target_obj_voxel,
                    crop_radius=self._crop_radius, voxposer_only_eval=self._voxposer_only_eval, ep_number=ep, no_voxposer=self._no_voxposer, no_acting_stabilizing=self._no_acting_stabilizing, baseline_name=self._baseline_name, gt_target_object_world_coords=self._gt_target_object_world_coords, kwargs=self._kwargs)
                try:
                    curr_transition = 0
                    for replay_transition in generator:
                        while True:
                            if self._kill_signal.value:
                                env.shutdown()
                                return
                            if (eval or self._target_replay_ratio is None or
                                    self._step_signal.value <= 0 or (
                                            self._current_replay_ratio.value >
                                            self._target_replay_ratio)):
                                break
                            time.sleep(1)
                            logging.debug(
                                'Agent. Waiting for replay_ratio %f to be more than %f' %
                                (self._current_replay_ratio.value, self._target_replay_ratio))

                        with self.write_lock:
                            if len(self.agent_summaries) == 0:
                                # NOTE: this strategy assumes the left arm always starts first
                                if curr_transition % 2 == 0 and self._left_arm_agent is not None:
                                    for s in self._left_arm_agent.act_summaries():
                                        self.agent_summaries.append(s)
                                else:
                                    # Only store new summaries if the previous ones
                                    # have been popped by the main env runner.
                                    for s in self._agent.act_summaries():
                                        self.agent_summaries.append(s)
                                curr_transition += 1
                        episode_rollout.append(replay_transition)
                except StopIteration as e:
                    continue
                except Exception as e:
                    env.shutdown()
                    raise e

                with self.write_lock:
                    for transition in episode_rollout:
                        self.stored_transitions.append((name, transition, eval))

                        new_transitions['eval_envs'] += 1
                        total_transitions['eval_envs'] += 1
                        stats_accumulator.step(transition, eval)
                        current_task_id = transition.info['active_task_id']

                self._num_eval_episodes_signal.value += 1

                task_name, _ = self._get_task_name()
                reward = episode_rollout[-1].reward
                lang_goal = env._lang_goal
                print(f"Evaluating {task_name} | Episode {ep} | Score: {reward} | Lang Goal: {lang_goal}")

                # helpful for debugging and figuring out where a policy performs poorly
                # if task_name == 'open_jar':
                #     is_lid_in_goal_loc = env._task._task.get_conditions()[0].condition_met()[0]
                #     is_semi_closed, is_closed_not_valid_grasp, is_obj_grasped = env._task._task.get_conditions()[1].get_status()
                #     print(f"Lid in goal loc: {is_lid_in_goal_loc} | left gripper semi-closed: {is_semi_closed} | left gripper closed (not a valid grasp): {is_closed_not_valid_grasp} | left gripper grasping bottle: {is_obj_grasped}")
                #     if is_lid_in_goal_loc:
                #         lid_open_and_reach_goal += 1
                #     if is_obj_grasped and is_semi_closed and (not is_closed_not_valid_grasp):
                #         jar_grasp_successfully += 1
                # elif task_name == 'open_drawer':
                #     has_bottom_drawer_handle_reach_goal = env._task._task.get_conditions()[0].condition_met()[0]
                #     is_hand_holding_down_the_drawer = env._task._task.get_conditions()[1].condition_met()[0]
                #     print(f"Is hand holding the drawer: {is_hand_holding_down_the_drawer} | has bottom drawer handle reached goal: {has_bottom_drawer_handle_reach_goal}")

                # save recording
                if self.rec_cfg.enabled:
                    success = reward > 0.99
                    record_file = os.path.join(seed_path, 'videos',
                                               '%s_w%s_s%s_%s.mp4' % (task_name,
                                                                      weight_name,
                                                                      eval_demo_seed,
                                                                      'succ' if success else 'fail'))

                    lang_goal = self._eval_env._lang_goal

                    try:
                        self.tr.save(record_file, lang_goal, reward)
                        self.tr._cam_motion.restore_pose()
                    except:
                        print('!!!!!!!!!!! Issue saving video... Continue to the next episode.')

            # report summaries
            summaries = []
            summaries.extend(stats_accumulator.pop())

            eval_task_name, multi_task = self._get_task_name()

            if self._left_arm_ckpt is not None:
                # log left arm's checkpoint number
                summaries.append(ScalarSummary('eval_envs/left_arm_steps', int(self._left_arm_ckpt.split('/')[-2])))
            if task_name == 'open_jar':
                summaries.append(ScalarSummary('eval_envs/lid_open_and_reach_goal', int(lid_open_and_reach_goal)))
                summaries.append(ScalarSummary('eval_envs/jar_grasp_successfully', int(jar_grasp_successfully)))

            if eval_task_name and multi_task:
                for s in summaries:
                    if 'eval' in s.name:
                        s.name = '%s/%s' % (s.name, eval_task_name)

            if len(summaries) > 0:
                if multi_task:
                    task_score = [s.value for s in summaries if f'eval_envs/return/{eval_task_name}' in s.name][0]
                else:
                    task_score = [s.value for s in summaries if f'eval_envs/return' in s.name][0]
            else:
                task_score = "unknown"

            print(f"Finished {eval_task_name} | Final Score: {task_score}\n")

            if self._save_metrics:
                with writer_lock:
                    writer.add_summaries(weight_name, summaries)

            self._new_transitions = {'train_envs': 0, 'eval_envs': 0}
            self.agent_summaries[:] = []
            self.stored_transitions[:] = []

        if self._save_metrics:
            with writer_lock:
                writer.end_iteration()

        logging.info('Finished evaluation.')
        env.shutdown()

    def kill(self):
        self._kill_signal.value = True
