from typing import Type, List

import numpy as np
from rlbench import ObservationConfig2Robots, ActionMode
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.observation_two_robots import Observation2Robots
from rlbench.backend.task_two_robots import Task2Robots
from yarr.agents.agent import ActResult, VideoSummary, TextSummary
from yarr.envs.rlbench_env_two_robots import RLBenchEnv2Robots, MultiTaskRLBenchEnv
from yarr.utils.observation_type import ObservationElement
from yarr.utils.transition import Transition
from yarr.utils.process_str import change_case

from pyrep.const import RenderMode
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.objects import VisionSensor, Dummy
from rlbench import tasks



class CustomRLBenchEnv2Robots(RLBenchEnv2Robots):
    def __init__(self,
                 task_class: Type[Task2Robots],
                 observation_config: ObservationConfig2Robots,
                 action_mode: ActionMode,
                 episode_length: int,
                 dataset_root: str = '',
                 channels_last: bool = False,
                 reward_scale=100.0,
                 headless: bool = True,
                 time_in_state: bool = False,
                 include_lang_goal_in_obs: bool = False,
                 record_every_n: int = 20,
                 train_cfg = None,
                 voxposer_only_eval = False,
                 eval_which_arm = '',
                 custom_ttt_file = ''):
        super(CustomRLBenchEnv2Robots, self).__init__(
            task_class, observation_config, action_mode, dataset_root,
            channels_last, headless=headless,
            include_lang_goal_in_obs=include_lang_goal_in_obs, train_cfg=train_cfg, voxposer_only_eval=voxposer_only_eval, eval_which_arm=eval_which_arm, custom_ttt_file=custom_ttt_file)
        self._reward_scale = reward_scale
        self._episode_index = 0
        self._record_current_episode = False
        self._record_cam = None
        self._previous_obs, self._previous_obs_dict = None, None
        self._recorded_images = []
        self._episode_length = episode_length
        self._time_in_state = time_in_state
        self._record_every_n = record_every_n
        self._i = 0
        self._error_type_counts = {
            'IKError': 0,
            'ConfigurationPathError': 0,
            'InvalidActionError': 0,
        }
        self._last_exception = None

    @property
    def observation_elements(self) -> List[ObservationElement]:
        obs_elems = super(CustomRLBenchEnv2Robots, self).observation_elements
        for oe in obs_elems:
            if oe.name == 'low_dim_state':
                oe.shape = (oe.shape[0] - 7 * 3 + int(self._time_in_state),)  # remove pose and joint velocities as they will not be included
                self.low_dim_state_len = oe.shape[0]
        return obs_elems

    def extract_obs(self, obs: Observation2Robots, t=None, prev_action=None):
        obs.joint_velocities_right = None
        grip_right_mat= obs.gripper_right_matrix
        grip_right_pose = obs.gripper_right_pose
        joint_pos_right = obs.joint_positions_right
        obs.gripper_right_pose = None
        obs.gripper_right_matrix = None
        obs.wrist_camera_matrix = None
        obs.joint_positions_right = None
        if obs.gripper_right_joint_positions is not None:
            obs.gripper_right_joint_positions = np.clip(
                obs.gripper_right_joint_positions, 0., 0.04)

        obs.joint_velocities_left = None
        grip_left_mat = obs.gripper_left_matrix
        grip_left_pose = obs.gripper_left_pose
        joint_pos_left = obs.joint_positions_left
        obs.gripper_left_pose = None
        obs.gripper_left_matrix = None
        obs.wrist2_camera_matrix = None
        obs.joint_positions_left = None
        if obs.gripper_left_joint_positions is not None:
            obs.gripper_left_joint_positions = np.clip(
                obs.gripper_left_joint_positions, 0., 0.04)

        obs_dict = super(CustomRLBenchEnv2Robots, self).extract_obs(obs)

        if self._train_cfg.method.arm_pred_input:
            # arm ID is defined in peract/helpers/demo_loading_utils.py
            obs_dict['low_dim_state_right_arm'] = np.concatenate(
                [obs_dict['low_dim_state_right_arm'], [0]]).astype(np.float32)
            obs_dict['low_dim_state_left_arm'] = np.concatenate(
                [obs_dict['low_dim_state_left_arm'], [1]]).astype(np.float32)
        elif self._train_cfg.method.arm_id_to_proprio:
            if self._time_in_state:
                time = (1. - ((self._i if t is None else t) / float(
                    self._episode_length - 1))) * 2. - 1.
                obs_dict['low_dim_state_right_arm'] = np.concatenate(
                    [obs_dict['low_dim_state_right_arm'], [time], [0]]).astype(np.float32)
                obs_dict['low_dim_state_left_arm'] = np.concatenate(
                    [obs_dict['low_dim_state_left_arm'], [time], [1]]).astype(np.float32)
        else:
            if self._time_in_state:
                time = (1. - ((self._i if t is None else t) / float(
                    self._episode_length - 1))) * 2. - 1.
                obs_dict['low_dim_state_right_arm'] = np.concatenate(
                    [obs_dict['low_dim_state_right_arm'], [time]]).astype(np.float32)
                obs_dict['low_dim_state_left_arm'] = np.concatenate(
                    [obs_dict['low_dim_state_left_arm'], [time]]).astype(np.float32)

        obs.gripper_right_matrix = grip_right_mat
        obs.joint_positions_right = joint_pos_right
        obs.gripper_right_pose = grip_right_pose

        obs.gripper_left_matrix = grip_left_mat
        obs.joint_positions_left = joint_pos_left
        obs.gripper_left_pose = grip_left_pose
        return obs_dict

    def launch(self):
        super(CustomRLBenchEnv2Robots, self).launch()
        self._task._scene.register_step_callback(self._my_callback)
        if self.eval:
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam_base = Dummy('cam_cinematic_base')
            cam_base.rotate([0, 0, np.pi * 0.75])
            self._record_cam = VisionSensor.create([320, 180])
            self._record_cam.set_explicit_handling(True)
            self._record_cam.set_pose(cam_placeholder.get_pose())
            self._record_cam.set_render_mode(RenderMode.OPENGL)

    def reset(self) -> dict:
        self._i = 0
        self._previous_obs_dict = super(CustomRLBenchEnv2Robots, self).reset()
        self._record_current_episode = False
        self._episode_index += 1
        self._recorded_images.clear()
        return self._previous_obs_dict

    def register_callback(self, func):
        self._task._scene.register_step_callback(func)

    def _my_callback(self):
        if self._record_current_episode:
            self._record_cam.handle_explicitly()
            cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
            self._recorded_images.append(cap)

    def _append_final_frame(self, success: bool):
        self._record_cam.handle_explicitly()
        img = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(img)
        final_frames = np.zeros((10, ) + img.shape[:2] + (3,), dtype=np.uint8)
        # Green/red for success/failure
        final_frames[:, :, :, 1 if success else 0] = 255
        self._recorded_images.extend(list(final_frames))

    def step(self, act_result: ActResult, which_arm: str) -> Transition:
        action = act_result.action
        success = False
        obs = self._previous_obs_dict  # in case action fails.

        try:
            obs, reward, terminal = self._task.step(action, which_arm)
            if reward >= 1:
                success = True
                reward *= self._reward_scale
            else:
                reward = 0.0
            obs = self.extract_obs(obs)
            self._previous_obs_dict = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminal = True
            reward = 0.0

            if isinstance(e, IKError):
                self._error_type_counts['IKError'] += 1
                print('Error in step function: IKError')
            elif isinstance(e, ConfigurationPathError):
                self._error_type_counts['ConfigurationPathError'] += 1
                print('Error in step function: configuration path error')
            elif isinstance(e, InvalidActionError):
                self._error_type_counts['InvalidActionError'] += 1
                print('Error in step function: invalid action error')

            self._last_exception = e

        summaries = []
        self._i += 1
        if ((terminal or self._i == self._episode_length) and
                self._record_current_episode):
            self._append_final_frame(success)
            vid = np.array(self._recorded_images).transpose((0, 3, 1, 2))
            summaries.append(VideoSummary(
                'episode_rollout_' + ('success' if success else 'fail'),
                vid, fps=30))

            # error summary
            error_str = f"Errors - IK : {self._error_type_counts['IKError']}, " \
                        f"ConfigPath : {self._error_type_counts['ConfigurationPathError']}, " \
                        f"InvalidAction : {self._error_type_counts['InvalidActionError']}"
            if not success and self._last_exception is not None:
                error_str += f"\n Last Exception: {self._last_exception}"
                self._last_exception = None

            summaries.append(TextSummary('errors', f"Success: {success} | " + error_str))
        return Transition(obs, reward, terminal, summaries=summaries)

    def step_custom_action_mode(self, act_result: ActResult, which_arm: str) -> Transition:
        action = act_result.action
        success = False
        obs = self._previous_obs_dict  # in case action fails.

        try:
            obs, reward, terminal = self._task.step_custom_action_mode(action, which_arm)
            if reward >= 1:
                success = True
                reward *= self._reward_scale
            else:
                reward = 0.0
            obs = self.extract_obs(obs)
            self._previous_obs_dict = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminal = True
            reward = 0.0

            if isinstance(e, IKError):
                self._error_type_counts['IKError'] += 1
                print('Error in step function: IKError')
            elif isinstance(e, ConfigurationPathError):
                self._error_type_counts['ConfigurationPathError'] += 1
                print('Error in step function: configuration path error')
            elif isinstance(e, InvalidActionError):
                self._error_type_counts['InvalidActionError'] += 1
                print('Error in step function: invalid action error')

            self._last_exception = e

        summaries = []
        self._i += 1
        if ((terminal or self._i == self._episode_length) and
                self._record_current_episode):
            self._append_final_frame(success)
            vid = np.array(self._recorded_images).transpose((0, 3, 1, 2))
            summaries.append(VideoSummary(
                'episode_rollout_' + ('success' if success else 'fail'),
                vid, fps=30))

            # error summary
            error_str = f"Errors - IK : {self._error_type_counts['IKError']}, " \
                        f"ConfigPath : {self._error_type_counts['ConfigurationPathError']}, " \
                        f"InvalidAction : {self._error_type_counts['InvalidActionError']}"
            if not success and self._last_exception is not None:
                error_str += f"\n Last Exception: {self._last_exception}"
                self._last_exception = None

            summaries.append(TextSummary('errors', f"Success: {success} | " + error_str))
        return Transition(obs, reward, terminal, summaries=summaries)

    def no_step_get_env_stats(self) -> Transition:
        obs, reward, terminal = self._task.no_step_get_env_stats()
        if reward >= 1:
            reward *= self._reward_scale
        else:
            reward = 0.0
        obs = self.extract_obs(obs)
        self._previous_obs_dict = obs

        summaries = []
        self._i += 1
        return Transition(obs, reward, terminal, summaries=summaries)

    def get_observation(self):
        obs = self._task.get_observation()
        obs = self.extract_obs(obs)
        self._previous_obs_dict = obs
        self._record_current_episode = False
        self._episode_index += 1
        self._recorded_images.clear()
        return obs

    def reset_to_demo(self, i):
        self._i = 0
        # super(CustomRLBenchEnv, self).reset()

        self._task.set_variation(-1)
        d, = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i)

        self._task.set_variation(d.variation_number)
        _, obs = self._task.reset_to_demo(d)
        self._lang_goal = self._task.get_task_descriptions()[0]

        # HACK if we don't do this step in scene, then
        # the obs we get would be the old obs.
        self._task._scene.step()
        obs = self._task.get_observation()

        self._previous_obs_dict = self.extract_obs(obs)
        self._record_current_episode = False
        self._episode_index += 1
        self._recorded_images.clear()

        return self._previous_obs_dict


    def reset_to_demo_voxposer(self, i, ep_number: int = -1):
        """
        reset to demo and update VoxPoser's VoxPoserRLBench2Robots variables
        """
        self._i = 0
        self._task.set_variation(-1)
        d, = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i)
        self._task.set_variation(d.variation_number)

        if self._rlbench_env._dominant_assitive_policy:
            # figure out the dominant arm in this episode
            self._rlbench_env.set_dominant_hand_for_ep_reset(ep_number)
            self._task.set_dominant(self._rlbench_env._dominant_arm_for_ep_reset)
        try:
            _, obs = self._task.reset_to_demo(d, self._rlbench_env._dominant_assitive_policy)
        except:
            # Fix "RuntimeError: Expected to be resetting 10 objects, but there were 9."
            # - shutdown and restart the simulator
            # - initialize the task
            self._rlbench_env.rlbench_env.shutdown()
            # NOTE: modify this when new task is added
            if self._task._task.__class__.__name__ == 'OpenJar':
                self._rlbench_env.load_task(tasks.OpenJar)
            elif self._task._task.__class__.__name__ == 'OpenDrawer':
                self._rlbench_env.load_task(tasks.OpenDrawer)
            elif self._task._task.__class__.__name__ == 'SweepToDustpan':
                self._rlbench_env.load_task(tasks.SweepToDustpan)
            elif self._task._task.__class__.__name__ == 'PutItemInDrawer':
                self._rlbench_env.load_task(tasks.PutItemInDrawer)
            elif self._task._task.__class__.__name__ == 'HandOverItem':
                self._rlbench_env.load_task(tasks.HandOverItem)
            else:
                raise NotImplementedError
            self._rlbench_env.task._scene.init_task()
            self._i = 0
            # update self._task with the newly initialized task
            self._task = self._rlbench_env.task
            # reset the environment again to episode i
            self._task.set_variation(-1)
            d, = self._task.get_demos(
                1, live_demos=False, random_selection=False, from_episode_number=i)
            self._task.set_variation(d.variation_number)

            if self._rlbench_env._dominant_assitive_policy:
                # figure out the dominant arm in this episode
                self._rlbench_env.set_dominant_hand_for_ep_reset(ep_number)
                self._task.set_dominant(self._rlbench_env._dominant_arm_for_ep_reset)

            _, obs = self._task.reset_to_demo(d, self._rlbench_env._dominant_assitive_policy)
            self._rlbench_env.load_objects()
            self.reload_voxposer_variables()
            # need this reset here to move the arms back to the starting positions NOTE: doesn't seem to be needed anymore
            # self._task._scene.reset()
            obs = self._task.get_observation()

        # preprocess obs for VoxPoserRLBench2Robots
        obs = self._rlbench_env._process_obs(obs)

        if self._rlbench_env._dominant_assitive_policy:
            self._rlbench_env.update_env_variables()
            # update obs in _rlbench_env for the following function calls
            self._rlbench_env.init_obs = obs
            self._rlbench_env.latest_obs = obs

            ###### for debugging
            # debug_front_rgb = np.clip((self._rlbench_env.latest_obs.front_rgb).astype(np.uint8), 0, 255)
            # from PIL import Image
            # debug_front_rgb = Image.fromarray(debug_front_rgb)
            # debug_front_rgb.show()

            # determine which arm to use
            self._rlbench_env.determine_dominant_hand()

            # set the dominant arm
            self._task.set_dominant(self._rlbench_env._dominant_arm)

            # get lagnauge instruction based on the determined dominant arm
            # language instruction is used in VoxPoser to control which arm to execute first (LLM prompting)
            self._lang_goal = self._task.get_task_descriptions_dominant_assistive()[0]
        else:
            self._lang_goal = self._task.get_task_descriptions()[0]

        # update VoxPoserRLBench2Robots variables
        self._rlbench_env.init_obs = obs
        self._rlbench_env.latest_obs = obs
        self._rlbench_env._update_visualizer()

        # NOTE: need to call scene.step and get the observation again; otherwise,
        # the environment would not be properly reset. HACK but couldn't find a more elegant fix.
        self._rlbench_env.task._scene.step()
        obs = self._task.get_observation()
        obs = self._rlbench_env._process_obs(obs)
        self._rlbench_env.init_obs = obs
        self._rlbench_env.latest_obs = obs
        self._rlbench_env._update_visualizer()

        try:
            if self._rlbench_env._dominant_assitive_policy:
                waypoints, waypoints_labels = self._task._scene.task.get_waypoints_dominant_assistive(dominant=self._rlbench_env._dominant_arm)
            else:
                waypoints, waypoints_labels = self._task._scene.task.get_waypoints()
            if len(waypoints) == 0:
                print('!!! Object cannot be reached due to infeasible path. Calling reset_to_demo_voxposer to re-place the object.')
                self.reset_to_demo_voxposer(i, ep_number)
        except:
            print('!!! Object cannot be reached due to infeasible path. Calling reset_to_demo_voxposer to re-place the object.')
            self.reset_to_demo_voxposer(i, ep_number)

        return self._lang_goal, self._previous_obs_dict

    def get_dominant_arm(self):
        return self._rlbench_env._dominant_arm

class CustomMultiTaskRLBenchEnv(MultiTaskRLBenchEnv):

    def __init__(self,
                 task_classes: List[Type[Task2Robots]],
                 observation_config: ObservationConfig2Robots,
                 action_mode: ActionMode,
                 episode_length: int,
                 dataset_root: str = '',
                 channels_last: bool = False,
                 reward_scale=100.0,
                 headless: bool = True,
                 swap_task_every: int = 1,
                 time_in_state: bool = False,
                 include_lang_goal_in_obs: bool = False,
                 record_every_n: int = 20):
        super(CustomMultiTaskRLBenchEnv, self).__init__(
            task_classes, observation_config, action_mode, dataset_root,
            channels_last, headless=headless, swap_task_every=swap_task_every,
            include_lang_goal_in_obs=include_lang_goal_in_obs)
        # self._reward_scale = reward_scale
        # self._episode_index = 0
        # self._record_current_episode = False
        # self._record_cam = None
        # self._previous_obs, self._previous_obs_dict = None, None
        # self._recorded_images = []
        # self._episode_length = episode_length
        # self._time_in_state = time_in_state
        # self._record_every_n = record_every_n
        # self._i = 0
        # self._error_type_counts = {
        #     'IKError': 0,
        #     'ConfigurationPathError': 0,
        #     'InvalidActionError': 0,
        # }
        # self._last_exception = None
        raise NotImplementedError

    @property
    def observation_elements(self) -> List[ObservationElement]:
        obs_elems = super(CustomMultiTaskRLBenchEnv, self).observation_elements
        for oe in obs_elems:
            if oe.name == 'low_dim_state':
                oe.shape = (oe.shape[0] - 7 * 3 + int(self._time_in_state),)  # remove pose and joint velocities as they will not be included
                self.low_dim_state_len = oe.shape[0]
        return obs_elems

    def extract_obs(self, obs: Observation2Robots, t=None, prev_action=None):
        obs.joint_velocities = None
        grip_mat = obs.gripper_matrix
        grip_pose = obs.gripper_pose
        joint_pos = obs.joint_positions
        obs.gripper_pose = None
        # obs.gripper_pose = None
        obs.gripper_matrix = None
        obs.wrist_camera_matrix = None
        obs.joint_positions = None
        if obs.gripper_joint_positions is not None:
            obs.gripper_joint_positions = np.clip(
                obs.gripper_joint_positions, 0., 0.04)

        obs_dict = super(CustomMultiTaskRLBenchEnv, self).extract_obs(obs)

        if self._time_in_state:
            time = (1. - ((self._i if t is None else t) / float(
                self._episode_length - 1))) * 2. - 1.
            obs_dict['low_dim_state'] = np.concatenate(
                [obs_dict['low_dim_state'], [time]]).astype(np.float32)

        obs.gripper_matrix = grip_mat
        # obs.gripper_pose = grip_pose
        obs.joint_positions = joint_pos
        obs.gripper_pose = grip_pose
        # obs_dict['gripper_pose'] = grip_pose
        return obs_dict

    def launch(self):
        super(CustomMultiTaskRLBenchEnv, self).launch()
        self._task._scene.register_step_callback(self._my_callback)
        if self.eval:
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam_base = Dummy('cam_cinematic_base')
            cam_base.rotate([0, 0, np.pi * 0.75])
            self._record_cam = VisionSensor.create([320, 180])
            self._record_cam.set_explicit_handling(True)
            self._record_cam.set_pose(cam_placeholder.get_pose())
            self._record_cam.set_render_mode(RenderMode.OPENGL)

    def reset(self) -> dict:
        self._i = 0
        self._previous_obs_dict = super(CustomMultiTaskRLBenchEnv, self).reset()
        self._record_current_episode = (
                self.eval and self._episode_index % self._record_every_n == 0)
        self._episode_index += 1
        self._recorded_images.clear()
        return self._previous_obs_dict

    def register_callback(self, func):
        self._task._scene.register_step_callback(func)

    def _my_callback(self):
        if self._record_current_episode:
            self._record_cam.handle_explicitly()
            cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
            self._recorded_images.append(cap)

    def _append_final_frame(self, success: bool):
        self._record_cam.handle_explicitly()
        img = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(img)
        final_frames = np.zeros((10, ) + img.shape[:2] + (3,), dtype=np.uint8)
        # Green/red for success/failure
        final_frames[:, :, :, 1 if success else 0] = 255
        self._recorded_images.extend(list(final_frames))

    def step(self, act_result: ActResult) -> Transition:
        action = act_result.action
        success = False
        obs = self._previous_obs_dict  # in case action fails.

        try:
            obs, reward, terminal = self._task.step(action)
            if reward >= 1:
                success = True
                reward *= self._reward_scale
            else:
                reward = 0.0
            obs = self.extract_obs(obs)
            self._previous_obs_dict = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminal = True
            reward = 0.0

            if isinstance(e, IKError):
                self._error_type_counts['IKError'] += 1
            elif isinstance(e, ConfigurationPathError):
                self._error_type_counts['ConfigurationPathError'] += 1
            elif isinstance(e, InvalidActionError):
                self._error_type_counts['InvalidActionError'] += 1

            self._last_exception = e

        summaries = []
        self._i += 1
        if ((terminal or self._i == self._episode_length) and
                self._record_current_episode):
            self._append_final_frame(success)
            vid = np.array(self._recorded_images).transpose((0, 3, 1, 2))
            task_name = change_case(self._task._task.__class__.__name__)
            summaries.append(VideoSummary(
                'episode_rollout_' + ('success' if success else 'fail') + f'/{task_name}',
                vid, fps=30))

            # error summary
            error_str = f"Errors - IK : {self._error_type_counts['IKError']}, " \
                        f"ConfigPath : {self._error_type_counts['ConfigurationPathError']}, " \
                        f"InvalidAction : {self._error_type_counts['InvalidActionError']}"
            if not success and self._last_exception is not None:
                error_str += f"\n Last Exception: {self._last_exception}"
                self._last_exception = None

            summaries.append(TextSummary('errors', f"Success: {success} | " + error_str))
        return Transition(obs, reward, terminal, summaries=summaries)

    def reset_to_demo(self, i, variation_number=-1):
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        self._i = 0
        # super(CustomMultiTaskRLBenchEnv, self).reset()

        # if variation_number == -1:
        #     self._task.sample_variation()
        # else:
        #     self._task.set_variation(variation_number)

        self._task.set_variation(-1)
        d = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i)[0]

        self._task.set_variation(d.variation_number)
        _, obs = self._task.reset_to_demo(d)
        self._lang_goal = self._task.get_task_descriptions()[0]

        self._previous_obs_dict = self.extract_obs(obs)
        self._record_current_episode = (
                self.eval and self._episode_index % self._record_every_n == 0)
        self._episode_index += 1
        self._recorded_images.clear()

        return self._previous_obs_dict