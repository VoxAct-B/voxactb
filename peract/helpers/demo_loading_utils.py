import logging
from typing import List

import numpy as np
from rlbench.demo import Demo


def _is_stopped(demo, i, obs, stopped_buffer, which_arm, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)

    if which_arm == 'right':
        gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_right_open == demo[i + 1].gripper_right_open and
             obs.gripper_right_open == demo[i - 1].gripper_right_open and
             demo[i - 2].gripper_right_open == demo[i - 1].gripper_right_open))
        small_delta = np.allclose(obs.joint_velocities_right, 0, atol=delta)
        stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    elif which_arm == 'left':
        gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_left_open == demo[i + 1].gripper_left_open and
             obs.gripper_left_open == demo[i - 1].gripper_left_open and
             demo[i - 2].gripper_left_open == demo[i - 1].gripper_left_open))
        small_delta = np.allclose(obs.joint_velocities_left, 0, atol=delta)
        stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    else:
        raise NotImplementedError
    return stopped

def _is_stopped_2arms(demo, i, obs, stopped_buffer_right, stopped_buffer_left, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)

    # if right gripper or left gripper satisfies the condition, then it's a keyframe
    gripper_state_no_change_right = (
        i < (len(demo) - 2) and
        (obs.gripper_right_open == demo[i + 1].gripper_right_open and
            obs.gripper_right_open == demo[i - 1].gripper_right_open and
            demo[i - 2].gripper_right_open == demo[i - 1].gripper_right_open))
    small_delta_right = np.allclose(obs.joint_velocities_right, 0, atol=delta)

    gripper_state_no_change_left = (
        i < (len(demo) - 2) and
        (obs.gripper_left_open == demo[i + 1].gripper_left_open and
            obs.gripper_left_open == demo[i - 1].gripper_left_open and
            demo[i - 2].gripper_left_open == demo[i - 1].gripper_left_open))
    small_delta_left = np.allclose(obs.joint_velocities_left, 0, atol=delta)

    stopped_right = (stopped_buffer_right <= 0 and (small_delta_right and
            (not next_is_not_final) and gripper_state_no_change_right))

    stopped_left = (stopped_buffer_left <= 0 and (small_delta_left and
            (not next_is_not_final) and gripper_state_no_change_left))

    # debugging
    # print('Is right stopped: ', stopped_right)
    # print('Is left stopped: ', stopped_left)
    # print('Is both stopped: ', stopped_right and stopped_left)
    return stopped_right, stopped_left

def keypoint_discovery(demo: Demo,
                       stopping_delta=0.1,
                       which_arm='right',
                       method='heuristic',
                       saved_every_last_inserted=0,
                       dominant_assistive_arm='',
                       use_default_stopped_buffer_timesteps=False,
                       stopped_buffer_timesteps_overwrite=0) -> List[int]:
    episode_keypoints = []
    if method == 'heuristic':
        if which_arm == 'right':
            gripper_open = demo[0].gripper_right_open
        elif which_arm == 'left':
            gripper_open = demo[0].gripper_left_open
        elif which_arm == 'dominant' or which_arm == 'assistive':
            gripper_right_open = demo[0].gripper_right_open
            gripper_left_open = demo[0].gripper_left_open
            prev_gripper_right_open = gripper_right_open
            prev_gripper_left_open = gripper_left_open
            stopped_buffer_right = 0
            stopped_buffer_left = 0
            labels = [] # only use if arm_id_to_proprio is True

            if stopped_buffer_timesteps_overwrite != 0:
                stopped_buffer_timesteps = stopped_buffer_timesteps_overwrite
            else:
                if which_arm == 'dominant' or use_default_stopped_buffer_timesteps:
                    stopped_buffer_timesteps = 4
                else:
                    # which_arm == 'assistive'
                    # ours implementation (tested in '11/16/2023 to 11/29 progress report': ours_v2 (best left-armed model)
                    stopped_buffer_timesteps = 12

            for i, obs in enumerate(demo):
                stopped_right, stopped_left = _is_stopped_2arms(demo, i, obs, stopped_buffer_right, stopped_buffer_left, stopping_delta)
                stopped_buffer_right = stopped_buffer_timesteps if stopped_right else stopped_buffer_right - 1
                stopped_buffer_left = stopped_buffer_timesteps if stopped_left else stopped_buffer_left - 1
                # If change in gripper, or end of episode.
                last = i == (len(demo) - 1)

                if dominant_assistive_arm == 'left' and i != 0 and (obs.gripper_left_open != prev_gripper_left_open or last or stopped_left):
                    episode_keypoints.append(i)
                    labels.append(1) # left-armed keyframe

                if dominant_assistive_arm == 'right' and i != 0 and (obs.gripper_right_open != prev_gripper_right_open or last or stopped_right):
                    episode_keypoints.append(i)
                    labels.append(0) # right-armed keyframe

                prev_gripper_right_open = obs.gripper_right_open
                prev_gripper_left_open = obs.gripper_left_open
            if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
                    episode_keypoints[-2]:
                episode_keypoints.pop(-2)
                labels.pop(-2)
            logging.debug('Found %d keypoints.' % len(episode_keypoints),
                        episode_keypoints)
            return episode_keypoints, labels
        else:
            if which_arm == 'multiarm':
                # ours implementation (tested in '11/16/2023 to 11/29 progress report': ours_v2 (best left-armed model)
                stopped_buffer_timesteps_left = 12
            else:
                # baseline #2 implementation
                stopped_buffer_timesteps_left = 4

            gripper_right_open = demo[0].gripper_right_open
            gripper_left_open = demo[0].gripper_left_open
            prev_gripper_right_open = gripper_right_open
            prev_gripper_left_open = gripper_left_open
            stopped_buffer_right = 0
            stopped_buffer_left = 0
            labels = []
            for i, obs in enumerate(demo):
                stopped_right, stopped_left = _is_stopped_2arms(demo, i, obs, stopped_buffer_right, stopped_buffer_left, stopping_delta)
                stopped_buffer_right = 4 if stopped_right else stopped_buffer_right - 1
                stopped_buffer_left = stopped_buffer_timesteps_left if stopped_left else stopped_buffer_left - 1
                # If change in gripper, or end of episode.
                last = i == (len(demo) - 1)
                if i != 0 and (obs.gripper_right_open != prev_gripper_right_open or
                               obs.gripper_left_open != prev_gripper_left_open or
                            last or stopped_right or stopped_left):
                    if obs.gripper_right_open != prev_gripper_right_open or last or stopped_right:
                        # this is a right-armed keyframe
                        # note that I consider last as right-armed action
                        labels.append(0)
                    else:
                        # this is a left-armed keyframe
                        labels.append(1)
                    episode_keypoints.append(i)
                prev_gripper_right_open = obs.gripper_right_open
                prev_gripper_left_open = obs.gripper_left_open
            if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
                    episode_keypoints[-2]:
                episode_keypoints.pop(-2)
                labels.pop(-2)
            logging.debug('Found %d keypoints.' % len(episode_keypoints),
                        episode_keypoints)
            return episode_keypoints, labels

        # right or left arm only
        prev_gripper_open = gripper_open
        stopped_buffer = 0
        if which_arm == 'left':
            # tested in 11/16/2023 to 11/29 progress report: ours_v2 (best left-armed model)
            stopped_buffer_timesteps = 12
        else:
            # default
            stopped_buffer_timesteps = 4

        last_inserted_counter = 0
        for i, obs in enumerate(demo):
            stopped = _is_stopped(demo, i, obs, stopped_buffer, which_arm, stopping_delta)
            stopped_buffer = stopped_buffer_timesteps if stopped else stopped_buffer - 1
            # If change in gripper, or end of episode.
            last = i == (len(demo) - 1)
            if which_arm == 'right':
                if i != 0 and (obs.gripper_right_open != prev_gripper_open or
                            last or stopped):
                    episode_keypoints.append(i)
                    last_inserted_counter = 0
                else:
                    last_inserted_counter += 1

                # save a keypoint every 'saved_every_last_inserted' steps for tasks that require multi-step contact-rich manipulation, like sweep_to_dustpan
                if saved_every_last_inserted > 0 and last_inserted_counter >= saved_every_last_inserted:
                        episode_keypoints.append(i)
                        last_inserted_counter = 0
                        print(f'saved_every_last_inserted at {i}') # for debugging
                prev_gripper_open = obs.gripper_right_open
            elif which_arm == 'left':
                if i != 0 and (obs.gripper_left_open != prev_gripper_open or
                            last or stopped):
                    episode_keypoints.append(i)
                prev_gripper_open = obs.gripper_left_open

        if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
                episode_keypoints[-2]:
            episode_keypoints.pop(-2)
        logging.debug('Found %d keypoints.' % len(episode_keypoints),
                      episode_keypoints)
        return episode_keypoints

    elif method == 'random':
        # Randomly select keypoints.
        # episode_keypoints = np.random.choice(
        #     range(len(demo)),
        #     size=20,
        #     replace=False)
        # episode_keypoints.sort()
        # return episode_keypoints
        raise NotImplementedError

    elif method == 'fixed_interval':
        # Fixed interval.
        # episode_keypoints = []
        # segment_length = len(demo) // 20
        # for i in range(0, len(demo), segment_length):
        #     episode_keypoints.append(i)
        # return episode_keypoints
        raise NotImplementedError

    else:
        raise NotImplementedError


def keypoint_discovery_no_duplicate(demo: Demo,
                       stopping_delta=0.1,
                       which_arm='right',
                       method='heuristic',
                       saved_every_last_inserted=0,
                       dominant_assistive_arm='',
                       use_default_stopped_buffer_timesteps=False,
                       stopped_buffer_timesteps_overwrite=0) -> List[int]:
    episode_keypoints = []
    if method == 'heuristic':
        if which_arm == 'right':
            gripper_open = demo[0].gripper_right_open
        elif which_arm == 'left':
            gripper_open = demo[0].gripper_left_open
        elif which_arm == 'dominant' or which_arm == 'assistive':
            gripper_right_open = demo[0].gripper_right_open
            gripper_left_open = demo[0].gripper_left_open
            prev_gripper_right_open = gripper_right_open
            prev_gripper_left_open = gripper_left_open

            gripper_right_pose = demo[0].gripper_right_pose
            gripper_left_pose = demo[0].gripper_left_pose
            prev_gripper_right_pose = gripper_right_pose
            prev_gripper_left_pose = gripper_left_pose

            stopped_buffer_right = 0
            stopped_buffer_left = 0
            labels = [] # only use if arm_id_to_proprio is True

            if stopped_buffer_timesteps_overwrite != 0:
                stopped_buffer_timesteps = stopped_buffer_timesteps_overwrite
            else:
                if which_arm == 'dominant' or use_default_stopped_buffer_timesteps:
                    stopped_buffer_timesteps = 4
                else:
                    # which_arm == 'assistive'
                    # ours implementation (tested in '11/16/2023 to 11/29 progress report': ours_v2 (best left-armed model)
                    stopped_buffer_timesteps = 12

            for i, obs in enumerate(demo):
                stopped_right, stopped_left = _is_stopped_2arms(demo, i, obs, stopped_buffer_right, stopped_buffer_left, stopping_delta)
                stopped_buffer_right = stopped_buffer_timesteps if stopped_right else stopped_buffer_right - 1
                stopped_buffer_left = stopped_buffer_timesteps if stopped_left else stopped_buffer_left - 1
                # If change in gripper, or end of episode.
                last = i == (len(demo) - 1)

                if dominant_assistive_arm == 'left' and i != 0 and np.allclose(obs.gripper_left_pose, prev_gripper_left_pose, atol=1e-3):
                    continue

                if dominant_assistive_arm == 'right' and i != 0 and np.allclose(obs.gripper_right_pose, prev_gripper_right_pose, atol=1e-3):
                    continue

                if dominant_assistive_arm == 'left' and i != 0 and (obs.gripper_left_open != prev_gripper_left_open or last or stopped_left):
                    episode_keypoints.append(i)
                    labels.append(1) # left-armed keyframe
                    prev_gripper_left_pose = obs.gripper_left_pose

                if dominant_assistive_arm == 'right' and i != 0 and (obs.gripper_right_open != prev_gripper_right_open or last or stopped_right):
                    episode_keypoints.append(i)
                    labels.append(0) # right-armed keyframe
                    prev_gripper_right_pose = obs.gripper_right_pose

                prev_gripper_right_open = obs.gripper_right_open
                prev_gripper_left_open = obs.gripper_left_open
            if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
                    episode_keypoints[-2]:
                episode_keypoints.pop(-2)
                labels.pop(-2)
            logging.debug('Found %d keypoints.' % len(episode_keypoints),
                        episode_keypoints)
            return episode_keypoints, labels
        else:
            raise NotImplementedError
        raise NotImplementedError

    elif method == 'random':
        # Randomly select keypoints.
        # episode_keypoints = np.random.choice(
        #     range(len(demo)),
        #     size=20,
        #     replace=False)
        # episode_keypoints.sort()
        # return episode_keypoints
        raise NotImplementedError

    elif method == 'fixed_interval':
        # Fixed interval.
        # episode_keypoints = []
        # segment_length = len(demo) // 20
        # for i in range(0, len(demo), segment_length):
        #     episode_keypoints.append(i)
        # return episode_keypoints
        raise NotImplementedError

    else:
        raise NotImplementedError

# find minimum difference between any two elements in list
def find_minimum_difference(lst):
    minimum = lst[-1]
    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] < minimum:
            minimum = lst[i] - lst[i - 1]
    return minimum