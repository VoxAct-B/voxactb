# Adapted from ARM
# Source: https://github.com/stepjam/ARM
# License: https://github.com/stepjam/ARM/LICENSE

import logging
from typing import List

import numpy as np
from rlbench.backend.observation_two_robots import Observation2Robots
from rlbench.observation_config_two_robots import ObservationConfig2Robots
import rlbench.utils as rlbench_utils
from rlbench.demo import Demo
from yarr.replay_buffer.prioritized_replay_buffer import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from yarr.replay_buffer.task_uniform_replay_buffer import TaskUniformReplayBuffer

from helpers import demo_loading_utils, utils
from helpers.preprocess_agent import PreprocessAgent
from helpers.clip.core.clip import tokenize
from agents.peract_bc.perceiver_lang_io import PerceiverVoxelLangEncoder, PerceiverVoxelLang2RobotsEncoder
from agents.peract_bc.qattention_peract_bc_agent import QAttentionPerActBCAgent, QAttentionPerActBCAgent2Robots
from agents.peract_bc.qattention_stack_agent import QAttentionStackAgent, QAttentionStackAgent2Robots

import torch
import torch.nn as nn
import multiprocessing as mp
from torch.multiprocessing import Process, Value, Manager
from helpers.clip.core.clip import build_model, load_clip, tokenize
from omegaconf import DictConfig

REWARD_SCALE = 100.0
LOW_DIM_DOMINANT_ASSISTIVE_SIZE = 7
LOW_DIM_SIZE = 4
SINGLE_ARM = ['right', 'left']

def create_replay(batch_size: int, timesteps: int,
                  prioritisation: bool, task_uniform: bool,
                  save_dir: str, cameras: list,
                  voxel_sizes,
                  image_size=[128, 128],
                  replay_size=3e5,
                  which_arm='right',
                  crop_target_obj_voxel=False,
                  arm_pred_loss=False,
                  arm_id_to_proprio=False):

    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = (3 + 1)
    gripper_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77
    lang_feat_dim = 1024
    lang_emb_dim = 512

    # low_dim_state
    observation_elements = []
    if which_arm == 'both':
        observation_elements.append(
            ObservationElement('low_dim_state_right_arm', (LOW_DIM_SIZE,), np.float32))
        observation_elements.append(
            ObservationElement('low_dim_state_left_arm', (LOW_DIM_SIZE,), np.float32))
    elif which_arm == 'dominant' or which_arm == 'assistive':
        # 7 scalar values: left-armed gripper open, left-armed left finger joint position, left-armed right finger joint position,
        # right-armed gripper open, right-armed left finger joint position, right-armed right finger joint position, and timestep
        if arm_id_to_proprio:
            # plus 1 for arm ID
            observation_elements.append(
                ObservationElement('low_dim_state', (LOW_DIM_DOMINANT_ASSISTIVE_SIZE+1,), np.float32))
        else:
            observation_elements.append(
                ObservationElement('low_dim_state', (LOW_DIM_DOMINANT_ASSISTIVE_SIZE,), np.float32))
    else:
        observation_elements.append(
            ObservationElement('low_dim_state', (LOW_DIM_SIZE,), np.float32))

    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(
            ObservationElement('%s_rgb' % cname, (3, *image_size,), np.float32))
        observation_elements.append(
            ObservationElement('%s_point_cloud' % cname, (3, *image_size),
                               np.float32))  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement('%s_camera_extrinsics' % cname, (4, 4,), np.float32))
        observation_elements.append(
            ObservationElement('%s_camera_intrinsics' % cname, (3, 3,), np.float32))

    # discretized translation, discretized rotation, discrete ignore collision, 6-DoF gripper pose, and pre-trained language embeddings
    if which_arm == 'both':
        observation_elements.extend([
            ReplayElement('trans_action_indicies_right', (trans_indicies_size,),
                        np.int32),
            ReplayElement('rot_grip_action_indicies_right', (rot_and_grip_indicies_size,),
                        np.int32),
            ReplayElement('ignore_collisions', (ignore_collisions_size,),
                        np.int32),
            ReplayElement('gripper_pose_right', (gripper_pose_size,),
                        np.float32),
            ReplayElement('trans_action_indicies_left', (trans_indicies_size,),
                        np.int32),
            ReplayElement('rot_grip_action_indicies_left', (rot_and_grip_indicies_size,),
                        np.int32),
            ReplayElement('gripper_pose_left', (gripper_pose_size,),
                        np.float32),
            ReplayElement('lang_goal_emb', (lang_feat_dim,),
                        np.float32),
            ReplayElement('lang_token_embs', (max_token_seq_len, lang_emb_dim,),
                        np.float32), # extracted from CLIP's language encoder
            ReplayElement('label', (1,),
                        np.int32),
            ReplayElement('task', (),
                        str),
            ReplayElement('lang_goal', (1,),
                        object),  # language goal string for debugging and visualization
        ])
    else:
        observation_elements.extend([
            ReplayElement('trans_action_indicies', (trans_indicies_size,),
                        np.int32),
            ReplayElement('rot_grip_action_indicies', (rot_and_grip_indicies_size,),
                        np.int32),
            ReplayElement('ignore_collisions', (ignore_collisions_size,),
                        np.int32),
            ReplayElement('gripper_pose', (gripper_pose_size,),
                        np.float32),
            ReplayElement('lang_goal_emb', (lang_feat_dim,),
                        np.float32),
            ReplayElement('lang_token_embs', (max_token_seq_len, lang_emb_dim,),
                        np.float32), # extracted from CLIP's language encoder
            ReplayElement('task', (),
                        str),
            ReplayElement('lang_goal', (1,),
                        object),  # language goal string for debugging and visualization
        ])

    if arm_pred_loss:
        observation_elements.extend([
            ReplayElement('label', (1,), np.int32)
        ])

    if crop_target_obj_voxel:
        # target object position in world coordinates
        observation_elements.append(
                ObservationElement('target_object_scene_bounds', (6,), np.float32))

    extra_replay_elements = [
        ReplayElement('demo', (), np.bool),
    ]

    replay_buffer = TaskUniformReplayBuffer(
        save_dir=save_dir,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(replay_size),
        action_shape=(8,),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=extra_replay_elements
    )
    return replay_buffer


def _get_action(
        obs_tp1: Observation2Robots,
        obs_tm1: Observation2Robots,
        rlbench_scene_bounds: List[float], # metric 3D bounds of the scene
        voxel_sizes: List[int],
        bounds_offset: List[float],
        rotation_resolution: int,
        crop_augmentation: bool,
        which_arm: str,
        keypoint_label: int,
        dominant_assistive_arm: str):
    if which_arm in SINGLE_ARM or which_arm == 'multiarm' or which_arm == 'dominant' or which_arm == 'assistive':
        if which_arm == 'right' or dominant_assistive_arm == 'right':
            gripper_pose = obs_tp1.gripper_right_pose
            gripper_open = obs_tp1.gripper_right_open
        elif which_arm == 'left' or dominant_assistive_arm == 'left':
            gripper_pose = obs_tp1.gripper_left_pose
            gripper_open = obs_tp1.gripper_left_open
        elif which_arm == 'multiarm':
            if keypoint_label == 0:
                # right-armed action
                gripper_pose = obs_tp1.gripper_right_pose
                gripper_open = obs_tp1.gripper_right_open
            elif keypoint_label == 1:
                # left-armed action
                gripper_pose = obs_tp1.gripper_left_pose
                gripper_open = obs_tp1.gripper_left_open
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        quat = utils.normalize_quaternion(gripper_pose[3:])
        if quat[-1] < 0:
            quat = -quat
        disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
        disc_rot = utils.correct_rotation_instability(disc_rot, rotation_resolution)

        attention_coordinate = gripper_pose[:3]
        trans_indicies, attention_coordinates = [], []
        bounds = np.array(rlbench_scene_bounds)
        ignore_collisions = int(obs_tm1.ignore_collisions)
        for depth, vox_size in enumerate(voxel_sizes): # only single voxelization-level is used in PerAct
            if depth > 0:
                if crop_augmentation:
                    shift = bounds_offset[depth - 1] * 0.75
                    attention_coordinate += np.random.uniform(-shift, shift, size=(3,))
                bounds = np.concatenate([attention_coordinate - bounds_offset[depth - 1],
                                        attention_coordinate + bounds_offset[depth - 1]])
            index = utils.point_to_voxel_index(
                gripper_pose[:3], vox_size, bounds)
            trans_indicies.extend(index.tolist())
            res = (bounds[3:] - bounds[:3]) / vox_size
            attention_coordinate = bounds[:3] + res * index
            attention_coordinates.append(attention_coordinate)

        rot_and_grip_indicies = disc_rot.tolist()
        grip = float(gripper_open)
        rot_and_grip_indicies.extend([int(gripper_open)])

        return trans_indicies, rot_and_grip_indicies, ignore_collisions, np.concatenate(
            [gripper_pose, np.array([grip])]), attention_coordinates
    else:

        ########## right arm ##########
        gripper_right_pose = obs_tp1.gripper_right_pose
        gripper_right_open = obs_tp1.gripper_right_open

        quat = utils.normalize_quaternion(gripper_right_pose[3:])
        if quat[-1] < 0:
            quat = -quat
        disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
        disc_rot = utils.correct_rotation_instability(disc_rot, rotation_resolution)

        attention_coordinate_right = gripper_right_pose[:3]
        trans_indicies_right, attention_coordinates_right = [], []
        bounds = np.array(rlbench_scene_bounds)
        ignore_collisions = int(obs_tm1.ignore_collisions)
        for depth, vox_size in enumerate(voxel_sizes): # only single voxelization-level is used in PerAct
            if depth > 0:
                if crop_augmentation:
                    shift = bounds_offset[depth - 1] * 0.75
                    attention_coordinate_right += np.random.uniform(-shift, shift, size=(3,))
                bounds = np.concatenate([attention_coordinate_right - bounds_offset[depth - 1],
                                        attention_coordinate_right + bounds_offset[depth - 1]])
            index = utils.point_to_voxel_index(
                gripper_right_pose[:3], vox_size, bounds)
            trans_indicies_right.extend(index.tolist())
            res = (bounds[3:] - bounds[:3]) / vox_size
            attention_coordinate_right = bounds[:3] + res * index
            attention_coordinates_right.append(attention_coordinate_right)

        rot_and_grip_indicies_right = disc_rot.tolist()
        grip_right = float(gripper_right_open)
        rot_and_grip_indicies_right.extend([int(gripper_right_open)])


        ########## left arm ##########
        gripper_left_pose = obs_tp1.gripper_left_pose
        gripper_left_open = obs_tp1.gripper_left_open

        quat = utils.normalize_quaternion(gripper_left_pose[3:])
        if quat[-1] < 0:
            quat = -quat
        disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
        disc_rot = utils.correct_rotation_instability(disc_rot, rotation_resolution)

        attention_coordinate_left = gripper_left_pose[:3]
        trans_indicies_left, attention_coordinates_left = [], []
        bounds = np.array(rlbench_scene_bounds)
        ignore_collisions = int(obs_tm1.ignore_collisions)
        for depth, vox_size in enumerate(voxel_sizes): # only single voxelization-level is used in PerAct
            if depth > 0:
                if crop_augmentation:
                    shift = bounds_offset[depth - 1] * 0.75
                    attention_coordinate_left += np.random.uniform(-shift, shift, size=(3,))
                bounds = np.concatenate([attention_coordinate_left - bounds_offset[depth - 1],
                                        attention_coordinate_left + bounds_offset[depth - 1]])
            index = utils.point_to_voxel_index(
                gripper_left_pose[:3], vox_size, bounds)
            trans_indicies_left.extend(index.tolist())
            res = (bounds[3:] - bounds[:3]) / vox_size
            attention_coordinate_left = bounds[:3] + res * index
            attention_coordinates_left.append(attention_coordinate_left)

        rot_and_grip_indicies_left = disc_rot.tolist()
        grip_left = float(gripper_left_open)
        rot_and_grip_indicies_left.extend([int(gripper_left_open)])

        return trans_indicies_right, rot_and_grip_indicies_right, ignore_collisions, np.concatenate(
            [gripper_right_pose, np.array([grip_right])]), attention_coordinates_right, trans_indicies_left, rot_and_grip_indicies_left, np.concatenate(
            [gripper_left_pose, np.array([grip_left])]), attention_coordinates_left


def _add_keypoints_to_replay(
        cfg: DictConfig,
        task: str,
        task_idx: int,
        replay: ReplayBuffer,
        inital_obs: Observation2Robots,
        demo: Demo,
        episode_keypoints: List[int],
        cameras: List[str],
        rlbench_scene_bounds: List[float],
        voxel_sizes: List[int],
        bounds_offset: List[float],
        rotation_resolution: int,
        crop_augmentation: bool,
        description: str = '',
        clip_model = None,
        device = 'cpu',
        labels = None,
        dominant_assistive_arm = ''):
    obs = inital_obs

    if type(rlbench_scene_bounds[0]) is float:
        # single task
        rlbench_scene_bounds_processed = rlbench_scene_bounds
    else:
        # multi task
        rlbench_scene_bounds_processed = rlbench_scene_bounds[task_idx]

    if type(cfg.method.crop_radius) is float or cfg.method.crop_radius == 'auto':
        # single task
        crop_radius = cfg.method.crop_radius
    else:
        # multi task
        crop_radius = cfg.method.crop_radius[task_idx]
    # print(f'task_idx: {task_idx}, rlbench_scene_bounds: {rlbench_scene_bounds_processed}, crop_radius: {crop_radius}') # for debugging

    for k, keypoint in enumerate(episode_keypoints):
        obs_tp1 = demo[keypoint]
        obs_tm1 = demo[max(0, keypoint - 1)]

        if cfg.method.crop_target_obj_voxel:
            # overwrite rlbench_scene_bounds with target object position + offsets
            if crop_radius == 'auto' and obs_tp1.auto_crop_radius != 0.0:
                rlbench_scene_bounds_processed = utils.get_new_scene_bounds_based_on_crop(obs_tp1.auto_crop_radius, obs_tp1.target_object_pos)
                # print('Auto crop radius: ', obs_tp1.auto_crop_radius) # for debugging
            else:
                rlbench_scene_bounds_processed = utils.get_new_scene_bounds_based_on_crop(crop_radius, obs_tp1.target_object_pos)

        if labels is not None:
            keypoint_label = labels[k]
        else:
            keypoint_label = -1

        if cfg.method.which_arm == 'both':
            trans_indicies_right, rot_grip_indicies_right, ignore_collisions, action_right, attention_coordinates_right, \
                 trans_indicies_left, rot_grip_indicies_left, action_left, attention_coordinates_left = _get_action(
                obs_tp1, obs_tm1, rlbench_scene_bounds_processed, voxel_sizes, bounds_offset,
                rotation_resolution, crop_augmentation, cfg.method.which_arm, keypoint_label)
        else:
            trans_indicies, rot_grip_indicies, ignore_collisions, action, attention_coordinates = _get_action(
                obs_tp1, obs_tm1, rlbench_scene_bounds_processed, voxel_sizes, bounds_offset,
                rotation_resolution, crop_augmentation, cfg.method.which_arm, keypoint_label, dominant_assistive_arm)

        terminal = (k == len(episode_keypoints) - 1)
        reward = float(terminal) * REWARD_SCALE if terminal else 0

        if cfg.method.which_arm == 'multiarm':
            left_arm_description, right_arm_description = utils.extract_left_and_right_arm_instruction(description)
            if keypoint_label == 0:
                # right-armed action
                which_arm = 'right'
                curr_description = right_arm_description
            elif keypoint_label == 1:
                # left-armed action
                which_arm = 'left'
                curr_description = left_arm_description

            if cfg.method.arm_pred_input:
                obs_dict = utils.extract_obs(obs, t=k,
                                            cameras=cameras, episode_length=cfg.rlbench.episode_length, which_arm=which_arm, keypoint_label=keypoint_label)
            else:
                obs_dict = utils.extract_obs(obs, t=k,
                                        cameras=cameras, episode_length=cfg.rlbench.episode_length, which_arm=which_arm)
            tokens = tokenize([curr_description]).numpy()
            token_tensor = torch.from_numpy(tokens).to(device)
            sentence_emb, token_embs = clip_model.encode_text_with_embeddings(token_tensor)
            obs_dict['lang_goal_emb'] = sentence_emb[0].float().detach().cpu().numpy()
            obs_dict['lang_token_embs'] = token_embs[0].float().detach().cpu().numpy()
        else:
            if cfg.method.arm_id_to_proprio:
                obs_dict = utils.extract_obs(obs, t=k,
                                        cameras=cameras, episode_length=cfg.rlbench.episode_length, which_arm=cfg.method.which_arm, keypoint_label=keypoint_label)
            else:
                obs_dict = utils.extract_obs(obs, t=k,
                                            cameras=cameras, episode_length=cfg.rlbench.episode_length, which_arm=cfg.method.which_arm)
            tokens = tokenize([description]).numpy()
            token_tensor = torch.from_numpy(tokens).to(device)
            sentence_emb, token_embs = clip_model.encode_text_with_embeddings(token_tensor)
            obs_dict['lang_goal_emb'] = sentence_emb[0].float().detach().cpu().numpy()
            obs_dict['lang_token_embs'] = token_embs[0].float().detach().cpu().numpy()

        if cfg.method.which_arm == 'right' or dominant_assistive_arm == 'right':
            gripper_pose = obs_tp1.gripper_right_pose
        elif cfg.method.which_arm == 'left' or dominant_assistive_arm == 'left':
            gripper_pose = obs_tp1.gripper_left_pose
        elif cfg.method.which_arm == 'multiarm':
            if which_arm == 'right':
                gripper_pose = obs_tp1.gripper_right_pose
            elif which_arm == 'left':
                gripper_pose = obs_tp1.gripper_left_pose
        else:
            gripper_pose_right = obs_tp1.gripper_right_pose
            gripper_pose_left = obs_tp1.gripper_left_pose

        if cfg.method.crop_target_obj_voxel:
            obs_dict['target_object_scene_bounds'] = rlbench_scene_bounds_processed

        others = {'demo': True}
        if cfg.method.which_arm == 'both':
            final_obs = {
                'trans_action_indicies_right': trans_indicies_right,
                'rot_grip_action_indicies_right': rot_grip_indicies_right,
                'gripper_pose_right': gripper_pose_right,
                'trans_action_indicies_left': trans_indicies_left,
                'rot_grip_action_indicies_left': rot_grip_indicies_left,
                'gripper_pose_left': gripper_pose_left,
                'task': task,
                'lang_goal': np.array([description], dtype=object),
                'label': [labels[k]],
            }
        elif cfg.method.which_arm == 'multiarm':
            final_obs = {
                'trans_action_indicies': trans_indicies,
                'rot_grip_action_indicies': rot_grip_indicies,
                'gripper_pose': gripper_pose,
                'task': task,
                'lang_goal': np.array([curr_description], dtype=object),
            }
        else:
            final_obs = {
                'trans_action_indicies': trans_indicies,
                'rot_grip_action_indicies': rot_grip_indicies,
                'gripper_pose': gripper_pose,
                'task': task,
                'lang_goal': np.array([description], dtype=object),
            }

        if cfg.method.arm_pred_loss:
            final_obs['label'] = [labels[k]]
            # print('Arm pred_loss_label: ', labels[k]) # for debugging

        others.update(final_obs)
        others.update(obs_dict)

        timeout = False
        if cfg.method.which_arm == 'both':
            # NOTE: action is actually not used, so for the two arms implementation, I'm not storing action_right and action_left
            replay.add(action_right, reward, terminal, timeout, **others)
        else:
            replay.add(action, reward, terminal, timeout, **others)
        obs = obs_tp1

    # final step
    if cfg.method.which_arm == 'multiarm':
        if cfg.method.arm_pred_input:
            obs_dict_tp1 = utils.extract_obs(obs_tp1, t=k + 1,
                                            cameras=cameras, episode_length=cfg.rlbench.episode_length, which_arm=which_arm, keypoint_label=keypoint_label)
        else:
            obs_dict_tp1 = utils.extract_obs(obs_tp1, t=k + 1,
                                            cameras=cameras, episode_length=cfg.rlbench.episode_length, which_arm=which_arm)
    else:
        if cfg.method.arm_id_to_proprio:
            obs_dict_tp1 = utils.extract_obs(obs_tp1, t=k + 1,
                                        cameras=cameras, episode_length=cfg.rlbench.episode_length, which_arm=cfg.method.which_arm, keypoint_label=keypoint_label)
        else:
            obs_dict_tp1 = utils.extract_obs(obs_tp1, t=k + 1,
                                            cameras=cameras, episode_length=cfg.rlbench.episode_length, which_arm=cfg.method.which_arm)
    obs_dict_tp1['lang_goal_emb'] = sentence_emb[0].float().detach().cpu().numpy()
    obs_dict_tp1['lang_token_embs'] = token_embs[0].float().detach().cpu().numpy()

    if cfg.method.crop_target_obj_voxel:
        obs_dict_tp1['target_object_scene_bounds'] = rlbench_scene_bounds_processed

    obs_dict_tp1.pop('wrist_world_to_cam', None)
    obs_dict_tp1.update(final_obs)
    replay.add_final(**obs_dict_tp1)

def swap_words(s, x, y):
    return y.join(part.replace(y, x) for part in s.split(x))

def fill_replay(cfg: DictConfig,
                obs_config: ObservationConfig2Robots,
                rank: int,
                replay: ReplayBuffer,
                task: str,
                task_idx: int,
                num_demos: int,
                demo_augmentation: bool,
                demo_augmentation_every_n: int,
                cameras: List[str],
                rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
                voxel_sizes: List[int],
                bounds_offset: List[float],
                rotation_resolution: int,
                crop_augmentation: bool,
                clip_model = None,
                device = 'cpu',
                keypoint_method = 'heuristic'):
    logging.getLogger().setLevel(cfg.framework.logging_level)

    if clip_model is None:
        model, _ = load_clip('RN50', jit=False, device=device)
        clip_model = build_model(model.state_dict())
        clip_model.to(device)
        del model

    logging.debug('Filling %s replay ...' % task)
    for d_idx in range(num_demos):
        # load demo from disk
        if cfg.method.is_real_robot:
            demo = rlbench_utils.get_stored_real_world_demos(
                amount=1, image_paths=False,
                dataset_root=cfg.rlbench.demo_path,
                variation_number=-1, task_name=task,
                obs_config=obs_config,
                random_selection=False,
                from_episode_number=d_idx,
                which_arm=cfg.method.which_arm)[0]
        else:
            demo = rlbench_utils.get_stored_demos(
                amount=1, image_paths=False,
                dataset_root=cfg.rlbench.demo_path,
                variation_number=-1, task_name=task,
                obs_config=obs_config,
                random_selection=False,
                from_episode_number=d_idx,
                which_arm=cfg.method.which_arm)[0]

        descs = demo._observations[0].misc['descriptions']

        # extract keypoints (a.k.a keyframes)
        dominant_assistive_arm = ''
        if cfg.method.which_arm == 'dominant':
            # we assume that the data is split in half into left arm being the dominant arm in the first half and right arm being the dominant arm in the other half
            dominant_eps = int(num_demos / 2)
            if num_demos == 1 or d_idx < dominant_eps:
                dominant = 'left'
            else:
                dominant = 'right'
            dominant_assistive_arm = dominant
            print(f'd_idx {d_idx}, dominant {dominant_assistive_arm}') # for debugging
        elif cfg.method.which_arm == 'assistive':
            # we assume that the data is split in half into right arm being the assistive arm in the first half and left arm being the assistive arm in the other half
            dominant_eps = int(num_demos / 2)
            if num_demos == 1 or d_idx < dominant_eps:
                assistive = 'right'
            else:
                assistive = 'left'
            dominant_assistive_arm = assistive
            print(f'd_idx {d_idx}, assistive {dominant_assistive_arm}') # for debugging

        labels = None
        if cfg.method.keypoint_discovery_no_duplicate:
            episode_keypoints, labels = demo_loading_utils.keypoint_discovery_no_duplicate(demo, which_arm=cfg.method.which_arm, method=keypoint_method, saved_every_last_inserted=cfg.method.saved_every_last_inserted, dominant_assistive_arm=dominant_assistive_arm, use_default_stopped_buffer_timesteps=cfg.method.use_default_stopped_buffer_timesteps, stopped_buffer_timesteps_overwrite=cfg.method.stopped_buffer_timesteps_overwrite)
        else:
            if cfg.method.which_arm in ['both', 'multiarm', 'dominant', 'assistive']:
                # load left and right arm keypoints into 'episode_keypoints'
                episode_keypoints, labels = demo_loading_utils.keypoint_discovery(demo, which_arm=cfg.method.which_arm, method=keypoint_method, saved_every_last_inserted=cfg.method.saved_every_last_inserted, dominant_assistive_arm=dominant_assistive_arm, use_default_stopped_buffer_timesteps=cfg.method.use_default_stopped_buffer_timesteps, stopped_buffer_timesteps_overwrite=cfg.method.stopped_buffer_timesteps_overwrite)
            else:
                episode_keypoints = demo_loading_utils.keypoint_discovery(demo, which_arm=cfg.method.which_arm, method=keypoint_method, saved_every_last_inserted=cfg.method.saved_every_last_inserted)

        if rank == 0:
            logging.info(f"Loading Demo({d_idx}) - found {len(episode_keypoints)} keypoints - {task}")

        for i in range(len(demo) - 1):
            if not demo_augmentation and i > 0:
                break
            if i % demo_augmentation_every_n != 0:
                continue

            obs = demo[i]
            desc = descs[0]


            # if our starting point is past one of the keypoints, then remove it
            while len(episode_keypoints) > 0 and i >= episode_keypoints[0]:
                episode_keypoints = episode_keypoints[1:]
            if len(episode_keypoints) == 0:
                break
            _add_keypoints_to_replay(
                cfg, task, task_idx, replay, obs, demo, episode_keypoints, cameras,
                rlbench_scene_bounds, voxel_sizes, bounds_offset,
                rotation_resolution, crop_augmentation, description=desc,
                clip_model=clip_model, device=device, labels=labels, dominant_assistive_arm=dominant_assistive_arm)
    logging.debug('Replay %s filled with demos.' % task)


def fill_multi_task_replay(cfg: DictConfig,
                           obs_config: ObservationConfig2Robots,
                           rank: int,
                           replay: ReplayBuffer,
                           tasks: List[str],
                           num_demos: int,
                           demo_augmentation: bool,
                           demo_augmentation_every_n: int,
                           cameras: List[str],
                           rlbench_scene_bounds: List[float],
                           voxel_sizes: List[int],
                           bounds_offset: List[float],
                           rotation_resolution: int,
                           crop_augmentation: bool,
                           clip_model = None,
                           keypoint_method = 'heuristic'):
    manager = Manager()
    store = manager.dict()

    # create a MP dict for storing indicies
    # TODO(mohit): this shouldn't be initialized here
    del replay._task_idxs
    task_idxs = manager.dict()
    replay._task_idxs = task_idxs
    replay._create_storage(store)
    replay.add_count = Value('i', 0)

    # fill replay buffer in parallel across tasks
    max_parallel_processes = cfg.replay.max_parallel_processes
    processes = []
    n = np.arange(len(tasks))
    split_n = utils.split_list(n, max_parallel_processes)
    for split in split_n:
        for e_idx, task_idx in enumerate(split):
            task = tasks[int(task_idx)]
            if cfg.ddp.cpu:
                model_device = torch.device('cpu')
            else:
                model_device = torch.device('cuda:%s' % (e_idx % torch.cuda.device_count())
                                            if torch.cuda.is_available() else 'cpu')
            p = Process(target=fill_replay, args=(cfg,
                                                  obs_config,
                                                  rank,
                                                  replay,
                                                  task,
                                                  int(task_idx),
                                                  num_demos,
                                                  demo_augmentation,
                                                  demo_augmentation_every_n,
                                                  cameras,
                                                  rlbench_scene_bounds,
                                                  voxel_sizes,
                                                  bounds_offset,
                                                  rotation_resolution,
                                                  crop_augmentation,
                                                  clip_model,
                                                  model_device,
                                                  keypoint_method))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


def create_agent(cfg: DictConfig):
    LATENT_SIZE = 64
    depth_0bounds = cfg.rlbench.scene_bounds
    cam_resolution = cfg.rlbench.camera_resolution

    num_rotation_classes = int(360. // cfg.method.rotation_resolution)
    qattention_agents = []
    for depth, vox_size in enumerate(cfg.method.voxel_sizes):
        last = depth == len(cfg.method.voxel_sizes) - 1

        if cfg.method.variant == 'one_policy_more_heads':
            perceiver_encoder = PerceiverVoxelLang2RobotsEncoder(
                depth=cfg.method.transformer_depth,
                iterations=cfg.method.transformer_iterations,
                voxel_size=vox_size,
                initial_dim = 3 + 3 + 1 + 3,
                low_dim_size=4,
                layer=depth,
                num_rotation_classes=num_rotation_classes if last else 0,
                num_grip_classes=2 if last else 0,
                num_collision_classes=2 if last else 0,
                input_axis=3,
                num_latents = cfg.method.num_latents,
                latent_dim = cfg.method.latent_dim,
                cross_heads = cfg.method.cross_heads,
                latent_heads = cfg.method.latent_heads,
                cross_dim_head = cfg.method.cross_dim_head,
                latent_dim_head = cfg.method.latent_dim_head,
                weight_tie_layers = False,
                activation = cfg.method.activation,
                pos_encoding_with_lang=cfg.method.pos_encoding_with_lang,
                input_dropout=cfg.method.input_dropout,
                attn_dropout=cfg.method.attn_dropout,
                decoder_dropout=cfg.method.decoder_dropout,
                lang_fusion_type=cfg.method.lang_fusion_type,
                voxel_patch_size=cfg.method.voxel_patch_size,
                voxel_patch_stride=cfg.method.voxel_patch_stride,
                no_skip_connection=cfg.method.no_skip_connection,
                no_perceiver=cfg.method.no_perceiver,
                no_language=cfg.method.no_language,
                final_dim=cfg.method.final_dim,
            )

            qattention_agent = QAttentionPerActBCAgent2Robots(
                layer=depth,
                coordinate_bounds=depth_0bounds,
                perceiver_encoder=perceiver_encoder,
                camera_names=cfg.rlbench.cameras,
                voxel_size=vox_size,
                bounds_offset=cfg.method.bounds_offset[depth - 1] if depth > 0 else None,
                image_crop_size=cfg.method.image_crop_size,
                lr=cfg.method.lr,
                training_iterations=cfg.framework.training_iterations,
                lr_scheduler=cfg.method.lr_scheduler,
                num_warmup_steps=cfg.method.num_warmup_steps,
                trans_loss_weight=cfg.method.trans_loss_weight,
                rot_loss_weight=cfg.method.rot_loss_weight,
                grip_loss_weight=cfg.method.grip_loss_weight,
                collision_loss_weight=cfg.method.collision_loss_weight,
                include_low_dim_state=True,
                image_resolution=cam_resolution,
                batch_size=cfg.replay.batch_size,
                voxel_feature_size=3,
                lambda_weight_l2=cfg.method.lambda_weight_l2,
                num_rotation_classes=num_rotation_classes,
                rotation_resolution=cfg.method.rotation_resolution,
                transform_augmentation=cfg.method.transform_augmentation.apply_se3,
                transform_augmentation_xyz=cfg.method.transform_augmentation.aug_xyz,
                transform_augmentation_rpy=cfg.method.transform_augmentation.aug_rpy,
                transform_augmentation_rot_resolution=cfg.method.transform_augmentation.aug_rot_resolution,
                optimizer_type=cfg.method.optimizer,
                num_devices=cfg.ddp.num_devices,
                wandb_run=cfg.framework.wandb_logging,
            )
        else:
            if cfg.method.which_arm == 'dominant' or cfg.method.which_arm == 'assistive':
                low_dim_size = LOW_DIM_DOMINANT_ASSISTIVE_SIZE
                if cfg.method.arm_id_to_proprio:
                    low_dim_size += 1
            else:
                low_dim_size = LOW_DIM_SIZE
            perceiver_encoder = PerceiverVoxelLangEncoder(
                depth=cfg.method.transformer_depth,
                iterations=cfg.method.transformer_iterations,
                voxel_size=vox_size,
                initial_dim = 3 + 3 + 1 + 3,
                low_dim_size=low_dim_size,
                layer=depth,
                num_rotation_classes=num_rotation_classes if last else 0,
                num_grip_classes=2 if last else 0,
                num_collision_classes=2 if last else 0,
                input_axis=3,
                num_latents = cfg.method.num_latents,
                latent_dim = cfg.method.latent_dim,
                cross_heads = cfg.method.cross_heads,
                latent_heads = cfg.method.latent_heads,
                cross_dim_head = cfg.method.cross_dim_head,
                latent_dim_head = cfg.method.latent_dim_head,
                weight_tie_layers = False,
                activation = cfg.method.activation,
                pos_encoding_with_lang=cfg.method.pos_encoding_with_lang,
                input_dropout=cfg.method.input_dropout,
                attn_dropout=cfg.method.attn_dropout,
                decoder_dropout=cfg.method.decoder_dropout,
                lang_fusion_type=cfg.method.lang_fusion_type,
                voxel_patch_size=cfg.method.voxel_patch_size,
                voxel_patch_stride=cfg.method.voxel_patch_stride,
                no_skip_connection=cfg.method.no_skip_connection,
                no_perceiver=cfg.method.no_perceiver,
                no_language=cfg.method.no_language,
                final_dim=cfg.method.final_dim,
                arm_pred_loss=cfg.method.arm_pred_loss,
            )

            qattention_agent = QAttentionPerActBCAgent(
                layer=depth,
                coordinate_bounds=depth_0bounds,
                perceiver_encoder=perceiver_encoder,
                camera_names=cfg.rlbench.cameras,
                voxel_size=vox_size,
                bounds_offset=cfg.method.bounds_offset[depth - 1] if depth > 0 else None,
                image_crop_size=cfg.method.image_crop_size,
                lr=cfg.method.lr,
                training_iterations=cfg.framework.training_iterations,
                lr_scheduler=cfg.method.lr_scheduler,
                num_warmup_steps=cfg.method.num_warmup_steps,
                trans_loss_weight=cfg.method.trans_loss_weight,
                rot_loss_weight=cfg.method.rot_loss_weight,
                grip_loss_weight=cfg.method.grip_loss_weight,
                collision_loss_weight=cfg.method.collision_loss_weight,
                include_low_dim_state=True,
                image_resolution=cam_resolution,
                batch_size=cfg.replay.batch_size,
                voxel_feature_size=3,
                lambda_weight_l2=cfg.method.lambda_weight_l2,
                num_rotation_classes=num_rotation_classes,
                rotation_resolution=cfg.method.rotation_resolution,
                transform_augmentation=cfg.method.transform_augmentation.apply_se3,
                transform_augmentation_xyz=cfg.method.transform_augmentation.aug_xyz,
                transform_augmentation_rpy=cfg.method.transform_augmentation.aug_rpy,
                transform_augmentation_rot_resolution=cfg.method.transform_augmentation.aug_rot_resolution,
                optimizer_type=cfg.method.optimizer,
                num_devices=cfg.ddp.num_devices,
                crop_target_obj_voxel=cfg.method.crop_target_obj_voxel,
                wandb_run=cfg.framework.wandb_logging,
                arm_pred_loss=cfg.method.arm_pred_loss,
                randomizations_crop_point=cfg.method.randomizations_crop_point,
            )

        qattention_agents.append(qattention_agent)

    if cfg.method.variant == 'one_policy_more_heads':
        rotation_agent = QAttentionStackAgent2Robots(
            qattention_agents=qattention_agents,
            rotation_resolution=cfg.method.rotation_resolution,
            camera_names=cfg.rlbench.cameras,
        )
    else:
        rotation_agent = QAttentionStackAgent(
            qattention_agents=qattention_agents,
            rotation_resolution=cfg.method.rotation_resolution,
            camera_names=cfg.rlbench.cameras,
        )
    preprocess_agent = PreprocessAgent(
        pose_agent=rotation_agent
    )
    return preprocess_agent
