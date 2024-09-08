import copy
import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from pytorch3d import transforms as torch3d_tf
from yarr.agents.agent import Agent, ActResult, ScalarSummary, \
    HistogramSummary, ImageSummary, Summary

from helpers import utils
from helpers.utils import visualise_voxel, visualise_gt_voxel, stack_on_channel, visualise_voxel_2robots
from voxel.voxel_grid import VoxelGrid
from voxel.augmentation import apply_se3_augmentation, apply_se3_augmentation_2Robots
from einops import rearrange
from helpers.clip.core.clip import build_model, load_clip

import transformers
from helpers.optim.lamb import Lamb

from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt

NAME = 'QAttentionAgent'


class QFunction(nn.Module):

    def __init__(self,
                 perceiver_encoder: nn.Module,
                 voxelizer: VoxelGrid,
                 bounds_offset: float,
                 rotation_resolution: float,
                 device,
                 training,
                 arm_pred_loss):
        super(QFunction, self).__init__()
        self._rotation_resolution = rotation_resolution
        self._voxelizer = voxelizer
        self._bounds_offset = bounds_offset
        self._qnet = perceiver_encoder.to(device)
        self._arm_pred_loss = arm_pred_loss
        self._is_training = training

        # distributed training
        if training:
            if device == 'cpu':
                self._qnet = DDP(self._qnet)
            else:
                self._qnet = DDP(self._qnet, device_ids=[device])
            

    def _argmax_3d(self, tensor_orig):
        b, c, d, h, w = tensor_orig.shape  # c will be one
        idxs = tensor_orig.view(b, c, -1).argmax(-1)
        # indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1) # original code
        # this fixes a warning which only comes up when a new PyTorch version is used
        indices = torch.cat([(torch.div((torch.div(idxs, h, rounding_mode='trunc')), d, rounding_mode='trunc')), (torch.div(idxs, h, rounding_mode='trunc')) % w, idxs % w], 1)
        return indices

    def choose_highest_action(self, q_trans, q_rot_grip, q_collision):
        coords = self._argmax_3d(q_trans)
        rot_and_grip_indicies = None
        ignore_collision = None
        if q_rot_grip is not None:
            q_rot = torch.stack(torch.split(
                q_rot_grip[:, :-2],
                int(360 // self._rotation_resolution),
                dim=1), dim=1)
            rot_and_grip_indicies = torch.cat(
                [q_rot[:, 0:1].argmax(-1),
                 q_rot[:, 1:2].argmax(-1),
                 q_rot[:, 2:3].argmax(-1),
                 q_rot_grip[:, -2:].argmax(-1, keepdim=True)], -1)
            ignore_collision = q_collision[:, -2:].argmax(-1, keepdim=True)
        return coords, rot_and_grip_indicies, ignore_collision

    def forward(self, rgb_pcd, proprio, pcd, lang_goal_emb, lang_token_embs,
                bounds=None, prev_bounds=None, prev_layer_voxel_grid=None):
        # rgb_pcd will be list of list (list of [rgb, pcd])
        b = rgb_pcd[0][0].shape[0]
        pcd_flat = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, 3) for p in pcd], 1)

        # flatten RGBs and Pointclouds
        rgb = [rp[0] for rp in rgb_pcd]
        feat_size = rgb[0].shape[1]
        flat_imag_features = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, feat_size) for p in rgb], 1)

        # construct voxel grid
        voxel_grid = self._voxelizer.coords_to_bounding_voxel_grid(
            pcd_flat, coord_features=flat_imag_features, coord_bounds=bounds)

        # swap to channels fist
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach()

        # batch bounds if necessary
        if bounds.shape[0] != b:
            bounds = bounds.repeat(b, 1)

        # forward pass
        if self._arm_pred_loss:
            q_trans, \
            q_rot_and_grip,\
            q_ignore_collisions, \
            arm_out = self._qnet(voxel_grid,
                                proprio,
                                lang_goal_emb,
                                lang_token_embs,
                                prev_layer_voxel_grid,
                                bounds,
                                prev_bounds)

            if self._is_training:
                return q_trans, q_rot_and_grip, q_ignore_collisions, voxel_grid, arm_out
            else:
                # don't need arm_out during evaluation
                return q_trans, q_rot_and_grip, q_ignore_collisions, voxel_grid
        else:
            q_trans, \
            q_rot_and_grip,\
            q_ignore_collisions = self._qnet(voxel_grid,
                                            proprio,
                                            lang_goal_emb,
                                            lang_token_embs,
                                            prev_layer_voxel_grid,
                                            bounds,
                                            prev_bounds)

            return q_trans, q_rot_and_grip, q_ignore_collisions, voxel_grid


class QAttentionPerActBCAgent(Agent):

    def __init__(self,
                 layer: int,
                 coordinate_bounds: list,
                 perceiver_encoder: nn.Module,
                 camera_names: list,
                 batch_size: int,
                 voxel_size: int,
                 bounds_offset: float,
                 voxel_feature_size: int,
                 image_crop_size: int,
                 num_rotation_classes: int,
                 rotation_resolution: float,
                 lr: float = 0.0001,
                 lr_scheduler: bool = False,
                 training_iterations: int = 100000,
                 num_warmup_steps: int = 20000,
                 trans_loss_weight: float = 1.0,
                 rot_loss_weight: float = 1.0,
                 grip_loss_weight: float = 1.0,
                 collision_loss_weight: float = 1.0,
                 include_low_dim_state: bool = False,
                 image_resolution: list = None,
                 lambda_weight_l2: float = 0.0,
                 transform_augmentation: bool = True,
                 transform_augmentation_xyz: list = [0.0, 0.0, 0.0],
                 transform_augmentation_rpy: list = [0.0, 0.0, 180.0],
                 transform_augmentation_rot_resolution: int = 5,
                 optimizer_type: str = 'adam',
                 num_devices: int = 1,
                 crop_target_obj_voxel: bool = False,
                 wandb_run = None,
                 arm_pred_loss: bool =False,
                 arm_loss_weight: float = 1.0,
                 randomizations_crop_point: bool = False,
                 ):
        self._layer = layer
        if type(coordinate_bounds[0]) is float:
            # single task
            self._coordinate_bounds = coordinate_bounds
        else:
            # multi task
            # NOTE: it doesn't matter which coordinate bound we choose because we're
            # overwriting it in the update and act functions
            self._coordinate_bounds = coordinate_bounds[0]
        self._perceiver_encoder = perceiver_encoder
        self._voxel_feature_size = voxel_feature_size
        self._bounds_offset = bounds_offset
        self._image_crop_size = image_crop_size
        self._lr = lr
        self._lr_scheduler = lr_scheduler
        self._training_iterations = training_iterations
        self._num_warmup_steps = num_warmup_steps
        self._trans_loss_weight = trans_loss_weight
        self._rot_loss_weight = rot_loss_weight
        self._grip_loss_weight = grip_loss_weight
        self._collision_loss_weight = collision_loss_weight
        self._include_low_dim_state = include_low_dim_state
        self._image_resolution = image_resolution or [128, 128]
        self._voxel_size = voxel_size
        self._camera_names = camera_names
        self._num_cameras = len(camera_names)
        self._batch_size = batch_size
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._transform_augmentation_xyz = torch.from_numpy(np.array(transform_augmentation_xyz))
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = transform_augmentation_rot_resolution
        self._optimizer_type = optimizer_type
        self._num_devices = num_devices
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = rotation_resolution
        self._crop_target_obj_voxel = crop_target_obj_voxel
        self._wandb_run = wandb_run
        self._arm_pred_loss = arm_pred_loss
        self._arm_loss_weight = arm_loss_weight
        self._randomizations_crop_point = randomizations_crop_point

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self._name = NAME + '_layer' + str(self._layer)

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device

        if device is None:
            device = torch.device('cpu')

        self._voxelizer = VoxelGrid(
            coord_bounds=self._coordinate_bounds,
            voxel_size=self._voxel_size,
            device=device,
            batch_size=self._batch_size if training else 1,
            feature_size=self._voxel_feature_size,
            max_num_coords=np.prod(self._image_resolution) * self._num_cameras,
        )

        self._q = QFunction(self._perceiver_encoder,
                            self._voxelizer,
                            self._bounds_offset,
                            self._rotation_resolution,
                            device,
                            training,
                            self._arm_pred_loss).to(device).train(training)

        grid_for_crop = torch.arange(0,
                                     self._image_crop_size,
                                     device=device).unsqueeze(0).repeat(self._image_crop_size, 1).unsqueeze(-1)
        self._grid_for_crop = torch.cat([grid_for_crop.transpose(1, 0),
                                         grid_for_crop], dim=2).unsqueeze(0)

        self._coordinate_bounds = torch.tensor(self._coordinate_bounds,
                                               device=device).unsqueeze(0)

        if self._training:
            # optimizer
            if self._optimizer_type == 'lamb':
                self._optimizer = Lamb(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                    adam=False,
                )
            elif self._optimizer_type == 'adam':
                self._optimizer = torch.optim.Adam(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                )
            else:
                raise Exception('Unknown optimizer type')

            # learning rate scheduler
            if self._lr_scheduler:
                self._scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                    self._optimizer,
                    num_warmup_steps=self._num_warmup_steps,
                    num_training_steps=self._training_iterations,
                    num_cycles=self._training_iterations // 10000,
                )

            # one-hot zero tensors
            self._action_trans_one_hot_zeros = torch.zeros((self._batch_size,
                                                            1,
                                                            self._voxel_size,
                                                            self._voxel_size,
                                                            self._voxel_size),
                                                            dtype=int,
                                                            device=device)
            self._action_rot_x_one_hot_zeros = torch.zeros((self._batch_size,
                                                            self._num_rotation_classes),
                                                            dtype=int,
                                                            device=device)
            self._action_rot_y_one_hot_zeros = torch.zeros((self._batch_size,
                                                            self._num_rotation_classes),
                                                            dtype=int,
                                                            device=device)
            self._action_rot_z_one_hot_zeros = torch.zeros((self._batch_size,
                                                            self._num_rotation_classes),
                                                            dtype=int,
                                                            device=device)
            self._action_grip_one_hot_zeros = torch.zeros((self._batch_size,
                                                           2),
                                                           dtype=int,
                                                           device=device)
            self._action_ignore_collisions_one_hot_zeros = torch.zeros((self._batch_size,
                                                                        2),
                                                                        dtype=int,
                                                                        device=device)
            if self._arm_pred_loss:
                self._action_arm_zeros = torch.zeros((self._batch_size,
                                                           2),
                                                           dtype=int,
                                                           device=device)

            # print total params
            logging.info('# Q Params: %d' % sum(
                p.numel() for name, p in self._q.named_parameters() \
                if p.requires_grad and 'clip' not in name))
        else:
            for param in self._q.parameters():
                param.requires_grad = False

            # load CLIP for encoding language goals during evaluation
            model, _ = load_clip("RN50", jit=False)
            self._clip_rn50 = build_model(model.state_dict())
            self._clip_rn50 = self._clip_rn50.float().to(device)
            self._clip_rn50.eval()
            del model

            self._voxelizer.to(device)
            self._q.to(device)

    def _extract_crop(self, pixel_action, observation):
        # Pixel action will now be (B, 2)
        # observation = stack_on_channel(observation)
        h = observation.shape[-1]
        top_left_corner = torch.clamp(
            pixel_action - self._image_crop_size // 2, 0,
            h - self._image_crop_size)
        grid = self._grid_for_crop + top_left_corner.unsqueeze(1)
        grid = ((grid / float(h)) * 2.0) - 1.0  # between -1 and 1
        # Used for cropping the images across a batch
        # swap fro y x, to x, y
        grid = torch.cat((grid[:, :, :, 1:2], grid[:, :, :, 0:1]), dim=-1)
        crop = F.grid_sample(observation, grid, mode='nearest',
                             align_corners=True)
        return crop

    def _preprocess_inputs(self, replay_sample):
        obs = []
        pcds = []
        self._crop_summary = []
        for n in self._camera_names:
            rgb = replay_sample['%s_rgb' % n]
            pcd = replay_sample['%s_point_cloud' % n]

            obs.append([rgb, pcd])
            pcds.append(pcd)
        return obs, pcds

    def _act_preprocess_inputs(self, observation):
        obs, pcds = [], []
        for n in self._camera_names:
            rgb = observation['%s_rgb' % n]
            pcd = observation['%s_point_cloud' % n]

            obs.append([rgb, pcd])
            pcds.append(pcd)
        return obs, pcds

    def _get_value_from_voxel_index(self, q, voxel_idx):
        b, c, d, h, w = q.shape
        q_trans_flat = q.view(b, c, d * h * w)
        flat_indicies = (voxel_idx[:, 0] * d * h + voxel_idx[:, 1] * h + voxel_idx[:, 2])[:, None].int()
        highest_idxs = flat_indicies.unsqueeze(-1).repeat(1, c, 1)
        chosen_voxel_values = q_trans_flat.gather(2, highest_idxs)[..., 0]  # (B, trans + rot + grip)
        return chosen_voxel_values

    def _get_value_from_rot_and_grip(self, rot_grip_q, rot_and_grip_idx):
        q_rot = torch.stack(torch.split(
            rot_grip_q[:, :-2], int(360 // self._rotation_resolution),
            dim=1), dim=1)  # B, 3, 72
        q_grip = rot_grip_q[:, -2:]
        rot_and_grip_values = torch.cat(
            [q_rot[:, 0].gather(1, rot_and_grip_idx[:, 0:1]),
             q_rot[:, 1].gather(1, rot_and_grip_idx[:, 1:2]),
             q_rot[:, 2].gather(1, rot_and_grip_idx[:, 2:3]),
             q_grip.gather(1, rot_and_grip_idx[:, 3:4])], -1)
        return rot_and_grip_values

    def _celoss(self, pred, labels):
        return self._cross_entropy_loss(pred, labels.argmax(-1))

    def _softmax_q_trans(self, q):
        q_shape = q.shape
        return F.softmax(q.reshape(q_shape[0], -1), dim=1).reshape(q_shape)

    def _softmax_q_rot_grip(self, q_rot_grip):
        q_rot_x_flat = q_rot_grip[:, 0*self._num_rotation_classes: 1*self._num_rotation_classes]
        q_rot_y_flat = q_rot_grip[:, 1*self._num_rotation_classes: 2*self._num_rotation_classes]
        q_rot_z_flat = q_rot_grip[:, 2*self._num_rotation_classes: 3*self._num_rotation_classes]
        q_grip_flat  = q_rot_grip[:, 3*self._num_rotation_classes:]

        q_rot_x_flat_softmax = F.softmax(q_rot_x_flat, dim=1)
        q_rot_y_flat_softmax = F.softmax(q_rot_y_flat, dim=1)
        q_rot_z_flat_softmax = F.softmax(q_rot_z_flat, dim=1)
        q_grip_flat_softmax = F.softmax(q_grip_flat, dim=1)

        return torch.cat([q_rot_x_flat_softmax,
                          q_rot_y_flat_softmax,
                          q_rot_z_flat_softmax,
                          q_grip_flat_softmax], dim=1)

    def _softmax_ignore_collision(self, q_collision):
        q_collision_softmax = F.softmax(q_collision, dim=1)
        return q_collision_softmax

    def update(self, step: int, replay_sample: dict) -> dict:
        action_trans = replay_sample['trans_action_indicies'][:, self._layer * 3:self._layer * 3 + 3].int()
        action_rot_grip = replay_sample['rot_grip_action_indicies'].int()
        action_gripper_pose = replay_sample['gripper_pose']
        action_ignore_collisions = replay_sample['ignore_collisions'].int()
        action_label = replay_sample.get('label', None)
        lang_goal_emb = replay_sample['lang_goal_emb'].float()
        lang_token_embs = replay_sample['lang_token_embs'].float()
        prev_layer_voxel_grid = replay_sample.get('prev_layer_voxel_grid', None)
        prev_layer_bounds = replay_sample.get('prev_layer_bounds', None)
        device = self._device

        # NOTE: it's okay to use bounds from replay_sample to overwrite existing bounds in VoxelGrid
        if self._crop_target_obj_voxel:
            self._coordinate_bounds = replay_sample['target_object_scene_bounds']
            if self._randomizations_crop_point:
                # v3
                # rand_x = np.random.uniform(low=-0.1, high=0.1)
                # rand_y = np.random.uniform(low=-0.1, high=0.1)
                # rand_z = np.random.uniform(low=-0.1, high=0.1)

                # v4
                rand_x = np.random.uniform(low=-0.05, high=0.05)
                rand_y = np.random.uniform(low=-0.05, high=0.05)
                rand_z = np.random.uniform(low=-0.05, high=0.05)

                self._coordinate_bounds[:, 0] += rand_x
                self._coordinate_bounds[:, 3] += rand_x
                self._coordinate_bounds[:, 1] += rand_y
                self._coordinate_bounds[:, 4] += rand_y
                self._coordinate_bounds[:, 2] += rand_z
                self._coordinate_bounds[:, 5] += rand_z
                # print('Coordinate bounds modified!!! ', self._coordinate_bounds) # for debugging...


        bounds = self._coordinate_bounds.to(device)
        if self._layer > 0:
            cp = replay_sample['attention_coordinate_layer_%d' % (self._layer - 1)]
            bounds = torch.cat([cp - self._bounds_offset, cp + self._bounds_offset], dim=1)

        proprio = None
        if self._include_low_dim_state:
            proprio = replay_sample['low_dim_state']

        # NOTE: right now, we're feeding wrist and wrist2 images
        obs, pcd = self._preprocess_inputs(replay_sample)

        # batch size
        bs = pcd[0].shape[0]

        # SE(3) augmentation of point clouds and actions
        if self._transform_augmentation:
            action_trans, \
            action_rot_grip, \
            pcd = apply_se3_augmentation(pcd,
                                         action_gripper_pose,
                                         action_trans,
                                         action_rot_grip,
                                         bounds,
                                         self._layer,
                                         self._transform_augmentation_xyz,
                                         self._transform_augmentation_rpy,
                                         self._transform_augmentation_rot_resolution,
                                         self._voxel_size,
                                         self._rotation_resolution,
                                         self._device)

        # forward pass
        if self._arm_pred_loss:
            q_trans, q_rot_grip, \
            q_collision, \
            voxel_grid, \
            arm_out = self._q(obs,
                            proprio,
                            pcd,
                            lang_goal_emb,
                            lang_token_embs,
                            bounds,
                            prev_layer_bounds,
                            prev_layer_voxel_grid)
        else:
            q_trans, q_rot_grip, \
            q_collision, \
            voxel_grid = self._q(obs,
                                proprio,
                                pcd,
                                lang_goal_emb,
                                lang_token_embs,
                                bounds,
                                prev_layer_bounds,
                                prev_layer_voxel_grid)

        # argmax to choose best action
        coords, \
        rot_and_grip_indicies, \
        ignore_collision_indicies = self._q.choose_highest_action(q_trans, q_rot_grip, q_collision)

        q_trans_loss, q_rot_loss, q_grip_loss, q_collision_loss, q_arm_loss = 0., 0., 0., 0., 0.

        # translation one-hot
        action_trans_one_hot = self._action_trans_one_hot_zeros.clone()
        for b in range(bs):
            gt_coord = action_trans[b, :].int()
            action_trans_one_hot[b, :, gt_coord[0], gt_coord[1], gt_coord[2]] = 1

        # translation loss
        q_trans_flat = q_trans.view(bs, -1)
        action_trans_one_hot_flat = action_trans_one_hot.view(bs, -1)
        q_trans_loss = self._celoss(q_trans_flat, action_trans_one_hot_flat)

        with_rot_and_grip = rot_and_grip_indicies is not None
        if with_rot_and_grip:
            # rotation, gripper, and collision one-hots
            action_rot_x_one_hot = self._action_rot_x_one_hot_zeros.clone()
            action_rot_y_one_hot = self._action_rot_y_one_hot_zeros.clone()
            action_rot_z_one_hot = self._action_rot_z_one_hot_zeros.clone()
            action_grip_one_hot = self._action_grip_one_hot_zeros.clone()
            action_ignore_collisions_one_hot = self._action_ignore_collisions_one_hot_zeros.clone()

            for b in range(bs):
                gt_rot_grip = action_rot_grip[b, :].int()
                action_rot_x_one_hot[b, gt_rot_grip[0]] = 1
                action_rot_y_one_hot[b, gt_rot_grip[1]] = 1
                action_rot_z_one_hot[b, gt_rot_grip[2]] = 1
                action_grip_one_hot[b, gt_rot_grip[3]] = 1

                gt_ignore_collisions = action_ignore_collisions[b, :].int()
                action_ignore_collisions_one_hot[b, gt_ignore_collisions[0]] = 1

            # flatten predictions
            q_rot_x_flat = q_rot_grip[:, 0*self._num_rotation_classes:1*self._num_rotation_classes]
            q_rot_y_flat = q_rot_grip[:, 1*self._num_rotation_classes:2*self._num_rotation_classes]
            q_rot_z_flat = q_rot_grip[:, 2*self._num_rotation_classes:3*self._num_rotation_classes]
            q_grip_flat =  q_rot_grip[:, 3*self._num_rotation_classes:]
            q_ignore_collisions_flat = q_collision

            # rotation loss
            q_rot_loss += self._celoss(q_rot_x_flat, action_rot_x_one_hot)
            q_rot_loss += self._celoss(q_rot_y_flat, action_rot_y_one_hot)
            q_rot_loss += self._celoss(q_rot_z_flat, action_rot_z_one_hot)

            # gripper loss
            q_grip_loss += self._celoss(q_grip_flat, action_grip_one_hot)

            # collision loss
            q_collision_loss += self._celoss(q_ignore_collisions_flat, action_ignore_collisions_one_hot)

        if self._arm_pred_loss:
            action_arm_one_hot = self._action_arm_zeros.clone()
            for b in range(bs):
                gt_arm = action_label[b, :].long()
                action_arm_one_hot[b, gt_arm] = 1
            q_arm_loss += self._celoss(arm_out, action_arm_one_hot)

        combined_losses = (q_trans_loss * self._trans_loss_weight) + \
                          (q_rot_loss * self._rot_loss_weight) + \
                          (q_grip_loss * self._grip_loss_weight) + \
                          (q_collision_loss * self._collision_loss_weight) + \
                          (q_arm_loss * self._arm_loss_weight)

        total_loss = combined_losses.mean()

        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        self._summaries = {
            'losses/total_loss': total_loss,
            'losses/trans_loss': q_trans_loss.mean(),
            'losses/rot_loss': q_rot_loss.mean() if with_rot_and_grip else 0.,
            'losses/grip_loss': q_grip_loss.mean() if with_rot_and_grip else 0.,
            'losses/collision_loss': q_collision_loss.mean() if with_rot_and_grip else 0.,
        }
        if self._arm_pred_loss:
            self._summaries['losses/arm_loss'] = q_arm_loss.mean()
            # print(f'q_arm_loss: {q_arm_loss}, action_label: {action_label}') # for debugging...

        if self._lr_scheduler:
            self._scheduler.step()
            self._summaries['learning_rate'] = self._scheduler.get_last_lr()[0]

        self._vis_voxel_grid = voxel_grid[0]
        self._vis_translation_qvalue = self._softmax_q_trans(q_trans[0])
        self._vis_max_coordinate = coords[0]
        self._vis_gt_coordinate = action_trans[0]

        # Note: PerAct doesn't use multi-layer voxel grids like C2FARM
        # stack prev_layer_voxel_grid(s) from previous layers into a list
        if prev_layer_voxel_grid is None:
            prev_layer_voxel_grid = [voxel_grid]
        else:
            prev_layer_voxel_grid = prev_layer_voxel_grid + [voxel_grid]

        # stack prev_layer_bound(s) from previous layers into a list
        if prev_layer_bounds is None:
            prev_layer_bounds = [self._coordinate_bounds.repeat(bs, 1)]
        else:
            prev_layer_bounds = prev_layer_bounds + [bounds]

        # debug by visualizing front_rgb...
        # from PIL import Image
        # front_rgb_np = np.clip((((replay_sample['front_rgb'] + 1.0) * 255.0) / 2.0).cpu().detach().numpy().astype(np.uint8), 0, 255)        
        # front_rgb_np = Image.fromarray(np.transpose(front_rgb_np[0], (1, 2, 0)))
        # front_rgb_np.save('front_rgb_np.jpg')

        # debug: make sure voxel grid looks alright
        # visualise_voxel(
        #     self._vis_voxel_grid.detach().cpu().numpy(),
        #     self._vis_translation_qvalue.detach().cpu().numpy(),
        #     self._vis_max_coordinate.detach().cpu().numpy(),
        #     self._vis_gt_coordinate.detach().cpu().numpy(),
        #     show=True)

        # debug: visualize just ground truth coordinate
        # visualise_gt_voxel(
        #     self._vis_voxel_grid.detach().cpu().numpy(),
        #     self._vis_gt_coordinate.detach().cpu().numpy(),
        #     show=True)

        return {
            'total_loss': total_loss,
            'prev_layer_voxel_grid': prev_layer_voxel_grid,
            'prev_layer_bounds': prev_layer_bounds,
        }

    def act(self, step: int, observation: dict,
            deterministic=False, which_arm=None, new_scene_bounds=None, dominant_assitive_policy=False, ep_number=0, is_real_robot=False) -> ActResult:
        deterministic = True
        if new_scene_bounds is not None:
            self._coordinate_bounds = torch.tensor(new_scene_bounds, device=self._device).unsqueeze(0)

        bounds = self._coordinate_bounds
        prev_layer_voxel_grid = observation.get('prev_layer_voxel_grid', None)
        prev_layer_bounds = observation.get('prev_layer_bounds', None)

        if which_arm in ['multiarm_left', 'multiarm_right']:
            if which_arm == 'multiarm_left':
                lang_goal_tokens = observation.get('lang_goal_tokens_left', None).long()
            else:
                # which_arm == multiarm_right
                lang_goal_tokens = observation.get('lang_goal_tokens_right', None).long()
        else:
            lang_goal_tokens = observation.get('lang_goal_tokens', None).long()

        # extract CLIP language embs
        with torch.no_grad():
            lang_goal_tokens = lang_goal_tokens.to(device=self._device)
            lang_goal_emb, lang_token_embs = self._clip_rn50.encode_text_with_embeddings(lang_goal_tokens[0])

        # voxelization resolution
        res = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        max_rot_index = int(360 // self._rotation_resolution)
        proprio = None

        if self._include_low_dim_state:
            if dominant_assitive_policy:
                # left gripper open, left joint positions, right gripper open, right joint positions, timestep
                proprio = torch.cat((observation['low_dim_state_left_arm'][:, :, :3], observation['low_dim_state_right_arm']), 2)
            elif which_arm == 'right' or which_arm == 'multiarm_right':
                proprio = observation['low_dim_state_right_arm']
            elif which_arm == 'left' or which_arm == 'multiarm_left':
                proprio = observation['low_dim_state_left_arm']
            else:
                proprio = observation['low_dim_state']

        # NOTE: right now, we're feeding wrist and wrist2 images
        obs, pcd = self._act_preprocess_inputs(observation)

        # correct batch size and device
        obs = [[o[0][0].to(self._device), o[1][0].to(self._device)] for o in obs]
        proprio = proprio[0].to(self._device)
        pcd = [p[0].to(self._device) for p in pcd]
        lang_goal_emb = lang_goal_emb.to(self._device)
        lang_token_embs = lang_token_embs.to(self._device)
        bounds = torch.as_tensor(bounds, device=self._device)
        prev_layer_voxel_grid = prev_layer_voxel_grid.to(self._device) if prev_layer_voxel_grid is not None else None
        prev_layer_bounds = prev_layer_bounds.to(self._device) if prev_layer_bounds is not None else None

        # inference
        q_trans, \
        q_rot_grip, \
        q_ignore_collisions, \
        vox_grid = self._q(obs,
                           proprio,
                           pcd,
                           lang_goal_emb,
                           lang_token_embs,
                           bounds,
                           prev_layer_bounds,
                           prev_layer_voxel_grid)

        # softmax Q predictions
        q_trans = self._softmax_q_trans(q_trans)
        q_rot_grip =  self._softmax_q_rot_grip(q_rot_grip) if q_rot_grip is not None else q_rot_grip
        q_ignore_collisions = self._softmax_ignore_collision(q_ignore_collisions) \
            if q_ignore_collisions is not None else q_ignore_collisions

        # argmax Q predictions
        coords, \
        rot_and_grip_indicies, \
        ignore_collisions = self._q.choose_highest_action(q_trans, q_rot_grip, q_ignore_collisions)

        rot_grip_action = rot_and_grip_indicies if q_rot_grip is not None else None
        ignore_collisions_action = ignore_collisions.int() if ignore_collisions is not None else None

        coords = coords.int()
        attention_coordinate = bounds[:, :3] + res * coords + res / 2

        # stack prev_layer_voxel_grid(s) into a list
        # NOTE: PerAct doesn't used multi-layer voxel grids like C2FARM
        if prev_layer_voxel_grid is None:
            prev_layer_voxel_grid = [vox_grid]
        else:
            prev_layer_voxel_grid = prev_layer_voxel_grid + [vox_grid]

        if prev_layer_bounds is None:
            prev_layer_bounds = [bounds]
        else:
            prev_layer_bounds = prev_layer_bounds + [bounds]

        observation_elements = {
            'attention_coordinate': attention_coordinate,
            'prev_layer_voxel_grid': prev_layer_voxel_grid,
            'prev_layer_bounds': prev_layer_bounds,
        }
        info = {
            'voxel_grid_depth%d' % self._layer: vox_grid,
            'q_depth%d' % self._layer: q_trans,
            'voxel_idx_depth%d' % self._layer: coords
        }
        self._act_voxel_grid = vox_grid[0]
        self._act_max_coordinate = coords[0]
        self._act_qvalues = q_trans[0].detach()

        print(f'{which_arm} pred coordinates: {coords[0]}') # for debugging

        # debug: make sure voxel grid looks alright
        # visualise_voxel(
        #     vox_grid[0].detach().cpu().numpy(),
        #     q_trans[0].detach().cpu().numpy(),
        #     coords[0].detach().cpu().numpy(),
        #     show=True)

        # for debugging...
        # if step == 0:
        #     voxel_grid_img = visualise_voxel(
        #                         vox_grid[0].detach().cpu().numpy(),
        #                         q_trans[0].detach().cpu().numpy(),
        #                         coords[0].detach().cpu().numpy())
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(voxel_grid_img)
        #     filename = f'{ep_number}_voxel_grid.png'
        #     plt.savefig(filename)
        #     print(f'{filename} saved!')

        # if is_real_robot:
            # coords_target = coords[0].detach().cpu().numpy()
            # # coords_target[1] += 11 # offset for visualization of the left arm
            # # coords_target[1] += 14 # offset for visualization of the right arm
            # breakpoint()
            # voxel_grid_img = visualise_voxel(vox_grid[0].detach().cpu().numpy(), None, coords_target, show=True)
            # plt.figure(figsize=(10, 10))
            # plt.imshow(voxel_grid_img)
            # filename = f'{ep_number}_voxel_grid.png'
            # plt.savefig(filename)
            # print(f'{filename} saved!')

        return ActResult((coords, rot_grip_action, ignore_collisions_action),
                         observation_elements=observation_elements,
                         info=info)

    def update_summaries(self) -> List[Summary]:
        try:
            summaries = [
                ImageSummary('%s/update_qattention' % self._name,
                            transforms.ToTensor()(visualise_voxel(
                                self._vis_voxel_grid.detach().cpu().numpy(),
                                self._vis_translation_qvalue.detach().cpu().numpy(),
                                self._vis_max_coordinate.detach().cpu().numpy(),
                                self._vis_gt_coordinate.detach().cpu().numpy())))
            ]
        except:
            # this exception can happen when the computer does not have a display or the display is not set up properly
            summaries = []

        wandb_dict = {}
        for n, v in self._summaries.items():
            summaries.append(ScalarSummary('%s/%s' % (self._name, n), v))
            if self._wandb_run is not None:
                wandb_dict['%s/%s' % (self._name, n)] = v

        for (name, crop) in (self._crop_summary):
            crops = (torch.cat(torch.split(crop, 3, dim=1), dim=3) + 1.0) / 2.0
            summaries.extend([
                ImageSummary('%s/crops/%s' % (self._name, name), crops)])

        for tag, param in self._q.named_parameters():
            # assert not torch.isnan(param.grad.abs() <= 1.0).all()
            summaries.append(
                HistogramSummary('%s/gradient/%s' % (self._name, tag),
                                 param.grad))
            summaries.append(
                HistogramSummary('%s/weight/%s' % (self._name, tag),
                                 param.data))

        return summaries, wandb_dict

    def act_summaries(self) -> List[Summary]:
        try:
            return [
                ImageSummary('%s/act_Qattention' % self._name,
                            transforms.ToTensor()(visualise_voxel(
                                self._act_voxel_grid.cpu().numpy(),
                                self._act_qvalues.cpu().numpy(),
                                self._act_max_coordinate.cpu().numpy())))]
        except:
            # this exception can happen when VoxPoserOnly is used which doesn't utilize the act function
            return []

    def load_weights(self, savedir: str):
        if self._device == 'cpu':
            device = torch.device('cpu')
        else:
            device = self._device if not self._training else torch.device('cuda:%d' % self._device)
        weight_file = os.path.join(savedir, '%s.pt' % self._name)
        state_dict = torch.load(weight_file, map_location=device)

        # load only keys that are in the current model
        merged_state_dict = self._q.state_dict()
        for k, v in state_dict.items():
            if not self._training:
                k = k.replace('_qnet.module', '_qnet')
            if k in merged_state_dict:
                merged_state_dict[k] = v
            else:
                if '_voxelizer' not in k:
                    logging.warning("key %s not found in checkpoint" % k)
        self._q.load_state_dict(merged_state_dict)
        print("loaded weights from %s" % weight_file)

    def load_weight(self, ckpt_file: str):
        if self._device == 'cpu':
            device = torch.device('cpu')
        else:
            device = self._device if not self._training else torch.device('cuda:%d' % self._device)
        state_dict = torch.load(ckpt_file, map_location=device)

        # load only keys that are in the current model
        merged_state_dict = self._q.state_dict()
        for k, v in state_dict.items():
            if not self._training:
                k = k.replace('_qnet.module', '_qnet')
            if k in merged_state_dict:
                merged_state_dict[k] = v
            else:
                if '_voxelizer' not in k:
                    logging.warning("key %s not found in checkpoint" % k)
        self._q.load_state_dict(merged_state_dict)
        print("loaded weights from %s" % ckpt_file)

    def save_weights(self, savedir: str):
        torch.save(
            self._q.state_dict(), os.path.join(savedir, '%s.pt' % self._name))

class QFunction2Robots(nn.Module):

    def __init__(self,
                 perceiver_encoder: nn.Module,
                 voxelizer: VoxelGrid,
                 bounds_offset: float,
                 rotation_resolution: float,
                 device,
                 training):
        super(QFunction2Robots, self).__init__()
        self._rotation_resolution = rotation_resolution
        self._voxelizer = voxelizer
        self._bounds_offset = bounds_offset
        self._qnet = perceiver_encoder.to(device)

        # distributed training
        if training:
            self._qnet = DDP(self._qnet, device_ids=[device])

    def _argmax_3d(self, tensor_orig):
        b, c, d, h, w = tensor_orig.shape  # c will be one
        idxs = tensor_orig.view(b, c, -1).argmax(-1)
        indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1)
        return indices

    def choose_highest_action(self, q_trans, q_rot_grip, q_collision):
        coords = self._argmax_3d(q_trans)
        rot_and_grip_indicies = None
        ignore_collision = None
        if q_rot_grip is not None:
            q_rot = torch.stack(torch.split(
                q_rot_grip[:, :-2],
                int(360 // self._rotation_resolution),
                dim=1), dim=1)
            rot_and_grip_indicies = torch.cat(
                [q_rot[:, 0:1].argmax(-1),
                 q_rot[:, 1:2].argmax(-1),
                 q_rot[:, 2:3].argmax(-1),
                 q_rot_grip[:, -2:].argmax(-1, keepdim=True)], -1)
            ignore_collision = q_collision[:, -2:].argmax(-1, keepdim=True)
        return coords, rot_and_grip_indicies, ignore_collision

    def forward(self, rgb_pcd, proprio_right, proprio_left, pcd, lang_goal_emb, lang_token_embs,
                bounds=None, prev_bounds=None, prev_layer_voxel_grid=None):
        # rgb_pcd will be list of list (list of [rgb, pcd])
        b = rgb_pcd[0][0].shape[0]
        pcd_flat = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, 3) for p in pcd], 1)

        # flatten RGBs and Pointclouds
        rgb = [rp[0] for rp in rgb_pcd]
        feat_size = rgb[0].shape[1]
        flat_imag_features = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, feat_size) for p in rgb], 1)

        # construct voxel grid
        voxel_grid = self._voxelizer.coords_to_bounding_voxel_grid(
            pcd_flat, coord_features=flat_imag_features, coord_bounds=bounds)

        # swap to channels fist
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach()

        # batch bounds if necessary
        if bounds.shape[0] != b:
            bounds = bounds.repeat(b, 1)

        # forward pass
        q_trans_right, \
        q_rot_and_grip_right,\
        q_ignore_collisions_right, \
        q_trans_left, \
        q_rot_and_grip_left,\
        q_ignore_collisions_left = self._qnet(voxel_grid,
                                         proprio_right,
                                         proprio_left,
                                         lang_goal_emb,
                                         lang_token_embs,
                                         prev_layer_voxel_grid,
                                         bounds,
                                         prev_bounds)

        return q_trans_right, q_rot_and_grip_right, q_ignore_collisions_right, voxel_grid, q_trans_left, q_rot_and_grip_left, q_ignore_collisions_left


class QAttentionPerActBCAgent2Robots(Agent):

    def __init__(self,
                 layer: int,
                 coordinate_bounds: list,
                 perceiver_encoder: nn.Module,
                 camera_names: list,
                 batch_size: int,
                 voxel_size: int,
                 bounds_offset: float,
                 voxel_feature_size: int,
                 image_crop_size: int,
                 num_rotation_classes: int,
                 rotation_resolution: float,
                 lr: float = 0.0001,
                 lr_scheduler: bool = False,
                 training_iterations: int = 100000,
                 num_warmup_steps: int = 20000,
                 trans_loss_weight: float = 1.0,
                 rot_loss_weight: float = 1.0,
                 grip_loss_weight: float = 1.0,
                 collision_loss_weight: float = 1.0,
                 include_low_dim_state: bool = False,
                 image_resolution: list = None,
                 lambda_weight_l2: float = 0.0,
                 transform_augmentation: bool = True,
                 transform_augmentation_xyz: list = [0.0, 0.0, 0.0],
                 transform_augmentation_rpy: list = [0.0, 0.0, 180.0],
                 transform_augmentation_rot_resolution: int = 5,
                 optimizer_type: str = 'adam',
                 num_devices: int = 1,
                 wandb_run = None,
                 ):
        self._layer = layer
        self._coordinate_bounds = coordinate_bounds
        self._perceiver_encoder = perceiver_encoder
        self._voxel_feature_size = voxel_feature_size
        self._bounds_offset = bounds_offset
        self._image_crop_size = image_crop_size
        self._lr = lr
        self._lr_scheduler = lr_scheduler
        self._training_iterations = training_iterations
        self._num_warmup_steps = num_warmup_steps
        self._trans_loss_weight = trans_loss_weight
        self._rot_loss_weight = rot_loss_weight
        self._grip_loss_weight = grip_loss_weight
        self._collision_loss_weight = collision_loss_weight
        self._include_low_dim_state = include_low_dim_state
        self._image_resolution = image_resolution or [128, 128]
        self._voxel_size = voxel_size
        self._camera_names = camera_names
        self._num_cameras = len(camera_names)
        self._batch_size = batch_size
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._transform_augmentation_xyz = torch.from_numpy(np.array(transform_augmentation_xyz))
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = transform_augmentation_rot_resolution
        self._optimizer_type = optimizer_type
        self._num_devices = num_devices
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = rotation_resolution
        self._wandb_run = wandb_run

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self._name = NAME + '_layer' + str(self._layer)

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device

        if device is None:
            device = torch.device('cpu')

        self._voxelizer = VoxelGrid(
            coord_bounds=self._coordinate_bounds,
            voxel_size=self._voxel_size,
            device=device,
            batch_size=self._batch_size if training else 1,
            feature_size=self._voxel_feature_size,
            max_num_coords=np.prod(self._image_resolution) * self._num_cameras,
        )

        self._q = QFunction2Robots(self._perceiver_encoder,
                            self._voxelizer,
                            self._bounds_offset,
                            self._rotation_resolution,
                            device,
                            training).to(device).train(training)

        grid_for_crop = torch.arange(0,
                                     self._image_crop_size,
                                     device=device).unsqueeze(0).repeat(self._image_crop_size, 1).unsqueeze(-1)
        self._grid_for_crop = torch.cat([grid_for_crop.transpose(1, 0),
                                         grid_for_crop], dim=2).unsqueeze(0)

        self._coordinate_bounds = torch.tensor(self._coordinate_bounds,
                                               device=device).unsqueeze(0)

        if self._training:
            # optimizer
            if self._optimizer_type == 'lamb':
                self._optimizer = Lamb(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                    adam=False,
                )
            elif self._optimizer_type == 'adam':
                self._optimizer = torch.optim.Adam(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                )
            else:
                raise Exception('Unknown optimizer type')

            # learning rate scheduler
            if self._lr_scheduler:
                self._scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                    self._optimizer,
                    num_warmup_steps=self._num_warmup_steps,
                    num_training_steps=self._training_iterations,
                    num_cycles=self._training_iterations // 10000,
                )

            # one-hot zero tensors
            self._action_trans_one_hot_zeros = torch.zeros((self._batch_size,
                                                            1,
                                                            self._voxel_size,
                                                            self._voxel_size,
                                                            self._voxel_size),
                                                            dtype=int,
                                                            device=device)
            self._action_rot_x_one_hot_zeros = torch.zeros((self._batch_size,
                                                            self._num_rotation_classes),
                                                            dtype=int,
                                                            device=device)
            self._action_rot_y_one_hot_zeros = torch.zeros((self._batch_size,
                                                            self._num_rotation_classes),
                                                            dtype=int,
                                                            device=device)
            self._action_rot_z_one_hot_zeros = torch.zeros((self._batch_size,
                                                            self._num_rotation_classes),
                                                            dtype=int,
                                                            device=device)
            self._action_grip_one_hot_zeros = torch.zeros((self._batch_size,
                                                           2),
                                                           dtype=int,
                                                           device=device)
            self._action_ignore_collisions_one_hot_zeros = torch.zeros((self._batch_size,
                                                                        2),
                                                                        dtype=int,
                                                                        device=device)

            # print total params
            logging.info('# Q Params: %d' % sum(
                p.numel() for name, p in self._q.named_parameters() \
                if p.requires_grad and 'clip' not in name))
        else:
            for param in self._q.parameters():
                param.requires_grad = False

            # load CLIP for encoding language goals during evaluation
            model, _ = load_clip("RN50", jit=False)
            self._clip_rn50 = build_model(model.state_dict())
            self._clip_rn50 = self._clip_rn50.float().to(device)
            self._clip_rn50.eval()
            del model

            self._voxelizer.to(device)
            self._q.to(device)

    def _extract_crop(self, pixel_action, observation):
        # Pixel action will now be (B, 2)
        # observation = stack_on_channel(observation)
        h = observation.shape[-1]
        top_left_corner = torch.clamp(
            pixel_action - self._image_crop_size // 2, 0,
            h - self._image_crop_size)
        grid = self._grid_for_crop + top_left_corner.unsqueeze(1)
        grid = ((grid / float(h)) * 2.0) - 1.0  # between -1 and 1
        # Used for cropping the images across a batch
        # swap fro y x, to x, y
        grid = torch.cat((grid[:, :, :, 1:2], grid[:, :, :, 0:1]), dim=-1)
        crop = F.grid_sample(observation, grid, mode='nearest',
                             align_corners=True)
        return crop

    def _preprocess_inputs(self, replay_sample):
        obs = []
        pcds = []
        self._crop_summary = []
        for n in self._camera_names:
            rgb = replay_sample['%s_rgb' % n]
            pcd = replay_sample['%s_point_cloud' % n]

            obs.append([rgb, pcd])
            pcds.append(pcd)
        return obs, pcds

    def _act_preprocess_inputs(self, observation):
        obs, pcds = [], []
        for n in self._camera_names:
            rgb = observation['%s_rgb' % n]
            pcd = observation['%s_point_cloud' % n]

            obs.append([rgb, pcd])
            pcds.append(pcd)
        return obs, pcds

    def _get_value_from_voxel_index(self, q, voxel_idx):
        b, c, d, h, w = q.shape
        q_trans_flat = q.view(b, c, d * h * w)
        flat_indicies = (voxel_idx[:, 0] * d * h + voxel_idx[:, 1] * h + voxel_idx[:, 2])[:, None].int()
        highest_idxs = flat_indicies.unsqueeze(-1).repeat(1, c, 1)
        chosen_voxel_values = q_trans_flat.gather(2, highest_idxs)[..., 0]  # (B, trans + rot + grip)
        return chosen_voxel_values

    def _get_value_from_rot_and_grip(self, rot_grip_q, rot_and_grip_idx):
        q_rot = torch.stack(torch.split(
            rot_grip_q[:, :-2], int(360 // self._rotation_resolution),
            dim=1), dim=1)  # B, 3, 72
        q_grip = rot_grip_q[:, -2:]
        rot_and_grip_values = torch.cat(
            [q_rot[:, 0].gather(1, rot_and_grip_idx[:, 0:1]),
             q_rot[:, 1].gather(1, rot_and_grip_idx[:, 1:2]),
             q_rot[:, 2].gather(1, rot_and_grip_idx[:, 2:3]),
             q_grip.gather(1, rot_and_grip_idx[:, 3:4])], -1)
        return rot_and_grip_values

    def _celoss(self, pred, labels):
        return self._cross_entropy_loss(pred, labels.argmax(-1))

    def _softmax_q_trans(self, q):
        q_shape = q.shape
        return F.softmax(q.reshape(q_shape[0], -1), dim=1).reshape(q_shape)

    def _softmax_q_rot_grip(self, q_rot_grip):
        q_rot_x_flat = q_rot_grip[:, 0*self._num_rotation_classes: 1*self._num_rotation_classes]
        q_rot_y_flat = q_rot_grip[:, 1*self._num_rotation_classes: 2*self._num_rotation_classes]
        q_rot_z_flat = q_rot_grip[:, 2*self._num_rotation_classes: 3*self._num_rotation_classes]
        q_grip_flat  = q_rot_grip[:, 3*self._num_rotation_classes:]

        q_rot_x_flat_softmax = F.softmax(q_rot_x_flat, dim=1)
        q_rot_y_flat_softmax = F.softmax(q_rot_y_flat, dim=1)
        q_rot_z_flat_softmax = F.softmax(q_rot_z_flat, dim=1)
        q_grip_flat_softmax = F.softmax(q_grip_flat, dim=1)

        return torch.cat([q_rot_x_flat_softmax,
                          q_rot_y_flat_softmax,
                          q_rot_z_flat_softmax,
                          q_grip_flat_softmax], dim=1)

    def _softmax_ignore_collision(self, q_collision):
        q_collision_softmax = F.softmax(q_collision, dim=1)
        return q_collision_softmax

    # baseline #2 v3
    def update(self, step: int, replay_sample: dict) -> dict:
        action_trans_right = replay_sample['trans_action_indicies_right'][:, self._layer * 3:self._layer * 3 + 3].int()
        action_rot_grip_right = replay_sample['rot_grip_action_indicies_right'].int()
        action_gripper_pose_right = replay_sample['gripper_pose_right']
        action_trans_left = replay_sample['trans_action_indicies_left'][:, self._layer * 3:self._layer * 3 + 3].int()
        action_rot_grip_left = replay_sample['rot_grip_action_indicies_left'].int()
        action_gripper_pose_left = replay_sample['gripper_pose_left']
        action_ignore_collisions = replay_sample['ignore_collisions'].int()
        lang_goal_emb = replay_sample['lang_goal_emb'].float()
        lang_token_embs = replay_sample['lang_token_embs'].float()
        prev_layer_voxel_grid = replay_sample.get('prev_layer_voxel_grid', None)
        prev_layer_bounds = replay_sample.get('prev_layer_bounds', None)
        device = self._device

        bounds = self._coordinate_bounds.to(device)
        if self._layer > 0:
            cp = replay_sample['attention_coordinate_layer_%d' % (self._layer - 1)]
            bounds = torch.cat([cp - self._bounds_offset, cp + self._bounds_offset], dim=1)

        proprio_right, proprio_left = None, None
        if self._include_low_dim_state:
            proprio_right = replay_sample['low_dim_state_right_arm']
            proprio_left = replay_sample['low_dim_state_left_arm']

        # NOTE: right now, we're feeding wrist and wrist2 images
        obs, pcd = self._preprocess_inputs(replay_sample)

        # batch size
        bs = pcd[0].shape[0]

        # SE(3) augmentation of point clouds and actions
        if self._transform_augmentation:
            # left and right arms need to have the same augmentations (only 1 pcd is returned).
            action_trans_right, \
            action_rot_grip_right, \
            action_trans_left, \
            action_rot_grip_left, \
            pcd = apply_se3_augmentation_2Robots(pcd,
                                         action_gripper_pose_right,
                                         action_trans_right,
                                         action_rot_grip_right,
                                         action_gripper_pose_left,
                                         action_trans_left,
                                         action_rot_grip_left,
                                         bounds,
                                         self._layer,
                                         self._transform_augmentation_xyz,
                                         self._transform_augmentation_rpy,
                                         self._transform_augmentation_rot_resolution,
                                         self._voxel_size,
                                         self._rotation_resolution,
                                         self._device)

        # forward pass
        q_trans_right, q_rot_grip_right, \
        q_collision_right, \
        voxel_grid, \
        q_trans_left, q_rot_grip_left, \
        q_collision_left = self._q(obs,
                             proprio_right,
                             proprio_left,
                             pcd,
                             lang_goal_emb,
                             lang_token_embs,
                             bounds,
                             prev_layer_bounds,
                             prev_layer_voxel_grid)

        # argmax to choose best action
        coords_right, \
        rot_and_grip_indicies_right, \
        ignore_collision_indicies_right = self._q.choose_highest_action(q_trans_right, q_rot_grip_right, q_collision_right)

        coords_left, \
        rot_and_grip_indicies_left, \
        ignore_collision_indicies_left = self._q.choose_highest_action(q_trans_left, q_rot_grip_left, q_collision_left)

        q_trans_loss, q_rot_loss, q_grip_loss, q_collision_loss = 0., 0., 0., 0.

        # translation one-hot
        action_trans_one_hot_right = self._action_trans_one_hot_zeros.clone()
        for b in range(bs):
            gt_coord = action_trans_right[b, :].int()
            action_trans_one_hot_right[b, :, gt_coord[0], gt_coord[1], gt_coord[2]] = 1

        action_trans_one_hot_left = self._action_trans_one_hot_zeros.clone()
        for b in range(bs):
            gt_coord = action_trans_left[b, :].int()
            action_trans_one_hot_left[b, :, gt_coord[0], gt_coord[1], gt_coord[2]] = 1

        # translation loss
        q_trans_flat_right = q_trans_right.view(bs, -1)
        action_trans_one_hot_flat_right = action_trans_one_hot_right.view(bs, -1)
        q_trans_loss += self._celoss(q_trans_flat_right, action_trans_one_hot_flat_right)

        q_trans_flat_left = q_trans_left.view(bs, -1)
        action_trans_one_hot_flat_left = action_trans_one_hot_left.view(bs, -1)
        q_trans_loss += self._celoss(q_trans_flat_left, action_trans_one_hot_flat_left)

        with_rot_and_grip_right = rot_and_grip_indicies_right is not None
        if with_rot_and_grip_right:
            # rotation, gripper, and collision one-hots
            action_rot_x_one_hot_right = self._action_rot_x_one_hot_zeros.clone()
            action_rot_y_one_hot_right = self._action_rot_y_one_hot_zeros.clone()
            action_rot_z_one_hot_right = self._action_rot_z_one_hot_zeros.clone()
            action_grip_one_hot_right = self._action_grip_one_hot_zeros.clone()
            action_ignore_collisions_one_hot = self._action_ignore_collisions_one_hot_zeros.clone()

            for b in range(bs):
                gt_rot_grip = action_rot_grip_right[b, :].int()
                action_rot_x_one_hot_right[b, gt_rot_grip[0]] = 1
                action_rot_y_one_hot_right[b, gt_rot_grip[1]] = 1
                action_rot_z_one_hot_right[b, gt_rot_grip[2]] = 1
                action_grip_one_hot_right[b, gt_rot_grip[3]] = 1

                gt_ignore_collisions = action_ignore_collisions[b, :].int()
                action_ignore_collisions_one_hot[b, gt_ignore_collisions[0]] = 1

            # flatten predictions
            q_rot_x_flat_right = q_rot_grip_right[:, 0*self._num_rotation_classes:1*self._num_rotation_classes]
            q_rot_y_flat_right = q_rot_grip_right[:, 1*self._num_rotation_classes:2*self._num_rotation_classes]
            q_rot_z_flat_right = q_rot_grip_right[:, 2*self._num_rotation_classes:3*self._num_rotation_classes]
            q_grip_flat_right =  q_rot_grip_right[:, 3*self._num_rotation_classes:]

            # rotation loss
            q_rot_loss += self._celoss(q_rot_x_flat_right, action_rot_x_one_hot_right)
            q_rot_loss += self._celoss(q_rot_y_flat_right, action_rot_y_one_hot_right)
            q_rot_loss += self._celoss(q_rot_z_flat_right, action_rot_z_one_hot_right)

            # gripper loss
            q_grip_loss += self._celoss(q_grip_flat_right, action_grip_one_hot_right)

            # collision loss
            q_collision_loss += self._celoss(q_collision_right, action_ignore_collisions_one_hot)

        with_rot_and_grip_left = rot_and_grip_indicies_left is not None
        if with_rot_and_grip_left:
            # rotation, gripper, and collision one-hots
            action_rot_x_one_hot_left = self._action_rot_x_one_hot_zeros.clone()
            action_rot_y_one_hot_left = self._action_rot_y_one_hot_zeros.clone()
            action_rot_z_one_hot_left = self._action_rot_z_one_hot_zeros.clone()
            action_grip_one_hot_left = self._action_grip_one_hot_zeros.clone()
            action_ignore_collisions_one_hot = self._action_ignore_collisions_one_hot_zeros.clone()

            for b in range(bs):
                gt_rot_grip = action_rot_grip_left[b, :].int()
                action_rot_x_one_hot_left[b, gt_rot_grip[0]] = 1
                action_rot_y_one_hot_left[b, gt_rot_grip[1]] = 1
                action_rot_z_one_hot_left[b, gt_rot_grip[2]] = 1
                action_grip_one_hot_left[b, gt_rot_grip[3]] = 1

                gt_ignore_collisions = action_ignore_collisions[b, :].int()
                action_ignore_collisions_one_hot[b, gt_ignore_collisions[0]] = 1

            # flatten predictions
            q_rot_x_flat_left = q_rot_grip_left[:, 0*self._num_rotation_classes:1*self._num_rotation_classes]
            q_rot_y_flat_left = q_rot_grip_left[:, 1*self._num_rotation_classes:2*self._num_rotation_classes]
            q_rot_z_flat_left = q_rot_grip_left[:, 2*self._num_rotation_classes:3*self._num_rotation_classes]
            q_grip_flat_left =  q_rot_grip_left[:, 3*self._num_rotation_classes:]

            # rotation loss
            q_rot_loss += self._celoss(q_rot_x_flat_left, action_rot_x_one_hot_left)
            q_rot_loss += self._celoss(q_rot_y_flat_left, action_rot_y_one_hot_left)
            q_rot_loss += self._celoss(q_rot_z_flat_left, action_rot_z_one_hot_left)

            # gripper loss
            q_grip_loss += self._celoss(q_grip_flat_left, action_grip_one_hot_left)

            # collision loss
            q_collision_loss += self._celoss(q_collision_left, action_ignore_collisions_one_hot)

        combined_losses = (q_trans_loss * self._trans_loss_weight) + \
                          (q_rot_loss * self._rot_loss_weight) + \
                          (q_grip_loss * self._grip_loss_weight) + \
                          (q_collision_loss * self._collision_loss_weight)
        total_loss = combined_losses.mean()

        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        self._summaries = {
            'losses/total_loss': total_loss,
            'losses/trans_loss': q_trans_loss.mean(),
            'losses/rot_loss': q_rot_loss.mean() if with_rot_and_grip_right else 0.,
            'losses/grip_loss': q_grip_loss.mean() if with_rot_and_grip_right else 0.,
            'losses/collision_loss': q_collision_loss.mean() if with_rot_and_grip_right else 0.,
        }

        if self._lr_scheduler:
            self._scheduler.step()
            self._summaries['learning_rate'] = self._scheduler.get_last_lr()[0]

        self._vis_voxel_grid = voxel_grid[0]
        self._vis_translation_qvalue_right = self._softmax_q_trans(q_trans_right[0])
        self._vis_max_coordinate_right = coords_right[0]
        self._vis_gt_coordinate_right = action_trans_right[0]
        self._vis_translation_qvalue_left = self._softmax_q_trans(q_trans_left[0])
        self._vis_max_coordinate_left = coords_left[0]
        self._vis_gt_coordinate_left = action_trans_left[0]

        # Note: PerAct doesn't use multi-layer voxel grids like C2FARM
        # stack prev_layer_voxel_grid(s) from previous layers into a list
        if prev_layer_voxel_grid is None:
            prev_layer_voxel_grid = [voxel_grid]
        else:
            prev_layer_voxel_grid = prev_layer_voxel_grid + [voxel_grid]

        # stack prev_layer_bound(s) from previous layers into a list
        if prev_layer_bounds is None:
            prev_layer_bounds = [self._coordinate_bounds.repeat(bs, 1)]
        else:
            prev_layer_bounds = prev_layer_bounds + [bounds]

        # debug: make sure voxel grid looks alright
        # visualise_voxel_2robots(
        #     self._vis_voxel_grid.detach().cpu().numpy(),
        #     self._vis_translation_qvalue_right.detach().cpu().numpy(),
        #     self._vis_max_coordinate_right.detach().cpu().numpy(),
        #     self._vis_gt_coordinate_right.detach().cpu().numpy(),
        #     self._vis_translation_qvalue_left.detach().cpu().numpy(),
        #     self._vis_max_coordinate_left.detach().cpu().numpy(),
        #     self._vis_gt_coordinate_left.detach().cpu().numpy(),
        #     show=True)

        return {
            'total_loss': total_loss,
            'prev_layer_voxel_grid': prev_layer_voxel_grid,
            'prev_layer_bounds': prev_layer_bounds,
        }

    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:
        deterministic = True
        bounds = self._coordinate_bounds
        prev_layer_voxel_grid = observation.get('prev_layer_voxel_grid', None)
        prev_layer_bounds = observation.get('prev_layer_bounds', None)
        lang_goal_tokens = observation.get('lang_goal_tokens', None).long()

        # extract CLIP language embs
        with torch.no_grad():
            lang_goal_tokens = lang_goal_tokens.to(device=self._device)
            lang_goal_emb, lang_token_embs = self._clip_rn50.encode_text_with_embeddings(lang_goal_tokens[0])

        # voxelization resolution
        res = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        max_rot_index = int(360 // self._rotation_resolution)
        proprio_right, proprio_left = None, None

        proprio_right = observation['low_dim_state_right_arm']
        proprio_left = observation['low_dim_state_left_arm']

        # NOTE: right now, we're feeding wrist and wrist2 images
        obs, pcd = self._act_preprocess_inputs(observation)

        # correct batch size and device
        obs = [[o[0][0].to(self._device), o[1][0].to(self._device)] for o in obs]
        proprio_right = proprio_right[0].to(self._device)
        proprio_left = proprio_left[0].to(self._device)
        pcd = [p[0].to(self._device) for p in pcd]
        lang_goal_emb = lang_goal_emb.to(self._device)
        lang_token_embs = lang_token_embs.to(self._device)
        bounds = torch.as_tensor(bounds, device=self._device)
        prev_layer_voxel_grid = prev_layer_voxel_grid.to(self._device) if prev_layer_voxel_grid is not None else None
        prev_layer_bounds = prev_layer_bounds.to(self._device) if prev_layer_bounds is not None else None

        # inference
        q_trans_right, q_rot_grip_right, \
        q_ignore_collisions_right, \
        vox_grid, \
        q_trans_left, q_rot_grip_left, \
        q_ignore_collisions_left = self._q(obs,
                           proprio_right,
                           proprio_left,
                           pcd,
                           lang_goal_emb,
                           lang_token_embs,
                           bounds,
                           prev_layer_bounds,
                           prev_layer_voxel_grid)

        # softmax Q predictions
        q_trans_right = self._softmax_q_trans(q_trans_right)
        q_trans_left = self._softmax_q_trans(q_trans_left)
        q_rot_grip_right =  self._softmax_q_rot_grip(q_rot_grip_right) if q_rot_grip_right is not None else q_rot_grip_right
        q_rot_grip_left =  self._softmax_q_rot_grip(q_rot_grip_left) if q_rot_grip_left is not None else q_rot_grip_left
        q_ignore_collisions_right = self._softmax_ignore_collision(q_ignore_collisions_right) \
            if q_ignore_collisions_right is not None else q_ignore_collisions_right
        q_ignore_collisions_left = self._softmax_ignore_collision(q_ignore_collisions_left) \
            if q_ignore_collisions_left is not None else q_ignore_collisions_left

        # argmax Q predictions
        coords_right, \
        rot_and_grip_indicies_right, \
        ignore_collisions_right = self._q.choose_highest_action(q_trans_right, q_rot_grip_right, q_ignore_collisions_right)

        coords_left, \
        rot_and_grip_indicies_left, \
        ignore_collisions_left = self._q.choose_highest_action(q_trans_left, q_rot_grip_left, q_ignore_collisions_left)

        rot_grip_action_right = rot_and_grip_indicies_right if q_rot_grip_right is not None else None
        ignore_collisions_action_right = ignore_collisions_right.int() if ignore_collisions_right is not None else None

        rot_grip_action_left = rot_and_grip_indicies_left if q_rot_grip_left is not None else None
        ignore_collisions_action_left = ignore_collisions_left.int() if ignore_collisions_left is not None else None

        coords_right = coords_right.int()
        attention_coordinate_right = bounds[:, :3] + res * coords_right + res / 2

        coords_left = coords_left.int()
        attention_coordinate_left = bounds[:, :3] + res * coords_left + res / 2

        # stack prev_layer_voxel_grid(s) into a list
        # NOTE: PerAct doesn't used multi-layer voxel grids like C2FARM
        if prev_layer_voxel_grid is None:
            prev_layer_voxel_grid = [vox_grid]
        else:
            prev_layer_voxel_grid = prev_layer_voxel_grid + [vox_grid]

        if prev_layer_bounds is None:
            prev_layer_bounds = [bounds]
        else:
            prev_layer_bounds = prev_layer_bounds + [bounds]

        observation_elements = {
            'attention_coordinate_right': attention_coordinate_right,
            'attention_coordinate_left': attention_coordinate_left,
            'prev_layer_voxel_grid': prev_layer_voxel_grid,
            'prev_layer_bounds': prev_layer_bounds,
        }
        info = {
            'voxel_grid_depth%d' % self._layer: vox_grid,
            'q_depth_right%d' % self._layer: q_trans_right,
            'voxel_idx_depth_right%d' % self._layer: coords_right,
            'q_depth_left%d' % self._layer: q_trans_left,
            'voxel_idx_depth_left%d' % self._layer: coords_left,
        }
        self._act_voxel_grid = vox_grid[0]
        self._act_max_coordinate_right = coords_right[0]
        self._act_qvalues_right = q_trans_right[0].detach()
        self._act_max_coordinate_left = coords_left[0]
        self._act_qvalues_left = q_trans_left[0].detach()

        # debug: make sure voxel grid looks alright
        # visualise_voxel_2robots(
        #     vox_grid[0].detach().cpu().numpy(),
        #     q_trans_right[0].detach().cpu().numpy(),
        #     coords_right[0].detach().cpu().numpy(),
        #     None,
        #     q_trans_left[0].detach().cpu().numpy(),
        #     coords_left[0].detach().cpu().numpy(),
        #     None,
        #     show=True)

        return ActResult((coords_right, rot_grip_action_right, ignore_collisions_action_right, coords_left, rot_grip_action_left, ignore_collisions_action_left),
                         observation_elements=observation_elements,
                         info=info)

    def update_summaries(self) -> List[Summary]:
        try:
            summaries = [
                ImageSummary('%s/update_qattention' % self._name,
                            transforms.ToTensor()(visualise_voxel_2robots(
                                self._vis_voxel_grid.detach().cpu().numpy(),
                                self._vis_translation_qvalue_right.detach().cpu().numpy(),
                                self._vis_max_coordinate_right.detach().cpu().numpy(),
                                self._vis_gt_coordinate_right.detach().cpu().numpy(),
                                self._vis_translation_qvalue_left.detach().cpu().numpy(),
                                self._vis_max_coordinate_left.detach().cpu().numpy(),
                                self._vis_gt_coordinate_left.detach().cpu().numpy())))
            ]
        except:
            # this exception can happen when the computer does not have a display or the display is not set up properly
            summaries = []

        wandb_dict = {}
        for n, v in self._summaries.items():
            summaries.append(ScalarSummary('%s/%s' % (self._name, n), v))
            if self._wandb_run is not None:
                wandb_dict['%s/%s' % (self._name, n)] = v

        for (name, crop) in (self._crop_summary):
            crops = (torch.cat(torch.split(crop, 3, dim=1), dim=3) + 1.0) / 2.0
            summaries.extend([
                ImageSummary('%s/crops/%s' % (self._name, name), crops)])

        for tag, param in self._q.named_parameters():
            # assert not torch.isnan(param.grad.abs() <= 1.0).all()
            summaries.append(
                HistogramSummary('%s/gradient/%s' % (self._name, tag),
                                 param.grad))
            summaries.append(
                HistogramSummary('%s/weight/%s' % (self._name, tag),
                                 param.data))

        return summaries, wandb_dict

    def act_summaries(self) -> List[Summary]:
        return [
            ImageSummary('%s/act_Qattention' % self._name,
                         transforms.ToTensor()(visualise_voxel_2robots(
                             self._act_voxel_grid.cpu().numpy(),
                             self._act_qvalues_right.cpu().numpy(),
                             self._act_max_coordinate_right.cpu().numpy(),
                             None,
                             self._act_qvalues_left.cpu().numpy(),
                             self._act_max_coordinate_left.cpu().numpy(),
                             None,
                             )))]

    def load_weights(self, savedir: str):
        device = self._device if not self._training else torch.device('cuda:%d' % self._device)
        weight_file = os.path.join(savedir, '%s.pt' % self._name)
        state_dict = torch.load(weight_file, map_location=device)

        # load only keys that are in the current model
        merged_state_dict = self._q.state_dict()
        for k, v in state_dict.items():
            if not self._training:
                k = k.replace('_qnet.module', '_qnet')
            if k in merged_state_dict:
                merged_state_dict[k] = v
            else:
                if '_voxelizer' not in k:
                    logging.warning("key %s not found in checkpoint" % k)
        self._q.load_state_dict(merged_state_dict)
        print("loaded weights from %s" % weight_file)

    def load_weight(self, ckpt_file: str):
        device = self._device if not self._training else torch.device('cuda:%d' % self._device)
        state_dict = torch.load(ckpt_file, map_location=device)

        # load only keys that are in the current model
        merged_state_dict = self._q.state_dict()
        for k, v in state_dict.items():
            if not self._training:
                k = k.replace('_qnet.module', '_qnet')
            if k in merged_state_dict:
                merged_state_dict[k] = v
            else:
                if '_voxelizer' not in k:
                    logging.warning("key %s not found in checkpoint" % k)
        self._q.load_state_dict(merged_state_dict)
        print("loaded weights from %s" % ckpt_file)

    def save_weights(self, savedir: str):
        torch.save(
            self._q.state_dict(), os.path.join(savedir, '%s.pt' % self._name))