from typing import List

import torch

from yarr.agents.agent import Agent, Summary, ActResult, \
    ScalarSummary, HistogramSummary, ImageSummary


class PreprocessAgent(Agent):

    def __init__(self,
                 pose_agent: Agent,
                 norm_rgb: bool = True):
        self._pose_agent = pose_agent
        self._norm_rgb = norm_rgb

    def build(self, training: bool, device: torch.device = None):
        self._pose_agent.build(training, device)

    def _norm_rgb_(self, x):
        return (x.float() / 255.0) * 2.0 - 1.0

    def update(self, step: int, replay_sample: dict) -> dict:
        # Samples are (B, N, ...) where N is number of buffers/tasks. This is a single task setup, so 0 index.
        replay_sample = {k: v[:, 0] if len(v.shape) > 2 else v for k, v in replay_sample.items()}
        for k, v in replay_sample.items():
            if self._norm_rgb and 'rgb' in k:
                replay_sample[k] = self._norm_rgb_(v)
            else:
                replay_sample[k] = v.float()
        self._replay_sample = replay_sample
        return self._pose_agent.update(step, replay_sample)

    def act(self, step: int, observation: dict,
            deterministic=False, which_arm=None, new_scene_bounds=None, dominant_assitive_policy=False, ep_number=0, is_real_robot=False) -> ActResult:
        # observation = {k: torch.tensor(v) for k, v in observation.items()}
        for k, v in observation.items():
            if self._norm_rgb and 'rgb' in k:
                observation[k] = self._norm_rgb_(v)
            else:
                observation[k] = v.float()

        if is_real_robot:
            return self._pose_agent.act(step, observation, deterministic, which_arm, new_scene_bounds, dominant_assitive_policy, ep_number, is_real_robot)

        act_res = self._pose_agent.act(step, observation, deterministic, which_arm, new_scene_bounds, dominant_assitive_policy, ep_number, is_real_robot)
        act_res.replay_elements.update({'demo': False})
        return act_res

    def update_summaries(self) -> List[Summary]:
        prefix = 'inputs'
        demo_f = self._replay_sample['demo'].float()
        demo_proportion = demo_f.mean()
        tile = lambda x: torch.squeeze(
            torch.cat(x.split(1, dim=1), dim=-1), dim=1)
        if 'low_dim_state' in self._replay_sample:
                sums = [
                    ScalarSummary('%s/demo_proportion' % prefix, demo_proportion),
                    HistogramSummary('%s/low_dim_state' % prefix,
                            self._replay_sample['low_dim_state']),
                    HistogramSummary('%s/low_dim_state_tp1' % prefix,
                            self._replay_sample['low_dim_state_tp1']),
                    ScalarSummary('%s/low_dim_state_mean' % prefix,
                            self._replay_sample['low_dim_state'].mean()),
                    ScalarSummary('%s/low_dim_state_min' % prefix,
                            self._replay_sample['low_dim_state'].min()),
                    ScalarSummary('%s/low_dim_state_max' % prefix,
                            self._replay_sample['low_dim_state'].max()),
                    ScalarSummary('%s/timeouts' % prefix,
                            self._replay_sample['timeout'].float().mean()),
                ]
        else:
            # assume it's two arms implementation
            sums = [
                ScalarSummary('%s/demo_proportion' % prefix, demo_proportion),
                HistogramSummary('%s/low_dim_state_right_arm' % prefix,
                        self._replay_sample['low_dim_state_right_arm']),
                ScalarSummary('%s/low_dim_state_right_arm_mean' % prefix,
                        self._replay_sample['low_dim_state_right_arm'].mean()),
                ScalarSummary('%s/low_dim_state_right_arm_min' % prefix,
                        self._replay_sample['low_dim_state_right_arm'].min()),
                ScalarSummary('%s/low_dim_state_right_arm_max' % prefix,
                        self._replay_sample['low_dim_state_right_arm'].max()),
                ScalarSummary('%s/timeouts' % prefix,
                        self._replay_sample['timeout'].float().mean()),
                HistogramSummary('%s/low_dim_state_left_arm' % prefix,
                        self._replay_sample['low_dim_state_left_arm']),
                ScalarSummary('%s/low_dim_state_left_arm_mean' % prefix,
                        self._replay_sample['low_dim_state_left_arm'].mean()),
                ScalarSummary('%s/low_dim_state_left_arm_min' % prefix,
                        self._replay_sample['low_dim_state_left_arm'].min()),
                ScalarSummary('%s/low_dim_state_left_arm_max' % prefix,
                        self._replay_sample['low_dim_state_left_arm'].max()),
            ]

        for k, v in self._replay_sample.items():
            if 'rgb' in k or 'point_cloud' in k:
                if 'rgb' in k:
                    # Convert back to 0 - 1
                    v = (v + 1.0) / 2.0
                sums.append(ImageSummary('%s/%s' % (prefix, k), tile(v)))

        if 'sampling_probabilities' in self._replay_sample:
            sums.extend([
                HistogramSummary('replay/priority',
                                 self._replay_sample['sampling_probabilities']),
            ])
        summaries, wandb_dict = self._pose_agent.update_summaries()
        sums.extend(summaries)
        return sums, wandb_dict

    def act_summaries(self) -> List[Summary]:
        return self._pose_agent.act_summaries()

    def load_weights(self, savedir: str):
        self._pose_agent.load_weights(savedir)

    def load_weight(self, ckpt_file: str):
        self._pose_agent.load_weight(ckpt_file)

    def save_weights(self, savedir: str):
        self._pose_agent.save_weights(savedir)

    def reset(self) -> None:
        self._pose_agent.reset()

