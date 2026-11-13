"""Pilot execution shell: bounded fallback, episode memory, and shadow foresight.

The learned predictor is diagnostic only. Its offline loss is not a calibrated
success/risk score and therefore cannot authorize physical recovery actions.
"""
import json
import logging
import os
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from policy_space.base import PolicyModel
from policy_space.runtime.server import load_policy_model
from latent_world_model import Dynamics
from prepare_training import relative_actions

LOG = logging.getLogger(__name__)


def state20(values):
    from wall_x._vendor.x2robot_utils.geometry import euler_to_matrix_zyx_6d_nb
    values = np.asarray(values, dtype=np.float64)
    shape = values.shape[:-1]
    values = values.reshape(-1, 14)
    parts = []
    for offset in (0, 7):
        parts += [values[:,offset:offset+3],
                  euler_to_matrix_zyx_6d_nb(values[:,offset+3:offset+6]),
                  values[:,offset+6:offset+7]]
    return np.concatenate(parts, -1).reshape(*shape, 20).astype(np.float32)


def validate_actions(actions, state):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != (33,14) or not np.isfinite(actions).all():
        raise ValueError('Policy must return finite [33,14] actions')
    if not np.allclose(actions[0], state, atol=1e-4):
        raise ValueError('First action must echo current state')
    if np.any(actions[:,[6,13]] < 0) or np.any(actions[:,[6,13]] > 1):
        raise ValueError('Gripper values must be normalized to [0,1]')
    return actions


class ShadowWorld:
    def __init__(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['horizon'] != 32:
            raise ValueError('World predictor horizon does not match the policy')
        self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder.fc = nn.Identity()
        self.encoder.cuda().eval()
        self.members = []
        for weights in checkpoint['members']:
            member = Dynamics().cuda().eval()
            member.load_state_dict(weights)
            self.members.append(member)
        self.zmean, self.zstd, self.amean, self.astd = (
            checkpoint[key].numpy() for key in ('zmean','zstd','amean','astd'))

    @torch.inference_mode()
    def encode(self, obs, state):
        pictures = []
        for key in ('camera_front','camera_left','camera_right'):
            image = obs['images'][key]
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f'Expected uint8 HWC RGB: {key}')
            pictures.append(cv2.resize(image, (224,224)))
        x = torch.from_numpy(np.stack(pictures)).cuda().permute(0,3,1,2).float()/255
        mean = torch.tensor([.485,.456,.406], device=x.device)[None,:,None,None]
        std = torch.tensor([.229,.224,.225], device=x.device)[None,:,None,None]
        visual = self.encoder((x-mean)/std).cpu().numpy().reshape(-1)
        return np.concatenate((visual,state20(state)))

    @torch.inference_mode()
    def predict(self, current, actions):
        action = relative_actions(state20(actions[1:])[None], current[-20:][None]).reshape(-1)
        x = np.concatenate(((current-self.zmean)/self.zstd, (action-self.amean)/self.astd))
        tensor = torch.from_numpy(x.astype(np.float32))[None].cuda()
        members = np.stack([model(tensor).cpu().numpy()[0] for model in self.members])
        return current + members.mean(0)*self.zstd, float(members.std(0).mean())


class Model(PolicyModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.base, metadata = load_policy_model(os.environ['WALLX_BASE_DEPLOY'])
        if metadata.get('action_dim') != 14 or metadata.get('action_horizon') != 33:
            raise ValueError('Requires the Arm-6A EEF adapter')
        self.world = ShadowWorld(os.environ['AGENTOS_WORLD_CHECKPOINT'])
        self.audit_path = Path(os.getenv('AGENTOS_AUDIT_PATH', 'outputs/agentos-events.jsonl'))
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._clear()

    def _clear(self):
        self.obs = None
        self.info = {}
        self.last_step = None
        self.cached = None
        self.pending = None
        self.fallback_count = 0
        self.memory = deque(maxlen=64)

    def _emit(self, event, **evidence):
        record = {'event':event, 'episode_id':self.info.get('episode_id'),
                  'step':(self.obs or {}).get('step'), **evidence}
        self.memory.append(record)
        LOG.info('agentos %s', json.dumps(record, allow_nan=False))
        with self.audit_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')

    def initialize_episode(self, episode_info=None):
        self._clear()
        self.info = episode_info or {}
        self.base.initialize_episode(episode_info)

    def ingest_observation(self, obs):
        self.obs = obs
        self.base.ingest_observation(obs)

    def infer_actions(self):
        if self.obs is None:
            raise RuntimeError('No observation')
        step = self.obs.get('step')
        if type(step) is int and step == self.last_step and self.cached is not None:
            return self.cached.copy()
        state = np.concatenate([np.asarray(self.obs['state'][key],dtype=np.float32)
                                for key in ('follow1_pos','follow2_pos')])
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError('Invalid arm state')
        instruction = self.obs.get('instruction') or self.info.get('instruction') or ''
        # The learned model was trained on this task only; do not extrapolate its
        # scores to unrelated objects, robots, or tasks.
        supported = instruction.strip().casefold().rstrip('.') in {
            'pick up the bottle and place it on the plate',
            'pick up the bottle and place it on the plate, then return the arm home'}
        current = self.world.encode(self.obs, state) if supported else None
        if self.pending is not None and type(step) is int:
            old_step, forecast = self.pending
            if step-old_step == 48 and current is not None:
                error = float(np.mean(((current-forecast)/self.world.zstd)**2))
                self._emit('forecast_verified', standardized_mse=error,
                           task_success=None, verification='observed_state_only')
            elif step > old_step:
                self._emit('forecast_unverified', reason='time_alignment_or_task_mismatch')
            self.pending = None
        candidate = self.base.infer_actions()
        skill = 'policy_chunk'
        try:
            action = validate_actions(candidate, state)
        except ValueError as error:
            if self.fallback_count >= 2:
                raise RuntimeError('Action gate rejected policy after two bounded retries') from error
            self.fallback_count += 1
            action = validate_actions(np.repeat(state[None],33,axis=0), state)
            skill = 'hold_pose'
            self._emit('fallback', reason=str(error), attempts=self.fallback_count)
        if current is not None:
            forecast, uncertainty = self.world.predict(current, action)
            hold = np.repeat(state[None],33,axis=0)
            _, hold_uncertainty = self.world.predict(current, hold)
            self._emit('candidate_forecasts', selected=skill, mode='shadow_only',
                       policy_disagreement=uncertainty, hold_disagreement=hold_uncertainty,
                       success_probability=None, collision_probability=None)
            if type(step) is int:
                self.pending = (step,forecast)
        self._emit('execute', skill=skill, world_model_supported=supported)
        self.last_step, self.cached = step, action.copy()
        return action

    def finalize_episode(self, episode_info=None):
        self._emit('episode_end', task_success=None)
        self.base.finalize_episode(episode_info)
        self._clear()

    def close(self):
        self.base.close()
