"""Train a small action-conditioned predictor of future visual features and arm state.

This is an offline, task-specific dynamics experiment. It does NOT estimate grasp
success or collision probability, and must not control recovery until closed-loop
validation establishes those properties. ImageNet ResNet18 is frozen.
"""
import json
import random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from prepare_training import relative_actions

ROOT = Path('/workspace/x2real-training')
CAMERAS = ['faceImg', 'leftImg', 'rightImg']
HORIZON = 32


class Dynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1556 + 640, 512), nn.LayerNorm(512), nn.SiLU(),
                                 nn.Dropout(.1), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 1556))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


@torch.inference_mode()
def extract():
    cache = ROOT / 'world_features.npz'
    if cache.exists():
        return np.load(cache)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model = model.cuda().eval()
    mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
    table = pd.read_parquet(ROOT/'data/bottle-v3/data/chunk-000/file-000.parquet')
    arrays = {key: [] for key in ('current', 'next', 'actions', 'episode', 'frame')}
    for episode, frames in table.groupby('episode_index', sort=True):
        state = np.stack(frames['observation.state']).astype(np.float32)
        action = np.stack(frames['action']).astype(np.float32)
        starts = np.arange(0, len(frames)-HORIZON, 16)
        wanted = sorted(set(starts.tolist() + (starts+HORIZON).tolist()))
        if not wanted:
            continue
        per_camera = []
        for camera in CAMERAS:
            path = ROOT/f'data/bottle-v3/videos/observation.images.{camera}/chunk-000/file-{episode:03d}.mp4'
            cap = cv2.VideoCapture(str(path))
            pictures, indices, embeddings = [], [], {}
            last = max(wanted)
            wanted_set = set(wanted)
            for index in range(last+1):
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f'Video ended early: {path}, frame {index}')
                if index in wanted_set:
                    image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (224, 224))
                    pictures.append(image)
                    indices.append(index)
            cap.release()
            for offset in range(0, len(pictures), 32):
                batch = torch.from_numpy(np.stack(pictures[offset:offset+32])).cuda().permute(0,3,1,2).float()/255
                encoded = model((batch-mean)/std).float().cpu().numpy()
                embeddings.update(zip(indices[offset:offset+32], encoded))
            per_camera.append(embeddings)
        z = {i: np.concatenate([features[i] for features in per_camera] + [state[i]]) for i in wanted}
        arrays['current'].append(np.stack([z[i] for i in starts]))
        arrays['next'].append(np.stack([z[i+HORIZON] for i in starts]))
        chunks = action[starts[:, None]+np.arange(HORIZON)]
        arrays['actions'].append(relative_actions(chunks, state[starts]).reshape(len(starts), -1))
        arrays['episode'].append(np.repeat(episode, len(starts)))
        arrays['frame'].append(starts)
        if episode % 25 == 0:
            print(f'FEATURES episode={episode}', flush=True)
    result = {key: np.concatenate(values) for key, values in arrays.items()}
    np.savez_compressed(cache, **result)
    return result


def metrics(predicted, target, current, latent_scale):
    xyz = [1536+i for i in (0,1,2,10,11,12)]
    grip = [1545, 1555]
    return {'visual_standardized_mse': float(np.mean(((predicted[:,:1536]-target[:,:1536])/latent_scale[:1536])**2)),
            'eef_position_rmse_m': float(np.sqrt(np.mean((predicted[:,xyz]-target[:,xyz])**2))),
            'gripper_mae': float(np.mean(np.abs(predicted[:,grip]-target[:,grip])))}


def main():
    torch.set_num_threads(2)
    features = extract()
    ep = features['episode']
    # Source episode order was shuffled once with seed 42 before conversion.
    train, val, test = ep < 240, (ep >= 240) & (ep < 270), ep >= 270
    current, future, actions = (features[k].astype(np.float32) for k in ('current', 'next', 'actions'))
    zmean, zstd = current[train].mean(0), np.maximum(current[train].std(0), .02)
    amean, astd = actions[train].mean(0), np.maximum(actions[train].std(0), .02)
    x = np.concatenate(((current-zmean)/zstd, (actions-amean)/astd), axis=-1)
    y = (future-current)/zstd
    xt, yt = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
    train_ids = torch.from_numpy(np.flatnonzero(train)).cuda()
    val_ids = torch.from_numpy(np.flatnonzero(val)).cuda()
    test_ids = torch.from_numpy(np.flatnonzero(test)).cuda()
    members, predictions = [], []
    output = ROOT/'outputs/world-pilot'
    output.mkdir(parents=True, exist_ok=True)
    for member in range(3):
        seed = 42+member
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = Dynamics().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        best, best_state = float('inf'), None
        for epoch in range(30):
            model.train()
            order = train_ids[torch.randperm(len(train_ids), device='cuda')]
            for ids in order.split(128):
                pred = model(xt[ids])
                # Give proprioception equal total weight to the larger visual latent.
                loss = (pred[:,:1536]-yt[ids,:1536]).square().mean() + (pred[:,1536:]-yt[ids,1536:]).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                pred = model(xt[val_ids])
                score = ((pred[:,:1536]-yt[val_ids,:1536]).square().mean() + (pred[:,1536:]-yt[val_ids,1536:]).square().mean()).item()
            if score < best:
                best = score
                best_state = {key: value.detach().cpu().clone() for key,value in model.state_dict().items()}
            if epoch % 10 == 0:
                print(f'WORLD member={member} epoch={epoch} val_loss={score:.6f}', flush=True)
        model.load_state_dict(best_state)
        model.eval()
        with torch.inference_mode():
            predictions.append(model(xt[test_ids]).cpu().numpy())
        members.append(best_state)
    ensemble = np.stack(predictions)
    predicted = current[test] + ensemble.mean(0)*zstd
    command_baseline = current[test].copy()
    last = actions[test].reshape(-1, HORIZON, 20)[:, -1]
    for offset in (0, 10):
        command_baseline[:,1536+offset:1539+offset] += last[:,offset:offset+3]
        command_baseline[:,1545+offset] = last[:,offset+9]
    report = {'data_revision': json.loads((ROOT/'data/bottle-v3/source.json').read_text())['revision'],
              'train_episodes': 240, 'validation_episodes': 30, 'test_episodes': 30,
              'test_transitions': int(test.sum()), 'horizon_seconds': 1.6,
              'persistence_baseline': metrics(current[test], future[test], current[test], zstd),
              'last_command_state_static_visual_baseline': metrics(command_baseline, future[test], current[test], zstd),
              'learned_predictor': metrics(predicted, future[test], current[test], zstd),
              'mean_ensemble_std': float(ensemble.std(0).mean()),
              'platform_success_rate': None, 'deployment_mode': 'shadow_only',
              'limitations': ['single bottle task', 'no grasp-success or collision labels',
                              'offline held-out trajectory test, not closed-loop evaluation',
                              'ensemble disagreement is not calibrated confidence']}
    torch.save({'members': members, 'zmean': torch.from_numpy(zmean), 'zstd': torch.from_numpy(zstd),
                'amean': torch.from_numpy(amean), 'astd': torch.from_numpy(astd),
                'horizon': HORIZON, 'encoder': 'resnet18_IMAGENET1K_V1_resize224_rgb',
                'report': report}, output/'dynamics.pt')
    (output/'metrics.json').write_text(json.dumps(report, indent=2))
    print('WORLD_TRAIN_COMPLETE', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
