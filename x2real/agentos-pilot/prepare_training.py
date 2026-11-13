"""Create bounded action-expert training config and train-only episode-safe statistics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path('/workspace/x2real-training')


def rotation_matrix(x):
    a = x[..., :3]
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    b = x[..., 3:6] - (a * x[..., 3:6]).sum(-1, keepdims=True) * a
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
    return np.stack((a, b, np.cross(a, b)), axis=-2)


def relative_actions(actions, state):
    result = actions.copy()
    for start in (0, 10):
        result[..., start:start+3] -= state[..., None, start:start+3]
        rot = rotation_matrix(actions[..., start+3:start+9])
        anchor = rotation_matrix(state[..., start+3:start+9])
        delta = rot @ np.swapaxes(anchor, -1, -2)[..., None, :, :]
        result[..., start+3:start+9] = delta[..., :2, :].reshape(*delta.shape[:-2], 6)
    return result


def main():
    data = ROOT / 'data/bottle-v3'
    source = json.loads((data / 'source.json').read_text())
    n = len(source['source_episode_indices'])
    split = int(n * .9)
    table = pd.read_parquet(data / 'data/chunk-000/file-000.parquet')
    states, chunks = [], []
    for episode, frames in table.groupby('episode_index', sort=True):
        if episode >= split:
            continue
        s = np.stack(frames['observation.state']).astype(np.float32)
        a = np.stack(frames['action']).astype(np.float32)
        indices = np.minimum(np.arange(len(a))[:, None] + np.arange(32), len(a)-1)
        states.append(s)
        chunks.append(relative_actions(a[indices], s).reshape(-1, 20))
    def stats(x):
        return {'mean': x.mean(0).tolist(), 'std': x.std(0).tolist(),
                'q01': np.quantile(x, .01, axis=0).tolist(),
                'q99': np.quantile(x, .99, axis=0).tolist()}
    norm = {'norm_stats': {'observation.state': stats(np.concatenate(states)),
                           'action': stats(np.concatenate(chunks))}}
    (ROOT / 'norm_stats.json').write_text(json.dumps(norm, indent=2))
    manifest = {'source_revision': source['revision'], 'seed': 42,
                'train_original_episodes': source['source_episode_indices'][:split],
                'test_original_episodes': source['source_episode_indices'][split:],
                'normalizer_uses_train_only': True, 'action_chunks_cross_episodes': False}
    (ROOT / 'split.json').write_text(json.dumps(manifest, indent=2))
    cfg = yaml.safe_load((ROOT / 'wall-x/workspace/example/maniparena_example.yml').read_text())
    cfg['model'].update(config_path=str(ROOT/'weights/config.json'),
                        processor_path=str(ROOT/'weights'), pretrained_path=None)
    cfg['task']['use_state_string_representation'] = True
    cfg['hyperparams'].update(num_epoch=1, batch_size_per_gpu=1, gradient_accumulation_steps=2, seed=42)
    cfg['hyperparams']['optimizer'].update(learning_rate=1e-5, action_expert_learning_rate=1e-5,
                                          train_action_expert_only=True)
    cfg['hyperparams']['scheduler'].update(num_training_steps=64, num_warmup_steps=8, min_lr=1e-6)
    cfg['distributed'].update(use_fsdp=True, use_gradient_checkpointing=True)
    cfg['data'].update(train_test_split=.9, num_workers=2, norm_stats_path=str(ROOT/'norm_stats.json'))
    cfg['data']['lerobot_config'] = {'repo_id': 'x2real/bottle-v3', 'root': str(data)}
    cfg['logging'].update(use_wandb=False, log_name='agentos-bottle-pilot', log_interval=1,
                          save_interval=1000000, val_interval=1000000, epoch_save_interval=1000000)
    cfg['checkpoint'].update(save_path=str(ROOT/'outputs/action-pilot'),
                             resume_from=str(ROOT/'weights/model.safetensors'))
    (ROOT / 'train.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f'CONFIG_READY train={split} heldout={n-split} max_optimizer_steps=64', flush=True)


if __name__ == '__main__':
    main()
