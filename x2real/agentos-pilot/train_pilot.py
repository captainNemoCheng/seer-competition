"""Bounded fine-tune with an episode holdout and an explicit final checkpoint.

Upstream creates a random frame validation split inside its selected train episodes.
Replace those subsets with the separate held-out episodes before dataloader creation.
"""
import json
import logging
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import wall_x.data.backends.lerobot.loader as loader

ROOT = Path('/workspace/x2real-training')
original_load = loader.load_lerobot_data


def episode_split_loader(config, lerobot_config, **kwargs):
    dataset, train_num = original_load(config, lerobot_config, **kwargs)
    train = dataset.hf_dataset
    total = train.meta.total_episodes
    split = int(total * .9)
    heldout = LeRobotDataset(repo_id=lerobot_config['repo_id'], root=lerobot_config['root'],
                            episodes=list(range(split, total)),
                            delta_timestamps={'action': [t / 20 for t in range(32)]},
                            video_backend='pyav')
    # Fixed, evenly spaced held-out frames keep pilot validation bounded.
    indices = np.linspace(0, len(heldout)-1, 32, dtype=int).tolist()
    dataset.train_dataset = train
    dataset.val_dataset = Subset(heldout, indices)
    dataset._train()
    print(f'EPISODE_SPLIT train={len(train.episodes)} test={len(heldout.episodes)} validation_frames={len(indices)}', flush=True)
    return dataset, train_num


loader.load_lerobot_data = episode_split_loader
from wall_x.config.loader import load_config
from wall_x.trainer.fsdp_trainer import FSDPTrainer, cleanup_distributed


def main():
    logging.basicConfig(level=logging.INFO)
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', '0')))
    torch.distributed.init_process_group('nccl', device_id=torch.device('cuda:0'))
    trainer = FSDPTrainer(load_config(str(ROOT/'train.yaml')))
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f'TRAINABLE_PARAMETERS={trainable}', flush=True)
    try:
        # Same heldout frames and RNG for the before/after diagnostic.
        torch.manual_seed(4242)
        trainer.val_loop()
        trainer.fit()
        completed = trainer.global_step
        # Upstream saves zero-based step indices and adds one when resuming.
        # fit() has already incremented the counter after the final update.
        trainer.global_step = completed - 1
        trainer.save_checkpoint(0, trainer.global_step)
        trainer.global_step = completed
        torch.manual_seed(4242)
        trainer.val_loop()
        result = {'optimizer_steps': trainer.global_step,
                  'checkpoint': str(ROOT/f'outputs/action-pilot/0_{completed-1}'),
                  'task_success_rate': None, 'platform_submitted': False}
        (ROOT/'TRAIN_COMPLETE.json').write_text(json.dumps(result, indent=2))
        import subprocess
        import sys
        subprocess.run([sys.executable,str(ROOT/'finalize_checkpoint.py')],check=True)
        print('TRAIN_COMPLETE', json.dumps(result), flush=True)
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
