"""Repair export metadata without changing trained tensors.

The typed upstream checkpoint writer omits the raw data.max_length setting,
causing inference to fall back to 768 and truncate image tokens for square views.
"""
import json
from pathlib import Path
import torch
import yaml

ROOT=Path('/workspace/x2real-training')
result=json.loads((ROOT/'TRAIN_COMPLETE.json').read_text())
checkpoint=Path(result['checkpoint'])
config_path=checkpoint/'config.yml'
config=yaml.load(config_path.read_text(), Loader=yaml.FullLoader)
old=config.get('data',{}).get('max_length')
config.setdefault('data',{})['max_length']=2048
config_path.write_text(yaml.safe_dump(config,sort_keys=False))
counter_path=checkpoint/'global_step.pth'
counter=torch.load(counter_path,weights_only=True)
old_counter=counter['global_step']
# The first pilot was started before the runner's zero-based export correction.
if old_counter==result['optimizer_steps']:
    torch.save({'global_step':result['optimizer_steps']-1},counter_path)
note={'trained_tensors_modified':False,'completed_optimizer_steps':result['optimizer_steps'],
      'exported_zero_based_step':result['optimizer_steps']-1,
      'inference_max_length':2048,
      'reason':'Preserve all image tokens and match upstream resume counter semantics'}
(checkpoint/'export-notes.json').write_text(json.dumps(note,indent=2))
print('CHECKPOINT_METADATA_READY',json.dumps(note))
