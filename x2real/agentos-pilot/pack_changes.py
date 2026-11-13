"""Lossless incremental weight backup against the pinned public base model."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
ROOT=Path('/workspace/x2real-training')
torch.set_num_threads(2)
checkpoint=Path(json.loads((ROOT/'TRAIN_COMPLETE.json').read_text())['checkpoint'])
output=ROOT/'outputs/action-changes';output.mkdir(exist_ok=True)
changed={};schema={};unchanged=0
with safe_open(str(ROOT/'weights/model.safetensors'),framework='pt') as base, safe_open(str(checkpoint/'model.safetensors'),framework='pt') as trained:
    basekeys=set(base.keys())
    for name in trained.keys():
        tensor=trained.get_tensor(name)
        schema[name]={'dtype':str(tensor.dtype),'shape':list(tensor.shape)}
        same=False
        if name in basekeys:
            original=base.get_tensor(name)
            same=original.shape==tensor.shape and torch.equal(original.to(tensor.dtype),tensor)
        if same: unchanged+=tensor.numel()
        else: changed[name]=tensor.clone()
save_file(changed,str(output/'changed.safetensors'))
manifest={'format':'exact_changed_tensors_v1','base_repo':'x-square-robot/wall-oss-0.5','base_revision':'b7dc23a82dbf70a65859c9fdd596703234597028','base_sha256':'37f8f327d3f9c27f056ed078db43c0e39b8caf3df0dc0163cdb8cf92c167968a','changed_parameters':sum(t.numel() for t in changed.values()),'unchanged_parameters':unchanged,'tensor_schema':schema,'restore':'For each tensor_schema entry, take changed.safetensors if present; otherwise take the pinned base tensor cast to the recorded dtype. No quantization or approximation is used.'}
(output/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('LOSSLESS_CHANGES_READY',manifest['changed_parameters'],manifest['unchanged_parameters'],flush=True)
