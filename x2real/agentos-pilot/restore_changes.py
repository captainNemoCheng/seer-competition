"""Restore exact full weights from a changed-tensor backup and its pinned base.

Requires enough CPU memory and disk for the full model (about 18 GB each).
Copy the checkpoint configuration/tokenizer files beside the output weight file.
The optimizer state is a separate optional training-resume artifact.
"""
import argparse, hashlib, json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
parser=argparse.ArgumentParser()
parser.add_argument('--base',type=Path,required=True)
parser.add_argument('--changes',type=Path,required=True)
parser.add_argument('--manifest',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
manifest=json.loads(args.manifest.read_text())
hasher=hashlib.sha256()
with args.base.open('rb') as stream:
 for block in iter(lambda: stream.read(8*1024*1024),b''):
  hasher.update(block)
digest=hasher.hexdigest()
if digest!=manifest['base_sha256']:
 raise ValueError('Base checkpoint hash does not match the pinned source')
if args.output.exists():
 raise FileExistsError(args.output)
tensors={}
with safe_open(str(args.base),framework='pt') as base, safe_open(str(args.changes),framework='pt') as delta:
 changed=set(delta.keys())
 for name,schema in manifest['tensor_schema'].items():
  tensor=delta.get_tensor(name) if name in changed else base.get_tensor(name).to(getattr(torch,schema['dtype'].split('.')[-1]))
  if list(tensor.shape)!=schema['shape'] or str(tensor.dtype)!=schema['dtype']:
   raise ValueError('Tensor schema mismatch: '+name)
  tensors[name]=tensor
 args.output.parent.mkdir(parents=True,exist_ok=True)
 save_file(tensors,str(args.output))
print('RESTORED',args.output)
