#!/usr/bin/env bash
set -euo pipefail
cd /workspace/x2real-training
end=$((SECONDS+2400))
until [[ -f SETUP_COMPLETE ]]; do
  if (( SECONDS>=end )); then echo 'Setup did not complete within 40 minutes'; exit 1; fi
  sleep 10
done
.venv/bin/python -u - <<'PY'
from pathlib import Path
import hashlib,json,subprocess
p=Path('backup'); expected={Path(n).name:h for h,n in (x.split() for x in (p/'SHA256SUMS').read_text().splitlines())}
f=p/'changed.safetensors.zst';h=hashlib.sha256()
with f.open('rb') as stream:
 for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
assert h.hexdigest()==expected[f.name]
subprocess.run(['zstd','-d','-f',str(f),'-o',str(p/'changed.safetensors')],check=True)
h=hashlib.sha256()
with (p/'changed.safetensors').open('rb') as stream:
 for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
assert h.hexdigest()==expected['changed.safetensors']
print('BACKUP_HASHES_VERIFIED',flush=True)
PY
if [[ ! -f outputs/action-pilot/0_64/model.safetensors ]]; then
 .venv/bin/python -u restore_changes.py --base weights/model.safetensors --changes backup/changed.safetensors --manifest backup/manifest.json --output outputs/action-pilot/0_64/model.safetensors
fi
.venv/bin/python -u - <<'PY'
from pathlib import Path
import json,torch
from safetensors import safe_open
p=Path('backup');m=json.loads((p/'manifest.json').read_text());torch.set_num_threads(2)
with safe_open(str(p/'changed.safetensors'),framework='pt') as delta,safe_open('weights/model.safetensors',framework='pt') as base,safe_open('outputs/action-pilot/0_64/model.safetensors',framework='pt') as full:
 changed=set(delta.keys())
 for name,s in m['tensor_schema'].items():
  restored=delta.get_tensor(name) if name in changed else base.get_tensor(name).to(getattr(torch,s['dtype'].split('.')[-1]))
  assert torch.equal(restored,full.get_tensor(name)),name
report={'restored_tensors':len(m['tensor_schema']),'exact_match':True,'new_training_steps':0,'source_training_steps':64,'optimizer_restored':False}
Path('outputs/migration-verification.json').write_text(json.dumps(report,indent=2))
print('RESTORED_WEIGHTS_VERIFIED',json.dumps(report),flush=True)
PY
bash run_policy.sh check
touch MIGRATION_CHECK_COMPLETE
printf 'MIGRATION_CHECK_COMPLETE\n'
