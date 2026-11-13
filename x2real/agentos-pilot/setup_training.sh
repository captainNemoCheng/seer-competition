#!/usr/bin/env bash
set -euo pipefail
root=/workspace/x2real-training
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export UV_NO_CACHE=1 PIP_NO_CACHE_DIR=1
export HF_HOME="$root/hf" HF_HUB_DISABLE_XET=1
export MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.6 FLASH_ATTN_CUDA_ARCHS=86
mkdir -p "$root"
cd "$root"
nvcc --version
if [[ ! -d wall-x/.git ]]; then
  git init -q wall-x
  git -C wall-x remote add origin https://github.com/X-Square-Robot/wall-x.git
  git -C wall-x fetch --depth 1 origin ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f
  git -C wall-x checkout --detach FETCH_HEAD
fi
if [[ ! -d policy-space/.git ]]; then
  git init -q policy-space
  git -C policy-space remote add origin https://github.com/X-Square-Robot/policy-space.git
  git -C policy-space fetch --depth 1 origin 6825ed0526692803c5bf9ac450780c27dcfb9d7b
  git -C policy-space checkout --detach FETCH_HEAD
fi
[[ -x .venv/bin/python ]] || uv venv --python 3.10 .venv
py="$root/.venv/bin/python"
uv pip install --python "$py" --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
uv pip install --python "$py" -r wall-x/requirements.txt setuptools wheel pandas 'av>=14,<16' 'draccus==0.10.0' 'deepdiff>=7,<9' 'jsonlines>=4,<5' termcolor 'diffusers>=0.27,<0.36'
uv pip install --python "$py" --no-deps lerobot==0.4.4
# Download the pinned public model while compiling the CUDA extensions.
"$py" -u - <<'PY' > weights-download.log 2>&1 &
from huggingface_hub import snapshot_download
snapshot_download('x-square-robot/wall-oss-0.5', revision='b7dc23a82dbf70a65859c9fdd596703234597028', local_dir='weights')
print('WEIGHTS_READY', flush=True)
PY
weights_pid=$!
(
  if [[ ! -f data/bottle-v3/source.json ]]; then
    "$py" -u prepare_bottle_data.py data/bottle-v3 --cache data/source-cache --episodes 300
  else
    "$py" -c 'import json; s=json.load(open("data/bottle-v3/source.json")); assert s["revision"] == "5a741385f9f196073cfe163b0f25033025e383a6" and len(s["source_episode_indices"]) == 300'
  fi
) > dataset-download.log 2>&1 &
data_pid=$!
uv pip install --python "$py" --no-build-isolation flash-attn==2.8.3
uv pip install --python "$py" --no-build-isolation --no-deps -e wall-x
uv pip install --python "$py" --no-deps -e policy-space
wait "$weights_pid"
wait "$data_pid"
"$py" -c 'import torch, flash_attn, lerobot, wall_x; from lerobot.datasets.lerobot_dataset import LeRobotDataset; print("ENV_READY", torch.__version__, torch.cuda.get_device_name())'
touch SETUP_COMPLETE
echo SETUP_COMPLETE
