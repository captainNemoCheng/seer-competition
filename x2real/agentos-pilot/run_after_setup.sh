#!/usr/bin/env bash
set -euo pipefail
cd /workspace/x2real-training
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/workspace/x2real-training/wall-x:/workspace/x2real-training:/workspace/x2real-training/policy-space/src:/workspace/x2real-training/policy-space/packages/protocol/src"
export WANDB_MODE=disabled
setup_pid="${1:?Pass the running setup PID}"
while [[ ! -f SETUP_COMPLETE ]]; do
  if ! kill -0 "$setup_pid" 2>/dev/null; then
    echo 'Setup exited without SETUP_COMPLETE; refusing to start training.'
    exit 1
  fi
  sleep 10
done
.venv/bin/python prepare_training.py
.venv/bin/python check_pilot.py
timeout --signal=TERM --kill-after=30s 45m .venv/bin/torchrun --standalone --nproc_per_node=1 train_pilot.py
