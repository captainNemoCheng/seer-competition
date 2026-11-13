#!/usr/bin/env bash
set -euo pipefail
cd /workspace/x2real-training
export WALLX_ROOT="$PWD/wall-x"
export WALLX_BASE_DEPLOY="$PWD/policy-space/policies/wallx/wall_oss_05_artixon_arm_6a_eef/deploy.yaml"
export WALLX_CHECKPOINT_PATH
WALLX_CHECKPOINT_PATH="$(.venv/bin/python -c 'import json; print(json.load(open("TRAIN_COMPLETE.json"))["checkpoint"])')"
export WALLX_NORM_KEY=x2real/bottle-v3 WALLX_BACKEND=standard
export AGENTOS_WORLD_CHECKPOINT="$PWD/outputs/world-pilot/dynamics.pt"
export PYTHONPATH="$PWD:$WALLX_ROOT:$PWD/policy-space/src:$PWD/policy-space/packages/protocol/src"
export OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
case "${1:-check}" in
  check)
    .venv/bin/python -m policy_space.cli check "$WALLX_BASE_DEPLOY"
    .venv/bin/python check_real_policy.py
    ;;
  serve)
    exec .venv/bin/python secure_serve.py
    ;;
  real-check)
    exec .venv/bin/python check_real_policy.py
    ;;
  *) echo 'Use check, real-check or serve' >&2; exit 2 ;;
esac
