#!/usr/bin/env bash
set -euo pipefail
cd /workspace/x2real-training
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export UV_NO_CACHE=1 PIP_NO_CACHE_DIR=1
export HF_HOME="$PWD/hf" HF_HUB_DISABLE_XET=1
export MAX_JOBS=8 TORCH_CUDA_ARCH_LIST=12.0
checkout() {
  if [[ ! -d "$3/.git" ]]; then
    git init -q "$3"
    git -C "$3" remote add origin "$1"
    git -C "$3" fetch --depth 1 origin "$2"
    git -C "$3" checkout --detach FETCH_HEAD
  fi
}
checkout https://github.com/X-Square-Robot/wall-x.git ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f wall-x
checkout https://github.com/X-Square-Robot/policy-space.git 6825ed0526692803c5bf9ac450780c27dcfb9d7b policy-space
# Inference migration uses the image's Blackwell-enabled torch 2.8, not a new training run.
[[ -x .venv/bin/python ]] || uv venv --python /usr/local/bin/python --system-site-packages .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
.venv/bin/python -c 'import torch; assert torch.__version__.startswith("2.8.0"); assert torch.cuda.get_device_capability()==(12,0); assert torch._C._GLIBCXX_USE_CXX11_ABI; print("BLACKWELL_TORCH_READY",torch.__version__)'
sed '/^torch==/d; /^torchvision==/d; /^torchaudio==/d' wall-x/requirements.txt > requirements-inference.txt
printf 'torch==2.8.0+cu128\ntorchvision==0.23.0+cu128\ntorchaudio==2.8.0+cu128\n' > constraints-inference.txt
uv pip install --python .venv/bin/python -c constraints-inference.txt -r requirements-inference.txt setuptools wheel einops==0.8.2 pandas 'av>=14,<16' 'draccus==0.10.0' 'deepdiff>=7,<9' 'jsonlines>=4,<5' termcolor 'diffusers>=0.27,<0.36'
uv pip install --python .venv/bin/python --no-deps lerobot==0.4.4
.venv/bin/python -u - <<'PY' > weights-download.log 2>&1 &
from huggingface_hub import snapshot_download
snapshot_download('x-square-robot/wall-oss-0.5',revision='b7dc23a82dbf70a65859c9fdd596703234597028',local_dir='weights')
print('WEIGHTS_READY',flush=True)
PY
weights_pid=$!
wheel='flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
curl -fL --retry 3 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl' -o "$wheel"
echo "61ab39b7038d6a05cb97664f3481a92ced300d11aa14529ae3b675300a99f5e6  $wheel" | sha256sum -c -
uv pip install --python .venv/bin/python --no-deps "$wheel"
uv pip install --python .venv/bin/python --no-build-isolation --no-deps -e wall-x
uv pip install --python .venv/bin/python --no-deps -e policy-space
.venv/bin/python -u - <<'PY'
import torch
from flash_attn import flash_attn_func
for d in (80,128):
 q=torch.randn(1,17,4,d,device='cuda',dtype=torch.bfloat16)
 out=flash_attn_func(q,q,q,causal=False)
 ref=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),q.transpose(1,2),q.transpose(1,2)).transpose(1,2)
 assert torch.isfinite(out).all()
 torch.testing.assert_close(out,ref,atol=.03,rtol=.03)
 print('FLASH_SM120_OK',d,flush=True)
PY
wait "$weights_pid"
uv pip freeze --python .venv/bin/python > outputs/environment-freeze.txt
touch SETUP_COMPLETE
printf 'INFERENCE_SETUP_COMPLETE\n'
