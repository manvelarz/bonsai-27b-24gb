#!/usr/bin/env bash
set -euo pipefail
uv venv --python 3.12 .venv
source .venv/bin/activate

git clone --branch dflash-quantized-context-kv https://github.com/akmorley/vllm-prism.git
git -C vllm-prism checkout d463094f1d97d95ac6c9596b92d3b7ce677f6fb9
git clone --branch pq2-bundled-clean https://github.com/akmorley/vllm-gguf-plugin-prism.git
git -C vllm-gguf-plugin-prism checkout 7520fd9bf2c51ff760096bc1f4cee40c79100e16
git clone https://github.com/PrismML-Eng/llama.cpp.git llama

export TORCH_CUDA_ARCH_LIST=8.6
export VLLM_FA_CMAKE_GPU_ARCHES=86-real
export MAX_JOBS="${MAX_JOBS:-8}"
uv pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install -r vllm-prism/requirements/build/cuda.txt
uv pip install --no-build-isolation ./vllm-prism
uv pip install --no-build-isolation ./vllm-gguf-plugin-prism
uv pip install --no-deps ./llama/gguf-py
