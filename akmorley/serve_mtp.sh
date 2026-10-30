#!/usr/bin/env bash
set -euo pipefail
model_dir="${MODEL_DIR:-$PWD/models}"
export GGUF_PQ2_KERNELS="${GGUF_PQ2_KERNELS:-tuned}"
export GGUF_PQ2_INT_GEMV=1 GGUF_PQ2_PREPARED=1 GGUF_PQ2_MMQ=1
args=(
  serve "${MODEL_FILE:-$model_dir/Ternary-Bonsai-2-27B-PQ2_0.gguf}"
  --tokenizer "$model_dir/qwen38-tokenizer"
  --served-model-name prism-27b
  --host "${SERVER_HOST:-127.0.0.1}" --port 8000
  --dtype bfloat16 --language-model-only
  --mamba-ssm-cache-dtype "${SSM_CACHE_DTYPE:-float32}"
  --attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head
  --max-model-len -1 --gpu-memory-utilization 0.92
  --max-num-seqs 4 --max-num-batched-tokens 2048
  --reasoning-parser qwen3
)
if [[ "${USE_DFLASH:-0}" == "1" ]]; then
  args+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$model_dir/bonsai-dflash2\",\"num_speculative_tokens\":5,\"quantization\":\"fp8\"}")
fi
if [[ -n "${MTP_DRAFT:-}" ]]; then
  args+=(--speculative-config "{\"method\":\"mtp\",\"model\":\"$MTP_DRAFT\",\"num_speculative_tokens\":${MTP_N:-3}}")
elif [[ -n "${MTP_N:-}" ]]; then
  args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_N}}")
fi
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SELF_DIR/.venv/bin/python" -c "from vllm.entrypoints.cli.main import main; main()" "${args[@]}" "$@"
