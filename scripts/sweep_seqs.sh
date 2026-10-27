#!/bin/bash
# Measure how --max-num-seqs and --max-num-batched-tokens change the KV pool.
# Boot-only sweep: launch, read the engine's own "GPU KV cache size" line, kill.
set -u
IMG=fraserpricee/bonsai-vllm:20260918
OUT=ab_tests/sweep_seqs.txt
: > "$OUT"

run_cfg() {                       # $1=label  $2..=extra args
  local label="$1"; shift
  sudo -n docker rm -f bonsai-vllm >/dev/null 2>&1
  sleep 2
  if ! sudo -n docker run -d --gpus all --ipc=host -p 8000:8000 -v bonsai:/cache \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --name bonsai-vllm "$IMG" \
      --max-model-len 32768 --max-num-seqs 64 --gpu-memory-utilization 0.92 \
      --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32]}' "$@" >/dev/null 2>&1; then
    echo "$label | docker run FAILED" >> "$OUT"; return
  fi
  for i in $(seq 1 26); do
    sleep 10
    curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
    st=$(sudo -n docker inspect -f '{{.State.Status}}' bonsai-vllm 2>/dev/null)
    [ "$st" = "exited" ] && break
  done
  local kv avail conc
  avail=$(sudo -n docker logs bonsai-vllm 2>&1 | grep -o "Available KV cache memory: .*" | tail -1)
  kv=$(sudo -n docker logs bonsai-vllm 2>&1 | grep -o "GPU KV cache size: [0-9,]* tokens" | tail -1)
  conc=$(sudo -n docker logs bonsai-vllm 2>&1 | grep -o "Maximum concurrency for 32,768 tokens per request: .*" | tail -1)
  echo "$label | ${avail:-no-kv-line} | ${kv:-none} | ${conc:-none}" >> "$OUT"
  echo "$label done: ${kv:-none}"
}

run_cfg "A seqs64 batch8192 (baseline)"
run_cfg "B seqs8  batch8192"   --max-num-seqs 8
run_cfg "C seqs4  batch8192"   --max-num-seqs 4
run_cfg "D seqs8  batch4096"   --max-num-seqs 8 --max-num-batched-tokens 4096
run_cfg "E seqs64 batch4096"   --max-num-batched-tokens 4096
run_cfg "F seqs16 batch2048"   --max-num-seqs 16 --max-num-batched-tokens 2048
echo "SWEEP COMPLETE" >> "$OUT"
