# Credits

This work stands on other people's work. Full attribution, with links:

## Model
- **Ternary-Bonsai-2-27B** — [Prism ML](https://prismml.com/) ([model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit)).
  Ternary-weight (~2-bit) LLM, based on Qwen3.8-27B, released in the MLX 2-bit pack.
- **Qwen3.8-27B** — [Qwen team, Alibaba](https://huggingface.co/Qwen/Qwen3.8-27B).
  Upstream architecture; also the source of the BF16 **MTP head** used for speculative
  decoding in the vLLM repack (every drafted token is verified by Bonsai itself).

## vLLM path
- **fraserprice** — [fraserprice/bonsai-vllm](https://github.com/fraserprice/bonsai-vllm):
  the `prism_ternary` vLLM plugin, custom CUDA kernels, model converter
  (`prism-ternary-convert`), docker image `fraserpricee/bonsai-vllm`, and the
  `fraserprice/Ternary-Bonsai-2-27B-vllm` HF repack. Unofficial, WIP, not affiliated
  with Prism ML. Bug reports from this work belong in that repo's issues.
- **vLLM project** — [vllm-project/vllm](https://github.com/vllm-project/vllm)
  ([docs](https://docs.vllm.ai)): serving engine, PagedAttention, continuous
  batching, CUDA graphs, prefix caching, FlashInfer attention backend integration.
- **FlashInfer** — [flashinfer-ai](https://github.com/flashinfer-ai/flashinfer):
  attention kernels used for the fp8-KV experiments.

## GGUF-native vLLM path (Part 10)
- **Isotr0py / vLLM project** — [vllm-project/vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin)
  (Apache-2.0, maintained by Isotr0py, a vLLM contributor). Out-of-tree GGUF
  quantization plugin written after in-tree GGUF support was deprecated
  ([vllm#39583](https://github.com/vllm-project/vllm/issues/39583)). The Triton
  dequantize backend and the plugin scaffolding are this project's work — by line
  count, roughly two thirds of the `triton/` tree.
- **Anthony Morley** — [akmorley/vllm-gguf-plugin-prism](https://github.com/akmorley/vllm-gguf-plugin-prism)
  and [akmorley/vllm-prism](https://github.com/akmorley/vllm-prism). His additions on top of
  the plugin are the ternary-specific path: PQ2 decode kernels, per-architecture kernel
  tables, INT8 tensor-core prefill, split-KV INT8 per-token-head attention, and the
  Triton/Gluon ports (16 commits, Oct 2026). In the vLLM fork he carries 7 commits
  (`qwen3_dflash.py`, `qwen3_5_mtp.py`, per-token-head Triton attention, KV-cache group size).
  His int8-KV kernel states in its own docstring that it "follows llama.cpp's quantised-KV
  vector kernel (`fattn-vec`)"; the Q8_1/Q8 paths explicitly disclaim bit-exact equivalence
  with llama.cpp.

## llama.cpp path
- **ggml-org / llama.cpp** — [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp):
  `llama-server`, OpenAI-compatible API, `--kv-unified`, KV-cache quantization
  (`-ctk/-ctv q8_0`), speculative decoding (`--spec-type draft-mtp`).
- **ProCreations** — the `PQ2_0 + MTP` GGUF packaging referenced by the launch script
  (`Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf`). Credit as stated in the local
  packaging notes; adjust the link here if the author publishes a repo.

## Tooling used for measurement
- **Python stdlib** (`urllib`, `json`, `concurrent.futures`) — all benchmark
  harnesses are dependency-free on purpose, so they run anywhere.
- **Docker** + **NVIDIA Container Toolkit** — vLLM image runtime.
- **Hugging Face Hub** — model weights and caching.

## Community
- **u/Major_Border149** (r/LocalLLM) — a comment there is where this line of enquiry started.
- **u/live4evrr** (r/LocalLLaMA) — vLLM benchmark post; source of the flag combination used here
  (fp8 KV + MTP speculative decoding + `qwen3_coder` parser at 262,144 context).
- **u/Fancy-Snow7**, **u/Creative-Regular6799** (r/LocalLLaMA) — independent quality and
  Terminal-Bench results on the same quant.

## Hardware / platform
- **NVIDIA RTX 3090 (24 GB, sm_86)** — the card every number here is measured on.
  Note: upstream documents the image as tested on RTX PRO 6000 (Blackwell) only;
  Ampere results and failure modes in this report are our own.
- Linux (Mint), Intel i7-10700K, 32 GB RAM.

## If you reuse this
Keep this file and the article's credits section intact, and link back to the
authors above. If you find a bug in the vLLM plugin, report it upstream to
`fraserprice/bonsai-vllm` rather than assuming the model or vLLM itself is at fault.
