# Running Ternary-Bonsai-2-27B on a single 24 GB RTX 3090: vLLM vs llama.cpp, measured

*A practical, numbers-first field report. All figures below were measured on the machine
described, not estimated. Raw outputs are in [`data/`](data/) and the test scripts in
[`scripts/`](scripts/).*

---

## TL;DR

| | llama.cpp (GGUF, q8_0 KV) | vLLM (docker, bf16 KV) |
|---|---|---|
| max context / request | **262,144** (native) | **114,688** (hard ceiling here) |
| KV capacity | 262k shared pool | 119,553 tokens |
| VRAM | 19.9 GB | 22.5 GB |
| prefill speed | ~1,085 tok/s | ~1,118 tok/s |
| decode, short ctx | 84 tok/s | 84 tok/s |
| decode, 38k ctx, long output | **60.7 tok/s** | **29.7 tok/s** |
| 2 concurrent long requests | 133 s | **39 s** |
| needle recall at depth | ✅ (228k depth tested) | ✅ (95k depth tested) |

**Use llama.cpp for long-context single-user work. Use vLLM for concurrent chat.**
The two are equally fast at short context; they separate only under long context
(KV-cache bandwidth) or concurrency (batching).

---

## Why this exists

Ternary-Bonsai-2-27B (Prism ML's 2-bit ternary Qwen3.8-27B) is a ~27B model that
fits in 24 GB only because its weights are ternary. Serving it at long context is
where things get interesting: KV cache, not weights, becomes the budget.

vLLM support is unofficial and new (`fraserprice/bonsai-vllm`, WIP, "tested on
RTX PRO 6000 Blackwell only"). This report documents what actually works on
Ampere (sm_86), what breaks, and how to get the most context out of 24 GB.

**Hardware:** RTX 3090 24 GB, i7-10700K, 32 GB RAM, Linux.
**Model variants:** `fraserpricee/bonsai-vllm:20260918` (vLLM, `prism_ternary`
quantization) and `Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf` (llama.cpp, with MTP
speculative decoding).

---

## The KV cache math that drives everything

From the engine's own startup log (vLLM, bf16 KV, 32k max-len):

```
Available KV cache memory: 10.45 GiB
GPU KV cache size: 94,391 tokens
Maximum concurrency for 32,768 tokens per request: 2.88x
```

=> **~116 KiB of KV per token.** For a 262k-token context that is **28.6 GiB** —
impossible on a 24 GB card with 16-bit KV. This single fact explains every
trade-off below: context, quantization, and decode speed are all KV-byte problems.

---

## Part 1 — vLLM: what works, what breaks

### Working configuration

```bash
docker run -d --name bonsai-vllm --gpus all --ipc=host -p 8000:8000 \
  -v bonsai:/cache -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  fraserpricee/bonsai-vllm:20260918 \
  --max-model-len 114688 --max-num-seqs 64 --gpu-memory-utilization 0.92 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32]}'
```

Verified limits on this card (bf16 KV):

| max-model-len | KV memory | KV tokens | concurrency @ max len |
|---|---|---|---|
| 32,768 | 10.45 GiB | 94,391 | 2.88x |
| 98,304 | 9.29 GiB | 115,810 | 1.18x |
| 114,688 | 9.29 GiB | 119,553 | 1.04x |

Note the counter-intuitive detail: **bigger max-model-len does not shrink KV
capacity** (it went *up*: 94k → 119k tokens), because vLLM's profiling reserves
less for CUDA graphs at larger sequence limits. 114,688 is the practical ceiling;
beyond it you exceed the KV pool.

### Things that do NOT work (with evidence)

1. **fp8 KV cache (`--kv-cache-dtype fp8_e4m3`)** — starts fine, single requests
   answer correctly (5/7 outputs byte-identical to bf16; the rest diverged only in
   reasoning path, same correct answers). **But under 4+ concurrent requests the
   engine dies with `CUDA error: an illegal memory access`** and force-kills itself.
   Likely cause: fp8 KV layout on hybrid GDN/full-attention layers and/or CUDA-graph
   replay with a recently-added dtype.
2. **turboquant presets (`turboquant_k8v4` etc.)** — startup fails:
   `ValueError: Unknown TurboQuant cache dtype: 'auto'` — hybrid layers request a
   native-dtype cache and the TurboQuant backend raises instead of returning the
   standard shape. Patching that yields a second failure in
   `_reshape_attention_kv_cache` (layout mismatch). Upstream bug, not config.
3. **Blackwell-only testing.** The image is documented as tested on RTX PRO 6000
   only. Ampere surprises (above) are consistent with that.

### Upstream status (checked 2026-10-26)

- Repo `fraserprice/bonsai-vllm`: no commits since 2026-09-18; no releases; one
  published image tag.
- Open PR/issue #1 (2026-10-13): bump base image vLLM 0.25.1 → 0.31.0 (adds
  `cuda-toolkit` + `ninja` for JIT kernels). The author independently reports
  "131k context causes a KV cache mismatch" on a 24 GB RTX 4090 and settles on
  `--max-model-len 65536`. Not merged.

---

## Part 2 — llama.cpp: 262k context on the same card

```bash
llama-server -m models/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf \
  --host 127.0.0.1 --port 8080 -ngl 99 -fa on \
  -c 262144 --parallel 2 --kv-unified \
  -ctk q8_0 -ctv q8_0 --jinja \
  --spec-type draft-mtp --spec-draft-n-max 2
```

Result: 2 slots × **262,144** (the model's native max, from GGUF
`qwen35.context_length`), shared KV pool, **19,959 MiB VRAM**, needle-in-haystack
recall verified at **228,100 tokens** depth.

### The two llama.cpp gotchas worth knowing

**`-c` is the TOTAL pool, not per request.** With `-c 32768 --parallel 8`, every
chat gets **4,096** tokens — the startup log says it plainly:
`n_slots = 8, n_ctx_slot = 4096`. `--kv-unified` changes the allocation model:
one shared buffer, so a single request may use the entire pool (and concurrent
requests share it dynamically, first-come-first-served).

**KV dtype is f16 unless you say otherwise.** No `-ctk/-ctv` flags = 116 KiB/token,
identical cost to vLLM bf16. `-ctk q8_0 -ctv q8_0` halves it with no measurable
quality loss in our tests, and — importantly — halves the memory traffic per
decoded token, which is why llama.cpp wins long-context decode.

---

## Part 3 — Apples-to-apples benchmark

Method: identical prompts, greedy (`temperature=0`), `max_tokens` fixed,
**thinking disabled on both engines**:

```json
"chat_template_kwargs": {"enable_thinking": false}
```

(`/no_think` as a prompt prefix does **not** work on the llama.cpp build — it
silently keeps generating hidden thinking tokens. The template kwarg works on both.)

| test | llama.cpp | vLLM |
|---|---|---|
| "capital of France" | 8 tok, TTFT 0.11 s | 8 tok, TTFT 0.09 s |
| ~400-token answer | 414 tok, 5.2 s, 84.1 tok/s | 421 tok, 5.1 s, 83.6 tok/s |
| 38k prompt + long answer | TTFT 35.1 s, 558 tok, **60.7 tok/s** | TTFT 34.1 s, 634 tok, **29.7 tok/s** |
| needle at end | ✅ | ✅ |

**A measurement trap we fell into:** the first long-context vLLM run reported
282 tok/s. It was an artifact — 45 tokens delivered in one 0.16 s burst after a
long TTFT. Token rates are only meaningful with **hundreds** of generated tokens.
With real long outputs, vLLM is ~2× slower than llama.cpp at long context —
exactly what you expect when each decoded token must read 2× the KV bytes.

---

## Decision guide

- **Long documents, single user, max context** → llama.cpp (262k, 20 GB, faster at depth).
- **Several users / chat UI / batching** → vLLM (2 concurrent 38k requests in 39 s vs 133 s).
- **Short-context throughput** → either; they are within 1% of each other.
- **Do not** enable fp8 KV under concurrency on Ampere today, and skip turboquant
  presets until the upstream fix lands.

---

## Reproduce it

- `scripts/ab_kv_test.py` — bf16 vs fp8 KV A/B (outputs + TTFT + t/s + load test)
- `scripts/test_vllm_max.py` — vLLM max-context needle + concurrency
- `scripts/test_max_ctx.py` — llama.cpp max-context needle (228k depth)
- `scripts/a2a_test.py` — apples-to-apples harness (`vllm` | `llama`)
- `data/` — raw JSON results, `RESULTS.md` lab notebook, launch script

---

## Credits

This work builds entirely on other people's work; full attribution with links is in
[`CREDITS.md`](CREDITS.md). In short:

- **Model:** Ternary-Bonsai-2-27B by [Prism ML](https://prismml.com/) — ternary
  (~2-bit) weights, based on the **Qwen3.8-27B** architecture from the Qwen team
  (Alibaba). The BF16 **MTP head** used for speculative decoding comes from the same
  Qwen3.8-27B release; every drafted token is verified by Bonsai.
- **vLLM path:** the `prism_ternary` plugin, custom CUDA kernels, model converter and
  docker image are by **fraserprice** ([fraserprice/bonsai-vllm](https://github.com/fraserprice/bonsai-vllm),
  unofficial and WIP, not affiliated with Prism ML), built on
  **[vLLM](https://github.com/vllm-project/vllm)** and its
  **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** attention backend.
- **llama.cpp path:** **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** —
  `llama-server`, KV-cache quantization and MTP speculative decoding; GGUF packaging
  credited to **ProCreations** as noted in the launch script.
- **Measurement:** dependency-free Python stdlib harnesses, Docker + NVIDIA Container
  Toolkit, Hugging Face Hub for weights and caching.
- **Hardware:** RTX 3090 (24 GB, sm_86) — all numbers in this report were measured
  there, not on the Blackwell card the upstream image was validated on.

If you reuse this material, keep these credits and link back to the authors. Bugs in
the vLLM plugin belong upstream in `fraserprice/bonsai-vllm`.

