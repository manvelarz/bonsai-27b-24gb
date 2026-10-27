# Running Ternary-Bonsai-2-27B on a single 24 GB RTX 3090: vLLM vs llama.cpp, measured

*A practical, numbers-first field report. All figures below were measured on the machine
described, not estimated. Raw outputs are in [`data/`](data/) and the test scripts in
[`scripts/`](scripts/).*

---

## TL;DR

| | llama.cpp (GGUF, q8_0 KV) | vLLM (bf16 KV) | vLLM **final** (fp8 KV + MTP n=3) |
|---|---|---|---|
| max context / request | **262,144** (native) | 114,688 (bf16 KV limit) | **262,144** (native) |
| KV capacity | 262k shared pool | 119,553 tokens | 262k–294k tokens (1.00–1.12×, boot variance) |
| VRAM | 19.9 GB | 22.5 GB | 23.5 GB |
| prefill speed | ~1,085 tok/s | ~1,118 tok/s | **~1,227 tok/s** (218k prompt in 178 s) |
| decode, short ctx | 84 tok/s | 84 tok/s | **83.3 tok/s** |
| decode, 38k ctx, long output | 60.7 tok/s | 29.7 tok/s | **67.0 tok/s** |
| 2 concurrent long requests | 133 s (2×40k) | **39 s** (2×38k) | 80.3 s (2×57k) |
| 4 concurrent short streams | — | 167 tok/s agg | **253.8 tok/s agg** |
| needle recall at depth | ✅ 228k | ✅ 95k | ✅ **218,559 tokens** |

> **Update 2026-10-27 — the "vLLM can't do 8-bit KV" finding below is obsolete.**
> The fp8 crash had nothing to do with fp8 or CUDA graphs: it was the *draft depth*
> of the image's baked-in MTP speculative decoding (7 tokens). At
> `num_speculative_tokens: 3`, fp8 KV + full CUDA graphs is stable, and the KV pool
> grows to the native 262,144 tokens at unchanged speed. See **Part 4**.
> The original bf16-only analysis is kept verbatim as a record of the path.

**Use llama.cpp for long-context single-user work on 20 GB of VRAM. Use vLLM (final
config) when you want the same 262k context *plus* batching — it is now the better of
the two on every axis except 1 GB of VRAM.** The engines are equally fast at short
context; they separate only under long context (KV-cache bandwidth) or concurrency.

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

*(Update 2026-10-27: with 8-bit KV the same 262k context costs ~9.7 GiB — the pool now
holds exactly one full-length request. See Part 4.)*

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
less for CUDA graphs at larger sequence limits. 114,688 is the practical ceiling **for
16-bit KV only** — with fp8 KV the same card reaches the full native 262,144 (Part 4), so
this limit is a property of the KV dtype, not of vLLM or the GPU. Beyond it you exceed
the KV pool.

### Things that do NOT work (with evidence)

1. **fp8 KV cache (`--kv-cache-dtype fp8_e4m3`) with the image's default MTP depth
   (`num_speculative_tokens: 7`)** — starts fine, single requests answer correctly
   (5/7 outputs byte-identical to bf16; the rest diverged only in reasoning path,
   same correct answers). **But under 4+ concurrent requests the engine dies with
   `CUDA error: an illegal memory access`** and force-kills itself. The cause was
   later localized to the speculative-decoding **draft depth**, not to fp8 or to
   CUDA graphs: at `num_speculative_tokens: 3` the same fp8 configuration runs
   stably under 4-way load — see **Part 4**. (Independent confirmations that it was
   not a graph-capture fluke: the crash reproduced with capture sizes `[1,2,4]`,
   with a 512 MB FlashInfer workspace, and with chunked prefill unchanged; x2 never
   crashed, x4 always did.)
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

## Part 4 — 2026-10-27: 8-bit KV *does* work on Ampere, and vLLM reaches 262k

The question that started this day: *"vLLM only runs these weights at 16-bit — can we
get 8-bit KV or some smarter way to buy context in 24 GB?"* Answer: the 16-bit part
was never the weights, and the fp8 blocker was not fp8.

### 4.1 What the engine is actually running (from its own boot log)

```
Checkpoint size: 9.54 GiB                      # 27B params => ~2.8 bit/param
quantization=prism_ternary  bits=2  group_size=128  hadamard_block=1024
dtype=torch.bfloat16  kv_cache_dtype=auto
WARNING marlin_utils_fp8.py:112 Your GPU does not have native support for FP8
        computation ... Weight-only FP8 compression will be used (Marlin)
```

Weights are already 2-bit ternary; the only 16-bit component is **activations + KV
cache**. On sm_86 there is no native FP8 *compute*, only weight-only FP8 — worth
knowing before chasing lower-precision weights.

### 4.2 The retest matrix (fresh container per row, 400-token greedy load test)

| KV dtype / graphs | KV pool @32k | solo tok/s | x2 agg | x4 agg | 4 parallel |
|---|---|---|---|---|---|
| bf16 + graphs | 94,391 | 83.1 | 129.8 | 167.1 | OK |
| fp8 + graphs (MTP n=7) | 135,952 | 75.1 | 108.0 | — | **IMA, engine dead** |
| fp8 + graphs, capture `[1,2,4]` | 121,311 | 39.0 | 61.3 | — | **IMA, engine dead** |
| fp8 + graphs, 512 MB FI workspace | 135,952 | 74.5 | 114.9 | — | **IMA, engine dead** |
| fp8 + `--enforce-eager` | 138,043 | 27.5 | 42.8 | 100.3 | OK |
| fp8 + compile, `cudagraph_mode=NONE` | 138,043 | 38.0 | 62.9 | 142.8 | OK |

Closed at boot time, not by configuration: `--attention-backend FLASH_ATTN`
(`kv_cache_dtype not supported`), `TRITON_ATTN` (`FP8 KV ... requires SM89+`),
`--no-enable-chunked-prefill` (`Chunked prefill is required for mamba cache mode
'align'`). FlashInfer is the only fp8-KV backend available on this card.

### 4.3 The actual fix: draft depth, not dtype

The image's entrypoint bakes `--speculative-config {"method":"mtp",
"num_speculative_tokens":7}`. Overriding the draft depth is a one-flag change
(a duplicated CLI arg wins) — and vLLM also sizes the KV pool by draft depth:

| n_spec | KV pool @32k | x4 agg | x4 result |
|---|---|---|---|
| 7 (image default) | 135,952 (3.70×) | — | **CUDA IMA, engine dead** |
| **3** | **162,929 (4.97×)** | **246.5 tok/s** | OK |
| 0 (MTP off) | 277,162 (8.46×) | 193.7 tok/s | OK |

MTP is *verified* speculative decoding, so lowering the draft depth changes speed and
memory, never output quality.

### 4.4 Final configuration — 262k context at unchanged speed

```bash
docker run -d --name bonsai-vllm --gpus all --ipc=host -p 8000:8000 \
  -v bonsai:/cache -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  fraserpricee/bonsai-vllm:20260918 \
  --max-model-len 262144 --max-num-seqs 64 --gpu-memory-utilization 0.92 \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32]}'
```

| max-model-len | KV pool | concurrency | measured proof |
|---|---|---|---|
| 32,768 | 162,929 | 4.97× | x4 short-stream load: 246.5 tok/s agg, 0 errors |
| 131,072 | 241,875 | 1.85× | 114,059-token needle ✅ (141 s); 2 × 57,059 parallel ✅ (80.3 s) |
| **262,144** | **262,144** | **1.00×** | **218,559-token needle ✅ (178 s, ~1,227 tok/s prefill)** |

Solo decode at the final config: **83.3 tok/s** — identical to the old bf16+MTP
configuration — while 4-stream aggregate rises from 167.1 to **253.8 tok/s**.

**The honest caveats**

- At 262k the pool is exactly 1.00×: one full-length conversation owns the whole
  cache and other requests queue behind it. Normal traffic is unaffected (262,144 /
  8,000 ≈ 32 concurrent short chats).
- fp8 KV stores K/V at 8-bit: single-request output parity was verified earlier
  (5/7 byte-identical, the rest same answers) and every needle test passed, but it
  is not bit-identical to bf16.
- If you need extra *concurrency headroom* rather than speed, dropping MTP entirely
  (override the entrypoint and omit `--speculative-config`) yields a 311,200-token
  pool (1.19× at 262k) at 58 tok/s solo.
- These weights are **text-only**: the checkpoint contains 0 vision tensors, and the
  engine logs `All limits of multimodal modalities supported by the model are set to
  0, running in text-only mode`. Do not expect image input from this release.
- Detection lesson: when the engine dies, the API server keeps answering
  `/v1/models` for a few seconds and in-flight streams return **zero tokens with no
  client-side exception**. Judge health on `docker inspect` state + `docker logs`,
  never on the health endpoint alone.

---

## Decision guide

- **Maximum context, both engines now reach it** → vLLM final config (262k + batching) or
  llama.cpp (262k, 1 GB less VRAM, faster long-context decode).
- **Long documents, single user** → llama.cpp still wins decode at depth (60.7 vs 29.7
  tok/s at 38k, bf16-KV vLLM measurement; the fp8 config has not been re-measured there).
- **Several users / chat UI / batching** → vLLM final config (253.8 tok/s agg at 4 streams).
- **Short-context throughput** → either; they are within 1% of each other.
- **Do not** use fp8 KV with `num_speculative_tokens: 7` on Ampere, and skip
  turboquant presets until the upstream fix lands.

---

## Reproduce it

- `scripts/ab_kv_test.py` — bf16 vs fp8 KV A/B (outputs + TTFT + t/s + load test)
- `scripts/test_vllm_max.py` — vLLM max-context needle + concurrency
- `scripts/test_max_ctx.py` — llama.cpp max-context needle (228k depth)
- `scripts/a2a_test.py` — apples-to-apples harness (`vllm` | `llama`)
- `scripts/load_sweep.py` — x2/x4 short-prompt concurrency sweep (`load_sweep.py <label> 2,4`)
- `scripts/long_conc.py` — N simultaneous long prompts with a per-stream needle
  (`long_conc.py <label> <n> <tokens>`) — this is the run that proves real multi-session
  capacity, not just a short-prompt load test
- `scripts/sweep_seqs.sh` — boot-only sweep of `--max-num-seqs` / `--max-num-batched-tokens`
  (both turned out **not** to change the KV pool on this model)
- `data/` — raw JSON results, `RESULTS.md` lab notebook, launch script
- [`TIMELINE.md`](../TIMELINE.md) — the day-by-day path, including the ruled-out hypotheses

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

