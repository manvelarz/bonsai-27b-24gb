# KV-cache dtype A/B results — Ternary-Bonsai-2-27B (vLLM docker, 2026-10-14)

Test rig: `ab_kv_test.py` (greedy temp=0 seed=0, 7 prompts incl. ~5k-token
needle-in-haystack, plus load tests x4 / x8 parallel 250-word generations).
Raw outputs: `ab_bf16.json`, `ab_fp8.json`.

## Config A — bf16 KV (kv_cache_dtype=auto) — PRODUCTION, stable
- KV cache: **10.45 GiB → 94,391 tokens → 2.88× @32k** (~116 KB/token)
- TTFT: 0.48–0.94 s short prompts, 2.03 s (reason), 4.31 s (5k-token needle)
- decode t/s (single): 139–1020 t/s on measurable passes
- Load: **x4 = 176.7 agg t/s**, **x8 = 154.9 agg t/s**, 0 errors
- needle → ZEBRA-4471 ✅

## Config B — fp8_e4m3 (8-bit KV) — WORKS SINGLE, CRASHES UNDER LOAD
- KV cache: 9.3 GiB → **121,311 tokens → 3.70× @32k** (+28.5% capacity, NOT
  full 2× — the fp8/FlashInfer path reserved ~1.1 GiB extra scratch memory)
- FlashInfer: `kv_cache_dtype=torch.float8_e4m3fn, arch=sm86` accepted ✅
- Outputs: **5/7 byte-identical**; 2 diverged in reasoning path only,
  both still correct (17 chickens ✅, "Yes" logic ✅); needle ✅
  → no intelligence degradation observed on single requests
- Speed singles: ≈ same as bf16 (TTFT within noise; 3.66s on first prompt =
  cold start)
- **Load x4/x8: `CUDA error: an illegal memory access` → engine force-killed,
  500s on all requests, server DOWN.** ❌ unusable for multi-chat.

## Config C — turboquant_k8v4 (K 8-bit / V 4-bit) — NOT USABLE on this image
- Startup failure #1: `ValueError: Unknown TurboQuant cache dtype: 'auto'` —
  hybrid GDN layers request standard-shape native cache; TQ backend raises
  instead of returning it (contradicts its own docstring).
- With test-only patch handling 'auto': startup failure #2 —
  `_reshape_attention_kv_cache`: `shape '[2,4320,16,4,256]' invalid for input
  of size 107274240` (layout mismatch in the fallback path).
- Verdict: needs upstream fix in fraserpricee/bonsai-vllm (or vllm TQ backend)
  before it can run on this hybrid arch. NOT a config problem.

## Conclusion
- Stay **bf16** for production: only config that survives concurrency.
- fp8 promise (2× KV) not realized (+28.5%) AND crashes concurrently → no win.
- k8v4 blocked by engine bug (evidence above, ready for an upstream issue).
- Test artifacts: `ab_kv_test.py` (reusable), `patch_tq_auto.py` (test-only,
  container discarded — production image unpatched).

---

# Max-context shootout (2026-10-26): llama.cpp 262k vs vLLM 112k

## llama.cpp (`start_llama_pq2mtp.sh`): -c 262144 --parallel 2 --kv-unified -ctk/-ctv q8_0
- 2 slots × 262,144 (native model max), VRAM 19,959 MiB
- short (62 tok): TTFT 0.1 s, 90.5 tok/s decode
- 64k prompt: TTFT 78 s (~918 tok/s prefill), 57.8 tok/s decode
- 228k prompt: TTFT 432.8 s (~570 tok/s prefill), 30.3 tok/s decode, needle ✅
- 2× 40k parallel: 133 s wall (both correct)

## vLLM (`bonsai-vllm`): --max-model-len 114688, bf16 KV
- KV capacity 119,553 tokens → concurrency 1.04× at max len; VRAM 22,516 MiB
- 96k variant: 115,810 tokens, 1.18×
- short: TTFT 1.94 s (cold), decode n/a (32 tok)
- 95k prompt: TTFT 108.8 s (~874 tok/s prefill), **282 tok/s decode**, needle ✅
- 2× 38k parallel: 39.3 s wall (both correct) → vLLM batching ~3.4× faster than llama
- Can NOT reach 262k: bf16 KV = 116 KiB/tok → 262k needs 28.6 GiB. fp8 would fit ~239k
  but crashes under concurrency (IMA, see above); turboquant_k8v4 broken on sm86.

## Verdict
- Context: llama wins (262,144 vs 114,688 = 2.3×), thanks to 8-bit KV + tighter memory use.
- Long-prompt speed: vLLM wins decode (282 vs 30 tok/s @~100k; 58 @64k llama), similar prefill.
- Concurrency: vLLM wins decisively (2×38k in 39 s vs 2×40k in 133 s).
- Practical split: document/analysis with huge context → llama; multi-user/chat → vLLM.

## CORRECTION + apples-to-apples (thinking disabled, same prompts, greedy)
Method: `chat_template_kwargs={"enable_thinking": false}` on both engines
(vLLM honors it; llama.cpp honors it since recent builds — `/no_think` prefix does NOT work).

| test | llama (q8_0 KV, 262k) | vLLM (bf16 KV, 112k) |
|---|---|---|
| short "Paris" | 8 tok, TTFT 0.11 s | 8 tok, TTFT 0.09 s |
| medium ~400 tok | 414 tok, 5.2 s, **84.1 t/s** | 421 tok, 5.1 s, **83.6 t/s** |
| 38k prompt + long output | TTFT 35.1 s (~1085 t/s prefill), 558 tok, **60.7 t/s**, needle ✅ | TTFT 34.1 s (~1118 t/s prefill), 634 tok, **29.7 t/s**, needle ✅ |

=> Earlier "vLLM 282 t/s decode at long ctx" was an ARTIFACT (45 tokens delivered in
0.16 s of bursty streaming). With real long outputs vLLM is ~2x SLOWER than llama at
long context (29.7 vs 60.7 t/s) — consistent with bf16 KV costing 2x memory traffic
per token vs llama's q8_0 KV. Short/medium speed is identical.
---

# Weight-precision wall, retest (2026-10-27) — "can vLLM do anything but 16-bit?"

Man's requirement for this round: 2 concurrent OK, 4 concurrent = happy, 8 not needed.

## What the engine actually runs (verified in the boot log, not computed)
- Checkpoint: `fraserprice/Ternary-Bonsai-2-27B-vllm`, `quantization_config.quant_method=prism_ternary`,
  bits=2, group_size=128, hadamard_block=1024. **Checkpoint size 9.54 GiB** (`weight_utils.py` line).
  => weights are NOT 16-bit; they are 2-bit ternary and already load that way.
- `dtype=torch.bfloat16, quantization=prism_ternary, kv_cache_dtype=auto` -> the 16-bit part is
  activations + KV cache, plus on sm86 the FP8 path is weight-only Marlin
  (`marlin_utils_fp8.py:112 Your GPU does not have native support for FP8 computation`).
- So the only precision lever left is **KV cache dtype** (fp8) — weights can't go lower than they are.

## Retest matrix (every run: fresh container, same 400-token greedy load prompts + long-prompt proof)
| config | KV pool (tokens) | solo t/s | x2 agg | x4 agg | x4 (4 parallel) |
|---|---|---|---|---|---|
| bf16 KV + CUDA graphs (prod @0.92) | 94,391 | 83.1 | 129.8 | 167.1 | OK |
| **bf16 KV + CUDA graphs @util 0.95** | **100,749** | **82.3** | **129.5** | **166.2** | **OK** |
| fp8 KV + CUDA graphs | 135,952 | 75.1 | 108.0 | **IMA crash** | engine DEAD |
| fp8 KV + graphs, capture sizes [1,2,4] | 121,311 | 39.0 | 61.3 | **IMA crash** | engine DEAD |
| fp8 KV + graphs + FI workspace 512 MB | 135,952 | 74.5 | 114.9 | **IMA crash** | engine DEAD |
| fp8 KV + `--enforce-eager` | 138,043 | 27.5 | 42.8 | 100.3 | OK |
| **fp8 KV + compile, `cudagraph_mode=NONE`** | **138,043** | **38.0** | **62.9** | **142.8** | **OK** |
| fp8 KV + FLASH_ATTN | — | boot fail: `kv_cache_dtype not supported` | | | |
| fp8 KV + TRITON_ATTN | — | boot fail: `FP8 KV requires SM89+` (we are sm86) | | | |
| fp8 + `--no-enable-chunked-prefill` | — | boot fail: `Chunked prefill is required for mamba cache mode 'align'` | | | |

Crash signature (3 reproductions, fp8+graphs only): `torch.AcceleratorError: CUDA error: an illegal
memory access` in the FlashInfer fp8 KV path; the API server keeps answering `/v1/models` for a few
seconds while every in-flight stream returns zero tokens — always judge by `docker inspect`
state + `docker logs`, never by the health endpoint. Logs kept: `*_logs.txt` next to this file.

## Long-prompt concurrency proof (the number that matters)
| config | 4 parallel streams, ~tokens each | total prompt tokens | wall | needles | errors |
|---|---|---|---|---|---|
| bf16 + graphs @0.92 | 4 x 20,009 | 80,036 | 58.5 s | 4/4 | 0 |
| bf16 + graphs @0.95 | 4 x 21,909 | 87,636 | 57.7 s | 4/4 | 0 |
| fp8 + no graphs @0.92 | 4 x 28,559 | 114,236 | 79.4 s | 4/4 | 0 |

## Verdict
- **4 concurrent sessions work TODAY on bf16 + CUDA graphs** — no fp8, no engine patch needed:
  - util 0.92 (current `bonsai vllm`): 94,391-token pool = 4 streams x ~20k prompt, 83 solo / 167 agg.
    Keeps ~2.3 GB free. **Production default.**
  - util 0.95: 100,749-token pool = 4 streams x ~22k prompt, 82 solo / 166 agg — but only ~50 MiB
    left on the card after the peak, so it is an opt-in, not the default.
- fp8 KV buys +37% pool (138k -> 4 x ~28k) **only with CUDA graphs off**, and that costs ~54% of
  solo decode speed (38 vs 82 t/s). Not worth it unless a stream needs >22k prompt.
- fp8 KV **with** graphs is unusable: reproducible IMA at 4 streams, independent of capture sizes,
  FlashInfer workspace size, and prefill mode. Other attention backends cannot do fp8 on sm86.
  => This is an engine/kernel bug (fraserprice/bonsai-vllm + sm86 FlashInfer fp8), not a config.

---

# BREAKTHROUGH (2026-10-27, same day): the blocker was MTP, not fp8

> **Read the FINAL section at the bottom of this file.** This section proved that
> *removing* MTP makes fp8 KV work (and reach 262k). The next step localized the real
> trigger to the draft depth: `num_speculative_tokens: 7` crashed, `3` is stable — so
> MTP was kept after all. Use the FINAL config.

Man's real ask: more context in 24 GB, ideally 8-bit KV. The fp8 wall above was NOT fp8 — it was
the image's baked-in **MTP speculative decoding** (`--speculative-config {"method":"mtp",
"num_speculative_tokens":7}` in the image ENTRYPOINT). Removing it (override `--entrypoint vllm`
and pass `serve ... ` yourself, no spec flag) makes fp8 KV + FULL CUDA graphs work.

## Kept constant: max-model-len 32768, util 0.92, capture sizes [1..32]
| config | KV pool (tokens) | x pool | solo t/s | x2 agg | x4 agg |
|---|---|---|---|---|---|
| bf16 KV + MTP (old prod) | 94,391 | 2.88x | 83.1 | 129.8 | 167.1 |
| bf16 KV, no MTP | 152,917 | 4.67x | 57.5 | 102.5 | 190.2 |
| fp8 KV, no MTP | **277,162** | **8.46x** | 57.8 | 104.3 | **193.7** |

fp8 KB/token drops to ~33.5 (theory for 16 full-attn layers = 32 KB) — MTP was ALSO inflating the
accounting ~1.8x (118 KB/token -> 70 KB bf16 / 33.5 KB fp8).

## Context-window results (fp8 KV, no MTP, graphs on)
| max-model-len | KV pool | concurrency | proof |
|---|---|---|---|
| 32,768 | 277,162 | 8.46x | 4 x 28,559-token prompts, 4/4 needles, 101 s |
| 131,072 | 305,834 | 2.33x | 1 x 114,059 (TTFT 139 s), 2 x 57,059 parallel (115 s), all needles |
| **262,144** | **311,200** | **1.19x** | 1 x **218,559-token** prompt, needle ok, TTFT 356 s (614 t/s prefill) |

=> **vLLM now serves the full native 262,144 context on this 3090** — same ceiling as llama.cpp's
q8_0 config, with vLLM's batching. Cost of dropping MTP: solo decode 83.1 -> 57.8 t/s (-30%), but
x4 aggregate IMPROVES (167.1 -> 193.7) because spec-decode overhead is gone.

## Production (`bonsai` command)
- `bonsai vllm-max` = fp8 KV, no MTP, 262k ctx (current live): 311,200-token pool, 2.5 GB free.
- `bonsai vllm`     = old fast shape (MTP, bf16, 32k, 83 t/s solo) — kept as the speed option.
- `bonsai llama`    = 262k llama.cpp (60.7 t/s at 38k, 2 slots).


---

# FINAL: MTP back + 262k context (2026-10-27)

Root cause of the fp8+MTP crash was the **draft depth**: the image entrypoint bakes
`num_speculative_tokens: 7`. At n=7 with fp8 KV, 4 concurrent streams hit the IMA. Overriding it
with `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` (duplicate CLI arg wins)
fixes it AND shrinks the KV accounting (vLLM sizes the pool by draft depth):

| n_spec | KV pool @32k len | x4 agg | x4 |
|---|---|---|---|
| 7 (image default) | 135,952 (3.70x) | CRASH | engine dead |
| **3** | **162,929 (4.97x)** | **246.5 t/s** | OK |
| 0 (MTP off) | 277,162 (8.46x) | 193.7 t/s | OK |

## MTP n=3 + fp8 KV, scaling max-model-len (all stable, graphs on)
| max-len | KV pool | concurrency | solo t/s | proof |
|---|---|---|---|---|
| 32,768 | 162,929 | 4.97x | 82.3 | x4 short 246.5 agg |
| 131,072 | 241,875 | 1.85x | — | 1x114k needle ok (141 s); 2x57k parallel 80.3 s |
| **262,144** | **262,144** | **1.00x** | **83.3** | **1x218,559-token prompt, needle ok, 178 s** |

=> **Goal reached: MTP stays, context = native 262,144, solo speed unchanged (83 t/s), and x4
aggregate is 253.8 t/s vs the old 167.1.** Trade-off at 262k: the pool holds exactly one
full-length request (1.00x) — a single 262k conversation occupies the whole cache and others queue;
short/medium chats still batch normally (262,144 / 8k = ~32 slots).

Current production: `bonsai vllm` = this shape (written into `~/.local/bin/bonsai`).
`bonsai vllm-max` = MTP off, 311,200-token pool (1.19x), 58 t/s solo — only for extra headroom.

## 2026-10-28 — closing the last gap: 38k ctx + long output on the FINAL config
Ran `a2a_test.py vllm` against `bonsai vllm` (fp8 KV + MTP n=3, max-len 262,144, graphs on).
Boot reported KV pool **294,362 tokens (1.12x)** this time (previous boot: 262,144 / 1.00x —
see boot variance note in the article).

| test | llama.cpp (q8_0, 262k) | vLLM bf16 (114k) | **vLLM final (fp8 + MTP n=3)** |
|---|---|---|---|
| short "Paris" | 8 tok, TTFT 0.11 s | 8 tok, TTFT 0.09 s | 8 tok, TTFT 0.70 s (cold) |
| ~350–420 tok answer | 414 tok, 84.1 t/s | 421 tok, 83.6 t/s | 349 tok, **91.1 t/s** |
| 38k prompt + long output | TTFT 35.1 s, 558 tok, 60.7 t/s | TTFT 34.1 s, 634 tok, 29.7 t/s | TTFT 38.2 s (~998 t/s prefill), 467 tok, **67.0 t/s** |
| needle at end | ✅ | ✅ | ✅ ZEBRA-4471 |

=> Correction of record: the final vLLM shape is NOT slower than llama.cpp at 38k — it is
slightly faster (67.0 vs 60.7 t/s) because fp8 KV halves the per-token KV bytes, the same
effect that made llama.cpp's q8_0 win over bf16. The earlier "vLLM is 2x slower at long
context" applied to the **bf16-KV** build only.
