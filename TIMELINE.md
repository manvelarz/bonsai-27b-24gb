# Timeline — getting Ternary-Bonsai-2-27B to 262k context on one 24 GB GPU

An append-only log of how this project actually went: what we tried each day, what it
proved, and what it ruled out. The article ([`article.md`](article.md)) is the distilled
result; this file is the road, including the wrong turns.

- Raw measurements per run: [`data/RESULTS.md`](data/RESULTS.md)
- Harnesses used: [`scripts/`](scripts/)
- Hardware throughout: RTX 3090 24 GB (sm_86), i7-10700K, 32 GB RAM, Linux.
  (The first two days were done on a 12 GB card, which shaped the early plan.)

Reading rule used everywhere below: a number is quoted only if it came out of a run on
this machine; engine-reported figures are labelled as such.

---

## 2026-10-05 — Research and environment (phase 0–1)

**Decided:** stage the work (inspect the GGUF contract → convert → verify numerically →
serve) instead of writing GPU kernels blind.

- Captured the GGUF format contract (`gguf_inspect.py`): 402 tensors of type-142
  (PQ2_0 ternary), 96 IQ4_NL, 353 F32, plus `prism.hadamard.*` metadata that the
  runtime depends on.
- Fixed four environment failures, each verified: PyPI vLLM ≥0.28 ships CUDA 13 wheels
  that need a R580 driver (ours was R535) → used the `vllm-0.31.0+cu129` wheel;
  uv multi-index refusal → `--index-strategy unsafe-best-match`; system python3.12
  without `Python.h` → uv-managed venv; FlashInfer JIT needing nvcc → launched with
  `VLLM_USE_FLASHINFER_SAMPLER=0`.

**Ruled out:** stock vLLM quantizations for this model — dense GPTQ is 4/8-bit only,
Marlin is 4-bit only, no GGUF loader. On the 12 GB card of that week the only route was
a ternary-aware plugin; moving to the 24 GB card later removed that constraint.

## 2026-10-05/06 — Converter, and a failure cascade that paid off

Built a PQ2_0 → F16 converter so the model could be validated outside llama.cpp, then
walked a chain of failures that each eliminated one hypothesis:

- F16 artifact loaded into vLLM: config/arch/851 tensor names+shapes all accepted, then
  plain CUDA OOM (54 GB artifact on a 12 GB card) ⇒ **the artifact structure is valid**.
- Q2_0 requant → garbage (PPL 7,384,251 vs 3.69): ruled out metadata (0/18 diffs),
  tokenizer (byte-identical), tensor names/shapes (851/851), and structure vs HF Qwen3.5.
- Q5_K (normally near-lossless) → also garbage ⇒ a **real conversion bug**, not
  quantization.
- Ground-truth checks all passed (PQ2_0 decode vs Prism's own F16 file: 6e-3 relative
  error; IQ4_NL decode vs their reference kernel: exact 0.0; unfold math vs runtime
  formula: 5e-7).
- Activation-trace diff localized it to layer 0, then to a **norm convention**: the
  converter added `+1` to `*norm.weight` where the llama-side format stores `w` already,
  doubling every RMSNorm. Fix: emit `(original − 1)` for exactly those tensors.

**Lesson recorded:** the cheap way to find a convention bug is a full-tensor roundtrip
audit against the original, not eyeballing generated text.

## 2026-10-06 — vLLM docker path starts serving

- Image `fraserpricee/bonsai-vllm:20260918` (vLLM 0.25.1, `prism_ternary` quantization),
  model id `Bonsai-2-27B` on `:8000`, weights in a docker volume.
- `--enforce-eager` was load-bearing at first: without it the engine died during
  CUDA-graph memory profiling (1.62 GiB allocation) even though weights were only
  9.5 GB. Lowering `--gpu-memory-utilization` or `--max-model-len` did not help that
  path — the KV pool is allocated after graph profiling.
- Later that day graphs became usable by giving explicit
  `--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32]}'`.
- Measured under eager: `GPU KV cache size: 95,858 tokens`, 31.4 tok/s single stream,
  56.9 tok/s aggregate over 8 streams.
- Also discovered the image already enables **MTP speculative decoding** in its
  entrypoint (`num_speculative_tokens: 7`) — nothing to add, only to tune.

## 2026-10-07 — llama.cpp MTP build, and a one-command backend switch

- Prism's llama.cpp fork was built from source with a hand-applied Hadamard-inverse patch
  (the published patch was malformed), CUDA 13 from the pip-vendored toolkit, and
  `systemd-run --user --scope` because the worker cgroup OOM-killed nvcc.
- Result with `--spec-type draft-mtp --spec-draft-n-max 2`: 66.8 → **72.7 tok/s** solo,
  135.1 → **175.6 tok/s** aggregate at 8 streams, in 10.7 GB VRAM.
- `bonsai llama|vllm|status` was written so the two backends can be swapped with one
  command (they are mutually exclusive on a 24 GB card), plus LiteLLM aliases
  `bonsai-llama` / `bonsai-vllm` so clients never change.

**Ruled out:** the PTQ1_0 MTP quant graft (52.0 tok/s vs 51.6 without MTP — the quant
was the regression), ngram speculative decoding (no gain), and stock GGUFs (no MTP
layers at all).

## 2026-10-10 — Public endpoint work

Four tunnel hostnames (main proxy, engine-agnostic `bonsai` group, plus per-engine
test endpoints), per-user keys, and per-instance Langfuse tracing so usage can be
attributed. Design contract that later mattered: **swapping engines must not require
any config edit** — a helper script and router behaviour were fixed rather than
special-casing the dead engine.

## 2026-10-14 — First 8-bit KV attempt (the wall)

- bf16 KV (production): 10.45 GiB → **94,391 tokens**, 2.88× at 32k, x4 = 176.7 tok/s
  aggregate, 0 errors.
- `--kv-cache-dtype fp8_e4m3`: engine accepts it (FlashInfer reports
  `kv_cache_dtype=torch.float8_e4m3fn, arch=sm86`), capacity grows to 121,311 tokens,
  single-request quality fine (5/7 outputs byte-identical, the rest same answers) —
  **but at 4 concurrent streams: `CUDA error: an illegal memory access`, engine killed.**
- `turboquant_k8v4`: two separate startup failures (hybrid GDN layers pass `'auto'` to a
  backend that rejects it; patching that hits a layout mismatch) ⇒ upstream bug.

**Conclusion that day:** stay on bf16; fp8 gives no usable win. (This conclusion was
wrong in its cause, right in its data — see 2026-10-27.)

## 2026-10-26 — Max-context shootout, and two corrections

- llama.cpp: `-c 262144 --parallel 2 --kv-unified -ctk q8_0 -ctv q8_0` → 2 slots ×
  262,144 native, 19,959 MiB VRAM, needle verified at 228,100 tokens depth.
- vLLM (bf16): hard ceiling `--max-model-len 114688` → 119,553 tokens (1.04×). bf16 KV
  costs ~116 KiB/token, so 262k would need 28.6 GiB — impossible with 16-bit KV.
- Apples-to-apples with thinking disabled: short/medium speed identical (~84 tok/s
  both); at 38k context with long output vLLM was **29.7 tok/s vs llama.cpp's 60.7**.
- Correction #1: an earlier "282 tok/s" vLLM figure was a short-output artifact (45
  tokens delivered in one 0.16 s burst). Rates need hundreds of generated tokens.
- Correction #2 (same day): the engine-group failover bug — retries were re-rolled onto
  the dead backend and cooldown never engaged. Fixed with `allowed_fails: 1` +
  `cooldown_time: 30` in the router settings, verified over 25 public calls.

**Ruled out:** bf16 at 262k context on this card, by arithmetic and by startup log.

## 2026-10-27 (morning) — "Can vLLM use anything but 16-bit?" retest

Re-ran the whole precision matrix from scratch, one container per configuration.

- Verified from the boot log that the **weights are already 2-bit** (checkpoint 9.54 GiB,
  `prism_ternary`, bits=2, group 128) and that the only 16-bit part is activations + KV.
  On sm_86 vLLM warns there is no native FP8 compute — only weight-only FP8 via Marlin.
- The fp8 IMA reproduced **three times**: default capture sizes, `[1,2,4]`, and a 512 MB
  FlashInfer workspace. x2 never crashed, x4 always did.
- Two other backends closed: `FLASH_ATTN` (`kv_cache_dtype not supported`), `TRITON_ATTN`
  (`FP8 KV ... requires SM89+`). Disabling chunked prefill is impossible for this hybrid
  model (`Chunked prefill is required for mamba cache mode 'align'`).
- Workarounds that did survive: fp8 with CUDA graphs off (138,043 tokens, 38 tok/s) and
  `--enforce-eager` (27.5 tok/s).
- Proved 4-way concurrency on bf16 so the requirement was nailed down: 4 × 20,009-token
  prompts, all needles correct, 58.5 s wall.

## 2026-10-27 (afternoon) — the two discoveries

1. **Removing MTP** (override the entrypoint, drop `--speculative-config`) made fp8 KV +
   full CUDA graphs stable: 277,162 tokens at 32k len, 311,200 at 262k len (1.19×), and a
   218,559-token needle test passed. Cost: solo decode 83 → 58 tok/s.
2. **The crash was the draft depth, not fp8.** With `num_speculative_tokens: 3`, fp8 KV +
   MTP + full graphs is stable at 4 streams: 246.5 tok/s aggregate at 32k len, and the KV
   pool also grows (135,952 tokens at n=7 → **162,929** at n=3), because vLLM sizes the
   pool by draft depth.

Final stacking, `--max-model-len 262144` with n=3:

| pool | concurrency | solo | x4 agg | proof |
|---|---|---|---|---|
| 262,144 tokens | 1.00× | 83.3 tok/s | 253.8 tok/s | 218,559-token needle ✅, 178 s |

⇒ **The goal of the whole thread:** 8-bit KV, MTP on, native 262k context, same solo
speed as before, and better throughput under load. Written into `bonsai vllm`
(`bonsai vllm-max` keeps the MTP-off 311,200-token variant for headroom).

---

## What each stage eliminated

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Weights can be made smaller than 2-bit | No — they already are 2-bit | boot log `Checkpoint size: 9.54 GiB`, `bits=2` |
| 4-bit/AWQ conversion is the way | Not on this card, and unnecessary | format research; ternary checkpoint loads natively |
| fp8 KV is broken on Ampere | Wrong — it was the MTP draft depth | n=7 crashes at x4, n=3 stable at x4, n=0 stable |
| CUDA graphs are the culprit | No — graphs are fine without MTP-n7 | fp8 + graphs + n=3 → 253.8 tok/s agg |
| A different attention backend fixes fp8 | No — sm_86 has only FlashInfer for fp8 KV | boot-time rejections from FLASH_ATTN / TRITON |
| 262k needs bf16 KV (28.6 GiB) | No — fp8 KV fits in 9.7 GiB | 262,144-token pool measured |
| Long-context decode must be slow in vLLM | Partly — llama.cpp still leads at depth | 60.7 vs 29.7 tok/s at 38k (bf16 vLLM) |
| Image/vision input is available | No — text-only release | 0 vision tensors; engine logs text-only mode |

## 2026-10-29
- Read akmorley's 10-chapter write-up (guide + CUDA checkpoint + Triton/Gluon final).
- Verdict recorded: his implementation is faster (97.4 vs 82.7 solo, 2,297 vs 1,219 prefill @1k)
  and it is a different weight format, plugin and KV scheme — not a tuning difference.
- Fetched his code blocks verbatim into `akmorley/` + `PROVENANCE.md` (pinned revisions).
- Confirmed we already had `Ternary-Bonsai-2-27B-PQ2_0.gguf` (6.8 GB) locally.
- Staged `Qwen/Qwen3.8-27B` tokenizer files into `models/qwen38-tokenizer` (his serve.sh needs them).
- Started `docker build -t bonsai-pq2-akmorley` from his Containerfile (CUDA 13.0 devel, vLLM built
  from source at his commits).

- **2026-10-29 13:29 host incident:** MAX_JOBS=10 build hung the machine (32 GB host, ~20 GB services, 2 GB swap);
  colleague power-cycled it. Both bonsai containers exited 255. Resumed from the surviving venv with
  MAX_JOBS=4 (12 GB used, 18 GB available) and documented the rule in `akmorley/PROVENANCE.md`.
- Post-reset: GPU driver intact (595.58.03, display_active Enabled) but the display came back on
  NVIDIA's fallback mode table with **no EDID** — max 1600x900, no 1920x1080/2560x1440. That is the
  graphics change visible through NoMachine.
- **Hang investigation (post-reset):** no OOM/panic/hung-task/Xid in the journal; journal goes silent
  for ~5 min before the reset; build container `memory.peak` = 10.4 GB (5.5 GB at MAX_JOBS=4); host is
  31 GB with a 2 GB swapfile, no zram, `systemd-oomd` inactive, `earlyoom` absent, `sysstat` disabled.
  Cause not proven — memory timeline was never recorded. Mitigations listed in `akmorley/PROVENANCE.md`.
- **2026-10-29 — Morley stack reproduced.** Built his forks (Containerfile needs python3.12-dev;
  runtime needs python3.12-dev + ninja + CUDA toolkit for FlashInfer/JIT). Full 262,144 context fits,
  429,916 KV tokens (1.64x). Prefill 1.4-1.7x faster than ours; long-context decode +13%; ours wins
  4-stream batching. Documented as Part 10.
- **2026-10-30 — MTP attempt on his stack.** MTP draft loads and the speculator runs, then
  `torch.compile` fails on a zero-width `split`: the GGUF `nextn` block ships only 4 tensors
  (eh_proj + 3 norms), so the MTP head shares the backbone body and the adapter assumes a
  self-contained block. Four blockers fixed first (draft config path, `qwen3_5_mtp` registration,
  architecture mapping, 54 MB trimmed draft). Details: `akmorley/MTP_ATTEMPT.md`.
