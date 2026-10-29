# Reddit post drafts — bonsai-27b-24gb

## Title (pick one)

1. A 27B ternary model at 262k context on one 3090: vLLM vs llama.cpp, measured — and what 2-bit actually costs against the full-precision original
2. We ran a 2-bit 27B at 262k on a single 24GB 3090. What worked, what crashed, and the three benchmarks that fooled us
3. 24GB VRAM, 262k context, 27B: real numbers for vLLM and llama.cpp (plus a bug we misdiagnosed for a day)

## Body

**TL;DR** — Ternary-Bonsai-2-27B (a 2-bit quant of Qwen3.8-27B, 7.1 GB of weights) runs at its native **262,144-token context on a single RTX 3090**, on both engines. Measured, not assumed:

- **200k prompt**: first token at **330 s (vLLM)** / **347 s (llama.cpp)**, prefill ~600 tok/s
- **38k prompt, long answer**: **67 tok/s** (vLLM) vs **60.7 tok/s** (llama.cpp)
- **4 concurrent short streams**: **254 tok/s** aggregate (vLLM) vs **64 tok/s** (llama.cpp)
- **Quality vs the original Qwen3.8-27B**: 28/30 vs 28/30 verifiable answers — i.e. **the compression costs nothing measurable**, and both hit 30/30 with thinking on

**Setup.** 3090 (24 GB, sm_86), i7-10700K, 32 GB RAM. Two stacks, same prompts, same needle at the end of a 200k document:

- vLLM, fp8 KV, MTP speculative decoding at depth 3, full CUDA graphs → native 262,144 ctx, ~23.5 GB VRAM
- llama.cpp, q8_0 KV, `-c 262144 --parallel 2 --kv-unified` → ~19.9 GB VRAM

**The bit that cost us a day.** fp8 KV crashed under load with `CUDA error: an illegal memory access`, so we blamed fp8 and went back to bf16, which caps you at 114,688 tokens on this card. It was never fp8 — it was the **speculative draft depth**: the image ships with `num_speculative_tokens: 7`, and at 7 it dies under concurrent load. Drop it to 3 and fp8 KV + full CUDA graphs is stable at the native 262k. That one flag is the difference between "24 GB can't do 8-bit KV on Ampere" and "24 GB does 262k".

**Draft depth, measured.** One boot per setting, solo decode: n=1 → 71.4 tok/s, n=2 → 79.7, **n=3 → 82.7**, n=5 → 79.0 but 500s under 4 streams, n=4 → won't even boot at 262k (needs 9.48 GiB of KV, 9.3 available), n=7 → crashes. n=3 is the deepest setting that's stable *and* the fastest.

**What 2-bit costs.** Same 45 prompts through the original Qwen3.8-27B on a pinned provider: 28/30 both, 0.81 text agreement. The disagreements were a wash — the full-precision model short-circuited `(47×13)−(96/4)` to 605 and the ternary one worked it out to 587; the ternary one then flubbed `'5' + '3'` where the original got it. With thinking on, **both** go to 30/30 — so thinking mode, not precision, is what separates 28/30 from 30/30, and it costs ~4× the output tokens.

**One thing we were wrong to worry about: the power cap.** Our card is limited to 336 W (default 420, max 462). Same 40k prompt, two runs at each limit: raising it to 462 W bought under 2% prefill and nothing measurable in decode, with memory clock unchanged at 9,501 MHz — decode at this quant is bandwidth-bound, so the watts have nowhere to go. **Every number below was measured at 336 W**, so they hold for a stock uncapped 3090 too. And if you're leaving a box inferring all day, running capped costs nothing and saves ~25% of the power.

**Three measurement traps** (this is the part I'd want someone to tell me):

- **Prefix caching fakes prefill.** Our first sweep reported 36,911 tok/s at 40k — it was a cache hit. Every prompt now carries a unique nonce.
- **Short outputs inflate decode under speculation.** A 40-token probe said 291 tok/s at 200k; a 320-token output on the same setup measures **55.5 tok/s**.
- **The KV pool changes between boots.** Same config reported 262,144 and 294,362 tokens on different boots. Quote ranges, not numbers.

**What it's actually good for.** 262k with a 5-minute time-to-first-token is not a chat interface. It's for batch extraction, whole-repo analysis, long-document retrieval — jobs where you send one huge prompt and wait. For interactive use, 38k at 67 tok/s is the sweet spot; for multi-user serving, vLLM's batching is the whole ballgame (254 vs 64 tok/s at four streams).

**Thanks** — this setup follows the vLLM flag combination that **u/live4evrr** posted in r/LocalLLaMA (fp8 KV + MTP speculative decoding + `qwen3_coder` tool parser at 262k), which is what pointed us at the right shape in the first place. Also worth reading: **u/Fancy-Snow7**'s independent quality/needle numbers on the same PQ2_0 quant, and **u/Creative-Regular6799**'s Terminal-Bench run — both conclusions line up with what we measured here.

**Honest limits:** one card, one model, one image build. Single-run concurrency numbers. 30 verifiable answers is a smoke test, not a benchmark suite. No tool-calling or multi-turn tests yet. All of the above is in the write-up, including the cells I couldn't fill.

Full write-up with charts, the prefill/decode curves and every raw JSON: https://smartphotocropper.com/blog/ternary-bonsai-27b-24gb-vllm-vs-llamacpp/

Harnesses and data (nothing behind a login): https://github.com/manvelarz/bonsai-27b-24gb

Happy to answer questions about the numbers — and if you've got a 4090 or an A6000, I'd love to see whether these hold.
