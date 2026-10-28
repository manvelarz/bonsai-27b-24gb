# Ternary-Bonsai-2-27B on a single 24 GB GPU (RTX 3090): measured notes

Field report: running Prism ML's ternary 27B model with **vLLM** and **llama.cpp**
on one 24 GB card — max context, KV-cache trade-offs, speed, concurrency, and the
failure modes we hit on Ampere.

- 📄 **Read the article:** [`article.md`](article.md)
- 🗓 **How we got here, day by day:** [`TIMELINE.md`](TIMELINE.md)
- 🧾 **Attribution:** [`CREDITS.md`](CREDITS.md)
- 📊 **Lab notebook + raw results:** [`data/RESULTS.md`](data/RESULTS.md), [`data/`](data/)
- 🧪 **Benchmark harnesses:** [`scripts/`](scripts/)

## Headline numbers

| | llama.cpp (q8_0 KV) | vLLM (bf16 KV) | vLLM final (fp8 KV + MTP n=3) |
|---|---|---|---|
| max context | 262,144 | 114,688 | **262,144** |
| KV pool | 262k shared | 119,553 tokens | 262k–294k tokens (1.00–1.12×) |
| VRAM | 19.9 GB | 22.5 GB | 23.5 GB |
| decode @38k, long output | 60.7 tok/s | 29.7 tok/s | **67.0 tok/s** |
| 4 concurrent short streams | 64.1 tok/s agg | 167 tok/s agg | **253.8 tok/s agg** |
| TTFT at ~200k prompt | 347 s | n/a (114k limit) | **330 s** |
| quality, 30 verifiable prompts | 26/30 | 26/30 | 28/30 (within noise) |
| solo decode, short ctx | 84 tok/s | 84 tok/s | 83.3 tok/s |

## What does the compression cost?

Same 45-prompt suite (30 verifiable answers), every configuration, thinking pinned:

| | thinking off | thinking on |
|---|---|---|
| original Qwen3.8-27B, bf16 | 28/30 | 30/30 |
| original Qwen3.8-27B, fp8 | 28/30 | 30/30 |
| **ternary 2-bit, this rig** | 28/30 | **30/30** |

The 2-bit build loses nothing measurable against the model it was compressed from, at
either thinking setting. Thinking itself costs ~4x the output tokens. Details, the four
disagreements and the caveats: [`article.md`](article.md) Part 8; raw results in
[`data/`](data/).

**2026-10-27 update:** vLLM now reaches the native 262k context with 8-bit KV *and*
speculative decoding on. The earlier "fp8 KV crashes under load" result was a
`num_speculative_tokens: 7` problem, not an fp8 problem. Details in
[`article.md`](article.md) → Part 4 and the full path in [`TIMELINE.md`](TIMELINE.md).

## Status

Published (companion repo). Canonical article lives on smartphotocropper.com once the
blog post ships; this repo carries the full text, raw measurements and harnesses.
Paths/hostnames sanitized, hardware details kept. License to be added; see `CREDITS.md`
for authors.

Every number above came from a run on this machine, including the 38k/long-output cell
(67.0 tok/s on the final fp8 + MTP n=3 config, needle verified).
