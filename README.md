# Ternary-Bonsai-2-27B on a single 24 GB GPU (RTX 3090): measured notes

Field report: running Prism ML's ternary 27B model with **vLLM** and **llama.cpp**
on one 24 GB card — max context, KV-cache trade-offs, speed, concurrency, and the
failure modes we hit on Ampere.

- 📄 **Read the article:** [`article.md`](article.md)
- 🧾 **Attribution:** [`CREDITS.md`](CREDITS.md)
- 📊 **Lab notebook + raw results:** [`data/RESULTS.md`](data/RESULTS.md), [`data/`](data/)
- 🧪 **Benchmark harnesses:** [`scripts/`](scripts/)

## Headline numbers

| | llama.cpp (q8_0 KV) | vLLM (bf16 KV) |
|---|---|---|
| max context | 262,144 | 114,688 |
| VRAM | 19.9 GB | 22.5 GB |
| decode @38k, long output | 60.7 tok/s | 29.7 tok/s |
| 2 concurrent long requests | 133 s | 39 s |

## Status

Local draft — not published yet. Paths/hostnames sanitized; hardware details kept.
Licensing to be decided before any public release; see `CREDITS.md` for authors.
