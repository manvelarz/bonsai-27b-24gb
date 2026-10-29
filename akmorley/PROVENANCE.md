# Provenance — akmorley's Prism/vLLM stack, reproduced on this machine

Everything in this directory is **Anthony Morley's work**, not ours:
- Guide: https://anthonymorley.com/projects/prism/running-it-locally (published 2026-10-14, updated 2026-10-19)
- Investigation: https://anthonymorley.com/projects/prism (chapter 10: /from-cuda-to-triton-and-gluon)
- Forks: https://github.com/akmorley/vllm-gguf-plugin-prism @ 7520fd9bf2c51ff760096bc1f4cee40c79100e16
         https://github.com/akmorley/vllm-prism @ d463094f1d97d95ac6c9596b92d3b7ce677f6fb9 (branch dflash-quantized-context-kv)
- GGUF reader: https://github.com/PrismML-Eng/llama.cpp (gguf-py)

`install.sh`, `serve.sh` and `Containerfile` are transcribed verbatim from that guide on 2026-10-29.
The only change we make is pinning our own paths and port (8001) so it can run beside the existing
service; if we alter behaviour we will record it below.

## Why we are doing this
Our report measured fraserprice's prebuilt image (prism-ternary 0.1.0, fp8 KV, MTP n=3) at 82.7 tok/s
solo and ~1,000 tok/s prefill at 40k. Morley reports 97.4 tok/s solo and 2,297 tok/s prefill at 1k
on a 3090 Ti at 300 W with *no* speculative decoding, using GGUF-native PQ2 kernels and
int8_per_token_head KV. Different weights format, different plugin, different KV scheme — so the
comparison is not apples to apples until his stack runs on this 3090.

## Open questions we are testing
1. What max context does `--max-model-len -1` actually pick on a 24 GB 3090 (not Ti)?
2. Does his throughput reproduce on a plain 3090 at 336 W?
3. Does MTP compound with his kernels, or is DFlash (his speculative path) the better pairing?

## Local deviations
(none yet)
