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

**2026-10-29 12:32 — `python3.12-dev` missing from the published Containerfile.** The documented build fails at
CMake configure:
`Could NOT find Python (missing: Python_INCLUDE_DIRS Interpreter Development.Module Development.SABIModule)`
→ `CMake Error at cmake/utils.cmake:10: Unable to find python matching: /opt/bonsai/.venv/bin/python`.
The Containerfile installs `python3` and `python3-venv` but not the headers, so `find_python_from_executable`
cannot resolve the interpreter. Failure occurs before any CUDA compilation, so it costs ~2 min, not an hour.

Fixes applied, in this order (each one only exposed the next):
1. `apt-get install python3.12-dev`  → headers present at `/usr/include/python3.12/Python.h`
2. `UV_VENV_CLEAR=1` (and remove `.venv`) → the failed run leaves a venv that `uv venv` refuses to overwrite
   (`error: Failed to create virtual environment`, exit 2).
3. Re-clone the three repos (they were left in a half-checked state by the earlier run).

Everything else is his: same pinned commits, same flags, same kernels.

## Host incident — 2026-10-29 13:29

The first successful compile attempt (MAX_JOBS=10) made this machine **stop responding**; the owner had
to power-cycle it. Two containers died with the reset (`Exited 255`), losing ~20 min of compile progress.

**Diagnosis — what the evidence does and does not show.**

Correction to an earlier entry: the journal *is* persistent (`/var/log/journal` exists); my first read
failed because `sudo -n journalctl` needs a password and returned nothing. Once read properly:

- **No OOM kill, no kernel panic, no hung-task, no soft lockup, no Xid** in the window before the reset.
- **Logging stops.** Between 12:38 and 12:46 the journal holds four lines total (pipewire audio warnings);
  before that the box was writing every few seconds (Telegram polling conflicts, sudo audit, etc.).
  A chatty system going silent for five minutes means processes were not getting scheduled, not that
  they were idle.
- **Build memory peak: 10,369 MB** (cgroup `memory.peak` for `bonsai-build`). With four jobs it sits at
  5.5 GB. Ten jobs therefore cost ~10.4 GB.
- **Host: 31 GB RAM, 2 GB swapfile on a SATA SSD, no zram, `systemd-oomd` inactive, `earlyoom` absent,
  `sysstat` installed but disabled.** So no userspace OOM protection and no historical memory series.

Most likely: 10.4 GB of compiler + ~20 GB of always-on services on a 31 GB box, with 2 GB of swap and
nothing to intervene, degraded into a reclaim/IO stall that made the machine unresponsive to the owner
through NoMachine. **Not proven** — the decisive evidence (a memory timeline) was never collected
because `sysstat` was off, and the kernel never logged an OOM event.

**Mitigations.** MAX_JOBS=4 (measured: 5.5 GB instead of 10.4 GB). Recommended and not yet applied:
`docker run --memory=12g` so the container dies instead of the host; a 4–8 GB zram device; `earlyoom`;
and `sudo systemctl enable --now sysstat` so the next incident has data.

**Rule adopted for this box:** `MAX_JOBS=4` for any vLLM/torch CUDA build. Verified after the change:
22 compiler processes, 12 GB used of 31 GB, 18 GB available.

**What survived the reset:** the container's venv and torch 2.13.0+cu130 (no re-download), the three
repos at their pinned commits, and the GGUF + Qwen tokenizer on the host mount.
