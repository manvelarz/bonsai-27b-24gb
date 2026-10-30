#!/usr/bin/env bash
# Patch the installed vllm_gguf_plugin so a qwen3_5_mtp draft config resolves
# to the Qwen3_5MTP model class. Idempotent; runs as root inside the container.
set -euo pipefail
F=/opt/bonsai/.venv/lib/python3.12/site-packages/vllm_gguf_plugin/config_parser.py
if grep -q "Qwen3_5MoeMTP" "$F"; then
  echo "[patch] already applied"
  exit 0
fi
python3.12 - "$F" <<'PY'
import sys, pathlib
f = pathlib.Path(sys.argv[1]); s = f.read_text()
old = "        architecture = get_adapter_architecture(config)"
new = old + """
        if architecture is None and config.model_type in ("qwen3_5_mtp", "qwen3_5_moe_mtp"):
            architecture = (
                "Qwen3_5MoeMTP" if config.model_type == "qwen3_5_moe_mtp" else "Qwen3_5MTP"
            )"""
assert old in s, "anchor not found"
f.write_text(s.replace(old, new, 1))
print("[patch] applied to", f)
PY
