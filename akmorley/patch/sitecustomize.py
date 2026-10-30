"""Register Qwen3.5 MTP config type for transformers (not shipped upstream).

Loaded via PYTHONPATH at interpreter start so vLLM can build the MTP draft
ModelConfig from the local GGUF directory.
"""
try:
    from transformers import AutoConfig
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    class Qwen3_5MTPConfig(Qwen3_5Config):
        model_type = "qwen3_5_mtp"

    AutoConfig.register("qwen3_5_mtp", Qwen3_5MTPConfig, exist_ok=True)
    AutoConfig.register("qwen3_5_moe_mtp", Qwen3_5MTPConfig, exist_ok=True)
except Exception as exc:  # pragma: no cover
    import sys
    print(f"[mtp-patch] skipped: {exc}", file=sys.stderr)
