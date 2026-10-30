# MTP on Anthony Morley's GGUF-native stack — attempt log

Goal: add speculative decoding (MTP) to his vLLM 0.25.1 fork, which he never tried.

## What his code already supports

- `vllm-gguf-plugin-prism/vllm_gguf_plugin/weights_adapter/qwen3_5.py` ships
  `Qwen35MtpGGUFAdapter` ("Qwen3.5/3.6 single-block MTP draft stored in a GGUF nextn block"),
  mapping `blk.N.nextn.eh_proj -> mtp.fc`, `enorm -> pre_fc_norm_embedding`,
  `hnorm -> pre_fc_norm_hidden`, `shared_head_norm -> mtp.norm`, block -> `mtp.layers.0.`
- `vllm-prism` registers model `Qwen3_5MTP` (`qwen3_5_mtp`) and accepts speculative
  `method: "mtp"` (in `MTPModelTypes`).
- His README documents the intended usage: `--speculative-config '{"method":"mtp",
  "num_speculative_tokens":N}'` with the **nextn block embedded in the same GGUF**.

## What the GGUFs actually contain

| file | size | nextn tensors |
|---|---|---|
| `Ternary-Bonsai-2-27B-PQ2_0.gguf` (his backbone) | 7.21 GB | **0** |
| `Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf` | 7.66 GB | **4** — `blk.64.nextn.{eh_proj,enorm,hnorm,shared_head_norm}` |
| `Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf` | 7.01 GB | 5 |

So his backbone cannot draft; the MTP variant carries a 4-tensor head.

## Obstacles, in order

1. **`Unrecognized model in /models`** — a draft with no explicit model inherits the target's
   config path, and a local `.gguf` target leaves it pointing at the parent directory.
2. **`model type qwen3_5_mtp ... Transformers does not recognize`** — transformers 5.8 knows
   `qwen3_5` but not `qwen3_5_mtp`; vLLM's `_CONFIG_REGISTRY` has no entry either.
   Fix: `patch/sitecustomize.py` registers a `Qwen3_5MTPConfig(Qwen3_5Config)`
   under both `qwen3_5_mtp` and `qwen3_5_moe_mtp` (loaded via `PYTHONPATH`).
3. **`Can't get gguf config for qwen3_5_mtp`** — `config_parser.py` resolves architecture via
   `get_adapter_architecture`, which does not know the MTP types.
   Fix: `patch_mtp.sh` inserts a fallback returning `Qwen3_5MTP` / `Qwen3_5MoeMTP`.
4. **`Prism Hadamard weight 'blk.59.attn_k.weight' is not present in the GGUF name map`** —
   pointing the draft at the full 7.66 GB file makes the loader run the Hadamard/quant
   transforms over the whole backbone. Fix: `extract_mtp_min.py` writes a **54 MB draft**
   holding only the four nextn tensors plus `block_count` / `nextn_predict_layers`.
5. **Engine accepts MTP, then dies in `torch.compile`:**

```
WARNING [speculative.py:1381] Enabling num_speculative_tokens > 1 will run multiple times
        of forward on same MTP layer
...
torch._dynamo.exc.TorchRuntimeError: RuntimeError when making fake tensor call
  call_method split(*(FakeTensor(..., size=(s47, 0), dtype=torch.bfloat16),
                      [12288, 1024, 1024]), ...
```

## Diagnosis of the last failure

The draft loads and the speculator runs, but the `nextn` block carries **only** the projection
and the three norms — no attention or MLP weights. The zero-width `split` is the speculator
building `qkv_proj`/`gate_up_proj` for `mtp.layers.0` from tensors that are not in the file.
This architecture's MTP head therefore **shares the backbone's transformer body**; the plugin's
adapter assumes a self-contained block. Making it work needs the draft's layer weights tied to
the backbone's last layer (code change in the adapter/speculator), not a flag.

## State

- our service restored on `:8000` (HTTP 200)
- his container stopped; nothing published claims MTP works
- artifacts: `extract_mtp_min.py`, `patch_mtp.sh`, `patch/sitecustomize.py`, `serve_mtp.sh`,
  `models/mtp-draft/{config.json,nextn-draft.gguf}` (54 MB, 4 tensors)
