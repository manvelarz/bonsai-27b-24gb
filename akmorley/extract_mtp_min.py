#!/usr/bin/env python3
"""Build a minimal GGUF draft containing only the nextn/MTP block.

Only the handful of metadata keys the GGUF loader needs are copied; the HF
config comes from config.json beside this file.
"""
import sys
from gguf import GGUFReader, GGUFWriter, GGUFValueType

src, dst = sys.argv[1], sys.argv[2]
r = GGUFReader(src)
nextn = [t for t in r.tensors if "nextn" in t.name]
assert nextn, "no nextn tensors in source"
arch = r.fields["general.architecture"].contents()
print("nextn tensors:", len(nextn), "arch:", arch)

w = GGUFWriter(dst, arch)
keep = ["qwen35.block_count", "qwen35.nextn_predict_layers", "general.file_type",
        "general.quantization_version", "general.name"]
for key in keep:
    if key not in r.fields:
        print("  missing", key); continue
    val = r.fields[key].contents()
    print("  copy", key, "=", val)
    try:
        if isinstance(val, str):
            w.add_key_value(key, val, GGUFValueType.STRING)
        else:
            w.add_key_value(key, int(val), GGUFValueType.UINT32)
    except Exception as exc:
        print("   skip", key, type(exc).__name__, exc)

for t in nextn:
    print("  tensor", t.name, t.tensor_type)
    w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote", dst)
