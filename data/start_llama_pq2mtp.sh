#!/bin/bash
# Launch the PATCHED llama-server (built from src/ with Hadamard MTP fix)
# running the ProCreations PQ2_0 + MTP GGUF. Own systemd scope.
export LD_LIBRARY_PATH=/path/to/vllm/venv/lib/python3.11/site-packages/nvidia/cu13/lib
cd .
pkill -f bin/llama-server || true; sleep 4
: > llama-server.log
exec systemd-run --user --scope -p Delegate=yes --slice=independent.slice \
  ./src/build/bin/llama-server -m models/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf \
  --host 127.0.0.1 --port 8080 -ngl 99 -fa on -c 262144 --parallel 2 --kv-unified \
  -ctk q8_0 -ctv q8_0 --jinja \
  --spec-type draft-mtp --spec-draft-n-max 2 > llama-server.log 2>&1
