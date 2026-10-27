#!/usr/bin/env python3
"""A/B KV-cache-dtype test for Bonsai-2-27B.
Metrics per prompt: TTFT, decode t/s, total latency, tokens.
Plus concurrency load test (4 and 8 parallel) -> aggregate t/s.
Greedy (temperature=0), seed=0. Usage: ab_kv_test.py <label>
"""
import json, sys, time, urllib.request, pathlib, threading
from concurrent.futures import ThreadPoolExecutor

LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"
URL = "http://127.0.0.1:8000/v1/chat/completions"
OUT = pathlib.Path(__file__).parent
OUT.mkdir(exist_ok=True)

filler = ("The quarterly report of the Glimmerbrook Trading Company was "
          "reviewed by the committee on Tuesday. " * 260)
needle = "IMPORTANT MEMO: the access code for the archive room is ZEBRA-4471. "
long_prompt = ("Read the following archive text and answer the question at the end.\n\n"
               + needle + filler
               + "\n\nQuestion: What is the access code for the archive room? Answer with just the code.")

PROMPTS = [
    ("fact", "What is the capital of Australia? One sentence."),
    ("math", "What is 17 * 23? Reply with just the number."),
    ("logic", "If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies? Answer yes/no and one short reason."),
    ("reason", "A farm has chickens and cows. 30 heads, 86 legs. How many chickens? Show one line of work."),
    ("creative", "Write a 4-line poem about the sea."),
    ("instruction", "List exactly 5 words starting with 'un', comma separated, nothing else."),
    ("needle_long", long_prompt),
]

def stream_call(prompt, max_tokens=300, timeout=300):
    """One streaming chat completion. Returns dict with text + timing metrics."""
    body = json.dumps({
        "model": "Bonsai-2-27B",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "top_p": 1.0, "max_tokens": max_tokens,
        "seed": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; text = []; nchunks = 0
    usage = {}; err = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"): continue
                payload = line[5:].strip()
                if payload == "[DONE]": break
                try: obj = json.loads(payload)
                except Exception: continue
                if obj.get("usage"): usage = obj["usage"]
                for ch in obj.get("choices", []):
                    delta = ch.get("delta", {}) or {}
                    frag = delta.get("content") or delta.get("reasoning_content") or ""
                    if frag:
                        if ttft is None: ttft = time.time() - t0
                        text.append(frag); nchunks += 1
    except Exception as e:
        err = str(e)
    total = time.time() - t0
    out = "".join(text)
    ctok = usage.get("completion_tokens") or 0
    decode_s = (total - (ttft or 0)) if ttft is not None else 0
    return {
        "output": out, "error": err,
        "ttft_s": round(ttft, 3) if ttft else None,
        "total_s": round(total, 2),
        "completion_tokens": ctok,
        "decode_tps": round(ctok / decode_s, 1) if decode_s > 0.05 and ctok else None,
        "stream_chunks": nchunks,
        "prompt_tokens": usage.get("prompt_tokens"),
    }

# ---- single-request passes ----
results = []
for name, prompt in PROMPTS:
    m = stream_call(prompt)
    results.append({"name": name, **m})
    print(f"[{LABEL}] {name}: ttft={m['ttft_s']}s total={m['total_s']}s "
          f"tok={m['completion_tokens']} decode={m['decode_tps']}t/s "
          f"-> {m['error'] or (m['output'][:70].replace(chr(10),' '))}")

# ---- concurrency passes (same prompt, N parallel) ----
load = {}
for n in (4, 8):
    pr = "Write a detailed 250-word explanation of how tides work."
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        ms = list(ex.map(lambda _: stream_call(pr, max_tokens=300), range(n)))
    wall = time.time() - t0
    tot = sum(m["completion_tokens"] for m in ms)
    errs = sum(1 for m in ms if m["error"])
    load[str(n)] = {"wall_s": round(wall, 2), "total_completion_tokens": tot,
                    "aggregate_tps": round(tot / wall, 1) if wall else None,
                    "errors": errs,
                    "avg_ttft": round(sum((m["ttft_s"] or 0) for m in ms) / n, 3)}
    print(f"[{LABEL}] load x{n}: wall={wall:.1f}s tokens={tot} agg={load[str(n)]['aggregate_tps']}t/s errs={errs}")

(OUT / f"ab_{LABEL}.json").write_text(
    json.dumps({"label": LABEL, "prompts": results, "load": load}, indent=2, ensure_ascii=False))
print(f"saved ab_{LABEL}.json")