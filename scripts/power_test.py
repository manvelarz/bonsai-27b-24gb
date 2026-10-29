#!/usr/bin/env python3
"""One identical long-context test, measured twice: at the current power cap and at max.

Reports TTFT, prefill rate, decode rate, and peak power draw observed during the run.

Usage: python3 power_test.py <label>
"""
import json, sys, time, urllib.request, threading, subprocess, uuid

LABEL = sys.argv[1]
PORT = 8000
BASE = f"http://127.0.0.1:{PORT}/v1"
MODEL = "Bonsai-2-27B"
TARGET_TOKENS = 40000
OUT_TOKENS = 600


def ntok(text):
    body = json.dumps({"model": MODEL, "prompt": text}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/tokenize", data=body,
                                 headers={"Content-Type": "application/json"})
    return len(json.load(urllib.request.urlopen(req, timeout=120))["tokens"])


SENT = ("The archive room holds paper records from the harbour authority, and the clerk "
        "notes the weather in the margin of every page. ")
NEEDLE = "The access code for the north vault is ZEBRA-4471."


def build(target):
    nonce = f"NONCE-{uuid.uuid4()}\n"
    approx = max(1, (target * 4) // len(SENT))
    doc = nonce + SENT * approx
    for _ in range(6):
        n = ntok(doc)
        if abs(n - target) / target < 0.03:
            break
        approx = int(approx * target / max(1, n))
        doc = nonce + SENT * approx
    return doc + "\n" + NEEDLE + "\n\nWrite a ~300 word summary of this document, then on a final line state the access code for the north vault."


def power_sample(stop, out):
    while not stop.is_set():
        try:
            v = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,clocks.sm,clocks.mem,temperature.gpu",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
            w, clk, mem, temp = [x.strip() for x in v.split(",")]
            out.append((float(w), int(clk), int(mem), int(temp)))
        except Exception:
            pass
        time.sleep(0.4)


prompt = build(TARGET_TOKENS)
tokens = ntok(prompt)

body = json.dumps({
    "model": MODEL,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0, "max_tokens": OUT_TOKENS, "seed": 42, "stream": True,
    "stream_options": {"include_usage": True},
    "chat_template_kwargs": {"enable_thinking": False},
}).encode()

stop = threading.Event(); samples = []
th = threading.Thread(target=power_sample, args=(stop, samples), daemon=True); th.start()

t0 = time.time(); first = None; last = None; text = []; usage = {}
req = urllib.request.Request(BASE + "/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                d = json.loads(p)
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if first is None:
                        first = time.time()
                    last = time.time()
                    text.append(piece)
finally:
    stop.set(); th.join(timeout=3)

ttft = (first - t0) if first else 0
ctok = usage.get("completion_tokens") or 0
dec = ctok / max(0.001, (last - first)) if (first and last and last > first) else 0
answer = "".join(text)
res = {
    "label": LABEL,
    "power_limit_w": float(subprocess.run(["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"],
                                          capture_output=True, text=True).stdout.strip()),
    "prompt_tokens": tokens,
    "ttft_s": round(ttft, 2),
    "prefill_tps": round(tokens / ttft) if ttft else None,
    "decode_tps": round(dec, 1),
    "completion_tokens": ctok,
    "needle_ok": "ZEBRA-4471" in answer,
    "peak_power_w": max((x[0] for x in samples), default=None),
    "avg_power_w": round(sum(x[0] for x in samples) / len(samples), 1) if samples else None,
    "peak_sm_mhz": max((x[1] for x in samples), default=None),
    "avg_mem_mhz": round(sum(x[2] for x in samples) / len(samples)) if samples else None,
    "peak_temp_c": max((x[3] for x in samples), default=None),
    "seconds": round(time.time() - t0, 1),
}
print(json.dumps(res, indent=2))
open(f"/home/manvel/ai/bonsai/power_{LABEL}.json", "w").write(json.dumps(res, indent=2))
