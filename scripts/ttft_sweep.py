#!/usr/bin/env python3
"""Uniform TTFT / prefill / decode sweep across context lengths (streaming).

Same filler, same needle at the end, same settings for every backend.
Usage: python3 ttft_sweep.py <label> <port> [5000,40000,110000]
"""
import json, sys, time, urllib.request, pathlib, uuid

LABEL = sys.argv[1]
PORT = sys.argv[2] if len(sys.argv) > 2 else "8000"
CTX = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "5000,40000,110000").split(",")]
BASE = f"http://127.0.0.1:{PORT}/v1"
MODEL = "Bonsai-2-27B"

FILLER = ("The archive room holds paper records from a shipping company. "
          "Each folder describes a voyage, its cargo and its crew. ")
TAIL = "\n\nIMPORTANT CODE: ZEBRA-4471. What is the important code? Answer with the code only.\n"

def ntok(text):
    # vLLM uses {"prompt": ...}, llama.cpp uses {"content": ...}
    for key in ("prompt", "content"):
        body = json.dumps({"model": MODEL, key: text}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/tokenize", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            toks = json.load(r).get("tokens") or []
        if toks:
            return len(toks)
    raise RuntimeError("tokenizer returned no tokens for both payload shapes")

def build(target):
    per = ntok(FILLER)
    reps = max(1, int(target / per))
    while True:
        # unique nonce at the very start busts any prefix/prompt cache
        text = f"NONCE-{uuid.uuid4().hex}. " + FILLER * reps + TAIL
        n = ntok(text)
        if abs(n - target) <= max(50, target // 100):
            return text, n
        reps = max(1, int(reps * (target - ntok(TAIL)) / max(1, n - ntok(TAIL))))

def stream_call(prompt):
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "max_tokens": 40, "seed": 42, "stream": True,
                       "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; text = ""; usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if first is None:
                        first = time.time()
                    text += piece
    end = time.time()
    return {"ttft": (first or end) - t0, "total": end - t0, "text": text.strip(), "usage": usage}

dest = pathlib.Path(__file__).with_name(f"sweep_{LABEL}.json")
out = {"label": LABEL, "port": PORT, "points": []}
done = set()
if dest.exists():
    prev = json.loads(dest.read_text())
    out["points"] = [p for p in prev.get("points", []) if "ttft_s" in p]
    done = {p["target"] for p in out["points"]}

for target in CTX:
    if target in done:
        print(f"[{LABEL}] {target} tok (already measured, skipping)", flush=True)
        continue
    prompt, n = build(target)
    try:
        r = stream_call(prompt)
        ttft = r["ttft"]; gen = max(0.001, r["total"] - ttft)
        ctok = r["usage"].get("completion_tokens") or max(1, len(r["text"].split()))
        ptok = r["usage"].get("prompt_tokens") or n
        rec = {"target": target, "prompt_tokens": ptok, "ttft_s": round(ttft, 2),
               "prefill_tps": round(ptok / ttft, 1), "decode_tps": round(ctok / gen, 2),
               "ctok": ctok, "needle_ok": "ZEBRA-4471" in r["text"]}
        print(f"[{LABEL}] {ptok:>7} tok | TTFT {ttft:7.1f}s | prefill {ptok/ttft:6.0f} t/s | "
              f"decode {rec['decode_tps']:5.1f} t/s | needle={rec['needle_ok']}", flush=True)
    except Exception as e:
        rec = {"target": target, "prompt_tokens": n, "error": str(e)[:200]}
        print(f"[{LABEL}] {n} tok ERROR {rec['error']}", flush=True)
    out["points"].append(rec)
    out["points"].sort(key=lambda p: p["target"])
    dest.write_text(json.dumps(out, indent=2))
print("saved", dest)
