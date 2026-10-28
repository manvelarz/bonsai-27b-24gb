#!/usr/bin/env python3
"""Long-context needle test against OpenRouter, same protocol as ttft_sweep.py.

Builds a filler document of roughly N tokens, puts the needle at the very end,
asks for it back. Reports TTFT-equivalent (total wall for a non-streaming call),
provider actually used, and whether the needle survived.

Usage: python3 or_needle.py <provider> <tokens,tokens,...>
"""
import json, sys, time, urllib.request, pathlib, re, uuid

PROVIDER = sys.argv[1] if len(sys.argv) > 1 else "Chutes"
TARGETS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "100000").split(",")]

key = re.search(r'(?m)^\s*(?:export\s+)?OPENROUTER_API_KEY\s*=\s*["\']?([^"\'\s]+)',
                (pathlib.Path.home() / ".hermes/.env").read_text()).group(1)

SENT = ("The archive room holds paper records from the harbour authority, and the clerk "
        "notes the weather in the margin of every page. ")
NEEDLE = "The access code for the north vault is ZEBRA-4471."
QUESTION = "What is the access code for the north vault? Answer with the code only."

def build(target_tokens):
    nonce = f"NONCE-{uuid.uuid4()}\n"
    approx = max(1, (target_tokens * 4) // len(SENT))
    return nonce + SENT * approx + "\n" + NEEDLE + "\n\n" + QUESTION

def ntok(text):
    body = json.dumps({"model": "qwen/qwen3.8-27b", "prompt": text}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/tokenize", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    return json.load(urllib.request.urlopen(req, timeout=120))

rows = []
for target in TARGETS:
    text = build(target)
    try:
        n = len(ntok(text)["tokens"])
    except Exception:
        n = round(len(text) / 3.9)  # provider tokenizer endpoint loosely available; approx
    body = json.dumps({
        "model": "qwen/qwen3.8-27b",
        "messages": [{"role": "user", "content": text}],
        "temperature": 0, "max_tokens": 30, "seed": 42,
        "reasoning": {"enabled": False},
        "provider": {"order": [PROVIDER], "allow_fallbacks": False},
    }).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
        dt = time.time() - t0
        if "error" in d:
            raise RuntimeError(str(d["error"])[:160])
        txt = d["choices"][0]["message"]["content"] or ""
        ok = "ZEBRA-4471" in txt
        rows.append({"target": target, "tokens": n, "wall_s": round(dt, 1),
                     "needle": ok, "provider": d.get("provider"), "answer": txt[:40]})
    except Exception as e:
        rows.append({"target": target, "tokens": n, "error": str(e)[:160]})
    print(json.dumps(rows[-1]), flush=True)

print("\n" + json.dumps(rows, indent=2))
import pathlib as _pl
(_pl.Path(__file__).with_name("or_needle.json")).write_text(json.dumps({"provider": PROVIDER, "rows": rows}, indent=2))
print("saved or_needle.json")
