#!/usr/bin/env python3
"""Apples-to-apples test, thinking DISABLED, long-context + long-output.
Usage: python3 a2a_test.py vllm|llama
"""
import json, sys, time, urllib.request

BE = sys.argv[1] if len(sys.argv) > 1 else "vllm"
if BE == "vllm":
    URL = "http://127.0.0.1:8000/v1/chat/completions"; MODEL = "Bonsai-2-27B"
else:
    URL = "http://127.0.0.1:8080/v1/chat/completions"
    MODEL = "models/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf"

def call(prompt, max_tokens, timeout=2400):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; parts = []; usage = {}; err = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"): continue
                p = line[5:].strip()
                if p == "[DONE]": break
                try: o = json.loads(p)
                except Exception: continue
                if o.get("usage"): usage = o["usage"]
                for ch in o.get("choices", []):
                    d = ch.get("delta", {}) or {}
                    f = d.get("content") or d.get("reasoning_content") or ""
                    if f:
                        if ttft is None: ttft = time.time() - t0
                        parts.append(f)
    except Exception as e:
        err = str(e)[:200]
    total = time.time() - t0; text = "".join(parts); ct = usage.get("completion_tokens") or 0
    dec = total - (ttft or 0)
    return {"ttft": round(ttft, 2) if ttft else None, "total": round(total, 1),
            "ptok": usage.get("prompt_tokens"), "ctok": ct,
            "tps": round(ct/dec, 1) if dec > 0.2 and ct else None,
            "err": err, "words": len(text.split()), "text": text}

s = ("The quarterly report of the Glimmerbrook Trading Company was reviewed "
     "by the committee on Tuesday. ")  # 20 tok/sentence

def long_prompt(n_tokens=40000):
    n = int(n_tokens / 20)
    body = [s] * n
    body.insert(n // 10, "IMPORTANT MEMO: the access code for the archive room is ZEBRA-4471. ")
    return ("Archive text follows.\n\n" + "".join(body) +
            "\n\nUsing the archive text above, describe what the committee reviewed and "
            "the recurring schedule in detail (aim for 500+ words), then state the archive "
            "room access code at the very end.")

tests = [
    ("short", "What is the capital of France? One sentence.", 100),
    ("medium", "Explain how ocean tides work. Aim for about 300 words.", 500),
    ("long_ctx_long_out", long_prompt(40000), 900),
]
out = {}
for name, prompt, mx in tests:
    r = call(prompt, mx)
    out[name] = {k: r[k] for k in ("ttft", "total", "ptok", "ctok", "tps", "words", "err")}
    print(f"[{BE}] {name}: {out[name]}")
    tail = (r["text"] or "")[-160:].replace("\n", " ")
    print(f"    tail: {tail!r}")
    if "long_ctx" in name:
        print("    needle_found:", "ZEBRA-4471" in (r["text"] or ""))

with open(f"a2a_{BE}.json", "w") as f:
    json.dump(out, f, indent=2)