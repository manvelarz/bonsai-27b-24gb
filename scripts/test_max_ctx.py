#!/usr/bin/env python3
"""Validate max context: ~240k-token needle recall + short prompt speed."""
import json, time, urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "models/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf"

def call(prompt, max_tokens, timeout=2000):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": max_tokens, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
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
        err = str(e)
    total = time.time() - t0; text = "".join(parts); ct = usage.get("completion_tokens") or 0
    dec = total - (ttft or 0)
    return {"text": text, "err": err, "ttft": round(ttft, 1) if ttft else None,
            "total": round(total, 1), "ptok": usage.get("prompt_tokens"), "ctok": ct,
            "tps": round(ct/dec, 1) if dec > 0.1 and ct else None}

needle = "ZEBRA-4471"
s = ("The quarterly report of the Glimmerbrook Trading Company was reviewed "
     "by the committee on Tuesday. ")  # ~17 tok
n = int(240000 / 20)  # 20 tokens/sentence measured via /tokenize
memo = f"IMPORTANT MEMO: the access code for the archive room is {needle}. "
body = [s] * n
body.insert(n // 10, memo)
prompt = ("Archive text follows.\n\n" + "".join(body) +
          "\n\nQuestion: What is the access code for the archive room? "
          "Answer with just the code, no explanation.")

print("### short prompt t/s")
a = call("What is the capital of France? One sentence.", 80)
print({k: a[k] for k in ("ttft", "total", "ptok", "ctok", "tps", "err")})

print("### ~240k-token needle")
b = call(prompt, 400)
found = needle in (b["text"] or "")
print({k: b[k] for k in ("ttft", "total", "ptok", "ctok", "tps", "err")})
print("needle_found:", found)
print("tail of answer:", repr((b["text"] or "")[-200:]))