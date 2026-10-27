#!/usr/bin/env python3
"""vLLM max-context test: short t/s, ~100k needle, 2-way concurrency."""
import json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Bonsai-2-27B"

def call(prompt, max_tokens, timeout=1800):
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
        err = str(e)[:160]
    total = time.time() - t0; text = "".join(parts); ct = usage.get("completion_tokens") or 0
    dec = total - (ttft or 0)
    return {"text": text, "err": err, "ttft": round(ttft, 2) if ttft else None,
            "total": round(total, 1), "ptok": usage.get("prompt_tokens"), "ctok": ct,
            "tps": round(ct/dec, 1) if dec > 0.1 and ct else None}

s = ("The quarterly report of the Glimmerbrook Trading Company was reviewed "
     "by the committee on Tuesday. ")  # 20 tok/sentence (tokenizer-verified)
needle = "ZEBRA-4471"

def needle_prompt(n_tokens):
    n = int(n_tokens / 20)
    body = [s] * n
    body.insert(n // 10, f"IMPORTANT MEMO: the access code for the archive room is {needle}. ")
    return ("Archive text follows.\n\n" + "".join(body) +
            "\n\nQuestion: What is the access code for the archive room? "
            "Answer with just the code, no explanation.")

print("### short prompt")
a = call("What is the capital of France? One sentence.", 80)
print({k: a[k] for k in ("ttft", "total", "ptok", "ctok", "tps", "err")})

print("### ~100k-token needle")
b = call(needle_prompt(100000), 400)
print({k: b[k] for k in ("ttft", "total", "ptok", "ctok", "tps", "err")})
print("needle_found:", needle in (b["text"] or ""))
print("tail:", repr((b["text"] or "")[-160:]))

print("### 2 parallel ~40k requests")
t0 = time.time()
with ThreadPoolExecutor(max_workers=2) as ex:
    res = list(ex.map(lambda _: call(needle_prompt(40000), 400), range(2)))
print(f"wall={time.time()-t0:.1f}s")
for i, r in enumerate(res):
    print(f" req{i}: ttft={r['ttft']}s total={r['total']}s ptok={r['ptok']} needle={needle in (r['text'] or '')} err={r['err']}")