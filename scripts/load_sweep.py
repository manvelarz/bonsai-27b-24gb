#!/usr/bin/env python3
"""Concurrency sweep for bonsai-vllm: n=2 and n=4 (what Man actually needs).

Per request: TTFT, decode t/s, completion tokens.
Per level: wall time, aggregate t/s, per-stream avg t/s, error count.
Also one solo reference call + server health check after each level.
Usage: python3 load_sweep.py <label> [2,4]
"""
import json, sys, time, urllib.request, pathlib
from concurrent.futures import ThreadPoolExecutor

LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"
LEVELS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "2,4").split(",")]
URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Bonsai-2-27B"
OUT = pathlib.Path(__file__).parent


def stream_call(prompt, max_tokens=400, timeout=600):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "top_p": 1.0, "max_tokens": max_tokens, "seed": 0,
        "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; text = []; usage = {}; err = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                p = line[5:].strip()
                if p == "[DONE]":
                    break
                try:
                    o = json.loads(p)
                except Exception:
                    continue
                if o.get("usage"):
                    usage = o["usage"]
                for ch in o.get("choices", []):
                    d = ch.get("delta", {}) or {}
                    f = d.get("content") or d.get("reasoning_content") or ""
                    if f:
                        if ttft is None:
                            ttft = time.time() - t0
                        text.append(f)
    except Exception as e:
        err = str(e)[:200]
    total = time.time() - t0
    ctok = usage.get("completion_tokens") or 0
    dec = (total - (ttft or 0)) if ttft else 0
    return {"ttft": round(ttft, 2) if ttft else None, "total": round(total, 2),
            "ctok": ctok, "ptok": usage.get("prompt_tokens"),
            "tps": round(ctok / dec, 1) if dec > 0.3 and ctok else None,
            "err": err, "words": len("".join(text).split())}


def health():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as r:
            return r.status
    except Exception as e:
        return f"DOWN: {e}"


PROMPT = "Write a detailed 250-word explanation of how ocean tides work."

res = {"label": LABEL, "solo": None, "levels": {}}

solo = stream_call(PROMPT, max_tokens=400)
res["solo"] = solo
print(f"[{LABEL}] solo: ttft={solo['ttft']}s ctok={solo['ctok']} t/s={solo['tps']} err={solo['err']}")

for n in LEVELS:
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        ms = list(ex.map(lambda _: stream_call(PROMPT, max_tokens=400), range(n)))
    wall = time.time() - t0
    tot = sum(m["ctok"] for m in ms)
    errs = [m["err"] for m in ms if m["err"]]
    lvl = {"n": n, "wall_s": round(wall, 2), "total_ctok": tot,
           "aggregate_tps": round(tot / wall, 1) if wall else None,
           "per_stream_tps": sorted([m["tps"] for m in ms if m["tps"]]),
           "ttft_s": [m["ttft"] for m in ms],
           "errors": len(errs), "error_sample": errs[0] if errs else None,
           "health_after": health()}
    res["levels"][str(n)] = lvl
    print(f"[{LABEL}] x{n}: wall={wall:.1f}s ctok={tot} agg={lvl['aggregate_tps']}t/s "
          f"per-stream={lvl['per_stream_tps']} errs={lvl['errors']} health={lvl['health_after']}")
    if errs:
        print(f"    first error: {errs[0]}")
    if str(lvl["health_after"]).startswith("DOWN"):
        print("    server died -> aborting remaining levels")
        break

(OUT / f"load_{LABEL}.json").write_text(json.dumps(res, indent=2))
print(f"saved load_{LABEL}.json")
