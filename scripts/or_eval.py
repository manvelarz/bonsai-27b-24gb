#!/usr/bin/env python3
"""Run the same 45-prompt quality suite through OpenRouter, with provider pinned.

Usage: python3 or_eval.py <label> <provider> [quant]
Saves eval_<label>.json in the same schema as quality_eval.py.

The prompts are not duplicated: they are read out of quality_eval.py so both
harnesses can never drift apart.
"""
import json, sys, time, urllib.request, pathlib, re

HERE = pathlib.Path(__file__).resolve().parent
LABEL = sys.argv[1]
PROVIDER = sys.argv[2] if len(sys.argv) > 2 else "Chutes"
QUANT = sys.argv[3] if len(sys.argv) > 3 else "fp8"
REASONING = (sys.argv[4].lower() == "on") if len(sys.argv) > 4 else False
MAXTOK = 1500 if REASONING else 400

# pull PROMPTS out of quality_eval.py without executing its main body
src = (HERE / "quality_eval.py").read_text()
head = src.split("def call(")[0]
ns = {}
exec(head, ns)
PROMPTS = ns["PROMPTS"]

key = re.search(r'(?m)^\s*(?:export\s+)?OPENROUTER_API_KEY\s*=\s*["\']?([^"\'\s]+)',
                (pathlib.Path.home() / ".hermes/.env").read_text()).group(1)

def call(prompt, max_tokens=None):
    max_tokens = max_tokens or MAXTOK
    body = json.dumps({
        "model": "qwen/qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": 42,
        "reasoning": {"enabled": REASONING},
        "provider": {"order": [PROVIDER], "allow_fallbacks": False,
                     "require_parameters": True},
    }).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=400) as r:
        d = json.load(r)
    dt = time.time() - t0
    if "error" in d:
        raise RuntimeError(str(d["error"])[:200])
    ch = d["choices"][0]["message"]
    text = (ch.get("content") or "").strip()
    return {"text": text, "ctok": d.get("usage", {}).get("completion_tokens"),
            "dt": round(dt, 2), "provider": d.get("provider"),
            "reasoning_tokens": (d.get("usage", {}).get("completion_tokens_details") or {}).get("reasoning_tokens")}

out = {"label": LABEL, "model": "qwen/qwen3.8-27b", "provider": PROVIDER,
       "quant": QUANT, "reasoning": REASONING, "results": []}
for p in PROMPTS:
    try:
        r = call(p["q"])
    except Exception as e:
        r = {"text": "", "error": str(e)[:200]}
    r.update({"id": p["id"], "cat": p["cat"], "expect": p["expect"]})
    out["results"].append(r)
    print(f'{p["id"]:3} {p["cat"]:11} {r.get("ctok")}t {r.get("dt")}s  {r["text"][:70].replace(chr(10)," ")}', flush=True)

dest = HERE / f"eval_{LABEL}.json"
dest.write_text(json.dumps(out, indent=2))
print("saved", dest)
print("providers used:", sorted({r.get("provider") for r in out["results"] if r.get("provider")}))
