#!/usr/bin/env python3
"""Real-world concurrency test: N simultaneous LONG prompts (each ~T tokens).

This is the test that matters for Man's requirement: 2 (nice) to 4 (happy)
parallel agent sessions, each with a large prompt.

Usage: python3 long_conc.py <label> <n_parallel> <tokens_per_prompt>
"""
import json, sys, time, urllib.request, pathlib
from concurrent.futures import ThreadPoolExecutor

LABEL = sys.argv[1]
N = int(sys.argv[2])
TOK = int(sys.argv[3])
URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Bonsai-2-27B"
OUT = pathlib.Path(__file__).parent

SENT = "The quarterly report of the Glimmerbrook Trading Company was reviewed by the committee on Tuesday. "
NEEDLE = "IMPORTANT MEMO: the access code for the archive room is ZEBRA-4471. "
PER_SENT = 20


def make_prompt(tok, tag):
    n = max(1, int(tok / PER_SENT))
    body = [SENT] * n
    # unique needle position per stream so prefix caching can't hide anything
    body.insert(int(n * (0.2 + 0.15 * tag)), f"MEMO {tag}: the access code for the archive room is ZEBRA-44{70 + tag}. ")
    return ("Archive text follows.\n\n" + "".join(body) +
            f"\n\nQuestion: what is the access code in MEMO {tag}? Answer with just the code.")


def call(tag, tok, max_tokens=120, timeout=1800):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": make_prompt(tok, tag)}],
        "temperature": 0, "max_tokens": max_tokens, "seed": 0, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
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
    out = "".join(text)
    ctok = usage.get("completion_tokens") or 0
    dec = (total - (ttft or 0)) if ttft else 0
    return {"stream": tag, "ttft": round(ttft, 1) if ttft else None,
            "total": round(total, 1), "ptok": usage.get("prompt_tokens"), "ctok": ctok,
            "tps": round(ctok / dec, 1) if dec > 0.3 and ctok else None,
            "needle_ok": f"ZEBRA-44{70 + tag}" in out,
            "out_tail": out[-40:].replace("\n", " "), "err": err}


t0 = time.time()
with ThreadPoolExecutor(max_workers=N) as ex:
    ms = list(ex.map(lambda i: call(i + 1, TOK), range(N)))
wall = time.time() - t0
ptot = sum(m["ptok"] or 0 for m in ms)
ctot = sum(m["ctok"] for m in ms)
res = {"label": LABEL, "n": N, "target_tokens": TOK, "wall_s": round(wall, 1),
       "total_prompt_tokens": ptot, "total_completion_tokens": ctot,
       "aggregate_decode_tps": round(ctot / wall, 1) if wall else None,
       "errors": [m["err"] for m in ms if m["err"]],
       "needles_ok": sum(1 for m in ms if m["needle_ok"]), "streams": ms}
for m in ms:
    print(f"[{LABEL}] stream{m['stream']}: ptok={m['ptok']} ttft={m['ttft']}s "
          f"ctok={m['ctok']} t/s={m['tps']} needle={m['needle_ok']} err={m['err']}")
print(f"[{LABEL}] N={N} x ~{TOK} tok: wall={wall:.1f}s prompt_tokens={ptot} "
      f"needles_ok={res['needles_ok']}/{N} errors={len(res['errors'])}")
print(f"    tail: {ms[0]['out_tail']!r}")

(OUT / f"long_{LABEL}.json").write_text(json.dumps(res, indent=2))
print(f"saved long_{LABEL}.json")
