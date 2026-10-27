#!/usr/bin/env python3
"""Score quality_eval outputs across configurations (v2 scoring).

- short answers: expected value must appear in the (normalised) answer
- i2: instruction following -> answer must contain no letter 'e'
- l4: classic mislabelled-boxes puzzle -> answer must be 'oranges'
"""
import json, glob, pathlib, re, difflib

def norm(s):
    s = (s or "").lower()
    s = re.sub(r'[`*_\'"]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

FIX = {"a1": "587"}          # (47*13) - (96/4) = 587; earlier expectation was wrong
CUSTOM = {"i2": "no_e", "l4": "oranges"}

def score(pid, res):
    txt = res.get("text", "")
    n = norm(txt)
    if pid in CUSTOM:
        rule = CUSTOM[pid]
        if rule == "no_e":
            body = re.sub(r'\s+', ' ', n)
            return ("e" not in body), "no letter 'e'"
        return (rule in n), rule
    exp = norm(FIX.get(pid, res.get("expect") or ""))
    if not exp:
        return None, None
    return (exp in n), exp

files = sorted(glob.glob(str(pathlib.Path(__file__).parent / "eval_*.json")))
data = {}
for f in files:
    d = json.loads(pathlib.Path(f).read_text())
    data[d["label"]] = {r["id"]: r for r in d["results"]}
labels = list(data)
ids = list(data[labels[0]].keys())

rows, tally = [], {l: [0, 0] for l in labels}
for i in ids:
    cells, exp = [], None
    for l in labels:
        ok, e = score(i, data[l].get(i, {}))
        exp = exp or e
        if ok is not None:
            tally[l][0] += bool(ok); tally[l][1] += 1
        cells.append("PASS" if ok else ("FAIL" if ok is False else "—"))
    rows.append((i, exp, cells))

print("| id | expected | " + " | ".join(labels) + " |")
print("|" + "---|" * (len(labels) + 2))
for i, exp, cells in rows:
    print(f"| {i} | {exp} | " + " | ".join(cells) + " |")
print()
for l in labels:
    c, n = tally[l]
    print(f"- {l}: {c}/{n} = {100*c/n:.0f}%")

print("\n### Pairwise text agreement (all prompts)")
for a in labels:
    for b in labels:
        if a >= b: continue
        sims, exact = [], 0
        for i in data[a]:
            if i not in data[b]: continue
            x, y = norm(data[a][i].get("text")), norm(data[b][i].get("text"))
            sims.append(difflib.SequenceMatcher(None, x, y).ratio()); exact += (x == y)
        print(f"- {a} vs {b}: mean similarity {sum(sims)/len(sims):.3f}, identical {exact}/{len(sims)}")

print("\n### Throughput during the suite")
for l in labels:
    toks = sum(r.get("ctok") or 0 for r in data[l].values())
    dts = sum(r.get("dt") or 0 for r in data[l].values())
    print(f"- {l}: {toks} tokens in {dts:.1f}s = {toks/dts:.1f} tok/s")
