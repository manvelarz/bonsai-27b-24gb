#!/usr/bin/env python3
"""Generate the report charts from the raw JSON measurements.

Inputs (same folder or SRC dir): sweep_*.json, load_*.json, eval_*.json
Outputs: ../charts/*.svg (and .png for social previews)
"""
import json, os, glob, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = pathlib.Path(os.environ.get("SRC", "/home/manvel/ai/bonsai"))
OUT = pathlib.Path(os.environ.get("OUT", str(pathlib.Path(__file__).parent.parent / "charts")))
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linestyle": "--", "font.family": "DejaVu Sans",
})
C = {"llama_q8_0": "#d97706", "bf16_mtp3": "#2563eb", "fp8_mtp3": "#059669"}
NAME = {"llama_q8_0": "llama.cpp · q8_0 KV · 262k",
        "bf16_mtp3": "vLLM · bf16 KV · 114k",
        "fp8_mtp3": "vLLM · fp8 KV + MTP n=3 · 262k"}

# ---------- 1. TTFT vs context ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for path in sorted(glob.glob(str(SRC / "sweep_*.json"))):
    d = json.loads(pathlib.Path(path).read_text())
    pts = [p for p in d["points"] if "ttft_s" in p]
    if not pts:
        continue
    x = [p["prompt_tokens"] for p in pts]
    ax1.plot(x, [p["ttft_s"] for p in pts], "o-", color=C.get(d["label"], "#888"),
             label=NAME.get(d["label"], d["label"]), lw=2, ms=6)
    ax2.plot(x, [p["prefill_tps"] for p in pts], "o-", color=C.get(d["label"], "#888"),
             label=NAME.get(d["label"], d["label"]), lw=2, ms=6)
    if d["label"] == "llama_q8_0":
        for p in pts:
            ax1.annotate(f'{p["ttft_s"]:.0f}s', (p["prompt_tokens"], p["ttft_s"]),
                         textcoords="offset points", xytext=(0, -14), fontsize=8, ha="center",
                         color=C["llama_q8_0"])

ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel("prompt tokens"); ax1.set_ylabel("time to first token (s)")
ax1.set_title("Time to first token vs prompt length")
ax1.legend(fontsize=7.5, frameon=False)
ax2.set_xscale("log")
ax2.set_xlabel("prompt tokens"); ax2.set_ylabel("prefill throughput (tok/s)")
ax2.set_title("Prefill throughput vs prompt length")
ax2.legend(fontsize=7.5, frameon=False)
fig.suptitle("One RTX 3090 (24 GB) · same filler + needle prompt, cache-busted", fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "ttft-vs-context.svg", bbox_inches="tight")
fig.savefig(OUT / "ttft-vs-context.png", bbox_inches="tight", dpi=140)
plt.close(fig)

# ---------- 2. aggregate throughput vs concurrency ----------
LOAD_SRC = {
    "llama_q8_0": str(REPO / "scripts/load_llama_q8_0.json"),
    "bf16_mtp3": str(REPO / "data/load_bf16_retest.json"),
    "fp8_mtp3": str(REPO / "data/load_fp8_mtp3_262k.json"),
}
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for label, path in LOAD_SRC.items():
    p = pathlib.Path(path)
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    lv = d.get("levels", {})
    xs = [1] + sorted(int(k) for k in lv)
    ys = [d.get("solo", {}).get("tps")] + [lv[str(k)]["aggregate_tps"] for k in sorted(int(x) for x in lv)]
    ax.plot(xs, ys, "o-", color=C[label], lw=2, ms=7, label=NAME[label])
    for x, y in zip(xs, ys):
        if y:
            ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 7),
                        fontsize=8, ha="center")
ax.set_xlabel("concurrent streams"); ax.set_ylabel("aggregate throughput (tok/s)")
ax.set_title("Aggregate throughput vs concurrency (short prompts)")
ax.set_xticks([1, 2, 4])
ax.set_xlim(0.7, 4.3)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "throughput-vs-concurrency.svg", bbox_inches="tight")
fig.savefig(OUT / "throughput-vs-concurrency.png", bbox_inches="tight", dpi=140)
plt.close(fig)

# ---------- 3. quality accuracy (computed from eval_*.json) ----------
import re
def norm(x):
    x = (x or "").lower()
    x = re.sub(r"[`*_\'\"]", "", x)
    return re.sub(r"\s+", " ", x).strip()

FIX = {"a1": "587"}
CUSTOM = {"i2": "no_e", "l4": "oranges"}

def score(pid, res):
    n = norm(res.get("text", ""))
    if pid in CUSTOM:
        if CUSTOM[pid] == "no_e":
            return "e" not in n
        return CUSTOM[pid] in n
    exp = norm(FIX.get(pid, res.get("expect") or ""))
    return (exp in n) if exp else None

scores, totals = {}, {}
for label in ["llama_q8_0", "bf16_mtp3", "fp8_mtp3"]:
    f = SRC / f"eval_{label}.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    ok = tot = 0
    for r in d["results"]:
        v = score(r["id"], r)
        if v is None:
            continue
        tot += 1
        ok += bool(v)
    scores[label] = (ok, tot)
    totals[label] = len(d["results"])

fig, ax = plt.subplots(figsize=(6.4, 3.8))
labels = list(scores)
vals = [100 * s[0] / s[1] for s in scores.values()]
import math
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z*z/n
    c_ = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return 100*(c_ - h), 100*(c_ + h)

errs = [wilson(*scores[l]) for l in labels]
bars = ax.bar(range(len(labels)), vals, color=[C[l] for l in labels], width=0.55,
              yerr=[[vals[i] - errs[i][0] for i in range(len(vals))],
                    [errs[i][1] - vals[i] for i in range(len(vals))]], capsize=5,
              error_kw={"ecolor": "#444", "lw": 1.2, "alpha": 0.8})
ax.set_xticks(range(len(labels)))
ax.set_xticklabels([NAME[l].replace(" · ", "\n") for l in labels], fontsize=8)
ax.set_ylabel("verifiable answers correct (%)")
ax.set_ylim(0, 100)
ax.set_title(f"Quality suite: {scores[labels[0]][1]} verifiable answers of {totals[labels[0]]} prompts")
for b, l in zip(bars, labels):
    c_, n_ = scores[l]
    ax.annotate(f"{c_}/{n_}", (b.get_x() + b.get_width() / 2, errs[labels.index(l)][1]),
                textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "quality-accuracy.svg", bbox_inches="tight")
fig.savefig(OUT / "quality-accuracy.png", bbox_inches="tight", dpi=140)
plt.close(fig)

# ---------- 4. decode throughput (long outputs only) ----------
dl = json.loads((REPO / "data/decode_long.json").read_text())
fig, ax = plt.subplots(figsize=(6.2, 4.0))
for cfg in ["llama_q8_0", "bf16_mtp3", "fp8_mtp3"]:
    pts = [p for p in dl["points"] if p["config"] == cfg]
    if not pts:
        continue
    ax.plot([p["ctx"] for p in pts], [p["decode_tps"] for p in pts], "o",
            color=C[cfg], ms=8, label=NAME[cfg])
    off = {"llama_q8_0": (0, 10), "bf16_mtp3": (0, -14), "fp8_mtp3": (0, 10)}[cfg]
    for p in pts:
        ax.annotate(f'{p["decode_tps"]:.0f} t/s', (p["ctx"], p["decode_tps"]),
                    textcoords="offset points", xytext=off, fontsize=8, ha="center", color=C[cfg])
ax.set_xscale("log")
ax.set_xlabel("context length (prompt tokens)")
ax.set_ylabel("decode throughput (tok/s)")
ax.set_ylim(0, 80)
ax.set_title("Decode speed at depth (outputs >= 300 tokens)")
ax.legend(fontsize=7.5, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "decode-vs-context.svg", bbox_inches="tight")
fig.savefig(OUT / "decode-vs-context.png", bbox_inches="tight", dpi=140)
plt.close(fig)

print("charts written to", OUT)
for f in sorted(OUT.glob("*")):
    print(" ", f.name, f.stat().st_size, "B")
