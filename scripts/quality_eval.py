#!/usr/bin/env python3
"""Quality eval: identical deterministic prompts across KV/quant configurations.

Usage: python3 quality_eval.py <label> [base_url]
Saves eval_<label>.json next to this script.
Thinking is disabled so only the answer is compared.
"""
import json, sys, time, urllib.request, pathlib

BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000/v1"
LABEL = sys.argv[1]
MODEL = "Bonsai-2-27B"

P = lambda i, cat, q, expect=None: {"id": i, "cat": cat, "q": q, "expect": expect}

PROMPTS = [
    # arithmetic
    P("a1", "math", "Compute (47 * 13) - (96 / 4). Answer with the number only.", "515"),
    P("a2", "math", "What is 17.5% of 240? Answer with the number only.", "42"),
    P("a3", "math", "A shirt costs 80 EUR. It is discounted 25%, then taxed 21% on the discounted price. Final price? Number only, 2 decimals.", "72.60"),
    P("a4", "math", "Sum the integers from 1 to 100. Number only.", "5050"),
    P("a5", "math", "Solve 3x + 7 = 31. Give x only.", "8"),
    # code
    P("c1", "code", "Write a Python function `is_palindrome(s)` that ignores case and non-alphanumeric characters. Code only.", None),
    P("c2", "code", "What does `print([i*i for i in range(5)])` output? Output only.", "[0, 1, 4, 9, 16]"),
    P("c3", "code", "Fix this buggy line and return only the fixed line: for i in range(len(lst)): print(lst[i+1])", None),
    P("c4", "code", "SQL: table users(id, email, created_at). Return emails created in 2025 ordered newest first. Query only.", None),
    # structured extraction
    P("j1", "json", 'Extract as JSON with keys name, amount, currency: "Invoice from Acme GmbH for 1.250,00 EUR due 2026-11-30". JSON only.', None),
    P("j2", "json", 'Return JSON {"sentiment": ..., "confidence": 0-1} for: "The update broke everything and support never answered."', None),
    P("j3", "json", 'Convert to JSON array of {city, temp_c}: "Madrid 21C, Oslo 4C, Lima 19C". JSON only.', None),
    # translation
    P("t1", "translation", "Translate to English: 'El servidor se cayó a las tres de la mañana y nadie lo notó hasta el desayuno.'", None),
    P("t2", "translation", "Translate to Spanish: 'The cache is warm, so the first token arrives almost instantly.'", None),
    P("t3", "translation", "Translate to German: 'Please restart the service before deploying.'", None),
    P("t4", "translation", "Translate to English: 'Necesito que revises los números del último informe.'", None),
    # summarization fidelity
    P("s1", "summary", "Summarize in exactly one sentence: 'The company reported quarterly revenue of 4.2 million euros, up 18 percent year over year, driven mainly by subscriptions, while hardware sales declined for the third consecutive quarter.'", None),
    P("s2", "summary", "List the three concrete numbers in: 'Latency dropped from 840 ms to 310 ms after the upgrade, and error rate fell to 0.4 percent.'", None),
    P("s3", "summary", "Summarize in max 10 words: 'Running 27B parameters on a single 24 GB GPU requires quantized weights and quantized KV cache.'", None),
    # instruction following
    P("i1", "instruction", "Reply with exactly three words, no punctuation.", None),
    P("i2", "instruction", "Write a sentence about rain that does not contain the letter 'e'.", None),
    P("i3", "instruction", "Give exactly 3 bullet points about backups. Use '-' as bullet marker.", None),
    P("i4", "instruction", "Answer in all uppercase: what is the capital of Portugal?", "LISBON"),
    # factual
    P("f1", "factual", "What is the capital of Australia? One word.", "Canberra"),
    P("f2", "factual", "How many bytes in one gibibyte? Number only.", "1073741824"),
    P("f3", "factual", "Which is larger: 0.1% of 1000 or 1% of 10? Answer with the larger value only.", "1"),
    P("f4", "factual", "In what year did the Berlin Wall fall? Year only.", "1989"),
    # logic
    P("l1", "logic", "All Bloops are Razzies. All Razzies are Lazzies. Are all Bloops Lazzies? Yes or no.", "Yes"),
    P("l2", "logic", "If today is Wednesday, what day is it 100 days from now? Day only.", "Friday"),
    P("l3", "logic", "A is taller than B. C is shorter than B. Who is tallest? Letter only.", "A"),
    P("l4", "logic", "There are 3 boxes: one has apples, one oranges, one mixed. All labels are wrong. You pick one fruit from the box labeled 'mixed'. It is an apple. What is in the box labeled 'apples'?", None),
    # --- extended short-answer set (added for stronger statistics) ---
    P("a6", "math", "What is 30% of 90? Number only.", "27"),
    P("a7", "math", "Compute 7/8 as a decimal, 3 decimals. Number only.", "0.875"),
    P("a8", "math", "Average of 12, 18, 21, 25? Number only.", "19"),
    P("a9", "math", "How many minutes in 2.75 hours? Number only.", "165"),
    P("a10", "math", "1000 EUR at 5% simple interest for 3 years: total interest? Number only.", "150"),
    P("a11", "math", "GCD of 84 and 126? Number only.", "42"),
    P("c5", "code", "What does `len(set('banana'))` return? Number only.", "3"),
    P("c6", "code", "What is the output of `print('5' + '3')`? Output only.", "53"),
    P("f5", "factual", "How many bits in one byte? Number only.", "8"),
    P("f6", "factual", "What is the chemical symbol for gold? Symbol only.", "Au"),
    P("f7", "factual", "Which planet is closest to the Sun? One word.", "Mercury"),
    P("f8", "factual", "How many degrees in a right angle? Number only.", "90"),
    P("l5", "logic", "If a train travels 120 km in 1.5 hours, what is its speed in km/h? Number only.", "80"),
    P("l6", "logic", "If no cats are reptiles, and all reptiles are animals, can a cat be an animal? Yes or no.", "Yes"),
]

def call(prompt):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 400,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(BASE.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    dt = time.time() - t0
    ch = d["choices"][0]["message"]
    text = (ch.get("content") or "").strip()
    u = d.get("usage", {})
    return {"text": text, "ctok": u.get("completion_tokens"), "dt": round(dt, 2)}

out = {"label": LABEL, "base": BASE, "model": MODEL, "results": []}
for p in PROMPTS:
    try:
        r = call(p["q"])
        r.update({"id": p["id"], "cat": p["cat"], "expect": p["expect"]})
    except Exception as e:
        r = {"id": p["id"], "cat": p["cat"], "expect": p["expect"], "text": "", "error": str(e)[:200]}
    out["results"].append(r)
    print(f'{p["id"]:3} {p["cat"]:11} {r.get("ctok")}t {r.get("dt")}s  {r["text"][:70].replace(chr(10)," ")}', flush=True)

dest = pathlib.Path(__file__).with_name(f"eval_{LABEL}.json")
dest.write_text(json.dumps(out, indent=2))
print("saved", dest)
