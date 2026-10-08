#!/usr/bin/env python3
"""Mia's README output claims on a running server, without retaining any text:
drafted replies equal "draft": false (serial) replies, and replies served together (a burst and a staggered wave)
equal the same requests served alone. Prints only per-case equality, token counts and finish reasons."""
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request

API = os.environ.get("API_URL", "http://127.0.0.1:8888").rstrip("/") + "/v1/chat/completions"
MODEL = os.environ.get("MODEL") or sys.exit("set MODEL to the served model id")
PY = "def merge(a, b):\n    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]\n"
OFF = {"chat_template_kwargs": {"enable_thinking": False}}
ON = {"reasoning_effort": "low"}
GREEDY = {"temperature": 0}
CASES = [
    ("prose-greedy", "Write a short paragraph about how rivers shape valleys.", OFF, GREEDY, 200),
    ("prose-seed11", "Write a short paragraph about winter in a mountain village.", OFF, {"temperature": 1.0, "top_p": 0.95, "seed": 11}, 200),
    ("prose-seed22", "Describe a busy morning market in four sentences.", OFF, {"temperature": 1.0, "top_p": 0.95, "seed": 22}, 200),
    ("code-greedy", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.", OFF, GREEDY, 256),
    ("code-seed33", "Write a Python function that checks whether a string is a palindrome, ignoring case and spaces.", OFF, {"temperature": 1.0, "top_p": 0.95, "seed": 33}, 256),
    ("quote-edit", "Return this Python code unchanged except add type hints to the function signature:\n\n" + PY, OFF, GREEDY, 256),
    ("json-greedy", "Return a JSON object with keys name, age and city for a fictional person. Output only the JSON.", OFF, GREEDY, 120),
    ("think-greedy", "What is 17 * 23? Answer with the number.", ON, GREEDY, 400),
    ("think-seed11", "A train leaves at 9:40 and arrives at 13:05. How long is the trip?", ON, {"temperature": 1.0, "top_p": 0.95, "seed": 11}, 600),
    ("list-greedy", "List five prime numbers greater than 100, comma separated.", OFF, GREEDY, 80),
    ("summary-greedy", "Summarize in one sentence: the committee met, reviewed three budget proposals, and approved the second.", OFF, GREEDY, 80),
]


def ask(case, draft=True):
    name, prompt, mode, sampling, max_tokens = case
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, **mode, **sampling}
    if not draft:
        body["draft"] = False
    req = urllib.request.Request(API, json.dumps(body).encode(), {"Content-Type": "application/json"})
    reply = json.load(urllib.request.urlopen(req, timeout=1800))
    msg = reply["choices"][0]["message"]
    text = (msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or "")
    return hashlib.sha256(text.encode()).hexdigest(), reply["usage"]["completion_tokens"], reply["choices"][0]["finish_reason"]


def together(stagger):
    out = [None] * len(CASES)
    def go(i):
        try:
            out[i] = ask(CASES[i])
        except Exception as exc:  # noqa: BLE001
            out[i] = ("error:" + type(exc).__name__, 0, "error")
    threads = []
    for i in range(len(CASES)):
        t = threading.Thread(target=go, args=(i,)); t.start(); threads.append(t)
        if stagger:
            time.sleep(stagger)
    for t in threads:
        t.join()
    return out


def main():
    serial = [ask(c, draft=False) for c in CASES]
    alone = [ask(c) for c in CASES]
    burst = together(0)
    stag = together(0.7)
    fails = 0
    for i, c in enumerate(CASES):
        s, a, b, g = serial[i], alone[i], burst[i], stag[i]
        ok = (a[0] == s[0], b[0] == a[0], g[0] == a[0])
        fails += not all(ok)
        print(f"{c[0]:15} tokens={a[1]:4} finish={a[2]:6} drafted==serial={ok[0]} burst==alone={ok[1]} staggered==alone={ok[2]}")
    n = len(CASES)
    print(f"exact: drafted==serial {sum(alone[i][0] == serial[i][0] for i in range(n))}/{n}, "
          f"burst==alone {sum(burst[i][0] == alone[i][0] for i in range(n))}/{n}, "
          f"staggered==alone {sum(stag[i][0] == alone[i][0] for i in range(n))}/{n}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
