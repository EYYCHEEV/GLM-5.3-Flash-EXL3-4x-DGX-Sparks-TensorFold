#!/usr/bin/env python3
"""Four-Spark fork: one autoresearch measurement against a running server, retaining no text.

Decode waves follow sparkDash's decode protocol (its fixed prose prompt and code tasks, 400 forced tokens, temperature
0, top_p 1, thinking off, a short warm-up, decode rate = (completion tokens - 1) / (last - first token)), so the numbers
compare with Mia's README tables. The loaded-arrival probe measures what a new user waits for when the server is full:
C - 1 streams decode while one fresh ~16k-token prompt arrives; its time to first token is reported.
Usage: python3 tools/tp4_bench.py C [--kinds prose,code|none] [--tokens 400] [--no-loaded] [--loaded-words 16000]
"""
import argparse
import http.client
import json
import os
import random
import statistics
import threading
import time
import urllib.parse

URL = urllib.parse.urlparse(os.environ.get("API_URL", "http://127.0.0.1:8888"))
HOST, PORT = URL.hostname, URL.port or 80

PROSE = ("Write a detailed step-by-step explanation of how a hash map works, "
         "including collision handling, resizing, and time complexity. Be thorough.")
TAIL = ("Output only Python source. No comments, no docstrings, no markdown fences. "
        "Then add tests and the helpers this needs. Keep writing code.")
TASKS = [
    ("binary_search", "def binary_search(nums, target) -> int: index of target in a sorted list, or -1."),
    ("merge_sort", "def merge_sort(nums) -> list: stable sort of a list of ints, returning a new list."),
    ("lru_cache", "class LRUCache: get(key) and put(key, value) with a fixed capacity, evicting the least recently used."),
    ("token_bucket", "class TokenBucket: allow(n) consumes n tokens refilled at a fixed rate, else returns False."),
    ("ring_buffer", "class RingBuffer: push and pop over a fixed-capacity array, raising on overflow and underflow."),
    ("dijkstra", "def dijkstra(graph, src) -> dict: shortest path weights from src on a non-negative weighted graph."),
    ("edit_distance", "def edit_distance(a, b) -> int: Levenshtein distance between two strings."),
    ("semver_cmp", "def semver_cmp(a, b) -> int: compare dotted numeric versions, negative if a < b."),
    ("url_parse", "def url_parse(url) -> dict: scheme, host, port, path, and query pairs. No extra libraries."),
    ("json_pointer", "def json_pointer(doc, pointer) -> object: follow an RFC 6901 pointer, or None if missing."),
    ("glob_match", "def glob_match(pattern, text) -> bool: * and ? wildcards, no character classes."),
    ("csv_parse", "def csv_parse(text) -> list: rows of fields, honoring double-quoted commas and escaped quotes."),
    ("rle", "def rle_encode(s) -> str and rle_decode(s) -> str: run-length encoding of single-byte runs."),
    ("top_k", "def top_k(nums, k) -> list: the k largest ints, unordered, using a bounded heap."),
    ("interval_merge", "def merge_intervals(spans) -> list: merge overlapping [start, end] pairs."),
    ("topo_sort", "def topo_sort(nodes, edges) -> list: a valid order, or None if the graph has a cycle."),
]
CODE = [f"{n}\n{s}\n{TAIL}" for n, s in TASKS]
WARM_CODE = f"warmup_noop\ndef warmup_noop(x): return x unchanged.\n{TAIL}"
WORDS = ("time year people way day thing life world school state family group country problem hand part place case "
         "week company system program question work government number night point home water room mother area money "
         "story fact month lot right study book eye job word business issue side kind head house service friend").split()


def model_id():
    c = http.client.HTTPConnection(HOST, PORT, timeout=30)
    c.request("GET", "/v1/models")
    return json.loads(c.getresponse().read())["data"][0]["id"]


MODEL = None


def body(prompt, max_tokens, force=True):
    b = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
         "temperature": 0, "top_p": 1, "stream": True, "stream_options": {"include_usage": True},
         "chat_template_kwargs": {"enable_thinking": False, "thinking": False, "thinking_mode": "disabled"}}
    if force:
        b.update(min_tokens=max_tokens, ignore_eos=True, stop=[])
    return b


def stream(b, on_first=None):
    r = {"t0": time.perf_counter(), "first": None, "last": None, "chunks": 0, "usage": None, "err": None}
    try:
        c = http.client.HTTPConnection(HOST, PORT, timeout=900)
        c.request("POST", "/v1/chat/completions", json.dumps(b), {"Content-Type": "application/json"})
        resp = c.getresponse()
        if resp.status != 200:
            resp.read()
            r["err"] = f"HTTP {resp.status}"
            return r
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                r["usage"] = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    now = time.perf_counter()
                    if r["first"] is None:
                        r["first"] = now
                        if on_first:
                            on_first()
                    r["last"] = now
                    r["chunks"] += 1
    except Exception as exc:  # noqa: BLE001
        r["err"] = type(exc).__name__
    return r


def run_all(bodies, on_first=None):
    out = [None] * len(bodies)

    def go(i):
        out[i] = stream(bodies[i], on_first)
    ts = [threading.Thread(target=go, args=(i,)) for i in range(len(bodies))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def rate(r):
    comp = (r["usage"] or {}).get("completion_tokens") or r["chunks"]
    span = (r["last"] or 0) - (r["first"] or 0)
    return max(0, comp - 1), (max(0, comp - 1) / span if span > 0 else 0.0)


def wave(n, kind, tokens):
    prompts = CODE[:n] if kind == "code" else ([PROSE] if n == 1 else [f"{PROSE} (stream {i + 1}/{n})" for i in range(n)])
    res = run_all([body(p, tokens) for p in prompts])
    ok = [r for r in res if not r["err"] and r["first"]]
    if not ok:
        return {"ok": 0, "failed": n}
    dec = [rate(r) for r in ok]
    window = max(r["last"] for r in ok) - min(r["first"] for r in ok)
    ttft = [(r["first"] - r["t0"]) * 1000 for r in ok]
    return {"ok": len(ok), "failed": n - len(ok), "agg": round(sum(d for d, _ in dec) / window, 1) if window > 0 else 0,
            "per_req_med": round(statistics.median(x for _, x in dec), 1), "per_req_min": round(min(x for _, x in dec), 1),
            "ttft_med_ms": round(statistics.median(ttft)), "ttft_max_ms": round(max(ttft))}


def loaded(n, words):
    """C - 1 long decoding streams, then one fresh prompt of ``words`` random words; its time to first token."""
    bg_n = max(0, n - 1)
    started = threading.Semaphore(0)
    bg = [body(f"{PROSE} (background {i + 1}/{bg_n}, {random.random():.6f})", 1500) for i in range(bg_n)]
    holder = {}
    th = threading.Thread(target=lambda: holder.update(res=run_all(bg, started.release)))
    th.start()
    for _ in range(bg_n):
        started.acquire(timeout=600)
    rng = random.Random(time.time_ns())
    text = f"[{rng.random():.9f}] " + " ".join(rng.choice(WORDS) for _ in range(words))
    probe = body("Read the notes below, then reply with one word: done.\n\n" + text, 16, force=False)
    r = stream(probe)
    th.join()
    bgr = [x for x in holder.get("res", []) if not x["err"] and x["first"]]
    return {"loaded_ttft_s": round((r["first"] - r["t0"]), 2) if r["first"] else None,
            "loaded_prompt_tokens": (r["usage"] or {}).get("prompt_tokens"), "loaded_err": r["err"],
            "bg_ok": len(bgr), "bg_failed": bg_n - len(bgr)}


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("c", type=int)
    ap.add_argument("--kinds", default="prose,code")
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--no-loaded", action="store_true")
    ap.add_argument("--loaded-words", type=int, default=16000)
    a = ap.parse_args()
    MODEL = model_id()
    out = {"c": a.c, "tokens": a.tokens}
    for kind in [k for k in a.kinds.split(",") if k and k != "none"]:
        # a C-wide warm-up: the first rounds at a new stream count pay a one-time cost (1.4 s first tokens at 12)
        run_all([body(f"{PROSE if kind == 'prose' else WARM_CODE} (warm {i})", 32, force=False) for i in range(a.c)])
        out[kind] = wave(a.c, kind, a.tokens)
    if not a.no_loaded:
        out.update(loaded(a.c, a.loaded_words))
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
