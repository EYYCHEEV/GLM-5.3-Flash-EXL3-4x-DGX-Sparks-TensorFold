#!/usr/bin/env python3
"""Bounded soak: WORKERS concurrent clients for MINUTES minutes against the private server, with a fixed synthetic mix
(short/long prompts, thinking on/off, streaming, tool calls, mid-stream disconnects). Retains no text: prints counts,
status classes, finish reasons, throughput and /health snapshots only. Exit 1 on any unexpected failure or when the
server does not return to idle."""
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = os.environ.get("API_URL", "http://127.0.0.1:8888").rstrip("/")
URL = BASE + "/v1/chat/completions"
MODEL = os.environ.get("MODEL") or sys.exit("set MODEL to the served model id")
MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 60
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
WORDS = ("time year people way day thing life world school state family group country problem hand part place case "
         "week company system program question work number night point home water room area money story fact month "
         "book job word business side kind head house service friend power hour game line end member law car city "
         "river mountain signal engine garden theory market winter method bridge letter window voice paper field").split()
TOOL = {"type": "function", "function": {"name": "lookup", "description": "Look up a record by id.",
        "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}}}

lock = threading.Lock()
stats = {"ok": 0, "cancelled": 0, "tool_calls": 0, "completion_tokens": 0, "prompt_tokens": 0, "cached_tokens": 0}
finish = {}
errors = {}
kinds = {}


def prose(tokens, rng):
    out = []
    while len(out) < tokens * 0.72:
        s = [rng.choice(WORDS) for _ in range(rng.randint(6, 16))]
        out += s[:-1] + [s[-1] + "."]
    return " ".join(out)


def bump(d, k, n=1):
    with lock:
        d[k] = d.get(k, 0) + n


def request(rng):
    r = rng.random()
    off = {"chat_template_kwargs": {"enable_thinking": False}}
    if r < 0.30:
        kind, body = "short", {"messages": [{"role": "user", "content": "Write a short paragraph about " + rng.choice(WORDS) + "."}], "max_tokens": 400, **off}
    elif r < 0.45:
        kind, body = "code", {"messages": [{"role": "user", "content": "Write a small Python function about " + rng.choice(WORDS) + " with a docstring."}], "max_tokens": 600, **off}
    elif r < 0.60:
        kind, body = "think", {"messages": [{"role": "user", "content": f"What is {rng.randint(100, 999)} * {rng.randint(10, 99)}? Answer with the number."}], "max_tokens": 2000, "reasoning_effort": "low"}
    elif r < 0.78:
        n = rng.choice([4000, 8000, 16000, 32000])
        kind, body = f"long{n // 1000}k", {"messages": [{"role": "user", "content": "Name the most common word in this text.\n" + prose(n, rng)}], "max_tokens": 128, **off}
    elif r < 0.88:
        kind, body = "tool", {"messages": [{"role": "user", "content": f"Look up record {rng.randint(1, 9999)}."}], "tools": [TOOL], "max_tokens": 300, **off}
    else:
        kind, body = "cancel", {"messages": [{"role": "user", "content": "Write a long essay about " + rng.choice(WORDS) + "."}], "max_tokens": 2000, **off}
    body["model"] = MODEL
    body["stream"] = kind == "cancel" or rng.random() < 0.5
    if body["stream"]:
        body["stream_options"] = {"include_usage": True}
    return kind, body


def one(rng):
    kind, body = request(rng)
    bump(kinds, kind)
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=1800)
    except urllib.error.HTTPError as exc:
        bump(errors, f"http{exc.code // 100}xx:{kind}"); return
    except Exception as exc:  # noqa: BLE001
        bump(errors, f"{type(exc).__name__}:{kind}"); return
    try:
        if not body["stream"]:
            reply = json.load(resp)
            ch = reply["choices"][0]
            bump(finish, ch["finish_reason"])
            if ch["message"].get("tool_calls"):
                bump(stats, "tool_calls")
            u = reply.get("usage") or {}
        else:
            events, done, u, fin = 0, False, {}, None
            for raw in resp:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True; break
                ev = json.loads(data)
                events += 1
                if ev.get("usage"):
                    u = ev["usage"]
                for c in ev.get("choices") or []:
                    if c.get("finish_reason"):
                        fin = c["finish_reason"]
                    if (c.get("delta") or {}).get("tool_calls"):
                        bump(stats, "tool_calls")
                if kind == "cancel" and events >= rng.randint(5, 40):
                    resp.close(); bump(stats, "cancelled"); return
            if not done:
                bump(errors, f"stream-no-done:{kind}"); return
            bump(finish, fin or "none")
        bump(stats, "ok")
        bump(stats, "completion_tokens", u.get("completion_tokens", 0))
        bump(stats, "prompt_tokens", u.get("prompt_tokens", 0))
        bump(stats, "cached_tokens", ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0))
    except Exception as exc:  # noqa: BLE001
        bump(errors, f"read-{type(exc).__name__}:{kind}")
    finally:
        resp.close()


def health():
    return json.load(urllib.request.urlopen(BASE + "/health", timeout=30))


def main():
    end = time.time() + MINUTES * 60
    def worker(seed):
        rng = random.Random(seed)
        while time.time() < end:
            one(rng)
    threads = [threading.Thread(target=worker, args=(1000 + i,)) for i in range(WORKERS)]
    t0 = time.time()
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(60)
        try:
            h = health()
            snap = f"running={h.get('requests_running')} decoding={h['streams'].get('decoding')} kept={h.get('kept_prompts')} pool_free={h.get('pool_free_tokens')}"
        except Exception as exc:  # noqa: BLE001
            snap = f"health-error={type(exc).__name__}"
        with lock:
            print(f"[{(time.time() - t0) / 60:5.1f} min] ok={stats['ok']} cancelled={stats['cancelled']} errors={sum(errors.values())} {snap}", flush=True)
    for t in threads:
        t.join()
    wall = time.time() - t0
    idle = False
    for _ in range(30):
        h = health()
        if not h.get("requests_running") and not h["streams"].get("decoding") and not h["streams"].get("prefilling"):
            idle = True; break
        time.sleep(2)
    print(f"soak: {wall / 60:.1f} min, {WORKERS} workers, requests by kind {dict(sorted(kinds.items()))}")
    print(f"soak: ok={stats['ok']} cancelled={stats['cancelled']} tool_calls={stats['tool_calls']} finish={finish} errors={errors}")
    print(f"soak: completion tokens {stats['completion_tokens']} ({stats['completion_tokens'] / wall:.1f} tok/s aggregate incl. prefill), "
          f"prompt tokens {stats['prompt_tokens']}, cached {stats['cached_tokens']}; returned to idle: {idle}")
    sys.exit(0 if not errors and idle else 1)


if __name__ == "__main__":
    main()
