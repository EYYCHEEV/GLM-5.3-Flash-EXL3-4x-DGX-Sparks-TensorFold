# Four Sparks (TP=4) and up to 16 requests: an unofficial fork

This fork of [Mia's AI Lab's recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) (v1.9.1, commit `2c9df0b`) serves GLM-5.3-Flash on four DGX Sparks joined by a switch, and raises the concurrent-request limit from 8 to 16.
It is an experiment on one home cluster, not affiliated with or reviewed by Mia's AI Lab.
Everything else, including the pinned image, weights and drafter, is Mia's recipe unchanged; read her README first.

## Results

Four Sparks at TP=4 on recipe v1.9.1 (the 16-request image, the 128-row shared verify window, `DENSE=fp8`, NCCL transport), measured on 2026-10-08 with the sparkDash protocol (fixed prose and code prompts, 400 tokens, temperature 0, thinking off) from the head Spark, on one boot after a warm-up.

**Decode** (aggregate across the concurrent requests, per request, and time to first token)

| Concurrent requests | Prose | Prose, per request | TTFT | Code | Code, per request | TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 75.1 tok/s | 75.1 tok/s | 110 ms | 119.4 tok/s | 119.4 tok/s | 141 ms |
| 2 | 121.0 tok/s | 62.2 tok/s | 176 ms | 161.7 tok/s | 82.1 tok/s | 185 ms |
| 4 | 155.9 tok/s | 39.4 tok/s | 186 ms | 203.2 tok/s | 56.3 tok/s | 234 ms |
| 6 | 193.4 tok/s | 32.9 tok/s | 214 ms | 240.2 tok/s | 45.0 tok/s | 297 ms |
| 8 | 218.0 tok/s | 28.1 tok/s | 259 ms | 273.1 tok/s | 38.0 tok/s | 344 ms |
| 10 | 242.2 tok/s | 25.2 tok/s | 429 ms | 282.9 tok/s | 32.1 tok/s | 410 ms |
| 12 | 269.5 tok/s | 23.2 tok/s | 319 ms | 300.7 tok/s | 27.1 tok/s | 452 ms |
| 14 | 295.0 tok/s | 21.7 tok/s | 419 ms | 327.2 tok/s | 25.7 tok/s | 504 ms |
| 16 | 292.1 tok/s | 18.8 tok/s | 401 ms | 348.0 tok/s | 23.4 tok/s | 555 ms |

Sixteen at once: 3.9 times one request's prose and 2.9 times its code.
Prose stops gaining at 14 requests while code still climbs; 16 is the most that keeps every stream above 18 tok/s.

**Against fewer Sparks** (total tok/s, prose / code; two and three Sparks from Mia's README, each from one boot)

| Concurrent requests | 2 Sparks, `PARALLEL=8` | 3 Sparks, `PARALLEL=8` | 4 Sparks, `PARALLEL=16` |
| ---: | ---: | ---: | ---: |
| 1 | | 65.7 / 100.4 | 75.1 / 119.4 |
| 2 | | 93.8 / 129.5 | 121.0 / 161.7 |
| 4 | 103.2 / 126.7 | 121.8 / 165.3 | 155.9 / 203.2 |
| 6 | 116.8 / 150.0 | 146.3 / 192.6 | 193.4 / 240.2 |
| 8 | 130.8 / 167.0 | 166.0 / 211.5 | 218.0 / 273.1 |
| 16 | | | 292.1 / 348.0 |

This is a rough guide, not a controlled comparison: Mia's runs use 4-bit dense weights (faster than FP8 in her measurements), sparkDash from another machine on the network, and, for two Sparks, GPU clocks capped at 2,200 MHz.
Three of these same Sparks with this fork's settings (v1.8, 8 requests, `DENSE=fp8`) gave 179.1 / 209.0.

- A fresh ~16k-token prompt arriving while 15 requests decode gets its first token in about 15 s.
- v1.8 and v1.9.1 measured the same at 8 and 16 requests (v1.8: 218.7 / 270.8 at 8, 292.2 / 347.5 at 16).
- At 16 requests on v1.9.1: the recipe's output-equality checks (drafted equals serial, concurrent equals alone) pass, also with two runs at once; the ~195k-token needle is found (prefill 91.4 s); a 62-minute 16-client soak (short, code, thinking, tool-call and 4k to 32k-token prompts, mid-stream disconnects) completed 1,831 requests with zero errors, zero switch drops and zero RDMA retransmissions.
- The full experiment ledger, including discarded settings, is [`four-sparks/results.tsv`](four-sparks/results.tsv).

## What changed

- `start-tp4.sh` and `scripts/`: a fourth rank (`WORKER3`, `FABRIC_PEER3`, `WORKER_HF_CACHE3`, `WORKER_WEIGHTS3`, `NFS_SERVER3`), 8 requests by default from TP=3 up, a 48 GiB pool cap at TP=4, `RESERVE_EXTRA_GIB` on top of the reserve formula, `PARALLEL_MOST` for the request limit, and pass-through of `NCCL_IB_TC` and `NCCL_BUFFSIZE` to every rank.
- `four-sparks/image/sixteen-streams.patch`: TensorFold's GLM engine takes up to 16 requests (`PARALLEL_MOST` and `segments.MAX_SEGS` from 8 to 16; 8 or fewer keep their tables), and `TF_GLM_MULTI_WINDOW_WIDE=1` allows shared verify windows of 72 to 128 rows. Two Python files, about 11 lines; no GPU code changes.
- `four-sparks/image/Dockerfile`: applies that patch to the recipe's published image (exactly, or the build fails) and keeps its kernel-cache label, so nothing recompiles.
- `tools/tp4_bench.py`, `tools/tp4_equality.py`, `tools/tp4_soak.py`: the benchmark, the output-equality check and the soak used for the results.

## Running it

1. Wire four Sparks to one switch (two rail subnets, as in NVIDIA's switch playbook), and set `scripts/local.sh` as for TP=3 plus the fourth rank:

   ```bash
   WORKER=<rank-1 ssh target>; WORKER2=<rank-2 ssh target>; WORKER3=<rank-3 ssh target>
   FABRIC_PEER=<rank-1 rail-A address>; FABRIC_PEER2=<rank-2 rail-A address>; FABRIC_PEER3=<rank-3 rail-A address>
   MASTER_ADDR=<head rail-A address>; SOCKET_IFNAME=<head rail-A netdev>
   WORKER_WEIGHTS=copy; WORKER_WEIGHTS2=copy; WORKER_WEIGHTS3=copy
   DENSE=fp8                # the setting measured here
   NCCL_IB_TC=106           # RoCE in the switch's ECN class (see below)
   NCCL_BUFFSIZE=524288     # needed through the switch (see below)
   ```

2. `./start-tp4.sh` stages the image and weights on every Spark on first start and serves 8 requests.
3. For up to 16: build the derived image on every Spark, `docker build -t tensorfold-glm53:v0.6.0-sixteen four-sparks/image` (it starts from `tensorfold-glm53:v0.6.0`, the name `prepare.sh` gives the published image; `--build-arg BASE=<name>` for another), then add:

   ```bash
   IMAGE=tensorfold-glm53:v0.6.0-sixteen
   PARALLEL_MOST=16; PARALLEL=16
   export TF_GLM_MULTI_WINDOW_WIDE=1 TF_GLM_MULTI_WINDOW=128
   PREPARE=0                # prepare.sh knows only the published image
   ```

4. Check before use: `MODEL=GLM-5.3-Flash-EXL3 API_URL=http://127.0.0.1:8888 python3 tools/tp4_equality.py`, then `API_URL=http://127.0.0.1:8888 python3 tools/tp4_bench.py 16`.

## Switch finding: RoCE drops with four Sparks, and the fix

Switch: MikroTik CRS812-8DS-2DQ-2DDQ, RouterOS 7.24.4, Spark lanes on 400G-to-2x200G breakout cables, MTU 1500.
RoCE gets an ECN-marking class and congestion notifications a strict-priority class:

```routeros
/interface ethernet switch qos profile add name=roce dscp=26 traffic-class=3
/interface ethernet switch qos profile add name=cnp dscp=48 traffic-class=6
/interface ethernet switch qos tx-manager queue set [find where tx-manager=default and traffic-class=3] schedule=high-priority-group weight=1 ecn=yes
/interface ethernet switch qos tx-manager queue set [find where tx-manager=default and traffic-class=6] schedule=strict-priority
/interface ethernet switch qos port set <the Spark lanes> trust-l3=keep
```

With that and `NCCL_IB_TC=106` on the Sparks, three Sparks ran drop-free.
Four did not: a 62-minute, 16-client soak dropped 1,343 packets, in bursts of 13 to 176 toward one receiver. Retransmission recovered them all and no request failed.

Cause: the switch lets one egress queue grow to about 3.3 MB (`/interface ethernet switch qos port print usage` shows `queue3-byte-max` there on every lane), and it marks ECN only once a queue passes its shared limit (MikroTik's QoS manual; there is no threshold setting), so marking starts late.
NCCL lets each sender have channels x `NCCL_BUFFSIZE` in flight per peer, 4 x 4 MiB by default, so three senders reaching a late receiver at once overflow the queue before congestion notifications slow them.
Decoding alone never dropped; the bursts came under mixed load with long prompts.

Fix: `NCCL_BUFFSIZE=524288` on every rank (2 MiB in flight per sender).
The same soak then had zero drops, zero retransmissions and 20 times fewer ECN marks, at the same throughput, and eight fixed-seed replies were byte-identical with either buffer size.
Six-minute trials: 256 KiB was also drop-free (no marking at all, 2 to 6% slower); 1 MiB still dropped.
Priority flow control would cover worse overloads, but MikroTik's fix for its CRS8xx PFC and lossless-buffer issues was only in 7.25 test releases at the time.

## Caveats

- Patch 0084 and the 128-row window are tested only on this cluster, with the checks above.
- The DFlash2 drafter is CC BY-NC-ND 4.0 (non-commercial); see `NOTICE`.
- This fork's own changes are Apache-2.0, like the recipe; `NOTICE` lists them.
