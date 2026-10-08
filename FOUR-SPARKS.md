# Four Sparks (TP=4) and up to 16 requests: an unofficial fork

This fork of [Mia's AI Lab's recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) (v1.9.1, commit `2c9df0b`) serves GLM-5.3-Flash on four DGX Sparks joined by a switch, and raises the concurrent-request limit from 8 to 16.
It is an experiment on one home cluster, not affiliated with or reviewed by Mia's AI Lab.
Everything else, including the pinned image, weights and drafter, is Mia's recipe unchanged; read her README first.

## Results

Measured with the sparkDash protocol (fixed prose and code prompts, 400 tokens, temperature 0, thinking off), `DENSE=fp8`, NCCL transport, after a warm-up.
Each cell: total tok/s, then the median for one request.

| Setup | Recipe | Prose | Code |
|---|---|---|---|
| Mia's TP=2, 8 requests (her README) | v1.8 | 130.8 / 17.0 | 167.0 / 22.8 |
| TP=3, 8 requests | v1.8 | 179.1 / 23.1 | 209.0 / 30.2 |
| TP=4, 8 requests | v1.9.1 | 217.7 / 28.1 | 264.7 / 37.8 |
| TP=4, 12 requests | v1.8 | 271.2 / 23.3 | 295.8 / 26.6 |
| TP=4, 14 requests, 128-row window | v1.8 | 296.7 / 21.8 | 331.5 / 26.0 |
| TP=4, 16 requests, 128-row window | v1.9.1 | 292.3 / 18.8 | 348.2 / 23.4 |

- One request alone at TP=4 (v1.9.1): prose 74.9, code 118.2 tok/s; four at once: 154.8 and 196.7 in all.
- A fresh ~16k-token prompt arriving while 15 requests decode gets its first token in about 15 s.
- Prose throughput is flat from 14 requests; 16 is the most that keeps every stream above 18 tok/s.
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
