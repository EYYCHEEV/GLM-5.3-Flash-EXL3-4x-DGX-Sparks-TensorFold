#!/usr/bin/env bash
# Serve GLM-5.3 Flash EXL3 on four DGX Sparks (TP=4, four-Spark fork, experimental): ./start.sh with TP=4, COMM=nccl by
# default. The TP-N engine (patches 0066-0068) already splits over 2, 3 or 4 ranks; this fork lets the launcher use a
# fourth: WORKER, WORKER2 and WORKER3 in scripts/local.sh, every rank sharing the switch's two rail subnets.
#
# Usage: ./start-tp4.sh [restart] [extra tensorfold serve args]   (as ./start.sh; ./stop.sh stops it)
#   DRY_RUN=1 ./start-tp4.sh            # print every rank's docker command, change nothing
# COMM defaults to nccl here, as in ./start-tp3.sh; set COMM on the command line to change it.
set -euo pipefail
case "${1:-}" in
  help|-h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; echo; echo "./start.sh's options:"; echo ;;
esac
export TP=4 COMM="${COMM:-nccl}"
exec "$(dirname "$(readlink -f "$0")")/start.sh" "$@"
