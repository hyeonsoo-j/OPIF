#!/usr/bin/env bash
# Run experiment configs sequentially (CV96 protocol). Usage: scripts/run_queue.sh cfg1 cfg2 ...
cd "$(dirname "$0")/.." || exit 1
for c in "$@"; do
  echo "=== $(date +%T) start $c"
  python3 run.py --exp "$c" > "logs/$c.log" 2>&1
  echo "=== $(date +%T) done  $c (exit $?)"
done
