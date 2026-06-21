#!/usr/bin/env bash
# Run the four NEW cells (B/D DBBench-withheld, E/F FinanceBench-consensus) on
# both worker models. A/C are LOADED from E_harness2 (not run). 4 lanes run in
# parallel (one per model×domain); cells within a lane run sequentially.
set -u
cd "$(dirname "$0")"
LOG=../results/logs

dbbench_lane() {  # $1 = model
  python3 run_dbbench_cell.py --cell B --model "$1"                 > "$LOG/B_$1.log" 2>&1
  python3 run_dbbench_cell.py --cell D --model "$1" --verifier haiku > "$LOG/D_$1.log" 2>&1
}
finbench_lane() {  # $1 = model
  python3 run_finbench_cell.py --cell E --model "$1" --judge haiku                 > "$LOG/E_$1.log" 2>&1
  python3 run_finbench_cell.py --cell F --model "$1" --judge haiku --verifier haiku > "$LOG/F_$1.log" 2>&1
}

dbbench_lane haiku  &
dbbench_lane sonnet &
finbench_lane haiku  &
finbench_lane sonnet &
wait
echo "ALL CELLS DONE"
