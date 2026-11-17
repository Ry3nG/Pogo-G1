#!/bin/sh
# After baseline.txt: push timing for each baseline
# seed at 4000 updates, and the command matrix of the 20000-update runs.
#   sh scripts/queues/followup.sh
py=${PYTHON:-"uv run --locked --extra train python"}
for seed in 0 1 2; do
    run=$(ls -d artifacts/runs/*-vel-obs-s$seed | tail -1)
    $py scripts/push_phase.py "$run" --ckpt model_4000.pt
    [ -f "$run/model_19999.pt" ] && $py scripts/capability.py "$run" --ckpt model_19999.pt
done
