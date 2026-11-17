#!/bin/sh
# Control for path.sh: window seed 0 resumed with its own scoring.
py=${PYTHON:-"uv run --locked --extra train python"}
run=$(ls -d artifacts/runs/*-window-steer-s0 | tail -1)
$py scripts/train.py --name window-resumed --resume "$run/model_4000.pt" --iters 1000
out=$(ls -d artifacts/runs/*-window-resumed-s* | tail -1)
$py scripts/watch_curves.py "$out" 500 5000
$py scripts/capability.py "$out" --ckpt model_5000.pt
