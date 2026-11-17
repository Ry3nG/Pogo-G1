#!/bin/sh
# Path or optimum?: resume each window (0.5 s)
# and per-hop policy at 4000 updates for 1000 updates with steering scored per
# step (plain weight), then score it like the others.
#   sh scripts/queues/path.sh
py=${PYTHON:-"uv run --locked --extra train python"}
for run in $(ls -d artifacts/runs/*-window-steer-s[012]) \
    $(for s in 0 1 2; do ls -d artifacts/runs/*-vel-obs-s$s | tail -1; done); do
    case "$run" in *window*) name=step-from-window ;; *) name=step-from-hop ;; esac
    $py scripts/train.py --name "$name" --resume "$run/model_4000.pt" --iters 1000 \
        --set steer_timescale='"step"'
    out=$(ls -d artifacts/runs/*-"$name"-s* | tail -1)
    $py scripts/watch_curves.py "$out" 500 5000
    $py scripts/capability.py "$out" --ckpt model_5000.pt
done
