#!/bin/sh
# Terrain step B: resume each baseline seed's
# 4000-update policy for 2000 updates on 10 cm stairs, then probe terrain and
# re-check the flat command matrix.
#   sh scripts/queues/terrain.sh
py=${PYTHON:-"uv run --locked --extra train python"}
for run in $(ls -d artifacts/runs/*-vel-obs-s[012]); do
    $py scripts/train.py --name stairs10 --resume "$run/model_4000.pt" --iters 2000 \
        --set terrain='"stairs"' --set terrain_height=0.1 \
        --set apex="(0.05,0.8)" --set apex_cap=0.8
    out=$(ls -d artifacts/runs/*-stairs10-s* | tail -1)
    $py scripts/terrain_probe.py "$out" --ckpt model_6000.pt
    $py scripts/capability.py "$out" --ckpt model_6000.pt
done
