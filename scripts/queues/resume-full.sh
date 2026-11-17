#!/bin/sh
# control for grip-half. Each baseline seed's 4000-update policy
# resumed for 2000 updates exactly as grip-half (ceiling held at 0.8 m) but at
# the rated hand, so the comparison isolates the derated hand from resuming.
#   sh scripts/queues/resume-full.sh
py=${PYTHON:-"uv run --locked --extra train python"}
for run in $(ls -d artifacts/runs/*-vel-obs-s[012]); do
    $py scripts/train.py --name resume-full --resume "$run/model_4000.pt" --iters 2000 \
        --set apex="(0.05,0.8)" --set apex_cap=0.8
    out=$(ls -d artifacts/runs/*-resume-full-s* | tail -1)
    $py scripts/capability.py "$out" --ckpt model_6000.pt
    $py scripts/studies/grip_strength.py "$out" model_6000.pt "$out/grip-strength-model_6000.json"
done
