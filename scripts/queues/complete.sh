#!/bin/sh
# centroidal per-step steering (centroidal.txt), then the grip margin:
# each baseline seed's 4000-update policy resumed for 2000 updates with the
# finger servos at half their rated torque (ceiling held at 0.8 m, as the
# stairs runs), scored at full strength and by hand strength.
#   sh scripts/queues/complete.sh
py=${PYTHON:-"uv run --locked --extra train python"}
sh scripts/queue_runs.sh scripts/queues/centroidal.txt
for run in $(ls -d artifacts/runs/*-vel-obs-s[012]); do
    $py scripts/train.py --name grip-half --resume "$run/model_4000.pt" --iters 2000 \
        --set grip_strength=0.5 --set apex="(0.05,0.8)" --set apex_cap=0.8
    out=$(ls -d artifacts/runs/*-grip-half-s* | tail -1)
    $py scripts/capability.py "$out" --ckpt model_6000.pt
    $py scripts/studies/grip_strength.py "$out" model_6000.pt "$out/grip-strength-model_6000.json"
done
