#!/bin/sh
# Grip sweep. Each baseline seed's 4000-update policy resumed for
# 2000 updates at finger strength STRENGTH, exactly as grip-half and its
# control resume-full (ceiling held at 0.8 m), then scored at full strength
# (capability map) and by hand strength (grip_strength.py --search).
#   sh scripts/queues/grip-sweep.sh STRENGTH NAME [PYTHON]
# BASE (default artifacts/runs) holds the baselines; SCORE=0 skips scoring.
py=${3:-"uv run --locked --extra train python"}
base=${BASE:-artifacts/runs}
for run in "$base"/*-vel-obs-s[012]; do
    $py scripts/train.py --name "$2" --resume "$run/model_4000.pt" --iters 2000 \
        --set grip_strength="$1" --set apex="(0.05,0.8)" --set apex_cap=0.8
    [ "${SCORE:-1}" = 0 ] && continue
    out=$(ls -d artifacts/runs/*-"$2"-s* | tail -1)
    sh scripts/queues/grip-score.sh "$out" "$py"
done
