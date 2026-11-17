#!/bin/sh
# walking yaw at width 0.15, pogo kernel, per step vs window
# 0.5 s, three seeds; each seed's pair runs concurrently, then
# every 500th checkpoint is scored.
#   sh scripts/queues/walk.sh
py=${PYTHON:-"uv run --locked --extra train python"}
w=scripts/studies/walk_timescale.py
for seed in 0 1 2; do
    $py $w train step0.15-s$seed --std 0.15 --score step --seed $seed --iters 3001 \
        > artifacts/walk/step0.15-s$seed.log 2>&1 &
    $py $w train window0.15-s$seed --std 0.15 --score window --tau 0.5 --seed $seed \
        --iters 3001 > artifacts/walk/window0.15-s$seed.log 2>&1
    wait
done
for run in artifacts/walk/step0.15-s* artifacts/walk/window0.15-s* artifacts/walk/step0.71-s0; do
    for it in 500 1000 1500 2000 2500 3000; do
        echo "$run $it $($py $w eval "$run" model_$it.pt 2>&1 | grep JSON)"
    done
done > artifacts/walk/eval.txt
