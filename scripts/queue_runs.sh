#!/bin/sh
# Train each "NAME SEED [k=v | --flag=v ...]" line of a queue file for UPDATES updates
# in turn, then score its checkpoints with watch_curves.py (learning curve) and
# capability.py (experiment 1's command matrix) at the last checkpoint.
#   [UPDATES=12000] scripts/queue_runs.sh QUEUE_FILE [PYTHON]
# PYTHON defaults to the uv environment.
# UPDATES (default 4000) sets the last checkpoint that is trained and scored.
queue=$1
py=${2:-"uv run --locked --extra train python"}
updates=${UPDATES:-4000}
while read -r name seed rest; do
    case "$name" in ''|'#'*) continue ;; esac
    sets=""
    for kv in $rest; do
        case "$kv" in --*) sets="$sets $kv" ;; *) sets="$sets --set $kv" ;; esac
    done
    $py scripts/train.py --name "$name" --seed "$seed" --iters $((updates + 1)) $sets
    run=$(ls -d artifacts/runs/*-"$name"-s"$seed" | tail -1)
    $py scripts/watch_curves.py "$run" 500 "$updates"
    $py scripts/capability.py "$run" --ckpt "model_$updates.pt"
done < "$queue"
