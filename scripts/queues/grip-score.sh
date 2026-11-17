#!/bin/sh
# Score one resumed grip run at model_6000 as grip-half: capability map at
# full strength, and the lowest hand strength that keeps success >= 0.95.
#   sh scripts/queues/grip-score.sh RUN [PYTHON]
py=${2:-"uv run --locked --extra train python"}
$py scripts/capability.py "$1" --ckpt model_6000.pt
$py scripts/studies/grip_strength.py "$1" model_6000.pt "$1/grip-strength-model_6000.json" --search
