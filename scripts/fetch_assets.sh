#!/usr/bin/env bash
# Fetch Unitree's G1 + BrainCo Revo2 description (URDF and meshes) at a pinned
# commit into data/unitree_ros. The Vurtego visual asset is manual (docs/ASSETS.md).
set -euo pipefail
COMMIT=ccfc6fd8430a17ba3dacef9a1e2faf64ff3b0aee
DEST=data/unitree_ros
if [ ! -d "$DEST/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/unitreerobotics/unitree_ros.git "$DEST"
fi
git -C "$DEST" sparse-checkout set robots/g1_with_brainco_hand
git -C "$DEST" fetch --depth 1 origin "$COMMIT"
git -C "$DEST" checkout --quiet "$COMMIT"
echo "unitree_ros at $(git -C "$DEST" rev-parse --short HEAD)"
