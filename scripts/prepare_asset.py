"""Convert the Vurtego Sketchfab archive into scaled OBJ meshes + manifest.json."""

import argparse
from pathlib import Path

from pogo_g1 import vurtego

ap = argparse.ArgumentParser()
ap.add_argument("--archive", type=Path, default=Path("data/assets/vurtego/source.zip"))
args = ap.parse_args()
root = args.archive.parent
original = vurtego.extract_archive(args.archive, root)
meta = vurtego.prepare(original, root / "small-v4-pro-v3")
print(
    {
        k: meta[k]
        for k in ["uncompressed_height_m", "grip_height_above_pedal_m", "grip_span_m"]
    }
)
