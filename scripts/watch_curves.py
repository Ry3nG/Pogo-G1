"""Evaluate every 500th checkpoint of a run (64 worlds x 10 s) to measure time-to-usable.

Every 1000th checkpoint also runs the mid-episode command-response check
(constant-command evaluations missed a lost forward response).

usage: watch_curves.py RUN_DIR [STEP] [LAST]   (exit once LAST is evaluated)
"""

import json
import subprocess
import sys
import time
from pathlib import Path

run = Path(sys.argv[1])
step = int(sys.argv[2]) if len(sys.argv) > 2 else 500
last = int(sys.argv[3]) if len(sys.argv) > 3 else None
conds = "apex_0.10,apex_0.30,apex_0.50,apex_0.75,fwd_0.4,back_0.3,left_0.3,turn_0.5,turn_-0.5"
out = Path("artifacts/curves") / f"{run.name}.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
done = (
    {json.loads(line)["iter"] for line in out.read_text().splitlines()}
    if out.exists()
    else set()
)
start = json.loads((run / "config.json").read_text())
t0 = (run / "config.json").stat().st_mtime
while True:
    ckpts = sorted((int(p.stem.split("_")[1]), p) for p in run.glob("model_*.pt"))
    todo = [(i, p) for i, p in ckpts if i % step == 0 and i not in done and i > 0]
    for i, p in todo:
        time.sleep(5)
        r = subprocess.run(
            [
                sys.executable,
                "scripts/eval_suite.py",
                str(run),
                "--ckpt",
                p.name,
                "--worlds",
                "64",
                "--seconds",
                "10",
                "--only",
                conds,
            ],
            capture_output=True,
            text=True,
        )
        rep = run / f"suite-{p.stem}-64x10s.json"
        res = (
            json.loads(rep.read_text())["conditions"]
            if rep.exists()
            else {"error": r.stderr[-500:]}
        )
        row = {
            "iter": i,
            "wall_h": round((p.stat().st_mtime - t0) / 3600, 2),
            **{
                c: {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in res[c].items()
                    if k in ("success", "tip_apex_err", "hop_vel_err", "hop_yaw_err")
                }
                for c in res
                if c != "error"
            },
        }
        if i % 1000 == 0:
            resp = subprocess.run(
                [
                    sys.executable,
                    "scripts/studies/command_response.py",
                    str(run),
                    p.name,
                ],
                capture_output=True,
                text=True,
            )
            js = [x for x in resp.stdout.splitlines() if x.startswith("JSON ")]
            row["response"] = json.loads(js[-1][5:]) if js else None
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        done.add(i)
    if last is not None and last in done:
        break
    time.sleep(60)
