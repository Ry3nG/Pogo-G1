"""Roll out one hop policy and export the poses that blender_scene.py renders.

Stills (one hop cycle):

    uv run --locked --extra train python scripts/render/export_poses.py \
        artifacts/runs/1012-0002-vel-obs-s0 model_4000.pt artifacts/render [APEX]

Clips (every 2nd policy step, 25 fps at real speed): a command program of
"t:apex,vx,vy,yaw" entries (t in s from the first exported frame), after
WARMUP s (default 2) of the first command with yaw = 0, which are not exported:

    uv run --locked --extra train python scripts/render/export_poses.py \
        artifacts/runs/1012-0002-vel-obs-s0 model_4000.pt artifacts/render/opening \
        clip "0:0.3,0,0,0 2.5:0.9,0,0,0 7.5:0.5,0.3,0,0.4" 12

Failure clips (clip mode only), set in the environment: SET="k=v ..." overrides
task fields after loading (e.g. SET="grip_strength=0.5"; load_task restores the
rated hand); NOISE=SEED samples actions as in training (the policy's own std,
torch seed SEED); FALL=S drops the failure terminations, reports the first one
that would have fired and stops S s after it (the rider is seen to fall);
EVERY=N exports every Nth policy step at 25 fps (default 2, real speed; 1 is
half speed). SET="terrain='stairs'" (or slope, waves with terrain_slope,
terrain_height) keeps a terrain; its boxes and heightfields are written as one
triangle mesh in world coordinates (terrain_v, terrain_f), cropped to 5 m around
the rider's path (render with blender_scene.py's "terrain" mode):

    EVERY=4 SET="terrain='stairs' seed=1" uv run --locked --extra train python \
        scripts/render/export_poses.py artifacts/runs/1016-0627-stairs10-s0 \
        model_6000.pt artifacts/render/terrain/stairs clip \
        "0:0.3,0,0,0 1:0.3,0.4,0,0" 16
    FALL=1.5 uv run --locked --extra train python scripts/render/export_poses.py \
        artifacts/runs/1019-1149-novel-s1 model_4000.pt artifacts/render/fail-vel \
        clip "0:0.5,0,0,0 2:1.0,0,0,0" 12

Path follower replay: `NUS_EXPORT=DIR scripts/demo_nus.py ...` writes DIR/model.mjb
and DIR/replay.npz (policy-rate qpos of the rendered world, the letter path and
the trail); this exports it sped up to about SECONDS (default 7) at 25 fps, from
1 s after the start to 1 s after the letter's end:

    NUS_LETTERS=N NUS_SCALE=2 NUS_EXPORT=artifacts/render/nus-N uv run --locked \
        --extra train python scripts/demo_nus.py artifacts/runs/1012-0002-vel-obs-s0 \
        model_4000.pt /dev/null hold
    uv run --locked --extra train python scripts/render/export_poses.py \
        artifacts/render/nus-N/replay.npz artifacts/render/nus-N 7

The stills' rollout is that of scripts/report_filmstrip.py (deterministic
policy, one world, in place, 0.5 m apex), recorded at the physics rate
(500 Hz). The six phases are those of the filmstrip, taken from the first full
landing-to-landing cycle after 6 s; every 20 ms of that cycle is also exported
as c000, c001, ...

Writes to OUT: model.mjb (compiled scene), scene.npz (visual meshes in their
own frames plus world poses of every visual geom per frame, read by Blender's
own numpy; clips name frames f0000, f0001, ... and add "fps") and, for stills,
hop-frames.json (frame times, tip height, compression, and the cycle's time
series).
"""

import ast
import dataclasses
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

BOX_FACES = [
    (0, 1, 3, 2),
    (4, 6, 7, 5),
    (0, 4, 5, 1),
    (2, 3, 7, 6),
    (0, 2, 6, 4),
    (1, 5, 7, 3),
]


def terrain_mesh(m, d, lo, hi):
    """Terrain boxes and heightfields as triangles in world coordinates, cropped
    to the xy box [lo, hi]; None on flat ground."""
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "terrain")
    verts, faces = [], []
    for g in np.flatnonzero(m.geom_bodyid == body):
        p, rot = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            corner = np.array(
                [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
            )
            v = p + (corner * m.geom_size[g]) @ rot.T
            if (v[:, :2].max(0) < lo).any() or (v[:, :2].min(0) > hi).any():
                continue
            f = [
                (a, b, c)
                for q in BOX_FACES
                for a, b, c in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))
            ]
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD:
            h = m.geom_dataid[g]
            nr, nc = m.hfield_nrow[h], m.hfield_ncol[h]
            rx, ry, top, _ = m.hfield_size[h]
            a = m.hfield_adr[h]
            x, y = np.linspace(-rx, rx, nc) + p[0], np.linspace(-ry, ry, nr) + p[1]
            c0, c1 = np.searchsorted(x, lo[0]), np.searchsorted(x, hi[0])
            r0, r1 = np.searchsorted(y, lo[1]), np.searchsorted(y, hi[1])
            zz = (
                m.hfield_data[a : a + nr * nc].reshape(nr, nc)[r0:r1, c0:c1] * top
                + p[2]
            )
            xx, yy = np.meshgrid(x[c0:c1], y[r0:r1])
            v = np.stack([xx, yy, zz], -1).reshape(-1, 3)
            k = np.arange(v.shape[0]).reshape(zz.shape)
            q = np.stack([k[:-1, :-1], k[:-1, 1:], k[1:, 1:], k[1:, :-1]], -1).reshape(
                -1, 4
            )
            f = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
        else:
            continue
        faces.append(np.asarray(f) + sum(len(x) for x in verts))
        verts.append(v)
    return (np.concatenate(verts), np.concatenate(faces)) if verts else None


def write_scene(m, frames, out, **extra):
    """Visual geoms (group 2): meshes once, world poses per named qpos; and the
    terrain, if any, around the poses."""
    d = mujoco.MjData(m)
    vis = [g for g in range(m.ngeom) if m.geom_group[g] == 2]
    arrays = {}
    names = []
    for g in vis:
        assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH, m.geom(g).name
        mid = m.geom_dataid[g]
        v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        f0, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        key = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid)
        arrays.setdefault(f"v/{key}", m.mesh_vert[v0 : v0 + nv])
        arrays.setdefault(f"f/{key}", m.mesh_face[f0 : f0 + nf])
        names.append(
            (
                key,
                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]),
                *m.geom_rgba[g],
            )
        )
    for n, q in frames.items():
        d.qpos[:] = q
        mujoco.mj_fwdPosition(m, d)
        arrays[f"pos/{n}"] = d.geom_xpos[vis].copy()
        arrays[f"mat/{n}"] = d.geom_xmat[vis].reshape(-1, 3, 3).copy()
    xy = np.concatenate([arrays[f"pos/{n}"][:, :2] for n in frames])
    mujoco.mj_kinematics(m, d)
    if (ground := terrain_mesh(m, d, xy.min(0) - 5, xy.max(0) + 5)) is not None:
        arrays["terrain_v"], arrays["terrain_f"] = ground
    arrays["geoms"] = np.array([json.dumps(x[:2]) for x in names])
    arrays["rgba"] = np.array([x[2:] for x in names])
    np.savez_compressed(out / "scene.npz", **arrays, **extra)


if sys.argv[1].endswith(".npz"):  # path follower replay from demo_nus.py
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 7.0
    out.mkdir(parents=True, exist_ok=True)
    rep = np.load(src)
    m = mujoco.MjModel.from_binary_path(str(src.parent / "model.mjb"))
    if src.parent != out:
        mujoco.mj_saveModel(m, str(out / "model.mjb"))
    # from 1 s into the 2 s in-place start to 1 s after the last pen-down step
    a, b = 50, min(np.flatnonzero(rep["pen"])[-1] + 50, len(rep["qpos"]))
    every = max(1, round((b - a) / (seconds * 25)))
    step = np.arange(a, b, every)
    frames = {f"f{j:04d}": rep["qpos"][i] for j, i in enumerate(step)}
    extra = {k: rep[k] for k in rep.files if k != "qpos"}
    write_scene(m, frames, out, step=step, fps=25.0, **extra)
    print(f"{len(frames)} frames, every {every} policy steps ({every / 2:g}x)")
    sys.exit(0)

import torch  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402

from pogo_g1 import mdp  # noqa: E402
from pogo_g1.env import make_env_cfg  # noqa: E402
from pogo_g1.evaluate import load_task  # noqa: E402
from pogo_g1.ppo import runner_cfg  # noqa: E402

run, ckpt, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
clip = len(sys.argv) > 4 and sys.argv[4] == "clip"
apex = 0.5 if clip or len(sys.argv) < 5 else float(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)

task = load_task(run)
task.worlds, task.airborne_prob, task.episode_s, task.apex_cap = 1, 0.0, 60.0, 0.0
task.terrain = "flat"
for kv in os.environ.get("SET", "").split():
    k, v = kv.split("=", 1)
    setattr(task, k, ast.literal_eval(v))
cfg = make_env_cfg(task)
fall = float(os.environ.get("FALL", "nan"))
failures = {}
if clip and fall == fall:  # watch the failure terms instead of resetting on them
    failures = {k: t for k, t in cfg.terminations.items() if not t.time_out}
    cfg.terminations = {k: t for k, t in cfg.terminations.items() if t.time_out}
env = ManagerBasedRlEnv(cfg, device="cuda:0")
w = RslRlVecEnvWrapper(env)
r = MjlabOnPolicyRunner(w, dataclasses.asdict(runner_cfg()), None, device="cuda:0")
r.load(str(run / ckpt), load_cfg={"actor": True}, strict=False)
pol = r.get_inference_policy(device="cuda:0")
hop = env.command_manager.get_term("hop")
m = env.scene.spec.copy().compile()
mujoco.mj_saveModel(m, str(out / "model.mjb"))

if clip:
    program = [
        (float(t), [float(x) for x in c.split(",")])
        for t, c in (e.split(":") for e in sys.argv[5].split())
    ]
    seconds = float(sys.argv[6])
    warmup = float(sys.argv[7]) if len(sys.argv) > 7 else 2.0
    every, noise = int(os.environ.get("EVERY", 2)), "NOISE" in os.environ
    if noise:
        torch.manual_seed(int(os.environ["NOISE"]))
    hop.set_command(*program[0][1][:3], 0.0)
    obs = w.get_observations()
    frames, n, failed = {}, round(warmup / env.step_dt), None
    with torch.inference_mode():
        for i in range(n + round(seconds / env.step_dt)):
            t = (i - n) * env.step_dt
            for t0, c in program:
                if abs(t - t0) < env.step_dt / 2:
                    hop.set_command(*c)
            obs, _, done, _ = w.step(pol(obs, stochastic_output=noise))
            assert not done[0], f"the world failed or timed out at {t:.2f} s"
            for k, term in failures.items():
                if failed is None and term.func(env, **term.params)[0]:
                    failed = (t, k)
                    print(f"{k} at {t:.2f} s")
            if i >= n and (i - n) % every == 0:
                frames[f"f{len(frames):04d}"] = env.sim.data.qpos[0].cpu().numpy()
            if failed and t >= failed[0] + fall:
                break
    assert not failures or failed, "FALL was set but nothing failed"
    write_scene(m, frames, out, fps=0.5 / env.step_dt, program=json.dumps(program))
    print(len(frames), "frames at", 0.5 / env.step_dt, "fps")
    sys.exit(0)

hop.set_command(apex, 0.0, 0.0, 0.0)
qpos = []  # every physics substep
sim_step = env.sim.step


def recording_step():
    sim_step()
    qpos.append(env.sim.data.qpos[0].clone())


env.sim.step = recording_step
obs = w.get_observations()
check = []  # policy-rate quantities of the filmstrip, to validate ours
with torch.inference_mode():
    for _ in range(int(12.0 / env.step_dt)):
        obs, _, done, _ = w.step(pol(obs))
        assert not done[0], "the world failed during the rollout"
        check.append((float(mdp.tip_clearance(env)[0]), bool(mdp.tip_contact(env)[0])))
qpos = torch.stack(qpos).cpu().numpy().astype(np.float64)
dt = env.physics_dt

d = mujoco.MjData(m)
tip_g, floor_g = m.geom("pogo/tip").id, m.geom("terrain").id
radius = m.geom_size[tip_g, 0]
comp_q = m.jnt_qposadr[m.joint("pogo/compression").id]
tip, comp, contact = [], [], []
for q in qpos:
    d.qpos[:] = q
    mujoco.mj_fwdPosition(m, d)
    tip.append(d.geom_xpos[tip_g, 2] - radius)  # flat ground at z = 0
    comp.append(q[comp_q])
    contact.append(
        any(
            {c.geom1, c.geom2} == {tip_g, floor_g} and c.dist <= 0
            for c in d.contact[: d.ncon]
        )
    )
tip, comp, contact = np.array(tip), np.array(comp), np.array(contact)
t = (np.arange(len(tip)) + 1) * dt
k = round(env.step_dt / dt)
ct, cc = np.array(check).T
print(
    "policy-rate check: max |tip - tip_clearance| =", np.abs(tip[k - 1 :: k] - ct).max()
)
print(
    "contact agreement at policy rate:", np.mean(contact[k - 1 :: k] == cc.astype(bool))
)

land = np.flatnonzero(contact[1:] & ~contact[:-1]) + 1
land = land[t[land] > 6.0]
a, b = land[0], land[1]
off = a + np.flatnonzero(~contact[a:b])[0]
picks = {
    "touchdown": a,
    "compressed": a + int(np.argmax(comp[a:off])),
    "takeoff": off,
    "rising": off + (b - off) // 4,
    "apex": off + int(np.argmax(tip[off:b])),
    "landing": b - 2 * k,  # 40 ms before the next touchdown, as in the filmstrip
}
window = np.arange(a, b + 1)
# Also every policy step of the cycle, as hero candidates ("c000", "c001", ...).
frames = {**picks, **{f"c{j:03d}": i for j, i in enumerate(window[::k])}}
doc = {
    "source": f"{run}/{ckpt}, apex command {apex} m, vx = vy = yaw = 0",
    "dt_s": dt,
    "frames": {
        n: {
            "t_s": float(t[i]),
            "tip_height_m": float(tip[i]),
            "compression_m": float(comp[i]),
        }
        for n, i in picks.items()
    },
    "cycle": {
        "t_s": t[window].round(4).tolist(),
        "tip_height_m": tip[window].round(5).tolist(),
        "compression_m": comp[window].round(5).tolist(),
        "tip_contact": contact[window].astype(int).tolist(),
    },
}
(out / "hop-frames.json").write_text(json.dumps(doc, indent=1))
write_scene(m, {n: qpos[i] for n, i in frames.items()}, out)
print({n: round(float(t[i]), 3) for n, i in frames.items()})
