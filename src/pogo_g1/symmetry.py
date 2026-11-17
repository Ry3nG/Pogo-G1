"""Left-right (sagittal-plane) mirror for rsl_rl symmetry augmentation.

Mirror = reflection y -> -y in the robot's heading frame. Polar vectors flip y;
axial vectors (angular velocities) flip x and z. Joints swap left/right with a
sign from their axes. The mounted pose is solved symmetric, so joint offsets
mirror without bias. Maps are built from the observation manager's term layout,
so they stay correct when terms or history length change.
"""

import numpy as np
import torch

from pogo_g1.robot import is_finger

POLAR = np.array([1.0, -1.0, 1.0])
AXIAL = np.array([-1.0, 1.0, -1.0])


def counterpart(name: str) -> str:
    if "left_" in name:
        return name.replace("left_", "right_")
    if "right_" in name:
        return name.replace("right_", "left_")
    return name


def joint_mirror(model, names, prefix="robot/"):
    """Permutation and signs such that q_mirror = sign * q[perm]."""
    perm, sign = [], []
    for n in names:
        o = counterpart(n)
        perm.append(names.index(o))
        a = model.joint(prefix + n).axis
        b = model.joint(prefix + o).axis * AXIAL
        s = float(a @ b)
        if not np.isclose(abs(s), 1.0, atol=1e-6):
            raise ValueError(f"joint {n} is not mirror-aligned")
        sign.append(s)
    return np.array(perm), np.array(sign)


def _term_map(name, dim, jperm, jsign):
    eye = np.arange(dim)
    if name in ("imu_ang_vel",):
        return eye, AXIAL
    if name in ("imu_acc", "gravity", "base_lin_vel"):
        return eye, POLAR
    if name in ("joint_pos", "joint_vel", "actions"):
        return jperm, jsign
    if name == "command":  # [h, vx, vy, yaw_rate]
        return eye, np.array([1.0, 1.0, -1.0, -1.0])
    if name == "privileged":
        from pogo_g1.mdp import PRIVILEGED_LAYOUT

        perm, sign, i = [], [], 0
        for kind, n in PRIVILEGED_LAYOUT:
            idx = np.arange(i, i + n)
            if kind == "scalar":
                perm += list(idx)
                sign += [1.0] * n
            elif kind == "polar":
                perm += list(idx)
                sign += list(POLAR)
            elif kind == "axial":
                perm += list(idx)
                sign += list(AXIAL)
            elif kind == "planar":
                perm += list(idx)
                sign += [1.0, -1.0]
            elif kind == "pair":
                perm += list(idx[::-1])
                sign += [1.0, 1.0]
            i += n
        assert i == dim, (i, dim)
        return np.array(perm), np.array(sign)
    raise KeyError(f"no mirror rule for observation term {name}")


class Mirror:
    def __init__(self, env):
        robot = env.scene["robot"]
        names = [n for n in robot.joint_names if not is_finger(n)]
        jperm, jsign = joint_mirror(env.sim.mj_model, names)
        act = env.action_manager.get_term("joint_pos")
        if list(act._target_names) != names:
            raise ValueError("action order differs from joint order")
        self.aperm = torch.as_tensor(jperm, device=env.device)
        self.asign = torch.as_tensor(jsign, device=env.device, dtype=torch.float32)
        self.groups = {}
        om = env.observation_manager
        for group in ("actor", "critic"):
            perm, sign, off = [], [], 0
            for term, shape in zip(
                om.active_terms[group], om.group_obs_term_dim[group], strict=True
            ):
                total = int(np.prod(shape))
                cfg = om._group_obs_term_cfgs[group][om.active_terms[group].index(term)]
                hist = max(1, cfg.history_length or 1)
                dim = total // hist
                p, s = _term_map(term, dim, jperm, jsign)
                for h in range(hist):
                    perm += list(off + h * dim + p)
                    sign += list(s)
                off += total
            self.groups[group] = (
                torch.as_tensor(perm, device=env.device),
                torch.as_tensor(sign, device=env.device, dtype=torch.float32),
            )

    def obs(self, group, x):
        perm, sign = self.groups[group]
        return x[..., perm] * sign

    def actions(self, a):
        return a[..., self.aperm] * self.asign


def augment(env, obs=None, actions=None):
    """rsl_rl contract: return [original; mirrored] along the batch dimension."""
    task = env.unwrapped
    if not hasattr(task, "_mirror"):
        task._mirror = Mirror(task)
    m = task._mirror
    if obs is not None:
        mirrored = obs.clone()
        for key in mirrored.keys():
            mirrored[key] = m.obs(key, obs[key])
        obs = torch.cat((obs, mirrored), dim=0)
    if actions is not None:
        actions = torch.cat((actions, m.actions(actions)), dim=0)
    return obs, actions
