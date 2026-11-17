import numpy as np
import pytest

pytest.importorskip("mjlab", reason="mjlab (train extra) not installed")

from pogo_g1 import assembly, robot, symmetry  # noqa: E402


def test_joint_mirror_is_involution_and_pose_symmetric():
    _, m, pose = assembly.build_scene()
    names = [n for n in pose.joints if not robot.is_finger(n)]
    perm, sign = symmetry.joint_mirror(m, names)
    q = np.random.default_rng(0).normal(size=len(names))
    assert np.allclose(sign * (sign * q[perm])[perm], q)
    q0 = np.array([pose.joints[n] for n in names])
    # The official left and right Revo2 models are not exact mirror images, so the
    # settled grasps differ slightly; the body pose stays symmetric within 0.6 deg.
    assert np.abs(q0 - sign * q0[perm]).max() < 1e-2
