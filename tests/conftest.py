from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def pytest_collection_modifyitems(config, items):
    """Tests that build the robot need the local assets as well as mjlab."""
    try:
        from pogo_g1 import pogo, robot
    except ImportError:
        return  # those modules are skipped at import (pytest.importorskip)
    if (ROOT / pogo.DEFAULT_ASSET / "manifest.json").exists() and (
        ROOT / robot.HAND_URDF
    ).exists():
        return
    for item in items:
        if item.module.__name__ in ("test_physics", "test_symmetry"):
            item.add_marker(pytest.mark.skip(reason="local assets not available"))
