"""Terrain presets backed by mjlab's heightfield generator.

Each preset pins one sub-terrain at a fixed difficulty. `height` is the wave
amplitude, the rough-ground noise ceiling, the block height or the stair riser;
`slope` is rise over run. Spawn origins come from the generator; reset_mixed
corrects the spawn height with a ray under the mounted tip.
"""

from mjlab.terrains import (
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
)
from mjlab.terrains.terrain_entity import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

KINDS = ("flat", "waves", "slope", "rough", "steps", "stairs", "stairs_down")


def terrain_cfg(kind: str, height: float, slope: float, seed: int):
    if kind == "flat":
        return TerrainEntityCfg(terrain_type="plane")
    if kind not in KINDS:
        raise ValueError(f"Unknown terrain {kind!r}; choose one of {KINDS}")
    if kind == "slope" and slope <= 0:
        raise ValueError("slope gradient must be positive")
    if kind != "slope" and height <= 0:
        raise ValueError(f"{kind} height must be positive")
    if kind == "waves":
        patch = HfWaveTerrainCfg(
            amplitude_range=(height, height),
            num_waves=3,
            horizontal_scale=0.05,
            vertical_scale=0.001,
        )
    elif kind == "slope":
        patch = HfPyramidSlopedTerrainCfg(
            slope_range=(slope, slope),
            platform_width=1.0,
            inverted=True,  # flat valley spawn; uphill in every outward direction
            horizontal_scale=0.05,
            vertical_scale=0.001,
        )
    elif kind == "rough":
        # 10 cm cells with heights in [0, height]: the tip lands on cell edges.
        patch = HfRandomUniformTerrainCfg(
            noise_range=(0.0, height),
            noise_step=0.01,
            horizontal_scale=0.1,
            vertical_scale=0.001,
        )
    elif kind == "stairs":
        # Flat 1 m spawn at the bottom; 0.6 m treads rise outward in every
        # direction, so hopping forward climbs.
        patch = BoxInvertedPyramidStairsTerrainCfg(
            step_height_range=(height, height), step_width=0.6, platform_width=1.0
        )
    elif kind == "stairs_down":
        # Flat 1 m spawn on top; the same treads fall away outward.
        patch = BoxPyramidStairsTerrainCfg(
            step_height_range=(height, height), step_width=0.6, platform_width=1.0
        )
    else:  # steps
        # Raised blocks of one height on a flat floor, flat 1 m spawn platform.
        patch = HfDiscreteObstaclesTerrainCfg(
            obstacle_height_mode="fixed",
            obstacle_height_range=(height, height),
            obstacle_width_range=(0.4, 1.0),
            num_obstacles=24,
            platform_width=1.0,
            horizontal_scale=0.05,
            vertical_scale=0.001,
        )
    return TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=seed, size=(24.0, 24.0), sub_terrains={kind: patch}
        ),
    )
