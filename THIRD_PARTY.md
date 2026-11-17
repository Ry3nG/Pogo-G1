# Third-party components

- **mjlab 1.6.0** (Apache-2.0): environment framework, G1 model and actuator constants.
  Version and hashes are pinned in `uv.lock`.
- **MuJoCo / MuJoCo Warp 3.15.0** (Apache-2.0), **rsl_rl** through mjlab (BSD-3-Clause).
- **Unitree `unitree_ros`** (BSD-3-Clause): G1 + BrainCo Revo2 hand URDF and meshes.
  Downloaded separately and not vendored (docs/ASSETS.md).
- **Vurtego Pogo Stick Model** by RoboHoloclone12 (CC BY 4.0): visual meshes only.
  See docs/ASSETS.md.
- **D3** and **Observable Plot** (ISC), vendored in `figures/src/vendor` for the figures.
- **Source Sans 3** (SIL OFL 1.1) and **Latin Modern Roman** (GUST Font License), in
  `figures/src/fonts`.

The renders in `figures/` and `docs/media/` show the adapted Vurtego model (CC BY 4.0)
and Unitree's G1 and Revo2 meshes.

Actuator constants describe a model, not certified hardware limits.
