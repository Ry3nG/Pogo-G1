# Assets

## Pogo stick: Vurtego Pogo Stick Model

- Author: RoboHoloclone12 (Joshua-Ibrahim De Guzman), licence **CC BY 4.0**.
- Source: https://sketchfab.com/3d-models/vurtego-pogo-stick-model-01b521f3c511437da40224ecaa7658fc
- Archive SHA256: `1687a9389e671a7fa38b147bd3b2a620c97294dac3ecdb9a4cf309128a18df18`.
  Save it as `data/assets/vurtego/source.zip` (not in git), then run
  `scripts/prepare_asset.py`, which writes `data/assets/vurtego/small-v4-pro-v3/`.
- The meshes are appearance only: zero mass, no collision. Collision and
  inertia use primitive proxies measured from the adapted mesh bounds. Physics
  is in `pogo.PogoParams` and DESIGN.md section 2.

The import adapts the model to the manufacturer's Small V4 Pro dimensions. It
is a visual adaptation, not CAD.

| Quantity | Value | Source |
|---|---|---|
| Extended height | 1.2446 m (49 in) | [product page][vurtego-sizes] |
| Upper grip surface to pedal top | 0.762 m (30 in) | [product page][vurtego-sizes] |
| Pedal top to tip | 0.4826 m | from the two above |
| Stroke | 0.432 m (17 in) | [product page][vurtego-sizes] |
| Handlebar outer diameter | 22.225 mm (7/8 in) | [handlebar page][vurtego-handle] |
| Grip radius | 14.576 mm | adapted mesh |
| Tube outer diameter | 80.2 mm | 76.2 mm bore with an estimated 2 mm wall |
| Total mass | 4.54 kg | [FAQ][vurtego-faq] |
| Frame / plunger split | 3.64 / 0.90 kg | estimate |

A single uniform scale cannot match both the bar diameter and the overall
height, so the importer uses independent part anchors and keeps circular bar
and grip sections. `manifest.json` records all bounds, drawing anchors, the
source checksum and generated-file checksums; the mounted-pose cache includes
it, so regenerated geometry never reuses a stale pose. Dimensions of
unmeasured subparts (wall, grip rubber, cap and shaft ends) are visual and
collision adaptations.

Attribution for any published figure or video: "Vurtego Pogo Stick Model" by
RoboHoloclone12, CC BY 4.0, adapted (scaled, re-rigged with simulation proxies).

[vurtego-faq]: https://web.archive.org/web/20210125042001/https://www.vurtegopogo.com/faq/
[vurtego-sizes]: https://web.archive.org/web/20210225001949/https://www.vurtegopogo.com/v4-pogo-stick/
[vurtego-handle]: https://web.archive.org/web/20210118210549/https://www.vurtegopogo.com/product/handlebars/

## Robot

- G1 29-DoF body, actuators and collisions come from mjlab 1.6.0 (`g1.xml`, Apache-2.0).
- BrainCo Revo2 hands: Unitree's `unitree_ros/robots/g1_with_brainco_hand`
  (`g1_29dof_mode_15_brainco_hand.urdf` and meshes, BSD-3-Clause), commit
  `ccfc6fd8430a17ba3dacef9a1e2faf64ff3b0aee`, sparse checkout under
  `data/unitree_ros` (`scripts/fetch_assets.sh`). Used: hand links, inertias,
  mounting on `wrist_yaw_link`, joint ranges, effort limits and mimic ratios. The
  zero-size fingertip marker links (fake fixed-angle joints) are dropped.
- Revo2 ratings: about 383 g, 50 N grip, 20 kg hook payload (BrainCo product pages).
