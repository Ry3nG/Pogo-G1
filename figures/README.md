# Figures

The seven figures of the report, with their sources.

| Figure | Shows |
|---|---|
| `hop` | the rider at apex, one hop cycle, and yaw error under per-step and per-hop scoring |
| `method` | observation, actor and critic, physics and reward |
| `capability` | success and tracking error at five heights and four commands |
| `signal` | the scored yaw rate under exploration noise |
| `learning` | learning to turn against the fluctuation of each scored quantity |
| `nus` | the 18 m path hopped under the path follower |
| `grip` | the grip's safety margin and what weakened-hand training buys |

```
data/<name>.json   numbers behind each figure
src/<name>.html    template drawn with Observable Plot and D3 (src/vendor)
renders/           Blender stills of simulation states (scripts/render)
<name>.pdf         vector output
png/<name>.png     220 dpi preview, shown in the top-level README
```

Build one figure or all of them:

```sh
python3 figures/src/build.py [name ...]
```

The build needs Chrome (set `CHROME` to its binary), poppler-utils and Pillow.
It inlines the data, fonts and renders into the template and prints it to a
one-page PDF.

`scripts/fig_data.py figures/data` regenerates `signal`, `learning`,
`capability`, `grip` and `nus` from evaluation results under `artifacts/`.
The committed files also carry values added for the final report (each file
names its source in a `readme`, `aggregation` or `needed_source` field), and
`hop.json` comes from `scripts/hop_cycle.py`.

Fonts: Source Sans 3 (SIL Open Font License 1.1) and Latin Modern Roman (GUST
Font License), converted to TrueType so Chrome embeds them as CID fonts.
