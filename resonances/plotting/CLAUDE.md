# plotting — figure conventions

User docs: `docs/plots.md`. Every rule below was decided with the author (dates in `dev/decisions.md`, Figures).

## Modules

- `plotting/mmr_combined.py`: the one-page v2 combined MMR figure (layout V1, 24 × 13.5 in): time panels left, every decision
  number right (`mmr_verdict_panel.py`; the column engine shared with the secular/ZLK figure is `combined_numbers.py`);
  the package's `combined` kind for MMR. W23: the numbers fit by layout first (`combined_numbers.FIT_LEVELS`: grey
  numbers out, legend under the steps, a fifth column, only then compact; overflow 0 on the benchmark), overlapping
  labels merge over their joint span (`merge_overlapping_labels`).
- `plotting/verdict_text.py`: the explanation of a verdict for the right column (layout A′, `combined.numbers=explained`):
  plaque, sentences (what σ does, the deciding rule, the confidence), tree steps, margin/cap rows (✓ ⚠ ✗ – ? · ▼).
  Pure; `combined_numbers.draw_explained` draws it; every printed comparison is tested against the record; margins as
  the tree read them (R-ALIAS-A: `alias_margins`), the union rule named (`union_kind`), "next check: N yr" (N8).
  `physics_sections`: block A (σ's own pendulum) and B (what makes the cycles: owner, identity, forced wave, second
  capture); `CAP_ALTERNATIVES`: every cap down to medium/low names the other reading and what decides it. A withheld
  G2a step and the direction of an a-swing change ("shrank"/"grew") are worded here.
- `plotting/mmr_plots.py`: `MMRDiagnostics` + `draw_*` panels + `MMRPlotter` (standalone panels: recurrence, FAIR, portrait,
  cycles; the combined figure is `mmr_combined`). `plotting/style.py` holds the `screen`/`paper` styles and `plot_options`
  defaults/validation.
- `plotting/secular_plots.py`: `SecularDiagnostics` + `draw_*` + `SecularPlotter` (secular and ZLK rows): kinds `angle`,
  `rate`, `portrait`, `combined` (the one-page V1 figure, `draw_long_period_combined`). `SecularPlotter` and
  `MMRPlotter` share `DiagnosticPlotter` (`plotting/base.py`); `DataManager.DIAGNOSTIC_PLOTTERS` maps resonance types to them;
  panel (f) draws the rate of each element in `Response.variables` (d sin(i/2)/dt for an Ω-only angle, with `r2_i`),
  (h) is sin(i/2) cos/sin σ there.
- `clone_plots.py`: the `clones` kind (angles of every clone coloured by status, status against initial a, shares
  with Wilson intervals).
- Evolution plots (flexible panel configurator), ecc-vector, cross-spectrum and free-omega plots.
- `resonance/oscillations.py`: cycles (max to max), recurrence distance, FAIR coordinates. Pure math feeding the MMR
  plots; changes no status.

## Rules

- The MMR diagnostic plots (recurrence D, cycles) are for reading figures: D uses the object's own scales and is not
  comparable between objects. Nothing drawn may feed a status.
- The text reads the same predicate as the decision (e.g. the moving-centre wording reads `orbit_type.centre_moves`,
  as the header does); no number is computed twice, every printed number is a ledger value or a CSV column.
- Plotting classes inherit `BasePlotter`; styles only through `rc_context`.
- Overflow 0 on the benchmark is tested; zero overflow does not mean no annotation overlaps.

## The author's figure decisions

- One page per row, layout V1; time series left, all decision numbers right, also in the CSV. Recurrence and cycles
  are not in the combined figure.
- Plaque `STATUS · angle_status · CONFIDENCE`, "ends: …" last.
- σ̇ axis symlog σ̇/2ω₀ for every record, the σ̇ = 0 line highlighted; the osculating a envelope clipped at the axis
  (triangle at the edge).
- Panel (d): three levels of a; a_res on top, style B (width 1.3, alpha 0.8, white halo); a* alpha 0.8
  (`AXIS_LEVEL_ALPHA`).
- Angle labels in [0°, 360°) (270°, not −90°).
- Portraits square, at the same scale, coloured per episode; the owner dashed and translucent; legend on the right.
- Header `episodes about X° (not this resonance)` for non_resonant/uncertain rows with an induced or kinematic origin.
- FAIR is not drawn for three-body or retrograde resonances; 33128 6N-7 is the benchmark's "FAIR fails" example.
- Frozen-potential overlays stay off by default (`dev/rejected.md`).

## Known defects (open)

- Panel (e) draws the single-sine windows even when CS12 selects the harmonic law (ND13: 0.18/0.19 drawn, 0.646
  decided).
- Three ▼-marker overlaps remain on the benchmark.
- `benchmark/tools/reclassify.py --figures` writes into `benchmark/data/examples/`; saved figures can be stale.
