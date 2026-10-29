# plotting — figure conventions

User docs: `docs/plots.md` (files, settings) and `docs/reading-figures.md` (how to read each panel). A change to what a
figure shows updates both and reruns `make docs-figures`. Every rule below was decided with the author (dates in `dev/decisions.md`, Figures).

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
  cycles; the combined figure is `mmr_combined`). The a of `cycles`, of `portrait` (y='axis') and of a0 is the
  analysis' a_f (`axis_analysis`), else the raw a, never `body.axis_filtered` (a 2000-yr low-pass erases a faster MMR
  libration: 1921 2J-1).
  `plotting/style.py` holds the `screen`/`paper` styles and `plot_options` defaults/validation.
- `plotting/secular_plots.py`: `SecularDiagnostics` + `draw_*` + `SecularPlotter` (secular and ZLK rows): kinds `angle`,
  `rate`, `portrait`, `combined` (the one-page V1 figure, `draw_long_period_combined`). `SecularPlotter` and
  `MMRPlotter` share `DiagnosticPlotter` (`plotting/base.py`); `DataManager.DIAGNOSTIC_PLOTTERS` maps resonance types to them;
  panel (f) draws the rate of each element in `Response.variables` (d sin(i/2)/dt for an Ω-only angle, with `r2_i`),
  (h) is sin(i/2) cos/sin σ there. ZLK (f): the drift of the osculating and the free ω from their first values, in
  turns, with a legend (`_draw_free_omega_drift`); the right column's "also measured" lines (e-i coupling, attribution,
  the centre's slow line, the node lock) come from `verdict_text.zlk_column_lines`, measured facts only; the cause of a
  kinematic libration is the third sentence (`verdict_text.zlk_cause`, the same columns; a fact it states leaves the
  grey line). Panel (a′) (`_draw_forcing`, `forcing_panel_drawn`): only for an attributed row or a forced-beat centre,
  under (a), eight panels then; every other row keeps seven, untouched. ω_f grey, arg z_forced − Ω orange, ϖ_j − Ω
  purple dashed, from `FreeElements.forced_vector` and the osculating node; unwrapped within a turn, else modulo 360°
  about the forced direction with ω_f inside the librating episodes only (624).
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
- Panel (e): the law that decided on the series it read (`law_series`, `_law_shift`): the windows' sine on ψ = σ − c(t)
  at the response cutoff, or the multiplet law (`MultipletFit.pooled`, `response.members_model`); at the raised cutoff
  of the line rule with `cutoff … (line …)`; none when the harmonic law (CS12) or the 1:1 law decided, the panel
  prints that law's numbers instead (W26).
- Panel (d): three levels of a; a_res on top, style B (width 1.3, alpha 0.8, white halo); a* alpha 0.8
  (`AXIS_LEVEL_ALPHA`).
- Angle labels in [0°, 360°) (270°, not −90°).
- Portraits square, at the same scale, coloured per episode; the owner dashed and translucent; legend on the right.
- Header `episodes about X° (not this resonance)` for non_resonant/uncertain rows with an induced or kinematic origin.
- FAIR is not drawn for three-body or retrograde resonances; 33128 6N-7 is the benchmark's "FAIR fails" example.
- Frozen-potential overlays stay off by default (`dev/rejected.md`).

## Known defects (open)

- Panel (e) draws the record's pooled fit, not the witness episode's when rule R took L4 from one.
- Three ▼-marker overlaps remain on the benchmark.
- `benchmark/tools/reclassify.py --figures` writes into `benchmark/data/examples/`; saved figures can be stale.
