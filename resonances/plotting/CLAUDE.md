# plotting — figure conventions

User docs: `docs/plots.md` (files, settings) and `docs/reading-figures.md` (how to read each panel). A change to what a
figure shows updates both and reruns `make docs-figures`. Every rule below was decided with the author (dates in `dev/decisions.md`, Figures).

## Modules

- The right column (2026-11-04, the author's decisions D1–D10: `dev/decisions.md`, Figures; reader's guide
  docs/reading-figures.md#the-right-column). `plotting/verdict_column.py` is the model: `Column` of ten blocks
  (`BLOCKS`: header/chip, subtitle, key facts, why, confidence/pills, checks/sub-checks, evidence, caveats, numbers,
  footer), the icons ✓ ⚠ ✗ ? – ◆, the chip kinds, the inline markup (`**b**`, `!!status!!`, `[[numbers]]`,
  `<<panels>>`), `column_text` (the complete text export) and `problems` (the contract, tested on every built column:
  one ◆, its rule = the footer's, header level = Confidence level, codes/pointers only in Numbers and the footer, the
  action said once, ⚠ pills name their input, ✓ only for a best state, – never for an input that holds the level).
  Builders: `verdict_column_mmr.py` (MMR, 1:1; checks `NEST` margin → step, Q first, `P` policy line when
  `pol_decided_by` ≠ tree, the leaf then 'overruled'; Numbers `GROUPS`/`LATER_GROUPS`), `verdict_column_long.py`,
  `verdict_column_why.py` (Why: templates keyed on family and reason; `would_change` only from computed clauses),
  `verdict_column_common.py` (header chip, pills from `pol_level_inputs`, footer with `W28 tree: status/level`
  always, D2; `unnest`/`tidy_checks`: sub-checks under the step that reads them, duplicates merged). A step's icon
  is the worst of its sub-checks, a ? only from a rule the step's answer reads (`_worst(children, step.ids)`). A builder never computes a number: it calls `verdict_text`'s predicates and clause functions.
- `plotting/verdict_column_draw.py` (2026-11-04, the right column redesign, step 2): draws a `verdict_column.Column` into
  the right area (first to last `text_columns`) as two sub-columns (~5.55 in each) with a fixed place per block (the
  author, 2026-11-05: figures are read in batches): header to Checks in the first, Evidence (the first shown block of
  `SECOND_SUBCOLUMN`) at the top of the second, then Caveats, Numbers, footer; neither part flows into the other's
  sub-column, a long part grows the figure (D8). Numbers groups in one order for every family (LIBRATION, response/
  gate, …, ORBIT, REFERENCE PERIODS/FREQUENCIES, …, STEP, FILTER last). FREQUENCIES cells sit at measured x (an empty header cell or a fallback glyph never shifts a
  column); a short footer item and a title's grey gloss never break inside.
  Measured with the Agg renderer at the figure dpi, and the same units are measured and drawn. D8: `fit_layout(column,
  layout)` → `layout.grown(delta)` (left rects move up, text columns grow down; `style.COLUMN_MAX_GROWTH_IN`, 6 in), then
  `draw_column(fig, column, layout, fit)`; beyond the cap `trim_levels`: Numbers groups from the last (DECISIVE, ORBIT
  stay), ✓ sub-checks collapsed per step, Evidence one line per row. `draw_column_figure` = the column alone. Colours:
  `style.STATUS_COLOURS` (D4 palette), `COLUMN_ICON_COLOURS`, `CHIP_COLOURS`; every artist has gid `verdict_column`.
  Only ASCII spaces break a line (a rule tag is glued with U+00A0); a character missing from DejaVu Sans Mono falls back
  to DejaVu Sans.
- `plotting/verdict_column_long.py` (2026-11-04, step 2): `long_column(analysis, settings, row, episodes, frequencies,
  family)` → the `Column` of a secular/ZLK row (Why from `verdict_column_why.long_why`). Steps Q (filter guard SZ.F; an
  unphysical orbit ◆ here), 0 shape (SZ.N/S/P, G5s; ZLK: the centre scale a sub-check), secular 1 centre scale / 2 phase
  error (G5p), ZLK 1 free-ω gate (SZ.G) / 2 the deciding episode's e-i (◆ on a veto zlk_integral); `deciding_step` reads
  `pol_decided_by`. Pills are `verdict_column_common.pills` in long-period words (veto, M, V; second_capture dropped);
  caveats from `verdict_vocab.FLAG_CAVEATS`, then `LONG_CAVEATS`; the secular e-response only in Numbers and the footer
  (D7). Free-ω turns are the column free_omega_revolutions (net).
- The wiring (2026-11-04, step 2): in the explained mode both figures draw the `verdict_column` column.
  `mmr_combined.draw_combined` → `prepare_column` (`verdict_column_mmr.figure_parts` + `verdict_column_draw.prepare`)
  before the figure exists, the figure built on the grown layout, then `verdict_column_draw.draw_prepared`;
  `secular_plots.draw_long_period_combined` the same with `verdict_column_long.figure_parts`.
  `fig.text_stats` = the drawing's statistics + `column` + `text` (the VerdictText). The package export is
  `verdict_column_mmr.figure_column(analysis, rows, settings)` / `verdict_column_long.figure_column(analysis, row,
  free_elements, settings)` (the very column the figure draws; `verdict_column.column_text` writes it). The old
  engine (`combined_numbers.draw_explained`, `explained_lines`, `secular_plots.draw_long_period_explained_v1`) is no
  longer drawn, kept until the cleanup. The contract (`verdict_column.problems` == []) is checked on every built
  column in the tests, and on the benchmark by `test_verdict_column_sweep.py` (every secular/ZLK example; one MMR
  example per kind of verdict of the last test-benchmark run, ~3 min). K3s decides at step 3 (`_kinematic_step`;
  37087 had no ◆); an analysis without a leaf (built by hand) gets one ◆ line 'R · Which rule decided? none recorded'.
- Time axis (the author, 2026-11-05): labelled under the last panel and again under (b) (`label_time_axes`,
  `panel_rects(..., labelled)`, `VerdictFigureLayout.time_label_gap` 0.15 in more below (b)), both figures.
- `plotting/mmr_combined.py`: the one-page v2 combined MMR figure (layout V1, 24 × 13.5 in): time panels left, every decision
  number right (`mmr_verdict_panel.py`; the column engine shared with the secular/ZLK figure is `combined_numbers.py`);
  the package's `combined` kind for MMR. The `all` mode and the old engine (W23): the numbers fit by layout first (`combined_numbers.FIT_LEVELS`: grey
  numbers out, legend under the steps, a fifth column, only then compact; overflow 0 on the benchmark), overlapping
  labels merge over their joint span (`merge_overlapping_labels`). Number labels: `combined_numbers.COLUMN_LABELS`
  (words + source, read first by `label_of`; W28 E11: T0 is 'from observed cycles', never 'theory') and
  `RANGE_LABELS` (min/max pairs as one 'lo–hi' line in the explained column); every other column prints its stem.
- `plotting/verdict_text.py`: the library of the right column's predicates and clause functions (`MMRFacts`,
  `VerdictText`, `mmr_steps`/`mmr_checks`, `physics_sections`, `number_emphasis`, `owner_confirmed`,
  `single_swing_clause`, `secular_reference`, `zlk_physics_line`, `shown_rule` (the column `geo_angle_rule`), …),
  read by the `verdict_column_*` builders. Every printed comparison is tested against the record; margins as the tree
  read them. The old drawn parts (slotted bullets `mmr_bullets`/`long_period_bullets`, `policy_block`, 'Numbers
  behind this verdict' with `↑` pointers, LIMIT ▼ rows, `action_tag`, `combined_numbers.draw_explained`/
  `explained_lines`/`behind_lines`, `secular_plots.draw_long_period_explained_v1`) are no longer drawn and stay until
  the pre-merge cleanup; their history is in `dev/decisions.md` (Figures, Oct 31).
- `plotting/verdict_vocab.py`: the words of that column, a registry without logic (W28 E2): `CAPS`/`CapText` (a WP that
  adds a cap adds its `CapText` here; `test_w20_cs11_physics`), the reason, flag, ledger and margin words, and
  `RULE_NAMES`: every printable id → short name, printed `short name [id]` (`named`), the subtitle `decided: circulation
  (C)` (`decided`); prefixes only where ids collide (`SZ.N`…, `cap:<key>`). A new id needs a name (completeness test in
  `test_verdict_text.py`) and a row in docs/libration-analysis.md#names (the same test reads it). The right column's
  tables: `GLANCE_PANELS` (D5: input that holds the level → panels of the `glance at` chip, per family; awaits the
  author's review; `GLANCE_OVERRIDES`: an unfitted ZLK co-orbital's V → (a), (d)) and `FLAG_CAVEATS` (flag → statuses →
  Caveats sentence; others go to the footer only). `MMR_MARGIN_FAILED`: a sub-check's name when its printed
  comparison fails (`verdict_column_mmr.failed_reading`: the operator `compare` printed, or the state where no
  comparison is printed), so no ⚠/✗/– row states the passing condition beside failing numbers; Q answers `Partly:`/`No:`
  in those words. Pills: N and core say how they count (`confidence_basis`) and which level their bar opens; – pills
  'alone it would cap the level at X'; circulation names the cutoff with turning points (geo_n_tp_cut_*).
- `plotting/mmr_plots.py`: `MMRDiagnostics` + `draw_*` panels + `MMRPlotter` (standalone panels: recurrence, FAIR, portrait,
  cycles; the combined figure is `mmr_combined`). The a of `cycles`, of `portrait` (y='axis') and of a0 is the
  analysis' a_f (`axis_analysis`), else the raw a, never `body.axis_filtered` (a 2000-yr low-pass erases a faster MMR
  libration: 1921 2J-1).
  `plotting/style.py` holds the `screen`/`paper` styles and `plot_options` defaults/validation.
- `plotting/secular_plots.py`: `SecularDiagnostics` + `draw_*` + `SecularPlotter` (secular and ZLK rows): kinds `angle`,
  `rate`, `portrait`, `combined` (the one-page V1 figure, `draw_long_period_combined`). `SecularPlotter` and
  `MMRPlotter` share `DiagnosticPlotter` (`plotting/base.py`); `DataManager.DIAGNOSTIC_PLOTTERS` maps resonance types to them;
  W29 (author d11): (d) is a (`_draw_axis`: osculating band, a filtered at P_c, min–max; a_P dashed and the
  co-orbital zone's edges a_P(1 ± k (m_P/3)^(1/3)) dotted when `coorbital_planet_geometric` names a planet by ⟨a⟩
  alone, ZLK_COORBITAL_HILL_FACTOR k, for drawing only: Trojans and secular rows too), so e, i, physics, centre,
  portraits are (e)…(j) (`*_LETTER`, `PORTRAIT_LETTERS`).
  panel (g) draws the rate of each element in `Response.variables` (d sin(i/2)/dt for an Ω-only angle, with `r2_i`),
  (i) is sin(i/2) cos/sin σ there. ZLK (g): the drift of the osculating and the free ω from their first values, in
  turns, with a legend (`_draw_free_omega_drift`); the right column's Evidence 'what holds ω?' (forced vector, centre
  wave, node lock, e-i coupling) reads `verdict_text`'s ZLK line functions, measured facts only. Panel (a′) (`_draw_forcing`, `forcing_panel_drawn`): only for an attributed row or a forced-beat centre,
  under (a), nine panels then; every other row keeps eight, untouched. ω_f grey, arg z_forced − Ω orange, ϖ_j − Ω
  purple dashed, from `FreeElements.forced_vector` and the osculating node; unwrapped within a turn, else modulo 360°
  about the forced direction with ω_f inside the librating episodes only (624).
  Right column (W27, author's request): ORBIT has a (`ctx_a_med_au`/`_min`/`_max`, columns of the long-period row);
  FREQUENCIES (`long_period_frequency_lines`, both numbers modes) is a table `name  ″/yr  period  note`, kept in one
  column (`NumberLine.keep`): secular: g, s, σ from mean rates, g5/g6/s6 + the angle's modes (`in σ ±k`), the forced
  waves g − g_j; ZLK: free g, s, g − s, every fitted mode with |c_j| (◀ the attributed mode), g − g_j of the two
  largest forced modes (`kinematic` for the attributed one), their mutual beat, the node lock. Phases not shown.
  The rows are `Frequency` items (`long_period_frequencies`), shared with the text.
  The old explanation (W28 E9: `long_period_bullets`, `SLOT_SCALE`, `↑ in the text`) is no longer drawn (above).
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
- Printed text is American English (the author, 2026-11-05); identifiers, ledger names, flags and CSV columns of the
  analysis keep British spelling (`centre_moves`, `geo_centre_*`, ledger keys such as 'G5a slow centre' in
  `verdict_vocab`). An identifier shown as words goes through `verdict_vocab.snake_words`, never a bare
  `.replace('_', ' ')`.

## The author's figure decisions

- One page per row, layout V1; time series left, all decision numbers right, also in the CSV. Recurrence and cycles
  are not in the combined figure.
- The right column (2026-11-04, D1–D10 + go; supersedes the W28 bullets, the plaque tag and the policy lines below
  where they describe the drawn column): ten blocks in one order for every family; one status word on a plaque in the
  D4 palette, the level as dots, one action chip (none grey, `glance at (…)` amber with D5's panels, review red,
  extend/resample blue with the length or step); numbers printed wherever needed, no pointers (D1); the W28 tree only
  in the footer, always, `W28 tree: status/level`, no agrees/differs (D2); rule ids as small grey tags on check rows
  and in DECISIVE, W-codes, column stems and flag names only in the footer (D3); the forced-wave veto worded from
  `origin_detail == forced_wave_background` (D6); the secular e-response only in Numbers and the footer (D7); the
  figure grows down, capped, then trims Numbers and ✓ sub-checks, Evidence compacted, never dropped (D8); the
  package's terms with glosses, 'overruled by X', 'not certain because …' (D9). The three-body e^q strength scale is a
  grey '– scale only', never ✗.
- σ̇ axis symlog σ̇/2ω₀ for every record, the σ̇ = 0 line highlighted; the osculating a envelope clipped at the axis
  (triangle at the edge).
- Panel (e): the law that decided on the series it read (`law_series`, `_law_shift`): the windows' sine on ψ = σ − c(t)
  at the response cutoff, or the multiplet law (`MultipletFit.pooled`, `response.members_model`); at the raised cutoff
  of the line rule with `cutoff … (line …)`; none when the harmonic law (CS12) or the 1:1 law decided, the panel
  prints that law's numbers instead (W26). It draws `analysis.response`, rule R's response: a witness episode's
  windows when one decided, the pool otherwise (W28 g11-7, `test_panel_e_witness.py`); `law_series` re-smooths at
  `Response.force_cutoff_yr` whenever it differs from the figure's response cutoff (taken at the decisive period: a
  witness that is not the decisive episode has its own, 34706 7J-2 87 against 75 yr, 22000 5J-2S-2 637 against 538 yr).
- The ▼ of the strip (b′) on (b) (W28 g11-8, `strip_marks`): x is the window's middle and never moves; a ▼ meeting the
  panel letter or the ALIAS plaque is raised over it, its number tried right/hanging, level, left; the boxes are
  measured in pixels (`text_box` with the plaque). The W23 census overlaps (1862 12V-35, 2000 YG4 7E-23, 2025 PN7
  20 kyr) are tested at their positions.
- Panels (i), (j) and the secular cause (W28 B10b, author g08-8: "если причина — в вековом цикле, это надо показывать
  линией … и везде"): (i) hatches the slow scale (`slow_intervals`, `SLOW_HATCH`, 'slow #id'); (j) overlays the
  detrended p·ϖ of the slow scale, orange dashed, '−9ϖ 7.11 kyr', whenever slow_cause is secular (`varpi_overlay`);
  the kinematic mode (`centre_mode`: a kinematic origin, or a non_resonant row with slow_origin kinematic) draws
  σ = λ part + pϖ in degrees at `kinematic_scale` (the decisive episode at its response cutoff, as K1e; the slow
  scale for K3s and slow-only rows, detrended per slow episode), never at W_c alone (285637: W_c 54 kyr erased the
  18-kyr cycle). Nothing drawn without a slow swing; nothing drawn feeds a status.
- Panel (d): three levels of a; a_res on top, style B (width 1.3, alpha 0.8, white halo); a* alpha 0.8
  (`AXIS_LEVEL_ALPHA`).
- Angle labels in [0°, 360°) (270°, not −90°).
- Portraits square, at the same scale, coloured per episode; the owner dashed and translucent; legend on the right.
- Header `episodes about X° (not this resonance)` for non_resonant/uncertain rows with an induced or kinematic origin.
- Header second line (W28 A3b, card g04-6 A: two words owner and driver, the reason unchanged): the role of the other
  resonance (`verdict_text.cause_line`), shrunk to the header's width (`fit_width`); the record line moves under it.
  Induced episodes (episodes.csv libration_origin induced, `induced_episode_ids`) hatched `\\\\` on (a)
  (`INDUCED_HATCH`, apart from the jump's `////`) and labelled `#id induced` on (l); LEGEND (a) names both hatches.
- FAIR is not drawn for three-body or retrograde resonances; 33128 6N-7 is the benchmark's "FAIR fails" example.
- No empty slot (W28 g03-7, 2026-10-31: "пустой слот рисовать — не хорошо"): without FAIR there are no (m) axes, the
  legend takes the slot under one dark-orange warning line with the reason in words (`fair_warning`,
  `FAIR_REASON_WORDS`; a 1:1 without the planet adds 'for a 1:1 FAIR = (M, σ) repeats (b)'); the header's
  'FAIR no (<reason>)' stays. FAIR is not restored for co-orbitals without saved planets.
- A covered edge is filled to the record's edge only when the covered gap reaches within a period of it; otherwise to
  the cut that opened the run's window (`_covered_to`, W31: 439908 was filled across 13.6 kyr of circulation).
- Panel (l) over the fill interval of (a) (W28 g03-6, answer A): `portrait_span` = `episode_spans` clipped to the
  window of (k), shared by (l) and `episode_rate_limits`; beyond the first/last turning point dashed
  (`portrait_pieces`, `PORTRAIT_EXTENSION_STYLE`). A return-closed episode over the record (2025 RB7) gives (l) = (k).
  The force-law window is unchanged. The narrow legend (FAIR drawn) is full at LEGEND_MIN_FONT_PT: a longer LEGEND
  line overflows it (`test_right_labels_and_legend_stay_in_their_boxes`).
- One 1:1 episode in parts (W28 C13, cards g03-1 A, g03-3 C, g03-8 A, g03-9 A; C7 `marks`, C8 residence and
  sub-regimes): (a) brackets each sub-regime over the panel with its well (`draw_subregime_brackets`; the same well
  after an amplitude change `new amplitude`), the inferred residence beyond the fill lighter, hatched `--`,
  `inferred` (`residence_pieces`; never on (l), which keeps the fill span); the labels of such a run merge where they
  meet (`merge_label_items`); (d) ▼ `kick` (`draw_kicks`); header `orbit: A → B` for a sequence of wells
  (`orbit_type_text`). Words: `verdict_text.coorbital_parts_clause` ('measured … ; stays … (inferred, not measured)',
  never 'exit' for a residence's end), EPISODES lines `mmr_verdict_panel.coorbital_episode_parts`. The narrow
  legend had room for 'brackets … hatched – inferred stay' only as '⊓ sub-regime, hatched – inferred stay'.
- Right column (W28, 2026-10-31: g10-2 A … g03-5 A; superseded on 2026-11-04 by the ten-block column above): kept
  from it: `short name [id]` with the ids of docs/libration-analysis.md, steps as questions, one radial-overlap line
  over the eight planets, pendulum numbers labelled in words with their source. Accepted runs the status does not count
  are never episodes.
- Frozen-potential overlays stay off by default (`dev/rejected.md`).
- The minimal policy (W29 SPEC §7, PRINCIPLES 6): (a) of both figures shows only what the status counts
  (`mmr_combined.policy_marks` from `analysis.policy`, else the verdict's id cells; `shade_policy_runs`): counted
  episodes filled (a covered edge to the record edge, the filter-trim stretch pale, `fill_counted`), a counted 1:1
  leg 'leg', candidates faint and dashed 'cand' (`fill_candidate`), no grey strip; (l) puts the counted episodes first,
  solid, the others dashed '#id cand'/'#id not counted' (`_policy_tag`); the EPISODES block counts the policy's
  `counted` (`mmr_verdict_panel._counted_text`); a counted strength run is no ' forced (strength)'; the legend's (a) line follows
  (`legend_lines`, `long_period_legend`). Without a policy, nothing changes. Numbers ≥ `SCI_FROM`
  (1e6) print as '9.63e35'. The standalone MMR panels' re-run (`mmr_plots._rerun_analysis`) carries the stored verdict, or re-runs with the scans on.

## Known defects (open)

- `benchmark/tools/reclassify.py --figures` writes into `benchmark/data/examples/`; saved figures can be stale.
- The W28 Codex right-column defects: S4, S5, S8, S9 fixed in W34 (textfix: `episode_ei_objection`, `capture_hull`,
  `_around_leg`, `filter_too_slow_text`); S2 (WK15) and S3 (2002 BG) no longer occur on the current analysis. The
  "both resonances at once?" test itself still runs over the owner hull (S5 is worded, not re-measured: an analysis
  change, outside plotting).
- An owner cover that is not measured (NaN at the record edge) prints as a dash in the roles line and the right
  column, with no words of its own (cosmetic).
- The docs gallery (`docs/physics/gallery.md`) has no secular transient example since 2306 became uncertain.
