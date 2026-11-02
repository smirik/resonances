# plotting — figure conventions

User docs: `docs/plots.md` (files, settings) and `docs/reading-figures.md` (how to read each panel). A change to what a
figure shows updates both and reruns `make docs-figures`. Every rule below was decided with the author (dates in `dev/decisions.md`, Figures).

## Modules

- `plotting/mmr_combined.py`: the one-page v2 combined MMR figure (layout V1, 24 × 13.5 in): time panels left, every decision
  number right (`mmr_verdict_panel.py`; the column engine shared with the secular/ZLK figure is `combined_numbers.py`);
  the package's `combined` kind for MMR. W23: the numbers fit by layout first (`combined_numbers.FIT_LEVELS`: grey
  numbers out, legend under the steps, a fifth column, only then compact; overflow 0 on the benchmark), overlapping
  labels merge over their joint span (`merge_overlapping_labels`). Number labels: `combined_numbers.COLUMN_LABELS`
  (words + source, read first by `label_of`; W28 E11: T0 is 'from observed cycles', never 'theory') and
  `RANGE_LABELS` (min/max pairs as one 'lo–hi' line in the explained column); every other column prints its stem.
- `plotting/verdict_text.py`: the explanation of a verdict for the right column (layout A′, `combined.numbers=explained`):
  plaque, sentences (what σ does, the deciding rule, the confidence), tree steps, margin/cap rows (✓ ⚠ ✗ – ? · ▼).
  MMR (W28 E8, cards g10-2 style (i)/g10-8/g05-1/g04-7): `VerdictText.bullets` (`mmr_bullets`, slots `SLOTS`: verdict,
  sigma, decisive, objection, cause, change, orbit) are drawn instead of `sentences` (which secular/ZLK still draw and
  older tests read). Cause lines: 'Driven by X [owner|driver: regimes]' (first, under the plaque) only when
  `owner_confirmed` (the decision's own F7 cover + P2 predicate on σ's decisive episode, `EpisodeCover.induced`), else
  'Candidate driver: X (…): not shown to make these cycles'; without a driver 'Other resonances: not tested (…)' (+ 'not
  needed: a is not driven' at a kinematic leaf); the kinematic term from kin_swing_share/kin_lambda_swing_share (the
  'swing is σ's pϖ term' claim only when it spans more than the λ part); the identity σ = φ/u + Ψ/u of an induced row
  from `own_psi_def` (`PSI_DEF`); the radial overlap from ctx_orbit_overlap (E6). A failed check the verdict did not
  read is an 'Objection' bullet and leaves the grey line. No bare 'turns' ('turning points', 'revolutions'). The
  'change' slot drops 'Without the cap…' and the other reading (the arithmetic and the LIMIT row say them: `said`
  reset).
  Roles (W28 A3b, cards g04-4..7, g05-1/3/5, g06-1/2/4): `cause_line(verdict, facts)` → (role, text) from A3a's
  columns (cause_role, cause_term, owner, neighbours, resonant_argument, commensurability) for the header's second line;
  `subtitle_role` for the subtitle; the owner step says owner / driver (circulates, P) / candidate driver / neighbour
  ('not the cause, the tree goes on'; never 'drives a' for a refused owner); a neighbour's verb ('librates', 'drives a
  forced wave', 'acts') comes from the owner's measured regimes in its block; `term_text` writes '-4varpi' as '−4ϖ'.
  'both resonances at once?' (`CAPTURE_LABEL`, never 'second capture'/'2 captures'): each comparison printed the way it
  went (sweep/energy, A2's F3 secular R² and sin-x R²), the tested window from `SecondCapture.t_start_yr/t_end_yr`, the
  slow x at its level (`slow_x_line`; in the bullets only when its cause is identified or possible). Rule C: the
  argument and the term the two angles differ by (`argument_line`); a member test E′ rejected for strength is 'a frame,
  not a force' (`frame_line`, UG22; only when the scan ran).
  The steps (W28 E3, cards g10-4/g10-4b/g04-7): header 'Decided at <leaf>: … → STATUS' over both columns, each step a
  question '<n> · <question> [ids]' + 'Why:' (dropped when compact), the branch letter on the leaf, ◌ not run (a gap)
  apart from ○ not needed (`step_not_needed`: the owner at a kinematic leaf, 'n/a: no force on a') and · n/a (1:1:
  'the owner search is not applied to a 1:1'); the steps after the leaf one pale line; one footer line of docs pages.
  The confidence block (W28 E4, cards g10-4b/g03-5): rows 'name [id]' with a kind (support/limit/na, `KIND_*`); the
  LIMIT rows ▼ first ('LIMIT: earlier a-wave [O1] → LOW', `limit_heading`), a failed cap-class margin merged into its
  cap (`merge_limits`, never ✗), the full margins one line (`full_line`), n/a one grey line (`na_line`); the arithmetic
  two lines 'Support: … → HIGH' / 'Limits: … → LOW'; K1/K1e without a value print their kinematic reason.
  'Numbers behind this verdict' (W28 E10, card g10-5 B; CS11 kept by a pointer, a choice recorded in F5): MMR
  `VerdictText.emphasis` (`number_emphasis`): DECISIVE = the decision-path comparisons of the leaf's ids (and of a
  step the rule's branch passed, '5d' of 5d/6; a circulation leaf reads the shape's G ids), matched by
  `entry_rule_id` ('3 B1 …' -> B1, '2 owner R2' -> W, '2 second capture' apart, '==' flags left out); UNMEASURED = such
  an id without a value; LIMIT = the caps whose ceiling is the shown level, with the merged margin's comparison and
  column (`CAP_MARGIN_COLUMNS`: O1 -> ep_onset_ratio); CONFLICT = the objection's failed checks (`OBJECTION_COLUMNS`);
  a number the physics block prints stays there (`_outside_physics`, the line names that block).
  `combined_numbers.behind_lines` (≤ BEHIND_LINES_MAX, each line with its `pointer`); `draw_explained` draws the lines
  with `emphasis` framed over the whole width, at the numbers' font (`_fit_numbers(reserve=…)`), frame style per kind
  (`BEHIND_FRAMES`), and per FIT level `point_to_block` turns the table rows of the kept lines' columns (or of one in
  their `also`) into '↑ decisive [B1]'. FIT levels: the block keeps only its lines with a number before the compact
  explanation, and goes (its numbers back in the table) only after it.
  B-series words (W28 E19a): `wave_labels` (mmr_combined) and the episode lines name accepted runs that are no
  libration of σ (`wave`, `forced (strength)` of a 5s row: the ground, never a driver; `ripple`, dotted); 5s says
  the driver is not identified (`strength_mark_sentence`), 5x is a diagnostic (`_scale_clause`); O1 prints its three
  parts (`foreign_wave_text`), the 5o clause never 'strong a-wave'; R0 'k of n eligible windows' (`window_count_text`);
  L4 by the direct law (`direct_force_text`, `DIRECT_FORCE_NAME`, the DECISIVE line from `_direct_law_parts`, panel (e)
  `draw_direct_law` from `response.direct_curves`); step 0 G1f/F6r/G5w (`subfloor_clause`, `ripple_clause`,
  `slow_swing_text`, `orbit_period_clause`), the subtitle and header name the insufficient_time rule (`shown_rule`);
  cause lines `line_share_line` (only where L4 fails/thin or the direct law decided, never at a kinematic leaf) and
  `kinematic_suspicion_line`; `extension_line` (A-FW: what a length-limited extend serves); `forced_wave_source`
  ('a forced wave from driver X', never 'capture').
  Secular cause (W28 B10b, cards g08-8 'here and everywhere', g05-2, g05-3; B10a's columns): `secular_driver_line`
  ('Driven by: secular cycle −4ϖ (the 2ω cycle π/|ω̇|, P 18 kyr) [measured | candidate]', first under the plaque at a
  kinematic secular_cycle row; measured = period within mmr_kin_period_tol and K1e passes, or on the slow scale the
  identified level); `kinematic_cause` adds the swings (`swing_words`: kin_pred_swing_deg against the episode's
  half-amplitude, or kin_meas_swing_deg on the slow scale); a K3s row (`slow_kinematic`: slow_origin kinematic and σ's
  own period matches no secular one) reads the slow scale everywhere (`secular_reference`, `slow_kinematic_cause`: span,
  cutoff, period, cycle, swings, failed identity); `slow_scale_line` ('Slow scale: …' at its cause level, any row with
  a slow swing; not on K3s rows nor where `slow_x_line` says it). Term names and periods: `SECULAR_TERM_NAMES`,
  `secular_term_period` (slow_secular_period_yr, own_x_secular_period_yr, kin_two_omega_period_yr,
  kin_modulation_period_kyr).
  Pure; `combined_numbers.draw_explained` draws it; every printed comparison is tested against the record; margins as
  the tree read them (R-ALIAS-A: `alias_margins`), the union rule named (`union_kind`), "next check: N yr" (N8).
  `physics_sections`: block A (σ's own pendulum) and B (what makes the cycles: owner, identity, forced wave, second
  capture); `CAP_ALTERNATIVES`: every cap down to medium/low names the other reading and what decides it. A withheld
  G2a step and the direction of an a-swing change ("shrank"/"grew") are worded here.
- `plotting/verdict_vocab.py`: the words of that column, a registry without logic (W28 E2): `CAPS`/`CapText` (a WP that
  adds a cap adds its `CapText` here; `test_w20_cs11_physics`), the reason, flag, ledger and margin words, and
  `RULE_NAMES`: every printable id → short name, printed `short name [id]` (`named`), the subtitle `decided: circulation
  (C)` (`decided`); prefixes only where ids collide (`SZ.N`…, `cap:<key>`). A new id needs a name (completeness test in
  `test_verdict_text.py`) and a row in docs/libration-analysis.md#names (the same test reads it).
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
  turns, with a legend (`_draw_free_omega_drift`); the right column's "also measured" lines (e-i coupling, attribution,
  the centre's slow line, the node lock) come from `verdict_text.zlk_column_lines`, measured facts only; the cause of a
  kinematic libration is the third sentence (`verdict_text.zlk_cause`, the same columns; a fact it states leaves the
  grey line). Panel (a′) (`_draw_forcing`, `forcing_panel_drawn`): only for an attributed row or a forced-beat centre,
  under (a), nine panels then; every other row keeps eight, untouched. ω_f grey, arg z_forced − Ω orange, ϖ_j − Ω
  purple dashed, from `FreeElements.forced_vector` and the osculating node; unwrapped within a turn, else modulo 360°
  about the forced direction with ω_f inside the librating episodes only (624).
  Right column (W27, author's request): ORBIT has a (`ctx_a_med_au`/`_min`/`_max`, columns of the long-period row);
  FREQUENCIES (`long_period_frequency_lines`, both numbers modes) is a table `name  ″/yr  period  note`, kept in one
  column (`NumberLine.keep`): secular: g, s, σ from mean rates, g5/g6/s6 + the angle's modes (`in σ ±k`), the forced
  waves g − g_j; ZLK: free g, s, g − s, every fitted mode with |c_j| (◀ the attributed mode), g − g_j of the two
  largest forced modes (`kinematic` for the attributed one), their mutual beat, the node lock. Phases not shown.
  The rows are `Frequency` items (`long_period_frequencies`), shared with the text.
  Explanation (W28 E9, cards g09-5 C/g10-8/g09-3/g09-6/g09-7): `verdict_text.long_period_bullets` in the MMR slots +
  `SLOT_SCALE`: the verdict in words; the angle by its librating episodes only (`librating_episode_rows`, the predicate
  of sec_n_librating_episodes) and the deciding one; the decisive argument in physics (gate outcome: node near s_j
  with d against the resolution, the co-orbital model not applying, D10's e-i sentence on `zlk_gate_unconfirmed`;
  secular: the swing within the phase uncertainty, a centre that turned back); objections (response fail, one centre
  estimate, a ZLK libration without the e-i exchange) leave the grey line; ZLK physics (i, e, Kozai's 39.2° as a
  reference, free-ω turns |g − s|/(1/T)); the time scales (1/T, |σ̇|/(1/T) secular, the search window max(F·P_c,
  floor)…T − 2·trim·P_c, the period seen identified against the FREQUENCIES rows within 1/T, or 'unidentified', and
  `PRECESSION_SENTENCE`). A column a bullet prints (`Bullet.columns`) reads `↑ in the text` in the table
  (`point_to_text`). `sentences` stay (older tests read them).
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
- Plaque `STATUS · angle_status · CONFIDENCE`, then the amber action tag (`verdict_text.action_tag`: `→ extend to 40 kyr`,
  `→ resample to 0.5 yr`, `→ review`, `→ spot check`; none for action none; the author asked for it on 2026-11-02, Q5),
  "ends: …" last. A tag that would cross the column's edge drops to a row under the plaque (`TAG_ROW`).
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
- Right column (W28, 2026-10-31: g10-2 A, g10-3 C, g10-4/4b A, g10-5 B, g10-6 B, g10-7 C, g10-8 A, g03-5 A):
  slotted bullets for MMR and secular/ZLK; `short name [id]` with the ids of docs/libration-analysis.md; steps as
  questions with "Why"; LIMIT ▼ first (a failed cap margin never ✗), two-line arithmetic; 'Numbers behind this
  verdict' framed by line style (DECISIVE solid, LIMIT dashed, CONFLICT dash-dot, UNMEASURED dotted), a number printed
  once (`↑ decisive [id]`); one radial-overlap line over the eight planets; pendulum numbers labelled in words with
  their source; the cause line a driver, a candidate or none. Accepted runs the status does not count are grey strips
  `#id < floor` / `#id centre turns` (g02-9 A, `uncounted_tags`), never episodes.
- Frozen-potential overlays stay off by default (`dev/rejected.md`).
- The minimal policy (W29 SPEC §7, PRINCIPLES 6): (a) of both figures shows only what the status counts
  (`mmr_combined.policy_marks` from `analysis.policy`, else the verdict's id cells; `shade_policy_runs`): counted
  episodes filled (a covered edge to the record edge, the filter-trim stretch pale, `fill_counted`), a counted 1:1
  leg 'leg', candidates faint and dashed 'cand' (`fill_candidate`), no grey strip; the legend's (a) line follows
  (`legend_lines`, `long_period_legend`). Under v2 or without a policy, nothing changes. The explanation opens with
  four `SLOT_POLICY` bullets (`verdict_text.policy_block`: status · level · reason (origin / owner), `basis:` from
  `confidence_basis`, `flags:`, `v2:`); every former cap read is `has_mark` (caps or flags). The standalone MMR
  panels' re-run (`mmr_plots._rerun_analysis`) carries the stored verdict, or re-runs with the scans on.

## Known defects (open)

- `benchmark/tools/reclassify.py --figures` writes into `benchmark/data/examples/`; saved figures can be stale.
- Right-column text defects found by the W28 Codex review, waiting for the author: WK15's coverage sentence (S2),
  2002 BG's ripple counted as an episode in the prose (S3), RS2's e-i sentence reads the record's coupling instead of
  `zlk_ei_episode_*` (S4), 2001 HW7 150 kyr's "both resonances at once?" read over the owner hull (S5), the support of a
  G4c transient taken from the circulation checks (S8: XP16, OE5, YU11), AC4 "not measured" where the cause is
  `filter_too_slow` (S9).
- An owner cover that is not measured (NaN at the record edge) prints as a dash in the roles line and the right
  column, with no words of its own (cosmetic).
- The docs gallery (`docs/physics/gallery.md`) has no secular transient example since 2306 became uncertain.
