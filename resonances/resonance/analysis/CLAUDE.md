# resonance/analysis — libration analysis v2

The only classifier, wired into `Simulation.identify_librations`. It gives the verdict of every type (MMR, secular,
ZLK) on shared geometry. Pure: no Body/Simulation imports, and no bare numbers in decision code. Formulas and the status
tree: `docs/physics/` (one home per topic; the rule IDs with their settings and modules: `docs/libration-analysis.md`; an example figure per status in `docs/physics/gallery.md`, figures in `docs/img/gallery/`, rebuilt by `make docs-figures`). MMR steps and rule IDs: `mmr/CLAUDE.md`. Secular and ZLK science:
`resonances/secular/CLAUDE.md`. Reasons, rejected variants and traps: `dev/`.

## Shared modules

- `verdict`: string `Status` (resonant, transient, near_separatrix, non_resonant, uncertain, invalid) /
  `AngleStatus`/`Reason`, `libration_origin` + `origin_detail` + the roles of the other resonance (`owner`,
  `neighbours`, `resonant_argument`, `cause_role`, `cause_term`, `commensurability`; MMR only, `mmr/roles.py`), `final_regime` (how the record ends),
  `status_code` (a column for comparison with old runs; `verdict_from_status_code` only converts old integer summary
  files). `angle_status` describes the whole record: librating episodes followed by another regime, or a capture after
  one, are `episodic`. Old files: `angle_status_from_text` (angle `transient` → `episodic`, 2026-10-17), `reason_from_text`
  (`dynamic_*` with origin unknown → `geometric_*`, 2026-10-26). `not_measured` is never `fail`.
- `ledger`: every decisive comparison with its margin U. It feeds the CSV columns, the right column of the figure,
  the verdict text and the confidence; the decision path and the printed numbers must come from it, not be recomputed.
- `settings`: `AnalysisSettings` holds every threshold. Fields = `SimulationConfig` attributes = lower-case `.env.dist`
  keys. Shared names `LIBRATION_*`/`FILTER_*`/`SEPARATRIX_*`/`CONFIDENCE_*`; type prefixes `MMR_`/`SECULAR_`/`ZLK_`. A
  new threshold is a field + an `.env.dist` key with a comment (a convention says so, with its physical meaning) + a
  `docs/config.md` line. `settings_hash` goes into simulation.json and summary.csv. Trap: `settings_from_config()`
  without a source returns the dataclass defaults, so environment variables do not reach the MMR reclassification
  (`benchmark/tools/reclassify.py`); the secular/ZLK one builds a `Simulation` and does read the configuration.
- `filtering` (`resonance/analysis/filtering.py`): Butterworth in second-order sections, forward-backward (never (b, a) form). W23 memo: `record_memo` /
  `memoised` memoise `band_pass` and `response.smoothed_series` per `analyse_mmr` record (2 entries, keyed by object
  identity plus scalars, read-only arrays; never unscoped: a session-wide memo raised one worker from 1280 to 2084 MB).
  `LOW_MEMORY_ALLOCATOR` (`config.low_memory_allocator`, `worker_allocator`: `MallocLargeCache=0` for spawned workers
  on macOS).
- `pendulum` (`resonance/analysis/pendulum.py`): the potential, energies and periods. W24 review: `extrema` (exact, roots of z^N V'(z)),
  `loop_through`/`period_through` (turning points bracketed between extrema; at the separatrix → inf),
  `well_holding`, `cycle_period`: R0's period law reads each cycle in the well that holds its fastest sample (one
  well: the old law; two harmonics: every loop of one energy has one period in either well, so only a cycle below its
  well's bottom changes). For a multi-harmonic V the action J and J_s come from quadratures in the selected well; the
  half-width 2√(V'')/|c| and the cosine action hold for one harmonic only. W25: a libration cycle's J/J_s and its slope
  are read in the well it occupies (`occupied_well`, `CycleEnergy.well_rad`, nan = the deepest; `WellActions` caches
  each well's separatrix and J_s), not the deepest: with two harmonics J_deep(E) − J_shallow(E) is a constant, so the
  deepest well's ratio of a shallow-well cycle is too large. One-well records keep the deepest path exactly.
  W25 (topology): above its well's lower saddle in a V of several wells a cycle's loop encloses both wells: no
  single-well J/J_s (nan, `libration_topology` merged_wells/rotation, class `not_measured`, never "in the zone";
  `_libration_action` never regenerates a clipped 1); its own J_loop (`loop_action_through`) and T_loop/2π are columns.
  A one-well V keeps J/J_s = 1 above its top (separatrix). The averaged route compares actions: J/J_s of each turning
  point's loop in the averaged profile itself (`averaged_potential.action_in_well`) against J_lo, never an h mapped
  through V_E (`h_in_barrier_units` is gone). J_lo is J0/<J_s> (the record's mean separatrix action), so each local
  ratio is carried by J_s(t)/<J_s> of the averaged well, the scale 3k²μn²a included (`zone.tp_energies`,
  `centre_track.mean_separatrix_action`; Codex W25 #3: at a quarter of the mean strength 0.6 local is 0.3); no mean
  (a failed profile on the record's grid) → the pendulum's classes stay. The pendulum route does the same with
  √m_cycle. The zone classes reach L2, Z, caps and leaf 5o: not columns only.
  W28 B3 (g07-1, g08-1), R0 with a: `fit_potential_from_force` has no free constant (a moving frame subtracts its own
  s'', `frame_acceleration`; `SEPARATRIX_VF_FREE_CONSTANT` restores the old fit for comparison); a window's V_E takes
  its N by held-out prediction (`fit_window_potential`, `SEPARATRIX_HOLDOUT_GAIN`; σ-only R0 keeps the in-sample
  gain); `with_slow_check`/`agree_fraction` count eligible windows only (none: not_measured). A free constant on the
  visited arc is worse than either alone (BG41 0.25, many passing rows fail): never combine them.
- `policy` (W29; docs/physics/status-tree.md#minimal-policy; the `v2` switch was removed in W34): the last step
  of `analyse_mmr` (outermost call only: an inner pass returns the raw v2 analysis; derived rows first, then the
  catalogue row, then the multiplet status), `analyse_secular` and `analyse_zlk` (after its columns). It re-reads the
  finished v2 analysis: status from the geometry (`LEAVES`/`HANDLERS` map every `rec_rule` of `mmr.decide`; a new
  leaf needs a mapping, `test_policy_mapping` fails otherwise; the single 1:1 legs are read only on G4c and rule C,
  `LEG_RULES`: beside an accepted episode a leg is no evidence; rule 5s's strength marks count again where the veto is
  not upheld, also as Episode rows for N: `_countable`; a status the policy moves re-reads the header's orbit type), non_resonant only by a measured veto (owner, kinematic,
  strength two-body, ZLK kinematic or free e < forced e; thin -> medium + `veto_thin`; a secular/ZLK insufficient_time
  angle that ends in circulation by an end travel below 1 + `CONFIDENCE_THIN_BAND` turns -> non_resonant at most
  medium, `_travel_thin`, input `end_travel`, W36 F5: the travel is a range with the swing in it), the level `policy.level` from
  N/M/V/S/F (N < `CONFIDENCE_MEDIUM_CYCLES` low; W33 F: an MMR row with none of L4, R0, F passing is at most
  `CONFIDENCE_GEOMETRY_ONLY_MAX`, the identity is no force; S = L1 full and no `filter_sensitive`; a subresonance parent's N is
  `(member)`; no caps: every cap becomes a flag, `confidence_raw = confidence`, `cell_*` blank, `review_reasons` without
  caps), the action through one extend predicate (`extend_clause`; none at certain, `_certain_clause`; clause C, author Q5: a `veto_needs_length` row whose
`pol_veto_need_yr` fits `MMR_NEXT_CHECK_SPAN_FACTOR` x T extends to it, beyond it review; an alias with C5 and C2 full
is certain, Q18), `PolicyResult` (counted and candidate episodes)
  on `MMRAnalysis.policy`/`LongPeriodAnalysis.policy` and episodes.csv `counted`/`candidate`; its comparisons go to
  the decision path as `policy:*`; the W28 verdict stays in `v2_verdict`. Source columns for the figure (2026-11-04,
  no status or level reads them): `level_inputs` (`PolicyResult.level_inputs`, `pol_level_inputs`: each input with
  its ceiling; the level is their lowest, tested against `level`/`_final_level` on grids; never change `level` to fit
  it), `pol_decided_by` (`decided_by`: 'tree' or the handler's `DECIDED_*`, `;not_upheld:<leaf>`), `pol_objection_check`
  (`OBJECTION_*`, non-empty iff V is ambiguous: `_mmr_veto_reading` names what it reads). A word the policy changes
  clears the tree's `angle_subtype` (`_write`).
  The G2a guard (`_analyse_mmr_record`) still chooses between cut and whole on the raw v2 (status, confidence).
- `confidence` (the W28 tree's level, in `v2_verdict`; the policy overrides it): the level from margins and caps; `certain` is shown. certain needs every applicable margin full; high
  allows thin or unmeasured evidence but no failed support, and (W28 q-2 b, MMR own dynamics) at least one physical
  support passing with its full margin (else cap `no_full_physics_support`); one failed support → medium, two → low; nothing physical
  measured and explicit caps limit it; uncertain/invalid have no level. The calibration cell's author-label counts are
  only reported (`cell_*`, `resonances/data/confidence_cells.json`, rebuilt by `benchmark/tools/calibrate.py`); they
  never cap a level. Every cap down to medium or low names its alternative (`verdict_text.CAP_ALTERNATIVES`).
- `rules` (W28, cards s-4, g10-3): `Rule` (a StrEnum named by meaning), `RULES` (`RuleInfo`: doc id, `pending` until
  the owner work package writes it; title; since), `rules_since`, `fire`, `ordered`. Every W28 rule name is
  pre-registered; a work package only calls `fire` where its rule acts. Column `rules_fired` (contract below).
- `orbit_type`: the header's tadpole/horseshoe/… text, descriptive only (`centre_moves` is shared with the text).
  Subtype `stationary` means the cycle bands overlap, not zero centre motion (2003 TX9 keeps it while its centre
  moves; the text says so).

## Geometry (`resonance/analysis/geometry.py`)

- Each regime between jumps of a is measured on its own samples, with its own turning-point prominence; smoothing
  stays inside the regime. Turning points and centre per run. Every run of provisional TPs (at the type's floor) gets
  its own prominence too (`local_thresholds`): MMR `mmr.analyse.regime_prominence`, secular/ZLK
  `secular.preprocess.run_prominence` (W36 F2, capped at the record's: a run's fast part can hold its own libration
  where interleaved waves hide its period from `libration_image`). Secular/ZLK only: a stretch keeps the record's
  prominence when its own would lose a librating episode the record's finds (`secular.analyse.measure_fine`: the
  run spans the slow approach to a libration, 455327 g-s-g5+s6); the MMR path has no such guard.
- An episode must close at least one full cycle (`closure_ratios`), or one cycle by its phase return between two TPs
  (`phase_return`: at the mid-level or a band of levels, `LIBRATION_RETURN_LEVELS`, `LIBRATION_RETURN_BAND_LEVELS`);
  a return-closed cycle caps the level at medium and cannot cross a confirmed jump. ≥ 3 TPs and a closed cycle, never a
  TP count, separate an episode from noise.
- Runs break at a half-amplitude (or a midpoint jump) of `TurningPoints.break_half_amplitude_rad`, stamped from
  `FilteredAngle.run_break_half_amplitude_rad` in `_find_episodes`/`_measure`: pi for MMR and secular, half a turn for a
  ZLK omega (`ZLK_RUN_BREAK_HALF_AMPLITUDE_DEG`, a 2 omega island; W28 D3, rule `zlk_half_turn_cut`). It only splits
  runs: no rule may read which island (90/270) an episode is in (island_switch, centre mod 180: `dev/rejected.md`).
- An exit/entry is observed, not covered, when the orbit-averaged raw angle (MMR) or σ_f itself (secular/ZLK,
  `FilteredAngle.edge_arc_filtered`, W36 F1: no orbit to average) runs ≥ 1 turn monotonically from the last TP (edge
  arc); secular/ZLK then skip the centre scale (`secular.decide.edge_observed`) and set `final_regime` circulation.
  For an MMR |σ_f − c| > π and the arc's crossing read the local centre at that edge (`Episode.edge_centre_rad`: the edge half-cycle's midpoint, or the well's c(t) at the edge TP in ψ; C9, W28 g03-2),
  never the episode's mean (2016 JA: HS swing then tadpole, the mean put TP0 206° away) nor the first full cycle's
  midpoint (still 190°); `_edge_reading` fires `local_centre_entry` when the mean reading differs. Secular/ZLK edges
  keep the mean centre (not measured there). A run whose centre moves is cut at a centre step or a stall of the slow centre (regime cuts).
  W28 B1, two joins (episode boundaries only, no status): rule q (`_frame_join`, `LIBRATION_FRAME_JOIN`) joins runs
  broken by a half-cycle over the top in σ that stays below π in ψ (track valid, joined run accepted; a slow
  `_split_regular` piece holding the join is classified in ψ too); rule s (`_stall_join`, `LIBRATION_STALL_JOIN`)
  joins a stall cut's leftover that has a cycle of the stall's period (`_same_period`: any cycle with two leftover
  TPs, not the median, since ripple adds TPs: 385266) and is accepted with it. Both fire into
  `Geometry.rules_fired` through the `fired` collector of `_find_episodes`. W28 C16 (q-5 i), rule `regime_cut_mark`
  (`_mark_regime_cuts` at the end of `_regime_pieces`, `LIBRATION_REGIME_CUT_MARKS`): two ACCEPTED neighbouring
  pieces of a standing step/stall cut join when their facing full cycles share a band and a period
  (`cycles_continue`); the joined run is `regime_change` (L6 not measured) only when it fails for `centre_moves`
  alone, and the cut goes to `Episode.marks` (`centre_step@t0:t1`, `stall@t0:t1`). Never join an accepted|rejected
  pair, and never drop the band test (VA14 60 kyr E17 and 2002 BG 20 kyr move: C15's naive reading).
- Secular/ZLK only, rule turning_centre (W36 F3, `FilteredAngle.turning_centre`, `SECULAR_TURNING_CENTRE`,
  `ZLK_TURNING_CENTRE`; `accept_turning_centre`, first in `_find_episodes`, before split_regular/trims): a run rejected
  only for `centre_moves` is `modulated` (L6 not measured) when it closes a cycle, its TPs span < a turn
  (`within_one_turn`), its cycle centre turns back ≥ 1 time (p_c) within < π, and one cycle's band holds every centre
  (`swing_covers_centres`). The last condition is what keeps a small wave on a slower motion out (363/2306/518
  g−g6+s−s6: 4–20° waves on 42–144° centre travel); without it those rows went uncertain → transient/certain. Never
  drop it for "rule A" (every max above every min: rejected).
- Secular only, slow waves (W36 F4, `secular.decide.slow_waves` → `Geometry.slow_wave_ids`, in `excluded_ids`,
  eligibility `SLOW_WAVE`; `SECULAR_SLOW_WAVES`): an accepted episode not covering both edges is a wave on a slower
  motion when σ_f between the centre turns bracketing it spans R < 2π, A_max ≤ `SECULAR_SLOW_SWING_RIPPLE_RATIO` ×
  R/2 and the swing carries it (the slow track's rate inside ≥ `SECULAR_SLOW_SWING_CARRY_RATIO` × its fastest rate
  beside, `stall_reading`: a parabolic turn reads ≥ 1/8, a plateau the filter's leak; Codex's two stationary
  librations joined by a transition were discarded without it), or R ≥ 2π and the slow track (low-pass at 3 P_ep)
  beside it moves ≥ A_max in as long as the episode while inside it keeps ≥ `SECULAR_STALL_RATE_RATIO` of that rate
  (a stall, not a capture). An arm the record cuts below `SECULAR_STALL_ARM_MIN_FRACTION` of the span is not read. Applied at every scale
  (`analyse._set_slow_waves`), edges re-read (`refresh_edges`). The "moves ≥ A_max beside" condition is what keeps the
  pieces of one libration (F2 off) from reading as stalls. Not on ZLK (591986/162474 would be hit). The centre scale
  of a secular angle tries the centre-turns window (MMR G2c second scale) after W_c (`SECULAR_CENTRE_TURNS_SCALE`).
- MMR 1:1 only, rule `coorbital_edge_turn` (W28 C4, `_edge_raw_turns`, model `FilteredAngle.edge_turn` =
  `mmr.coorbital_geometry.CoorbitalEdgeTurn`, wired in `mmr.analyse._measure`): a reversal of the raw angle averaged
  over `MMR_COORBITAL_EDGE_TURN_ORBITS` orbits in the filter's edge trim, after the last (before the first) TP, is
  added as a TP only when it closes a cycle of a run that was no episode (never extends an accepted one: that would
  move its coverage); prominence ≥ `MMR_COORBITAL_EDGE_TURN_MIN_DEG` on both sides, a reflection site (planet/L3), a
  crosses a_P, |ln out/in| ≤ `MMR_COORBITAL_TURN_ENERGY_TOL` (sides shrunk to the record left). Columns
  `geo_edge_raw_tp_t`, `geo_edge_raw_read`, episodes `edge_raw_tp`. The centre scale drops the model; the x2 filter
  guard keeps it (its trim is twice as wide, so an edge episode it cannot see makes `filter_sensitive`: BZ147, DJ1).
- MMR 1:1 only, rule `coorbital_well_cut` (W28 C7, cards g03-1 A, g03-9 A; `FilteredAngle.coorbital_wells` =
  `mmr.coorbital_geometry.CoorbitalWells`, `MMR_COORBITAL_WELL_EPISODES`): an episode ends only at a separatrix. A run
  the band test rejects is accepted whole (`_accept_in_wells`, before split_regular/trims) when every full cycle's swing
  holds a minimum of the run window's averaged potential to within the run's TP prominence (`cycle_wells`; XO3's
  transitional cycles miss by 4-8° at 11°, BB2's drift past L5 by 7-25° at 4°) and its cycle centres stay within one
  turn; two ACCEPTED neighbours join across a half-cycle below pi that does not circulate (`_join_in_wells`; a rejected
  neighbour never: that is C8's residence, joining it moved TY11/277617/SZ2/PN/BT8/AP1), or across a swing of one
  compound well whose facing centres differ by < π (`_compound_swing`, W31: 439908 20 kyr); no regime cut for 1:1 with the
  model. `coorbital_marks` (end of `finish_geometry`, needs the jumps) writes `Episode.wells` ('qs>compound>horseshoe')
  and `Episode.marks` (`well_change@t0:t1`, `amplitude_change@t0:t1`, `kick@t`; episodes.csv); a run of several wells
  gets shape REGIME_CHANGE (a horseshoe band holds a QS band: 164207), `FloorWell.holds` reads any of its cycle centres
  (the floor of its fastest well; G1q keeps the outside episodes it contains, `analyse._kept_in_regime_change`), L5
  not_applicable (`decide`). Trap: without the G1q/L5 parts 164207 and 469219 fell to transient/medium.
- MMR 1:1 only, rule `coorbital_residence` (W28 C8, cards g03-3 C, g03-8 A, g03-9 A, g02-3 A;
  `MMR_COORBITAL_RESIDENCE`; `Residence`, `SubRegime`, `AveragedLegs`, `residence_of`, `coorbital_residences` after
  `coorbital_marks` in `finish_geometry` and again (residence only) in `refresh_edges`): every accepted run's core is its
  TP span (`Episode.core_t0_yr/core_t1_yr`, what coverage, edges, margins and the status read); `subregimes` cut it at
  the marks (each with its own 1:1 law from `WellsModel.law` for >= 2 pieces); the inferred `residence` walks the legs
  of the 3-orbit averaged raw angle (`WellsModel.legs`, reversals of `MMR_COORBITAL_EDGE_TURN_MIN_DEG`) out from the
  core until a leg sweeps `LIBRATION_EDGE_ARC_REV` turns ('circulation') or the record ends ('edge'); an observed
  exit/entry bounds it at the core. |da|/da_max (`zone_max`) is reported only: TY11 20 kyr's accepted core starts at
  a turn of zone 2.2 (as a bound it would cut the g03-8 A residence). Nothing reads it (test: decision equal with the
  switch off). `well_sequence` feeds the 1:1 orbit type and `geo_orbit_type_sequence`.
- MMR only, W28 C6 (leaves G4c/G4h read them; `MMR_COORBITAL_SINGLE_LEG`, `MMR_SINGLE_HALF_LIBRATION`):
  `coorbital_single_legs` (end of `finish_geometry`, after the residences) stores `Geometry.single_legs` (`SingleLeg`:
  the legs of `WellsModel.single_legs` outside every accepted run, period ≥ the floor less the thin band, with
  `LegTurn` out/in and zone and the local law; columns `geo_leg_n`, `geo_leg_read`); `half_librations` stores the
  2-TP half-swings of another MMR between passages with their one-harmonic law (`HalfLibration`,
  `geo_half_libration_read`; needs a finite `identity_coefficient`, so secular/ZLK never get one).
  `short_runs(runs, coorbital)` adds over_the_top 2-TP 1:1 runs whose half-swings cross at most one of 0 and pi
  (`crosses_planet_and_l3`).
- MMR only, W28 C14 (rule `end_arc_extend`, `MMR_RECORD_EDGE_STRETCH`): `edge_stretches` (end of `finish_geometry`)
  stores `Geometry.edge_stretches` (`EdgeStretch`, one per window edge, the start mirrored in time): the edge pass of
  σ_f from the slow centre's nearest turn, `slowdown` (≥ 1 revolution, then `duration_yr` without another, against the
  pass's longest revolution `last_rev_yr`) or `reflection` (< 1 revolution after a turn, ≥ 1 beyond it); columns
  `geo_start_edge_*`, `geo_end_edge_*`. Measurement only; `mmr.decide.record_edges` judges it. Cost trap: the
  revolutions are found by bisection on the running maximum; an argmax per revolution was O(N × revolutions) and
  took 100–260 s on circulating high-order rows (27103 14M+27, 18118 29M+49; FIX-perf 2026-11-01).
- Episodes without circulation between them cover the record together (`regime_change`, `covering_union`): 1:1 rule
  G2, any other MMR G2u; every gap bounded: each σ_f sample inside a neighbour's own band c ± A_max or the band carried
  linearly between their facing centres (± max A_max), up to `LIBRATION_UNION_BAND_MARGIN_DEG` (`gap_band_overshoot`;
  never the low-pass centre track, which follows a slow excursion; the carried band alone splits 524522 1V-1), whatever the gap's length (A7, cancels D1b); edges reached within a period.
- After any cut (a jump of a or an action step) the record's entry is read on its first run (`Geometry.record_entry`,
  columns `record_entry_*`, flag `entry_suspected`; `Geometry.entry_suspected()` covers both).
- Derived views (`Geometry.derived`, held in `derived_views`) are emptied by `forget_derived` wherever runs or their
  edges change.
- **G2a action step (W24)**: `action_steps`/`ActionStep`. An accepted σ-frame run (not 1:1, no jump of a inside)
  whose half-amplitudes step between separated bands, with the a swing and the cycle energy changing the same way
  within one half-cycle (`LIBRATION_ACTION_STEP_*`), is cut by a confirmed `Jump(action_step=True)` whose zone is the
  transitional half-cycle (`Jump.zone`; regimes, rule 10, G1e/G1l, `JumpGuard` read it).
  - Test 7 `regime_periods` (made robust after the W24 review): the two sides' periods differ by >
    `LIBRATION_ACTION_STEP_PERIOD_TOL` (else `isochronous`: one forcing frequency explains both), and at every cutoff
    of `LIBRATION_ACTION_STEP_CUTOFFS` (0.075/0.1/0.15 × the shorter period, same number of wells) V_E of the larger
    swings (each of their disjoint cycles held out; one cycle: in sample, provisional) holds every disjoint cycle of
    both sides in a loop of its own energy around the well of its fastest sample (`pendulum.loop_through`, no loop
    vetoes), with its period (`_PERIOD_TOL`) and its turning points (`_TP_TOL` × half-swing), each side's energy
    spread ≤ `_ENERGY_DRIFT` × the step; `shape_failed` = 'subtest@factor'; V_F reported only.
  - Provisional (≤ medium, cap `action_step_provisional`) with one energy cycle on a side or one larger-side cycle.
  - `analyse._action_step_guard`: a row that cut a step of its own (`ActionStep.key`, `_ROW_KEY`) and leaves a check
    failed on the whole record unmeasured, R0 included (no exemption), with another verdict → the whole record's
    analysis is kept, `geo_action_step_failed` = withheld. Columns `geo_action_step_*`, `jump_action_step`.
  - 2003 TX9 3E+2J-10: tests 1–6 pass, test 7 fails (late cycles: law 1.61, turning points 0.74 off), not cut,
    uncertain; the author accepted uncertain (2026-10-27).

## Contracts

- Analysis arrays and Body arrays can share storage: never mutate them in place (planet memmaps and memo arrays are
  read-only). Copy instead (e.g. the 1:1 R0 through `response.with_coorbital_potential`).
- An analysis' arrays are dropped right after its verdict when the row will not be plotted (`drop_arrays` freezes the
  CSV rows first).
- Tests of decision code build real result objects (`tests/resonances/analysis/builders.py`), not stubs; decision
  code reads dataclass fields directly.
- Rule flags (`rules.py`): a rule fires where its predicate holds and it acts on an intermediate result (a cut, a
  merge, a cap removed or added, an outcome), even when the verdict stays the same; a rule only evaluated does not
  fire. Geometry-level rules go into `Geometry.rules_fired`; other result objects carry frozensets of their own
  (secular `AngleDecision`/`StatusDecision.rules_fired`). The decide steps merge everything into
  `Verdict.rules_fired` (`mmr.decide.rules_fired`, `secular.decide.rules_fired`: deduplicated, registry order); a
  new carrier is added to that merge call by the work package that adds it. A second pass that replaces the verdict
  (MMR G2c, G1q; the secular centre scale) keeps the first pass's geometry rules. `to_flat_dict`/`from_flat_dict`
  round-trip the column; names outside the registry survive, after the registered ones. Not a setting
  (settings_hash unchanged), no decision reads it, no figure prints it (`combined_numbers.SPECIAL_COLUMNS`). The W28
  review folder's `same_rule/` is built from it, so firing too broadly inflates it.
- Secular and ZLK: `analysis/secular/` (`analyse_secular`: cutoff from the forced waves, centre scale,
  `decide_status` with the free-omega gate's outcome, `drift` only for a monotone slow centre (W28 D6: one that
  turned back, `mmr.decide.interior_turns` ≥ 1, is `insufficient_time`/`few_cycles`), margin P thin below
  1 + `CONFIDENCE_THIN_BAND` turns, `extend.recommend`: the action and `extend_to_years` of an uncertain verdict, `slow`: the centre's slow line, a column) and `analysis/zlk/` (`analyse_zlk`; `diagnostics`: the record's e-i coupling, attribution and the centre-scale guard `centre_guard`, columns only; `episode_ei`: the deciding episode's e-i, which decides an episodic row without a gate verdict, W28 D10; the guard runs only with the gate enabled, a split and no unfitted co-orbital, and rejects only a measured forced beat that explains the centre). Gate disabled ≠ gate
  unavailable: disabled keeps the geometric status with `check_zlk_free_omega` not_measured.

## Do not retry here

Jump candidates from cycle energy / peak distance / steepest a; deleting small TP pairs before P_cyc; rule A (every
max above every min); TP counts as a physical criterion; a session-wide memo; G2a cuts that hide a failed check. Why:
`dev/rejected.md`.
