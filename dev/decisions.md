# Decisions (index by topic)

An English map of `benchmark/DECISIONS.md` (the author's log, Russian, append-only) and of the decisions recorded in
the wave reports. Each line: date, the decision, why. "Superseded" marks an entry a later one replaced; the log keeps
both. "(author)" means the author decided; "(measured)" means the decision followed a measurement he accepted.
The exact current behaviour is in the code and in `docs/physics/`; the rule IDs are indexed in `docs/libration-analysis.md` and are explained in
`resonances/resonance/analysis/mmr/CLAUDE.md`.

## Status model

- **Oct 13 (author).** `status` ∈ resonant, transient, near_separatrix, non_resonant, uncertain, invalid. `invalid`
  replaces `chaotic`: classification impossible with these parameters (alias, unphysical orbit, undefined phase), the
  cause in `reason`. transient = a resonant object, if only for a while; near_separatrix = not a resonance but
  resembling one, a status of its own.
- **Oct 13 (author).** Separate columns: `angle_status` (geometry only), `confidence` (certain / high / medium / low;
  empty for uncertain/invalid), `checks` per resonance type (pass / fail / not_measured / not_applicable), `reason`,
  `action` (none, spot_check, review, extend, resample). `not_measured` is never `fail`. `slow_circulation` dropped:
  ≥ 1 turn is circulation, < 1 is insufficient_time.
- **Oct 13 (author).** Everything can be recomputed from the saved CSV/JSON without integrating (classification and
  figures). Integer statuses are gone; `status_code` is a comparison column only.
- **Oct 13 (author).** Kinematic libration (the angle bounded by geometry, ϖ̇ or a forced oscillation of a, without
  dynamics) is the same for every type: angle libration, `non_resonant`, `kinematic_libration`; ZLK −2 belongs here.
  A drift with a proven kinematic mechanism → non_resonant; without one → uncertain + extend.
- **Oct 13 (author).** A long libration whose three turning points do not fit the record → uncertain + extend. ZLK
  gate unavailable → uncertain (−9), `gate_unavailable`.
- **Oct 15 (author).** Origin of a libration is its own column `libration_origin` (dynamic, induced, subresonance,
  kinematic, unknown) with `owner` and `origin_detail`. A subresonance member found in a multiplet gets its own
  summary row and figure.
- **Oct 15 evening (author, after an independent review).** Status by observed behaviour: accepted libration episodes
  → transient/resonant; near_separatrix only without episodes, with a responding a and energy at the separatrix. The
  separatrix physics is columns and flags and does not change the status (52356 → transient/medium plus a flag).
- **Oct 16 (author).** angle_status describes the whole record: accepted episodes followed by another regime, or a
  capture after circulation, make the angle word episodic; the ending regime goes into `final_regime`. Supersedes
  "82831 angle_status = circulation" of Oct 15.
- **Oct 17 (author).** angle_status `transient` renamed `episodic` (the angle word no longer equals the status word);
  old files are read through `verdict.angle_status_from_text`.
- **Oct 26 (author, W24 Q3).** Leaf 5u/6 (no force measured, origin unknown) writes `geometric_libration` /
  `geometric_episode` instead of `dynamic_*`; rows with a measured force keep `dynamic_*`. Old files: `dynamic_*`
  with origin unknown is read as `geometric_*` (`verdict.reason_from_text`).

## Geometry and episodes

- **Oct 13 (author).** An episode needs ≥ 3 turning points (not 4). "Libration over the whole interval" means the
  episode starts within one libration period of the start and ends within one of the end (not "90 %").
- **Oct 15 (author).** transient and near_separatrix are separated by physics (energy relative to the separatrix,
  regime change), never by a turning-point count. Every convention is a setting with its physical meaning documented.
- **Oct 15 (author).** A visible exit in the last period: libration, ≤ medium, flag `exit_suspected`. An arc > 2π is
  not part of a libration. a's response at amplitude < 45° is not_measured.
- **Oct 16 (W9).** Each regime between jumps of a is measured on its own samples; a capture after a jump (G1l) is
  transient with `final_regime` libration.
- **Oct 17 (author).** A closed cycle is mandatory for an episode (1864 3E-5S-5 is non_resonant). The scales of L4
  and R0 are physical, not tied to P_c.
- **Oct 18 (W18).** Moving centre: a run rejected for centre motion is tested in ψ = σ − c(t) (c the occupied minimum
  of the averaged potential); L4 in ψ, R0 in the full well. `separatrix_chaos` renamed
  `episodes_in_separatrix_zone` (the model zone does not measure chaos).
- **Oct 19 (author, q5).** A cycle closed only by phase return (2 TP) caps the level at medium (`one_return_cycle`).
- **Oct 20 (W20).** Edge rule E1 (variant a): an exit/entry is observed when the orbit-averaged raw angle runs ≥ 1 turn
  monotonically from the last TP. Regime cuts at a centre step or stall (K3 stall cut left to the author, still open).
- **Oct 24 (author, D1).** The union rule extends from 1:1 (G2) to every MMR (G2u): episodes without circulation
  between them cover the record together; gaps ≤ 1 × (P_l + P_r). A G2u union read as episodic is promoted to
  resonant only from a dynamic transient (6u).
- **Oct 24 (author, D14).** G4b: a slow centre whose net shift is ≥ 2 turns is circulation.
- **Oct 25 (author).** TX9's episodes were found wrongly; episode-detection tests from the author's history were asked
  for → the episode gate (W23, 79 cases after E16 was removed on Oct 26).
- **Oct 26 (W23).** Rule (a): at the centre scale (G2c) turning points are taken at the centre-turn prominence p_c that
  qualified the scale. Rules (b) and (c) of the design were not shipped (see [rejected.md](rejected.md)).
- **Oct 27 (W24, author accepted 12:35).** G2a action step: implemented with test 7 (`regime_periods`) after review;
  TX9 fails test 7 and stays one episode, uncertain. The author accepted uncertain.
- **Oct 28 (W25, author approved both; Codex's conservative variant).** Separatrix zone: (1) above the lower saddle of
  its well a cycle's loop encloses several wells and has no single-well J/J_s (nan, topology `merged_wells`, class
  `not_measured`, never "in the zone"; J_loop and T_loop/2π reported); (2) the averaged route compares actions (J/J_s
  of each turning point's loop in the averaged profile) with J_lo, not h with an h_lo mapped through V_E. Measured on
  all 719 MMR examples (frozen trees, settings 59c557f7a08b): no status changes; 45 rows change flags, caps or confidence;
  medium → high for 2015 SO2, 2019 GM1, 2020 SB, 2024 TY11 (20 kyr), 2025 PN7, 439908 (20 kyr) (cap
  `episodes_in_separatrix_zone` gone), high → medium for 82831 13M-24 (both rows: its turning points' J/J_s
  0.106–0.119 ≥ J_lo 0.1 at h 0.105–0.124 below the old edge 0.125). The zone classes are not columns only: L2, Z, caps
  and leaf 5o read them.
- **Oct 28 (W25, Codex final review #3).** The averaged route compared J/J_s(t) (the profile's units at each time) with
  J_lo = J0/<J_s> (the record's mean separatrix action; `numeric_zone`: J/J_s(t0) = J0/√m0). Action at fixed h scales
  as √strength, so at m = 0.25 a local 0.6 is 0.3. The pre-W25 h-route had the same mismatch (local h against h(J_lo)).
  Now each ratio is carried by J_s(t)/<J_s> of the averaged well itself (absolute J_s = √(3k²μn²a) × the profile's,
  <J_s> its time mean on MMR_SEPARATRIX_AVGPOT_TP_MAX even times, the basin of the nearest turning point's swing
  middle); no mean (a failed profile) → the pendulum's classes. Not the zone model's √m: J_lo and the classes would
  then share a model the averaged route exists to replace (2004 TK14: depth ×100). Measured on 719 MMR examples
  against test-benchmark: no status or confidence changes; 23 rows change cycle classes, 4 change flags/thin margins:
  2000 SF8 8M-9 (3 cycles deep → zone, its episode in the well's strong phase, J_s(t)/<J_s> 1.6–3.05: L2 thin),
  2003 UG22 10E-17 and 2004 TK14 31M-34 0.5 yr (zone → deep: flag libration_in_separatrix_zone and thin L2 gone),
  87555 4N-5 (deep → zone: zone_edge_cycle gone); 3233 30M-53 loses the route (a failed profile on the grid). 82831
  13M-24 (both rows, J/<J_s> ≤ 0.47 ≥ J_lo 0.1) and the 1:1 rows (no averaged route) are unchanged.

## Response, force and strength (MMR)

- **Oct 13 (author).** Gallardo's resonance strength is not used for statuses. Confirmed by measurement on Oct 26
  (W24): his normalisation is arbitrary for this purpose.
- **Oct 16 (W9).** L4 (force law ȧ = K sin(σ − φ₀)) decided by R² without a φ₀ scatter test; 7343 1M+1J-2 resonant/high.
- **Oct 16 (W9).** Strength F counts only swings inside the well of ⟨R⟩ around the theoretical centre.
- **Oct 16 (W10/W11).** Rule R: the response is measured on every librating episode and a witness is chosen; a
  measured L3 failure disqualifies a witness, not_measured does not. The tree and the text read the same response.
- **Oct 17 (W12).** `MMR_RESPONSE_CUTOFF_CYCLES` 0.05 → 0.1 (derivative noise averaged on a physical cycle scale).
- **Oct 17 (author).** The owner search tests a candidate by the multiplet law (coherent coefficients of
  φ_o + j(ϖ − ϖ_P)); L4 contaminated by a forced line of the angle is not_measured and does not lead to dynamic.
- **Oct 17 (author).** The "L4 and R0 not measured" branch splits by physics: support / measured objection / nothing.
- **Oct 20 (W20, author Oct 19).** F6 forced waves; K2 kinematic drift of the slow centre; F8 baseline N from the
  deciding scale; subresonance members σ ± 2mω in one invariable frame; `certain` for far circulations.
- **Oct 23 (independent reviews).** A failed O1 = uncertain (a real pendulum capture gives O1 ≈ 0.5); the earlier
  recommendation to the contrary was wrong.
- **Oct 24 (author, D3–D15).** R-C4 (C4 not_measured when no force law explains ȧ), R-WJ (1:1 on one cycle with a
  failed energy R² ≤ medium), R-L3K/R-FK (L3 n/a and F read differently for kinematic verdicts), R-O1-OWNER,
  R-KDRIFT (K2b), R-ALIAS-A, owner route 'identity', R-DIV (B ≤ n·min(P, P_cyc)/2π), R-F (5u/6 cap
  `strength_scale_only` ≤ medium when the e^q scale passes).
- **Oct 24 (author, D2).** R-IS and R-SP-IS (lift caps for an implausible strength) dropped after he changed the
  2001 PG14 label.
- **Oct 26 (W23).** CS12 multi-harmonic force law: lifts L4, never lowers it; never for 1:1 or a null test.

## Co-orbitals (1:1)

- **Oct 16 (W4).** Judged by the averaged direct + indirect force law with the planet's own mass (K/K_th = 1), no owner
  search; W5 added the retrograde 1:1 law.
- **Oct 19 (author, q2).** No `transient_unexplained` cap for 1:1 (a co-orbital capture is a close encounter; the
  pendulum separatrix model does not apply).
- **Oct 20 (W20 CS5).** Quasi-satellite pass G1q: a second pass at the period floor and cutoff taken from the averaged
  potential.
- **Oct 19 (author, q3) → superseded Oct 27.** 2003 TX9 3E+2J-10 labelled resonant/medium (double well, capture into
  the second well after Mars encounters); after CS12 and G2a measured no per-regime physics, the author accepted
  uncertain.
- **Oct 17 (author).** "All Jupiter Trojans uncertain" was a bug; fixed in W16.

## Sampling and aliasing

- **Oct 13 (author).** An alias whose expected rate from mean a is unambiguously far from the resonance →
  non_resonant, `alias`; otherwise invalid/alias + resample. The finder warns before integrating.
- **Oct 15 (author).** `OUTPUT_STEP=auto` is an option; the dry-run always warns.
- **Oct 17 (W15).** The alias gate reads the λ part of the angle, not the full angle across low-e perihelion flips.
- **Oct 20 (W20 round D).** `short_period_aliased` stays on every record whose step folds short periods; the record
  never lifts it (`MMR_SHORT_PERIOD_CAP_RATIO` = 0). Measured on 47 fine re-integrations: folded lines are negligible
  in σ and matter through a.
- **Oct 23 (author, Q1d).** No cap on a geometric circulation without forced-wave runs. The cap is made actionable:
  `smp_fold_free_step_yr` and action `resample`.
- **Oct 24 (author, D11).** R-ALIAS-A: on a folding step with F direct passing and a's short-period scatter above
  `MMR_SHORT_PERIOD_A_RATIO`, a failed L4/L3 is not_measured.

## Confidence

- **Oct 13 (author).** certain = never needs a manual check; high = all checks pass, at least one thin, ≥ 97–98 %;
  medium ≥ 90 % (spot check); low = no promise.
- **Oct 15 (author).** high allows any number of thin margins and no failure; nothing physical measured → ≤ medium.
- **Oct 15 → superseded Oct 16.** "certain/high need ≥ 100 labels per cell": dropped; the level comes from margins and
  caps and `certain` is shown; calibration cells are reported only (`cell_*`, `resonances/data/confidence_cells.json`).
- **Oct 19 (author, q1).** Every cap down to medium or low names the other reading and what decides it
  (`verdict_text.CAP_ALTERNATIVES`).

## Secular and ZLK

- **August (measured).** The free-omega gate decides the ZLK status; e–i coherence is a flag. −2 (measured kinematic)
  and −9 (no verdict) are never merged. Calibrated at 500 kyr.
- **Oct 12.** Secular angles from the planets' mode longitudes and the invariable plane, reduced by default.
- **Oct 13.** The secular study uses the 21 literature resonances; no broad order-≤4 scans for status (the finder
  still scans the 75 catalogue formulas by divisor).
- **Oct 16 (W4).** Secular/ZLK moved to analysis v2; the secular cutoff includes the angle's longest forced wave;
  secular e/i response and e–i coherence are diagnostics only. The secular status is decided in
  `analysis.secular.decide.decide_status` (no hook after the analysis).
- **Oct 17 (author).** 162474 over 1.5 Myr is an ordinary transient (island 270° → 90°); no `island_switch` flag.
- **Oct 27 (author, W24 Q5).** The secular/ZLK author verdicts are gated in `make test-benchmark` (14/14); 189865 and
  477492 lose the "invalid" expectation (it came only from SABA failing on planet-crossers).
- **Oct 28 (author, W25 items 7, 8).** An uncertain secular/ZLK verdict names its next check (`analysis/secular/
  extend.py`, MMR's `next_check` with N by cause: baseline 1.5 P + 2 P_c; gate short_window FREE_GATE_MIN_CYCLES
  P_cyc; planetary_residual 1.25 × 1296000/d_min, review beyond 100 Myr); undefined_phase, unavailable and
  unusable_series now ask for review instead of extend (no length cures them). No status changes. The dry-run has
  advisory secular/ZLK rows (baseline 500 kyr / 9.4 kyr, the ZLK gate's 3 quadrupole times). Panel (f) draws the
  analysis' own response fit (it refitted at the fine P_c while printing the decisive scale's R²).
- **Oct 28 (author, W25 items 3, 4, 5, 10 and the ZLK double-scale studies).** ZLK ρ_max is local: max over the
  record of rollmed(|z_free|, W)/rollmed(|z_forced|, W), W = max(beat window, `ZLK_RHO_WINDOW_CYCLES` P_cyc); the old
  median ratio stays a column (`zlk_gate_rho_global`). Co-orbital guard, variant (a): only for planets whose modes are
  not fitted (Mercury to Mars), outcome unavailable (`coorbital_unfitted_forcing`), never not_applicable; Trojans
  covered. c1, c2, h_quad in the invariable plane. e–i coupling at the decisive period for every ZLK row
  (`zlk_ei_coupling`: zlk / forced_beat / none / not_measured; `confirmed_ei` = zlk). Attribution of a kinematic
  libration (`zlk_forcing_*`), the centre's slow line (`zlk_centre_*`, 591986: g6−g5), and a scale guard: a
  centre-scale libration at a forced beat with ρ_max < 1 never decides. n_cycles is T/P_cyc of the osculating ω (the
  figure said "free ω"). No status changed on the benchmark. Pending (author): a node-lock flag (Ω − ϖ5, ≥ 3 Myr).
- W26 (author, 2026-10-29: "state the presumed cause of an induced ZLK libration; overlay g5/g6 in one more panel only
  for such rows"): the cause sentence (`verdict_text.zlk_cause`, measured columns only), panel (a′) (ω_f, arg z_forced
  − Ω, ϖ_j − Ω; drawn only for an attributed row or a forced-beat centre, others keep seven panels), the flag
  `zlk_node_lock` (approved 2026-10-28; resolved only with a full cycle of Ω − ϖ_j, `ZLK_NODE_LOCK_MIN_CYCLES`),
  the centre origin `forced_direction`, and the attribution of an episodic row inside its episodes (scope
  `episodes`, 624). Columns only: no status changes.

## Figures

- **Oct 15 (author).** One combined figure per row (layout V1, "everything on one screen"); every decision number on
  the right, also in the CSV.
- **Oct 16–17 (author).** σ̇ axis symlog σ̇/2ω₀ with the zero line highlighted; the plaque reads
  `STATUS · angle_status · CONFIDENCE`, "ends: …" last.
- **Oct 17 (author).** 33128 6N-7 kept as the example where FAIR fails.
- **Oct 23 (author).** a_res on top of panel (d); angle labels in [0°, 360°); header `episodes about X° (not this
  resonance)` for induced/kinematic rows.
- **Oct 24–25 (author).** a_res style B (width 1.3, alpha 0.8, halo); a* alpha 0.8 (`AXIS_LEVEL_ALPHA`).
- **Oct 27 (author, W24 Q6/Q8).** `PLOTS=combined` by default; output layout `confidence/status` by default; a command
  to regenerate the documentation figures is required.

## Benchmark and process

- **Oct 13 (author).** A label belongs to its exact record (integrator, span, step, epoch); labels from other runs are
  hints only.
- **Oct 15 (author).** All configuration in `SimulationConfig`/`AnalysisSettings` (no separate parameter objects, no
  legacy code, readable names); benchmark code in the main repository; tests written by a context-free subagent.
- **Oct 15 (author).** Examples may be added freely, with honest label sources; reruns under 10 min need no approval,
  runs of an hour or more wait for the final stage.
- **Oct 16.** Agreement is measured against all author verdicts, not only the recent labels.
- **Oct 24 (author).** No new iteration until every disagreement is discussed; an iteration starts on his explicit go.
- **Oct 24 (author, D13).** R-AV: an old `author_verdict` in `candidates.csv` is not compared where a newer author
  label exists for the same record.
- **Oct 27 (author, W24 Q4).** Clones: interface B (orbit solution object + ensemble), 100 clones by default, weak
  NEAs start at the fit epoch (`.eq0`), non-gravitational parameters only flagged, the finder takes the cloud's
  candidates.
- **Oct 27 (author, W24 Q3).** MPC source: query by elements as for AstDyS; a separate module that can become its own
  package; a body at another epoch is integrated to the run date by default.
- **Oct 28 (author + main session, W25).** Clones: a run with clones resumes as a whole (one checkpoint
  `clones_state/`, cleared with simulation_state.json only after ordinary part and clones succeed: the old early clear
  re-ran every ordinary batch after a stop in the clones); the identity covers what was run, the starts are kept and
  compared bit for bit, a mismatch refuses. Q1: a finished one-simulation ordinary part is skipped (verdicts from
  summary.csv, coverage checked), never moved into another grouping. Q2: `ensemble-<parent>-results.npz` kept. Q3:
  restore reclassifies the nominal only and rewrites clones.csv's nominal columns; clones keep stored verdicts. No
  triage (docs/clones.md "Limits"). MPC: `MPC_LOOKUP=file_then_api` (default `file`) takes a missing identifier's
  nominal from get-orb without a covariance; names are documented by the MPC API, so no name mapping was added.
- **Oct 29 (Claude, W26 known gaps; not yet seen by the author).** Restore with recompute also rewrites the nominal's
  summary.csv rows (and mmr_numbers/episodes/candidates) in the run's folder, from the same verdict as clones.csv's
  nominal columns (`clone_store._rewrite_nominal_rows`): one writer of the verdict (`Simulation._finalise_verdict`),
  two projections that cannot disagree on disk. Clones are still not reclassified (Q3 kept): with `SAVE` other than
  `all` the saved clone series are a status-selected subset, so a recompute of those alone would bias the shares.
