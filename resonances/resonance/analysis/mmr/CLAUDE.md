# resonance/analysis/mmr — the MMR analysis

`analyse_mmr` runs the steps below on one record; `decide.py`'s module docstring is the full tree (G0–G6, the 5x/6x
leaves). Each rule ID here names the date the author accepted it (`benchmark/DECISIONS.md`) and the setting that
holds its convention. Reasons and rejected alternatives: `dev/decisions.md`, `dev/rejected.md`.

## Steps (one line each)

context → sampling → preprocess → geometry (shared) → response → coorbital / quasi_satellite → kinematics, strength,
centre_track → zone, multiplet, subresonances → owner, owner_potential, owner_identity → decide → analyse (scales,
guards) → io.

- `response`: L4, L3, B1 pooled and on every librating episode; rule R: the tree reads a witness episode's response
  (`deciding_response`); a measured L3 failure disqualifies a witness, not_measured does not.
- `coorbital`: the 1:1 force law from the averaged direct + indirect potential (K/K_th = 1 is the planet's own mass;
  retrograde 1:1 too). The owner search, the e^q strength and `transient_unexplained` do not apply to 1:1; one
  predicate decides "is 1:1" everywhere (`preprocess.law_planet`).
- `quasi_satellite` (rule G1q): a second pass at the 1:1 quasi-satellite floor from that potential.
  W28 C5 trigger (cards g01-2, g02-8; rule `qs_trigger`, `MMR_COORBITAL_QS_TRIGGER_PERIODS`): the pass is tried when the
  orbit-averaged angle turns in the well (`turn_values`, reversal ≥ `MMR_TP_PROMINENCE_FLOOR_DEG`) during a stay of
  ≥ 1 loop of the averaged potential through the stay's lowest/highest turn (`coorbital_geometry.loop_period`; the
  longest P_QS along the stay when nan); acceptance unchanged; ledger entry `QS_TURN` is third (verdict_text reads the
  first two). Trap: the small-amplitude P_QS overstates a low-e loop (2002 PN: 32.5 yr vs a 17.8-yr loop with
  K/K_th 1.00). 2021 XA1 (Venus, e 0.41): P_QS ≈ tadpole period, the pass repeats the first one (not a G1q defect).
- `coorbital_geometry` (C2, W28, cards g01-8, g02-1/2/6, g03-1/9; pure helpers, no status, no threshold): the profile
  R~(θ) of a window's mean elements (`profile_at`, J2000 planet without saved elements; tables through `table_for`
  and the `record_tables` budget), `turn_site` (qs / tadpole_L4 / tadpole_L5 below the well's barrier, planet / l3
  above it), `well_of` (the minima a swing holds: horseshoe, compound), `loop_da`, `da_max` (C1's arc definition:
  the top before the planet or L3 ahead; C6's zone tolerance is calibrated on it), `loop_period` (T_hs(θ_t), tadpole,
  QS; inf at a separatrix, nan when no loop joins the turns), `turn_conservation` (out/in), `TurnPhysics`,
  `leg_columns` (`cpl_*`, wired by C3) and `local_law` (`coorbital_law` on one window at its own cutoff). Trap: R~
  is interpolated as 1/R~ (a spline of R~ rings 1.2 % from the one-node planet peak of a near-circular orbit).
- **W28 C7 1:1 boundaries** (`coorbital_geometry.CoorbitalWells`/`coorbital_wells`, the geometry's `WellsModel`;
  rule `coorbital_well_cut`, `MMR_COORBITAL_WELL_EPISODES`; cards g03-1 A, g03-9 A, g11-5): a 1:1 episode ends only at
  a separatrix; wells of a swing on the run window's profile (`well_of`), levels R~ at turning points. The geometry
  side (acceptance in wells, joins, marks) is in `analysis/CLAUDE.md`. Here: `analyse._measure` wires the model,
  `analyse._kept_in_regime_change` lets G1q count the first pass's horseshoe inside a joined QS episode as kept,
  `decide` sets L5 not_applicable for an episode of several wells. Moves (259 co-orbitals on/off): see dev/decisions.md.
- **W28 C8 core / residence / sub-regimes** (`CoorbitalWells.legs`, `zone_max`, `law`, `reversal_indices`; models
  `residence_columns`, `described_coorbital_columns` (summary `geo_orbit_type_sequence`, `geo_residence_*` in
  `SUMMARY_FIXED_COLUMNS`); `orbit_type._coorbital_sequence` reads the sub-regimes' wells first; rule
  `coorbital_residence`; geometry side in `analysis/CLAUDE.md`): data only, no status moves (259 co-orbitals on/off,
  scratch impl-C8). The sub-regime law uses `FilteredAngle.angle_raw` (angle_unwrapped), not angle_clean: a
  description, never a decision. 2009 BE81 20 kyr's sub-regime laws pass except the horseshoe's energy (R² 0.02); the
  author's low (g02-3 A) is in KNOWN_DISAGREEMENTS (medium stays: no physical criterion gives low).
- **W28 C3 1:1 profile floor** (`coorbital_geometry.CoorbitalRunFloor`, `fast_reference_yr`,
  `fastest_horseshoe_period`, `averaged_turns`; `geometry.RunFloor`/`FilteredAngle.run_floor`/`Geometry.run_floors`;
  `decide.period_floor`, `near_episode_floor`, `run_floor_acts`; rule `coorbital_floor_profile`,
  `MMR_COORBITAL_PROFILE_FLOOR`, `MMR_COORBITAL_MIN_TURN_HILL`, `MMR_COORBITAL_TURN_SIDE_WINDOW`; cards g02-1 A, g01-6
  A, g01-8 A, q-3 A): every 1:1 run of >= 2 TPs is read on the orbit-averaged angle (the filter rounds a planet turn
  off: YN107 39/319 deg filtered, 2.0/358.3 averaged); when every turn reflects on a top (planet, l3) its floor is
  0.5 x the loop through its turns (median over <= 9 pairs: a 20-kyr QS run has 5000 turns, 163 s before the cap),
  provided the local law on the run +- one half-cycle has K/K_th > 0; else the tadpole floor (a QS-well episode keeps
  G1q's). The floor may be above the tadpole floor (a cycle faster than half its own loop is no loop of this
  potential; 53 of the 102 reflection runs of the 259 co-orbitals, none rejected). The fast residual's reference is min(tadpole floor, 0.5 x the fastest horseshoe loop turning 3 r_H
  out): measured, T_hs(3 r_H) alone is above the tadpole floor everywhere (Jupiter 235 vs 74 yr), so it would raise
  every 1:1 prominence. The near-episode needs the floor x (1 - CONFIDENCE_THIN_BAND), for every MMR. The cutoff is
  unchanged. `preprocess.profile_period_floor` has the same 1:1 branch for one episode. Moves: see dev/decisions.md.
- **W28 C6 single legs, leaves G4c / G4h** (`coorbital_geometry.CoorbitalWells.single_legs`, `leg_topology`;
  `geometry.SingleLeg`/`LegTurn`/`coorbital_single_legs`, `HalfLibration`/`half_librations`/`harmonic_half_test`;
  `decide._g4c`, `leg_outcome`, `deciding_leg`, `leg_scatters`, `structural_near_episode`, `_g4h`,
  `deciding_half_libration`, `single_leg_columns`; rules `coorbital_leg`, `coorbital_near_episode`,
  `single_half_libration`; reasons `one_coorbital_leg`, `one_half_libration` (action review); cap `one_coorbital_leg`;
  `MMR_COORBITAL_SINGLE_LEG`, `MMR_COORBITAL_LEG_WINDOW_FACTOR`, `_LEG_CUTOFF_ORBITS`, `_LEG_CUTOFF_FRACTION`,
  `MMR_COORBITAL_TURN_ZONE_MAX`, `MMR_COORBITAL_LEG_FORCE_GATE` (off: q-4 A literally), `MMR_SINGLE_HALF_LIBRATION`,
  `MMR_HALF_LIBRATION_WINDOW_FACTOR`; cards g01-1/4/5/8, g02-4..7, q-4 A, q-5 i): a 1:1 circulation without an episode
  is read on the legs of its 3-orbit averaged angle (C1's measurement): clean + local law (K > 0) + energy + both turns
  reflect → transient (≤ medium); one discrepancy → uncertain; a clean edge leg whose turn reflects → extend; both
  turns kicks → circulation + near_episode; strongest leg wins; legs overlapping an accepted run are never read (C1
  rule 7: re-reading 3-TP episodes demoted 16 transients, XW2 among them). Not 1:1: a 2-TP half-swing between passages
  with its own one-harmonic law (C15's test) → uncertain (42946 43S-6 the only one, at the record end: extend).
  `short_runs(coorbital=True)` also takes over_the_top 2-TP runs whose half-swings cross at most one of 0 and pi (the
  structural near-episode, g01-8 A). Moves (259 co-orbitals): 31 rows, see dev/decisions.md; the rule against the
  author's per-object answers: 17 of 36 differ (W28 review page, d06/d07). Traps: a kick right after a turn
  barely changes |da| (the potential, not the kick, sets the speed there); the force gate moves 6 rows (138175, WZ12,
  JX1, CL1, VH2, TD); 2022 AP1's legs overlap its accepted (below-floor) run, so it stays non_resonant.
- `strength`: two-body = curvature of the directly averaged potential at the occupied centre (`averaged_potential`,
  method `direct_averaging`; only swings inside that well count), or the resonant harmonic in extended precision
  (`direct_harmonic`) when double precision resolves no well; `DIRECT_METHODS` holds these two (a measurement).
  Three-body and 1:1 = `eq_scale`, an order estimate μμe^q used only for median e < `MMR_STRENGTH_E_MAX` (0.3) and no
  crossing of the term's planets; implausible only when the residual ratio / B² of the small-divisor envelope still
  exceeds the implausibility factor (B ≤ n·min(P, P_cyc)/2π since W22, R-DIV; rule 5s, else 5s* uncertain).
- `centre_track`: the moving minimum c(t) of that potential along the record; a run whose centre moves is tested in
  ψ = σ − c(t) (subtype `moving_centre`, its L4 fitted in ψ; R0's period law in the full well). At a σ turning point
  ψ̇ = −ċ, so R0 carries ċ²/2.
- `subresonances` (`resonance/analysis/mmr/subresonances.py`, W20): members σ ± 2mω with the nodes, nodal members built in one (invariable) frame, the ranked
  table `sub_top` (E, W, N, A/W; by E and W only under `sub_strength_caveat`; `centre_pass_rad` per member in the scan
  window, card g06-8b A).
- **W28 A6a cheap numbers + T7/T8** (`subresonances.cheap_subresonances`, `theory_numbers`, `cheap_list`,
  `cheap_window`, `diagnostic_columns`; `models.SubresonanceCheap` = `MMRAnalysis.subresonance_cheap`; card g06-8a A,
  rules `subresonance_t7`/`subresonance_t8`, `MMR_SUBRESONANCE_CENTRE_MOVE_RATIO`): every first-pass prograde non-1:1
  row gets the theory (two-body: planar shares, Rouché of the catalogue member, |P|/max|P| in the gaps and at the edges
  of accepted episodes, the travel of π − arg P + j_cat β per episode) and the cheap list (planar members only, E′ and
  S1 on the centre of the longest accepted episode, the regimes before the last jump included). T7 (travel >
  ratio × half-amplitude) is a scan trigger; T8 (an accepted run with overlapping bands whose centre shift exceeds
  ratio × half-amplitude, or pending origin_unknown) alone runs `run_subresonances(diagnostic=True)`, whose result
  goes only to `sub_diag_*` (never to `decide`: `analyse._run_subresonances` returns None for it). Cost: 0.3–7 % of a
  record (145480 1 Myr 0.09 of 1.4 s). Moves: none on 12218, 31935, UG22 40 kyr, ND, EH16 (T7 rides on T1/T6 there;
  ND/EH16 diagnostics only). Trap: 12218's gap |P|/max 9e-5; UG22's flattest frames are +5+2 then +6+1 (factors
  8e-9, 2e-10), not +6+1 alone.
- **W28 A6b selection** (`subresonances.evaluate`, `_veto`, `_choose`, `TheoryCentres`, `theory_centres`,
  `episode_windows`, `scan_windows`, `_pick`; `models.SUBRESONANCE_CHOICE_*`, `SubresonanceResult.choice`/
  `episode_arguments`; `decide.composite_regime_detail`; `analyse._episode_arguments`; card g06-8c A, rule
  `subresonance_theory_select`; no new setting): every veto (step 1, E′, L, A, B; E on the window) before the choice,
  so a member B rejects hands it to the next; the choice is the flattest theoretical centre c_j = π − arg P + jβ over
  the window when the measured centre follows (D's RMS ≤ `MMR_SUBRESONANCE_THEORY_RMS_DEG`; the same RMS for every
  planar member), ties within `MMR_SUBRESONANCE_TIE_FRACTION` by W, W alone as the fallback (three-body, crossing,
  no follow; nodal members by W against the theory's winner). One window per accepted episode (its regular stretch,
  its regime's centre); the decisive episode (`status_basis`) gives the row's member, the others only episodes.csv
  `resonant_argument` (when the verdict names one) and, where their window cannot decide, `episode_undecided` (its
  flags cap the row like the decisive window's: 78085 5J-4S-1, whose decisive window passes, keeps
  `subresonance_undetermined` from its first episode, the old single window); no accepted episode → the old
  longest-TP-run window. 'composite': nothing admissible, the theory follows, range(β) >
  `MMR_SUBRESONANCE_CENTRE_MOVE_RATIO` × half-amplitude (the members are told apart) and the two flattest
  theoretical member centres tie within `MMR_SUBRESONANCE_TIE_FRACTION` (equal weights do) → check pass, an own
  libration gets origin_detail composite_multiplet and `resonant_argument` 'composite' (`roles`). The theory is now
  computed in every two-body window (also without a kept member). Measured (184 scanned rows): no status or accepted
  member moved; the theory's flattest = the accepted member on 12218, 27011, 31935, 65366, 67495 (82831 crosses Mars:
  W); 27011's first episode has D RMS 26° (no theory choice there); 12218's second episode (76–100 kyr) names none:
  β moves 63° < the half-amplitude 94°, step 1 keeps no member. Traps: "theory travel > half-amplitude" as the
  composite test flags clean 3J-2/3J-1 librations (1162: 66° over 100 kyr against 51°); "flattest ≥ ½ range(β)" flags
  2003 SL215 8J-3 (61° against 162° for the next): a dominant member is told by its neighbours, not by its own drift.
- `owner`: catalogue longitudes when a planet's λ is not saved (confidence ≤ medium); `owner_potential`: the owner's
  averaged force predicts ȧ (the potential route); `owner_identity` (route 'identity', W22): a three-body σ needing
  ≥ 10³ × μμe^q follows a two-body φ that librates over its episode with a passing direct strength; not run without
  saved longitudes. A candidate owner is tested by the multiplet law (coherent coefficients), never by a bare frequency
  coincidence.
- `decide`: rule C, an accepted member of the multiplet gives the catalogue row its status (the member of the decisive
  episode's window, A6b).
- `roles` (A3a, W28, cards g04-6, g06-1..3, g05-3): splits the decision's other key (`StatusDecision.owner`) by role
  after the status: `owner` only an owner or driver that covers the deciding episode; an owner F7/P2 refused →
  `neighbours` (blocks of its regimes before/during/after the own episodes, by midpoint); a rule C member →
  `resonant_argument`; kinematic → `cause_term` = the secular part of σ (`-4varpi`); `commensurability`. Changes no
  status; `StatusDecision.owner` keeps the raw key (decision tests read it).

## Rule glossary (`resonance/analysis/mmr/decide.py` unless stated)

- **F6** `forced_wave_episodes`: an accepted run on a lattice line with a failed identity (or a wrong-sign identity)
  is a forced wave, no episode for the status (`Geometry.forced_waves`, edges re-read by `geometry.refresh_edges`); the
  angle word keeps its loops (`forced_wave_word`). Author 2026-10-19.
- **F7 per episode + P2** (A1, W28, cards g04-0..g04-3; `owner.py` `owner_intervals`, `cover_episodes`,
  `induced_split`, `decide.owner_in_episode`): the owner's time is only its measured regimes (no slip bridge); its
  cover ≥ `MMR_OWNER_COVER_MIN` is checked on every accepted episode (`own_episode_covers`), and a librating owner
  must turn Ψ/u less than `MMR_OWNER_PSI_TURNS_PER_CYCLE_MAX` per σ cycle (P2, flag `owner_psi_too_fast`). Covered +
  P2 = induced. All induced → rule 2; none → flag owner_outside_episode, the tree goes on; mixed → a second pass
  with the induced runs out of the geometry (`Geometry.induced_ids`, `analyse._without_episodes`, `_InducedPass`: no
  G1q/G2c again), episodes.csv names the owner on the induced runs. The second capture's test window stays the hull
  of the owner intervals meeting σ's episodes (`capture_spans`, 78085 3.4–81.6 kyr), never owner time. The mixed path
  is not on the benchmark (synthetic test only).
- **Second capture, route F3** (A2, W28, cards g05-6, g05-2; `owner.second_capture`, `slow_x`, `well_selection`,
  `kinematics.secular_forcing_fit`): besides the sweep-and-energy route, the cap `second_capture_unresolved` goes when
  ẍ (x = Ψ/u low-passed at 3 P) follows the harmonics of 2ω or ϖ − g_j t (j = 5..8, mode rates, phase-free) with
  R² ≥ `MMR_OWNER_R2_MIN` and not [1, sin x, cos x] (R² < `MMR_OWNER_CAPTURE_SELF_R2_MAX`); rule
  `second_capture_f3`. Only 78085 5J-4S-1 moves (0.98 vs 0.001); HW7 stays (self 0.63). A secular term is read only
  when `MMR_OWNER_X_MIN_CYCLES` of its cycles fit in the span: a longer one fits any trend (survey 2026-10-31: R² 0.99
  'causes' at 150 kyr on 20-kyr spans before the limit). `own_x_*` (slow x and its cause) and F2's Rayleigh
  lengths (`own_capture_R_*`) are columns only. `secular_forcing_fit` is shared with B10a.
- **Both resonances at once, confirmed** (A2b, W28, card g05-8 B; `owner.joint_outcome`, `x_cycles`,
  `secular_periods`, `secular_offset`; `decide.joint_capture`, `joint_capture_origin`, `_own_read`; rule
  `both_resonances_confirmed`; no new setting): x bounded (sweep < `MMR_OWNER_CAPTURE_SWEEP_DEG`), energy R² and
  self R² ≥ `MMR_OWNER_R2_MIN`, P_x (median full period of its turns at `LIBRATION_CENTRE_TP_PROMINENCE_DEG`) off
  every secular period and harmonic by > `MMR_KIN_PERIOD_TOL`, ≥ `LIBRATION_CENTRE_MIN_ESTIMATES` cycles → `_foreign`
  steps aside, the tree's own reading gets origin joint / joint_capture / owner = the owner's key at the end of
  `decide_status` (only when it claims the libration; `own_dynamics_claimed` counts joint like unknown); one cycle →
  uncertain/baseline_needed, origin joint, rule 2j. Every block must confirm. Measured 2026-11-01 (45 owner rows):
  none confirmed. Traps: a monotonic x passes both R² trivially (single-valued ẋ(x): TW135 1.00/0.99, no turn), so
  the cycles carry the test; a small 2ω-forced sinusoid passes both fits like a pendulum, only the period refuses it;
  independent fast librations near the 3 P cut leak into ẍ (self R² drops: safe direction).
- **The rule in words** (A9, W28, card g05-7; `docs/physics/response.md`, 'Two librations are not two resonances',
  refs NM98 §2 pp. 3030–3031, Gallardo, Coito & Badano 2016 p. 89): membership is the term that moves a. A second
  librating angle, a residual after the owner, a slow centre swing at a secular period (ẍ a function of 2ω, not of x)
  or a ϖ-dominated angle is never by itself a second resonance or a veto; only the confirmed joint outcome says
  'both at once', and an owner never excludes an extra restoring term (hence unresolved caps, not exclusions).
- **K2** `slow_centre_is_kinematic`: episodes that ripple a slow centre following λ + p ϖ while every force law fails
  (`force_law_fails`, halves and multiplet) → non_resonant/kinematic_drift, graded as a drift. **K2b**
  `_kinematic_centre_moves`: a kinematic libration whose centre moves > `MMR_KIN_DRIFT_SHIFT_RATIO` × its amplitude
  goes the same way (R-KDRIFT, D10). **G4b** `centre_net_circulates`: net shift ≥ `LIBRATION_CIRCULATION_NET_REV` rev
  is circulation, also K2's word (R-NET, D14).
- **K1e** (A5, W28, card g05-3 B; `kinematics.episode_kinematics`, `decide.kinematic_episode_margin`,
  `physics_margins_of`): on the decisive episode σ and pϖ are low-passed at its response cutoff, cut and detrended;
  `kin_k1_episode` = their correlation, `kin_swing_share`/`kin_lambda_swing_share` the half-range shares. The support
  margin K1e (kinematic_libration only) is full when K1e ≥ `MMR_KIN_CORR_MIN`, B1 < `MMR_KIN_B1_CLEAR` (0.01, a
  convention) and the period offset ≤ `MMR_KIN_PERIOD_TOL`, else not_measured (a part that fails argues for no other
  status); it is a physics margin of a kinematic verdict, so a full one lifts `all_physics_not_measured` (rule
  `kinematic_high`). Moves: 285637, 481085 medium → high; 41588 (B1 0.095) stays. `kin_check_reason`: why the
  slow-angle correlation (K1) has no number (E4 prints it).
- **N8 / baseline** `baseline_period`: the estimate N (`base_extend_to_years`) from the scale that decides
  (`base_period_source`: tp_spacing, arc_slow, two_omega, stall_rate, …); `extend_years`/`next_check`: an
  insufficient_time verdict asks for the next check round_up(max(2T, min(N, 4T))) in `extend_to_years`
  (`MMR_EXTEND_MIN_SPAN_FACTOR`, `MMR_NEXT_CHECK_SPAN_FACTOR`); a budget, not a proof. No physical 1:1 exit estimate.
- **W22 rules**:
  - `promote_union`: a G2u union read as episodic becomes resonant only from a dynamic transient (6u; `union_read`);
    the rule becomes the leaf's + 'u' (`rec_rule` 5d/6u, 5c/6u; the prefix keeps the branch for `_l4`).
  - R-ALIAS-A `short_period_unmeasured`: on a folding step with F direct passing and a's short-period scatter >
    `MMR_SHORT_PERIOD_A_RATIO`, a failed L4/L3 is not_measured for the tree and the level.
  - R-F `_geometry_cap` (5u/6): reasons `geometric_libration`/`geometric_episode` since 2026-10-26 (old files'
    `dynamic_*` with origin unknown read so by `verdict.reason_from_text`); `strength_scale_only` ≤ medium when the e^q
    scale passes, `geometry_only` ≤ low otherwise.
  - R-WJ `one_cycle_energy_caps`: 1:1 on one cycle with the witness's energy R² failed → ≤ medium.
  - R-O1-OWNER `owner_librates_before`: O1 n/a where the owner librates over its background.
  - R-C4 `force_law_rejected`: C4 not_measured when no force law explains ȧ.
  - R-FK `kinematic_strength_margin` (F of a kinematic verdict; `kinematic_f_applies`, the text reads it too); R-L3K:
    L3 n/a for kinematic reasons.
  - A failed O1 is uncertain (a real pendulum capture gives O1 ≈ 0.5).
- **W28 B4 O1 narrowed** (`decide.foreign_wave`, `response._onset_wave`/`OnsetWave`, cards g03-4 A = g08-3 A): a ratio
  ≥ `MMR_O1_RATIO_FAIL` fails O1 (cap `onset_failed`) only when the background a-wave is coherent at P (O1c R²), in phase
  with the episode's first 2P and NOT explained by the episode's own law (O1l: K, φ0 fixed, no refit; the 1:1 law via
  `coorbital.law_model`); otherwise thin. Leaf 5o keeps the ratio alone (`onset_ratio_fails`): narrowed it would move
  the VA14 60 kyr and TG2 100 kyr labels. Trap: 17346 (coherent, in phase, own law 0.69 of 0.79) loses the cap, against
  the s-1 control label (in KNOWN_DISAGREEMENTS).
- **W28 B6b ripple on an arc** (`decide.ripple_waves`, `_g6_ripple`, card g08-4 A, rule `ripple_on_arc`,
  `MMR_RIPPLE_ON_ARC`): before G2 an accepted run < 45° whose own identity is measured and fails, on a slow centre that
  turns back 1–2 times within a turn (`unfinished_slow_swing`), goes to `Geometry.forced_waves` (route
  `unconfirmed_ripple`; not F6: `_f6_waves` and the word skip it); G6 counts the TPs outside it, none → G5a/G6 on the
  slow centre. Never a record-covering run or 1:1. Trap: "≥ 1 turn" alone would take 2003 UC22 16M-27 (43-TP 35°
  libration, identity failing through aliasing on 1 yr, 4 turns, author resonant/high). Moves: YG1 20 kyr →
  insufficient_time/arc extend 50 kyr; 2002 BG 1E-10S-1 kinematic_drift → implausible_strength (still non_resonant);
  523683 ripple set aside (B6c finishes it).
- **W28 B6c G5w slow swing in the well** (`slow_swing.py`, `decide._g5w`, `baseline_cycles`; cards g07-5/g07-6 A,
  rule `slow_swing_in_well`, `MMR_SLOW_SWING_IN_WELL`, `MMR_EXTEND_SLOW_CYCLES`): two interior turns of the slow
  centre within a turn, the slow identity passing and the half-swing's a swing and period 2 (t2 − t1) within
  √`MMR_STRENGTH_CHECK_RATIO` of the directly averaged pendulum at its amplitude (T0_th F(A), 2 ω0 sin(A/2)/|c|) →
  insufficient_time/arc, after G5a (instead of drift) and before the ripple/few_cycles G6; the extension asks for 3
  slow periods. Unmeasured (three-body, 1:1, one turn) changes nothing: 101703/142694 stay drift. Moves: 523683
  extend 400 kyr; 385458 9N-17 100 kyr few_cycles 200 kyr → arc 300 kyr.
- **W28 B5 G1f sub-floor pass** (`analyse._subfloor_pass`, `subfloor_eligible`, `subfloor_conditions`; card g08-2 B,
  rule `subfloor_pass`, `MMR_SUBFLOOR_PASS`): the 150-yr floor stays; a two-body non-1:1 angle without a libration
  (after G1q, before G2c, first pass only) is analysed again at max(T0_th/√`MMR_STRENGTH_CHECK_RATIO`, 2 dt ×
  `FILTER_MIN_LIBRATION_CUTOFFS`), kept only when its decisive cycle is below the floor, F passes by direct averaging,
  the theoretical centre lies inside every observed swing (offset_max ≤ smallest half-amplitude) and L4 passes.
  T0_th comes from `preprocess.profile_period_floor` (the shared hook; the 1:1 branch is C3's, below; three-body None):
  `strength.theory_period` on `strength.record_swings` (σ at P_c/2 cut in snapshot parts) — the at-floor strength
  has no episode to read (VC2: no_required_strength), so the theory is taken without one. The cap
  `short_period_aliased` and its fold-free step keep the angle's own floor (`SubfloorPass.own_floor_yr`). A binding
  realisable bound gives P_c = 2 dt: no pass. Moves: 2001 VC2 16E-17 → resonant/libration/medium; of 230 non-libration
  two-body rows only 1627 11E-28, VC2, WK15, XA were eligible (WK15, XA stay A7's: their passes find no cycle below the
  floor).
- **W28 B8 direct averaged-force law** (`response.direct_force_law`, `period_large`, `judge_direct`,
  `with_direct_force`; `strength.at_large_period`; `decide.taken_direct_law`, `theory_only_caps`; card g07-2 A, rule
  `direct_force_law`, `MMR_DIRECT_FORCE_*`): after the strength, where the harmonic L4 fails or is thin on a two-body,
  non-1:1 angle with a direct strength (direct_harmonic: not measured), not on a record whose step folds short periods
  into a (R-ALIAS-A's predicate), the term's Lagrange a_dot (real planet mass, each window's elements,
  `owner_potential.derivative_profile`) and the observed a_dot are low-passed alike at 0.35 P_large and fitted with one
  scale per window; taken with two disjoint windows, a null, R² ≥ `MMR_FORCE_R2_FAIL`, every s in [1/√r, √r], R² −
  null ≥ `MMR_FORCE_NULL_GAIN`. Lifts L4 (episodes, before-jump pool, record), never lowers it; cap
  `force_law_theory_only` (≤ medium); F reads mu_eff of the large turns. P_large: the period of the turns of σ at
  0.35 P whose TP count survives the doubled cutoff (else the doubled cutoff's period is tried). Moves: 59358 8N-13 →
  transient/medium (R² 0.75, s 1.14–1.29, F 5.9 → 1.6); 2004 PY27 2E-7 L4 lifted, level unchanged. Trap: GQ107
  40M-57's E01 passes (R² 0.70/0.78, s 0.73) on two overlapping windows only: the disjoint-window guard keeps it
  uncertain (the author's label); a longer E01 would pass.
- **W28 B8b line shares** (`forced_lines.line_share`, `response.line_share_on`/`with_line_share`, card g07-3; columns
  `rsp_line_*` only): on the decisive window (`analyse.decisive_window`), ȧ at the base response cutoff is fitted with
  sin/cos of every S1 line in [0.5 cutoff, T] at once (lines closer than 1/T are one regressor, all named; each
  frequency refined within ±0.5/T: the body's n on the episode and the lines' secular terms are known only to that),
  plus P_orb and the residual's strongest peak (`P_lib` or `unidentified`). Lines at the measured n of the window, not
  n_c. Only the libration's fundamental is named, never P/h (dense on a short episode: GQ107 107 yr ≈ P/11). GQ107
  E01: 7M-10 = 57 P_M, share 0.93; 59358: 3N-5 + 5N-8 = 13 P_N, 0.44 (lattice 0.83). Text: E19a.
- **W28 B9 length-limited extend and leaf 5x** (`decide.length_limited_extension`, `EXTEND_LEAVES`,
  `force_windows_short`, `last_episode_at_end`, `force_window_extension`, `cycles_extension`; `analyse._length_limited_action`;
  cards g08-5 B, g08-6 A, g06-6 A, g06-7 A; rules `extend_length_limited`, `eq_scale_not_objection`): an uncertain row
  left at review whose leaf 5e/5g/5x is decided while L4 has < `MMR_FORCE_MIN_WINDOWS` windows, or 5n, gets extend when
  the last librating episode reaches the record end: A-FW = round_up(t_start + `MMR_EXTEND_FORCE_WINDOWS` × 3.5 P) of
  that episode, 5n the N8 next check of its missing TPs (its decisive episode must be that last one); an L4 missing
  only for a swing < 45° keeps review. Leaf 5x (before 5n/5u): a failed eq_scale is no objection (5o keeps O1 and
  the direct F). Moves (action only): TX9 ×2 30 kyr, ND 60 kyr, EH16 60 kyr, 79653 200 kyr (5o → 5x), VA14 20 kyr
  40 kyr, 23240 200 kyr, 385533 400 kyr. Trap: the kinematic point count is no cause (the slow-angle test does not
  judge a librating angle).
- **W28 C14 the record's edges** (`decide.record_edges`, `RecordEdges`, `edge_sources`, `edge_stretch_holds`;
  `geometry.EdgeStretch`/`edge_stretches`; `analyse` after `full_support_caps`; card g01-7 A, rule `end_arc_extend`,
  `MMR_RECORD_EDGE_STRETCH`, `MMR_END_ARC_SLOWDOWN_FACTOR` (2, convention), `MMR_END_ARC_MIN_FLOOR_PERIODS` (1, read
  as 2D ≥ floor)): an end slowdown after circulation (D > 2 × the pass's longest revolution, no centre turn, no
  librating episode in it) → final_regime arc/stall, flag `extend_time` forward, cap `record_ends_unresolved` (≤
  medium) only on non_resonant/circulation; a start slowdown, one edge reflection, a G4c LEG_EXTEND edge leg and a G4h
  edge half-libration → the flag only (`Verdict.extend_direction` forward/backward/both, `extend_to_years` =
  round_up(2T) unless action extend). No status or level from the flag (scratch ledger). Traps: the last revolution as
  the reference made 2017 XF62 20 kyr (certain) a slowdown at 1.005 ×; the factor 1 is the bound of a steady
  circulation only.
- **W28 B7 G2c scale ladder** (`analyse._at_ladder_scale`, `_libration_period`, `_slow_scale_cap`;
  `geometry.slow_scale_rung`, `fast_period_yr`, `slow_extrema_shift`, `ladder_cutoffs`; card g07-4 A, rule
  `scale_ladder`, `MMR_SCALE_LADDER`, `MMR_SCALE_TP_TIME_TOL`): after G2c (and its retry at P / 2, which now reads the
  phase-return period when P_cyc is nan) found a slow libration whose W_c window holds fewer than
  `LIBRATION_CENTRE_MIN_ESTIMATES` slow periods, rungs W_top × f down to max(P_c, `LIBRATION_CENTRE_CYCLES` × P_fast)
  (P_fast = shortest full period of the fine TPs) are walked while their TPs at p_c match W_c and the rung above (count,
  kind, shift ≤ tol × P_slow); the last that holds is analysed when it shows more slow extrema than W_c, kept for a
  libration; L4 and R0 both not_measured there → cap `slow_scale_provisional` (≤ medium, action spot_check). Columns
  `geo_scale_ladder_*`; `geo_centre_scale_cutoff_yr` is the rung. Moves: 385533 7N-12 uncertain 5n → resonant/medium
  (W_c 22.7 → 7.95 kyr, 2 → 4 extrema, F direct T0_th 33.9–36.8 kyr). Without the window trigger 136108 (13.9 → 6.9 kyr)
  and TX9 (episode end 17.5 → 18.9 kyr) would move, verdicts equal; 182223's rung (5.98 kyr) decides no libration. A
  shift on a count-matched rung was never above 0.022 P on the 21 G2c rows: the count is what separates.
- **W28 B10a slow scale + K3s** (`kinematics.SlowTrack`, `measure_slow_scale`, `centre_scale_track`,
  `centre_track_slow`, `slower_than`, `predicted_swing_rad`; `analyse._slow_scale` (G2c's centre-scale geometries
  collected by `seen` in `_decide_at_scale`, before the covering filter); `decide.secular_cycle_kinematic`; models
  `SlowScale`; cards g08-7 A, g08-8 A, g05-2 A; rules `slow_scale`, `secular_cycle_kinematic`;
  `MMR_SLOW_SCALE_REPORT`, `MMR_SECULAR_CYCLE_KINEMATIC`): the slower libration of the centre is stored at its own
  scale: G2c's accepted runs at W_c (covering or not; episodes.csv scale='slow', `episode_rows(include_slow=True)`
  only: every reader of the main scale gets the default) slower than libration_centre_cycles × the main libration,
  else ≥ 3 interior turns of c(t) of a librating angle. Columns `slow_*`: closed cycles, period (nan below
  LIBRATION_CENTRE_MIN_ESTIMATES), identity on that scale, B1 in its band, nearest secular period (fundamental
  only; among matching g − g_j the one ċ follows best), origin (kinematic / dynamic_candidate / unresolved) and
  cause with a level (identified: period match + (ċ R² ≥ MMR_OWNER_R2_MIN or kin_centre_corr_rate ≥
  MMR_KIN_CORR_MIN); possible: one; unknown); an owner row takes A2's own_x cause first. Description only, but
  K3s: non_resonant/circulation (origin none, not 1:1) + secular offset ≤ MMR_KIN_PERIOD_TOL + slow identity fails
  + centre max monotone pass ≤ 1 + LIBRATION_SLIP_TOLERANCE → kinematic_libration/secular_cycle (`rec_rule` K3s),
  graded as the circulation (`_origin_margins` adds no K1/K1e/B5, no K1e physics). Moves: 37087 ×2 circulation →
  kinematic_libration, medium kept. Traps: without the slip test 285637 2 Myr (4.7 rev passes, author circulation/
  high) took K3s and lost the cap exemption (→ medium); a centre-scale run of a librating angle at its own period is
  that libration again (6489 3J-1, W_c 182 yr). 153 3J-2 (resonant/high) has a kinematic slow scale (g − g5,
  identified): reported, never demoted.
- **W28 q-2 b** `full_support_caps`: an own-dynamics resonant/transient/near_separatrix verdict is high only when one of
  `FULL_SUPPORT_MARGINS` (L4 incl. the 1:1 law, R0, F direct) is full, else cap `no_full_physics_support` (≤ medium;
  `MMR_HIGH_NEEDS_FULL_SUPPORT`); skipped when all physics is not_measured (that cap already says so).
- **W23 S2** `TreeView` (`tree_view`, built once after `union_read`: the angle, rule R's response and the regime's
  energy as the tree read them; margins, checks, caps, baseline and the confidence's word read it);
  `StatusDecision.withdrawn` (margin id → `Withdrawal`: the checks the filter guard and R-ALIAS-A made not_measured for
  the tree; `margins_as_tree_read` feeds the level, `withdrawn_checks` the two columns); `librating_episodes`,
  `librating_union`, `near_episode_tp` once per geometry and settings (`Geometry.derived`, emptied by `forget_derived`
  where the runs or their edges change); the 1:1 R0 is a copy from the deciding response
  (`response.with_coorbital_potential`, no in-place mutation).
- **CS12** (`resonance/analysis/mmr/response.py`, W23) `harmonic_law`: when L4 fails or passes thin (never 1:1, never in a null test),
  ȧ = Σ_{k≤N}(A_k sin kψ + B_k cos kψ) + C in the single law's windows; N grows while R² (smaller half median) gains
  ≥ `MMR_FORCE_MULTI_GAIN`, up to `MMR_FORCE_HARMONICS_MAX`; taken only with N ≥ 2 after coherence, shifted null and
  independent halves (`independent_halves_r2`, nan when a direction is unmeasured); lifts L4, never lowers it; K2's
  `force_laws` and the L4 margin read it; columns `rsp_force_harm_*` (2001 ND13 1V-3 → transient).
- **G2c rule (a)** (`resonance/analysis/mmr/analyse.py`, W23) `_decide_at_scale`: the centre-scale rerun takes its turning points at the
  centre-turn prominence p_c that qualified the scale (`LIBRATION_CENTRE_SCALE_TP_AT_CENTRE`; children inherit W_c and
  the raised floor); `_union_guard`/`_rejected_columns`: the G2c analysis the union guard rejected is kept in
  `geo_cs_rej_*`, `geo_union_word_centre_scale` = 1 only when the word was rewritten (2003 TX9: one episode
  2.21–17.49 kyr, uncertain by 5e). Fragile: at p_c 7° or 16° TX9's W_c changes (strict xfail).
- **`far_circulation`**: a non_resonant circulation ≥ `MMR_FAR_FROM_RES_FACTOR` a-scales from a* is not graded on the
  10° step rule: it can reach certain.
- **`short_period_caps`** (`analyse.py`): the cap `short_period_aliased` stays on every record whose step folds short
  periods (dt ≥ min(P)/2, or a harmonic up to `MMR_SHORT_PERIOD_HARMONICS` folded slower than the period floor; one
  predicate `sampling.short_period_trigger`), except a geometric circulation without forced-wave runs
  (`MMR_SHORT_PERIOD_CAP_CIRCULATION` off, author 2026-10-23); the record never lifts it
  (`MMR_SHORT_PERIOD_CAP_RATIO` = 0, W20 round D); the scatter of a against the libration's a swing warns
  (`a_short_period_scatter`) and gates R-ALIAS-A (`decide.short_period_unmeasured`); `smp_fold_free_step_yr` is the
  record's own step that folds nothing (`sampling.fold_free_step`), and when the cap holds the level down
  (`confidence.cap_binds`) the action is `resample` at it (W21 A; no status or confidence changes).
- **Filter guard Q9** (`resonance/analysis/mmr/preprocess.py` `filter_sensitivity`): the ×2 filter guard compares the decisive episode with
  its counterpart at 2 P_c (largest time overlap; none = not reproduced), else the whole interval; it runs before the
  tree: a passed R0 it withdraws is not_measured for the tree (`decide.energy_for_tree`, `flt_withdrawn_before_tree`),
  a failed one stays an objection (W21 Q9). The ×2 check uses the ×1 threshold profile.
- **G2a guard** (`analyse._action_step_guard`): see `../CLAUDE.md`.

## Physics anchors

- The a↔σ identity makes "σ and a share a period" automatic; V_E ≈ V_F is necessary, not sufficient; the period law
  (R0) separates real pendulums (T_obs/T_pend ≈ 1) from kinematic swings (0.61).
- Below a 45° half-amplitude, L4 and R0 with a are not_measured (sin σ ≈ σ: indistinguishable from the identity).
  R0 with a applies it per window too (W28 B3, g08-1; `zone.window_eligible`: a window counts with a half-swing ≥ 45°
  or a turn) and compares V_E and V_F on the visited arc (`zone._window_correlation`, `SEPARATRIX_VEVF_ARC`); rules
  `r0_windows`/`potential_on_arc` fire when either part changes the check (`zone.r0_with_a`, carried by
  `EnergyAnalysis.rules_fired`, merged via `zone.energy_rules`; none when the 1:1 law replaced R0). Moves: 137295
  1N-2 → resonant/high, 2004 BG41 → resonant/medium; TX9 (σ-only period law) and 82831 unchanged.
- Three-body strength is known only to a coefficient of 1–100; a computed three-body strength does not exist yet
  (2348 4J-2S-1 is medium only for that reason; see `dev/status.md`).
- Kinematic traps: 285637 (−4ϖ beating at the 2ω cycle, B1 6·10⁻⁵: a carries almost nothing; 481085 7E-10M-6 is
  the +9ϖ pair, B1 3.2·10⁻³), 37087 3J+8S-3 (secular cycle, K3s on the slow scale), 101703 41J-14 (driven by
  3J-1), a member with (e_E/e)^7 ≈ 6e-11.
- Alias validation reads the λ part of the angle; ghosts at n_res + 2πm/(kΔt).

## Traps

- Never certify an alias from the turn count the predictive unwrap inserted.
- `min(finite, nan)` passes a one-sided check; one window copied into an empty half is not two halves.
- The saved 1:1 `angle_raw` of the benchmark is λ_planet − λ; signs go through `angle_sign`.
- Do not lift caps by counts or by coefficients learned from benchmark siblings.
- Confirm `settings_hash` before reading a benchmark run.
- Co-orbital retabulation tolerances (e 0.005, i 0.5°) and the coorbital table limits are measured; do not loosen
  them for speed.
- Open bugs (W28 Codex review, not fixed until the author discusses them): `geometry.half_librations` takes a
  half-libration with no neighbour on one side as cut by the record edge without counting σ_f's turns to the window
  edge, so G4h asks 42946 43S-6 for a futile extend (8.1 turns follow; ≥ 1 turn should read as a passage, S1); the
  fixed `MMR_COORBITAL_EDGE_TURN_MIN_DEG` = 20° rejects 2007 VU6's 18.8° edge turn that closes its last cycle (S7, E67).
- Per-row cost: a geometry step that scans the record per revolution is O(N·revs) on high-order angles
  (C14 `edge_stretch` took 100–206 s on 8 rows until it used `searchsorted`); time the slowest mmr2 rows after a change.
