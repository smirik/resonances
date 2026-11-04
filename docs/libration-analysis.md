# Rule index

!!! note "Start with the physics"
    This page lists the rule IDs that appear in the combined figure, in `decision_path`, in `rec_rule` and in the
    output columns, each with a one-line meaning, its settings, the page that explains it and the code that runs it.
    The physics and the formulas are on the pages of [How the status is determined](physics/overview.md); the
    values of the settings are in [Configuration](config.md) and `resonances/.env.dist`.
    The column `rules_fired` names the rules that acted on a row by meaning; their doc ids are the IDs on this page.

Code paths are relative to `resonances/resonance/analysis/`. MMRs (two-body, three-body, retrograde, co-orbital) go
through `analyse_mmr` (`mmr/analyse.py`; the order of its steps: [Overview](physics/overview.md#the-pipeline)).
Secular and Lidov-Kozai angles share the geometry, the ledger and the confidence rule and have their own signal step
and physics ([Secular and Lidov-Kozai](physics/secular-zlk.md)).

## The verdict {#the-verdict}

The verdict fields (`status`, `angle_status`, `angle_subtype`, `reason`, `libration_origin`, `origin_detail`,
`confidence`, `action`, …) are described on the [Overview](physics/overview.md#the-verdict); every reason with its
status and origin is in [Every reason](physics/status-tree.md#every-reason); the old integer `status_code` is in
[Legacy integer codes](physics/status-tree.md#legacy-integer-codes). Every column of the output files is in
[Output data](output.md).

## Angle word: G rules {#angle-status}

The first rule that fires gives `angle_status` (`mmr/decide.py`, `decide_angle`; the geometry in `geometry.py`).
Full table: [From geometry to angle_status](physics/geometry.md#from-geometry-to-angle_status).

| rule | meaning | settings | physics | code |
|---|---|---|---|---|
| G0 | the record cannot be read: unphysical orbit, unwrap residual over its limits, $\varpi$ undefined, $P_c \le 2\Delta t$ → `undefined` / `invalid` | `MMR_UNWRAP_MAX_RESID_*`, `MMR_JITTER_MAX_RAD`, `MMR_VARPI_UNDEFINED_MAX_SHARE` | [signal](physics/signal.md#counting-the-turns-the-predictive-unwrap), [can the record be read](physics/status-tree.md#before-the-tree-can-the-record-be-read) | `mmr/sampling.py`, `mmr/models.py` (`domain_problems`), `mmr/analyse.py` (`_g0`) |
| G1 | a confirmed jump of $a$ (or a step of the action): the rules run on the last regime | `LIBRATION_JUMP_*` | [jumps](physics/geometry.md#jumps-of-a-and-the-last-regime) | `geometry.py` (`jump_candidates`, `confirm_jumps`) |
| G1c | after the last turning point the centre completes a revolution → `episodic`, `final_regime = circulation` | — | [jumps](physics/geometry.md#jumps-of-a-and-the-last-regime) | `mmr/decide.py` |
| G1e | the last regime holds no episode, episodes precede the jump → `episodic`; cap `episode_before_jump` | — | [jumps](physics/geometry.md#jumps-of-a-and-the-last-regime) | `mmr/decide.py` |
| G1l | a libration after a jump, none before it (a capture) → `episodic`, `final_regime = libration` | — | [jumps](physics/geometry.md#jumps-of-a-and-the-last-regime) | `mmr/decide.py` |
| G1q | a 1:1 angle that turns in the quasi-satellite well during a stay of one loop through its turns: analysed again at the quasi-satellite floor (rule `qs_trigger`) | `MMR_COORBITAL_QS_*` | [co-orbitals](physics/pendulum.md#co-orbitals-11) | `mmr/quasi_satellite.py` |
| G1f | a two-body angle without a libration, analysed again at the floor of its own averaged potential $T_{0,\text{th}}/\sqrt{x}$; kept when its cycle is below the floor and F (direct), the centre and L4 pass (rule `subfloor_pass`) | `MMR_SUBFLOOR_PASS`, `MMR_STRENGTH_CHECK_RATIO` | [status tree](physics/status-tree.md), [signal](physics/signal.md#the-cutoff-what-is-libration-and-what-is-noise) | `mmr/analyse.py` (`_subfloor_pass`), `mmr/preprocess.py` (`profile_period_floor`), `mmr/strength.py` (`theory_period`) |
| G2 | a librating episode: covering → `libration`, else `episodic`; for a 1:1 angle also a union of episodes; a 1:1 run whose turns reflect on the planet's or L3's top has the floor of the loop through its own turns (rule `coorbital_floor_profile`, `MMR_COORBITAL_PROFILE_FLOOR`); a 1:1 run that misses the turning point the filter's edge trim hides takes it from the orbit-averaged raw angle, only to close a cycle (rule `coorbital_edge_turn`); a 1:1 episode ends only at a separatrix, its changes of well and amplitude are `episodes.csv` marks (rule `coorbital_well_cut`) and every accepted 1:1 run stores its core, sub-regimes and inferred residence (rule `coorbital_residence`); an accepted run that does not count for the status (forced wave, induced, below the period floor, a centre that turns, before the last jump) has `eligible` false and its `ineligible_reason` (rule `eligible_reason`); an exit or entry is read against the local centre at the edge (rule `local_centre_entry` where the mean centre would read it otherwise) | `LIBRATION_*` (episodes, coverage), `MMR_COORBITAL_EDGE_TURN*`, `MMR_COORBITAL_TURN_ENERGY_TOL`, `MMR_COORBITAL_WELL_EPISODES`, `MMR_COORBITAL_RESIDENCE` | [runs and episodes](physics/geometry.md#runs-and-episodes), [coverage](physics/geometry.md#coverage-exits-and-entries), [edge turn](physics/geometry.md#a-co-orbital-turn-in-the-edge-trim), [1:1 boundary](physics/geometry.md#a-co-orbital-episode-ends-only-at-a-separatrix), [1:1 residence](physics/geometry.md#core-residence-and-sub-regimes-of-a-co-orbital-episode), [exits and entries](physics/geometry.md#coverage-exits-and-entries), [eligible](output.md) | `geometry.py` (`_edge_raw_turns`, `_accept_in_wells`, `_join_in_wells`, `coorbital_marks`, `residence_of`, `episode_eligibility`, `_edge_reading`), `mmr/coorbital_geometry.py` (`CoorbitalEdgeTurn`, `CoorbitalWells`) |
| G2z | a Lidov-Kozai $\omega$: a run breaks at half a turn (a $2\omega$ island; rule `zlk_half_turn_cut`) | `ZLK_RUN_BREAK_HALF_AMPLITUDE_DEG` | [secular and ZLK](physics/secular-zlk.md#geometry-and-status) | `geometry.py` (`split_runs`), `secular/preprocess.py` |
| G2q | MMR: two runs broken by one half-cycle over the top in $\sigma$ but not in $\psi = \sigma - \sigma_c(t)$ are one run (rule `frame_join`) | `LIBRATION_FRAME_JOIN` | [joins](physics/geometry.md#runs-and-episodes) | `geometry.py` (`_frame_join`) |
| G2s | MMR: a leftover of a stall cut that carries the stall's period and passes the band test with it joins the stall (rule `stall_join`) | `LIBRATION_STALL_JOIN`, `LIBRATION_REGULAR_PERIOD_RATIO` | [joins](physics/geometry.md#runs-and-episodes) | `geometry.py` (`_stall_join`) |
| G2m | MMR, not 1:1: two accepted pieces of a centre-step or stall cut whose facing full cycles share a band and a period are one episode; the cut is an `episodes.csv` mark (`centre_step@t0:t1`, `stall@t0:t1`; rule `regime_cut_mark`) | `LIBRATION_REGIME_CUT_MARKS`, `LIBRATION_REGULAR_PERIOD_RATIO` | [joins](physics/geometry.md#runs-and-episodes) | `geometry.py` (`_mark_regime_cuts`, `cycles_continue`) |
| G2u | a union of librating episodes of a non-1:1 angle, joined where $\sigma_f$ stays inside its neighbours' bands through every gap, however long (rule `bounded_gap_merge`): reported `libration` / `regime_change`, read as episodic by the tree (6u) | `LIBRATION_UNION_BAND_MARGIN_DEG` | [coverage](physics/geometry.md#coverage-exits-and-entries) | `geometry.py` (`covering_union`) |
| G2a | a step of the libration action within one half-cycle cuts the record as a jump of $a$ | `LIBRATION_ACTION_STEP*` | [action step](physics/geometry.md#a-step-of-the-libration-action) | `geometry.py` (`action_steps`) |
| G2c | no libration at $P_c$ but the centre librates: analysed again at the centre scale; a slow libration the centre scale trims to < 2 periods at the shortest cutoff of the scale ladder that keeps its slow extrema (rule `scale_ladder`, cap `slow_scale_provisional`); G2c's slow runs are stored and judged at their own scale (`episodes.csv` `scale = slow`, rule `slow_scale`) | `LIBRATION_CENTRE_CYCLES`, `LIBRATION_CENTRE_SCALE_TP_AT_CENTRE`, `MMR_SCALE_LADDER`, `MMR_SCALE_TP_TIME_TOL` | [centre scale](physics/geometry.md#the-centre-scale), [scale ladder](physics/signal.md#a-shorter-cutoff-for-a-slow-libration-the-scale-ladder-g2c), [slow scale](physics/response.md#kinematic-origins) | `geometry.py` (`centre_scale_angle`), `mmr/analyse.py`, `mmr/kinematics.py` (`measure_slow_scale`) |
| G3 | slips (pattern A) or stalls (pattern B) at the top of the well → `near_separatrix` | `LIBRATION_SLIP_*`, `LIBRATION_PATTERN_A_*`, `LIBRATION_STALL_*` | [near the separatrix](physics/geometry.md#near-the-separatrix) | `geometry.py` |
| G4, G4b | the centre makes a revolution, or ≥ 2 net → `circulation` | `LIBRATION_CIRCULATION_NET_REV` | [centre and circulation](physics/geometry.md#the-centre-and-circulation) | `geometry.py` |
| G4c, G4h | a circulation without an episode judged on its single legs: 1:1 → transient (one leg), uncertain/one_coorbital_leg, extend or the near-episode (rules `coorbital_leg`, `coorbital_near_episode`); not 1:1, a half-swing with its own law → uncertain/one_half_libration (rule `single_half_libration`) | `MMR_COORBITAL_SINGLE_LEG`, `MMR_COORBITAL_LEG_*`, `MMR_COORBITAL_TURN_ENERGY_TOL`, `MMR_COORBITAL_TURN_ZONE_MAX`, `MMR_SINGLE_HALF_LIBRATION` | [single leg](physics/response.md#local-force-and-energy-on-a-single-leg), [status tree](physics/status-tree.md) | `geometry.py`, `mmr/coorbital_geometry.py`, `mmr/decide.py` |
| end_arc_extend | after the tree: an arc or stall after circulation at the record's end (slower than 2 × its longest revolution and than the slow centre's reversals) → final_regime arc/stall, flag `extend_time` forward, a non_resonant circulation capped `record_ends_unresolved`; the start mirror, one 1:1 edge reflection, a 1:1 edge leg or an edge half-libration → the flag only (forward/backward) | `MMR_RECORD_EDGE_STRETCH`, `MMR_END_ARC_SLOWDOWN_FACTOR`, `MMR_END_ARC_MIN_FLOOR_PERIODS` | [the record's edges](physics/status-tree.md#how-the-records-edges-end) | `geometry.py`, `mmr/decide.py` |
| G5, G5a | runs with a moving centre → `drift`; one slow turn back → `insufficient_time` / `arc` | `LIBRATION_DRIFT_MIN_WINDOW_FRACTION` | [centre and circulation](physics/geometry.md#the-centre-and-circulation) | `mmr/decide.py` |
| G5w | MMR: the slow centre turns back twice within a turn and the half-swing between the turns passes the slow identity, with its $a$ swing and period $2(t_2-t_1)$ within $\sqrt{x}$ of the directly averaged pendulum at its amplitude → `insufficient_time` / `arc` instead of `drift` or `few_cycles`; extension `MMR_EXTEND_SLOW_CYCLES` slow periods (rule `slow_swing_in_well`) | `MMR_SLOW_SWING_IN_WELL`, `MMR_STRENGTH_CHECK_RATIO`, `MMR_IDENTITY_*`, `MMR_EXTEND_SLOW_CYCLES` | [steps](physics/status-tree.md#steps-in-order) | `mmr/slow_swing.py`, `mmr/decide.py` (`_g5w`) |
| G5s | secular and ZLK: `drift` only for a monotone slow centre; a centre that turned back → `insufficient_time` / `few_cycles` (rule `centre_turned_insufficient`) | `LIBRATION_DRIFT_MIN_WINDOW_FRACTION` | [secular and ZLK](physics/secular-zlk.md#geometry-and-status) | `secular/decide.py` |
| G5p | secular under linear mode phases: a libration or episode swinging no more than the phases can stray over the record, $2A_{\max} \le$ factor × $\sum_j \lvert k_j\rvert\,\mathrm{range}_j(T)$ → `insufficient_time` / `phase_unresolved`, action review (rule `phase_unresolved`) | `SECULAR_PHASE_UNCERTAINTY_FACTOR` | [secular and ZLK](physics/secular-zlk.md#geometry-and-status) | `secular/decide.py` (`phase_check`), `secular/phase_uncertainty.py` |
| G6 | turning points but no episode, or none → `insufficient_time` (`few_cycles`, `arc`, `stall`) | `LIBRATION_MIN_TURNING_POINTS` | [centre and circulation](physics/geometry.md#the-centre-and-circulation) | `mmr/decide.py` |

## Response tests {#response}

The checks that ask whether $a$ answers $\sigma$ as the pendulum of this resonance requires
([Is it this resonance?](physics/response.md)).

| ID | meaning | settings | physics | code |
|---|---|---|---|---|
| L3 | the identity $\dot\sigma \simeq c\,(a - a_*)$ (`check_mmr_identity`) | `MMR_IDENTITY_*` | [identity](physics/response.md#the-identity) | `mmr/response.py` |
| L4 | the force law $\dot a = R\sin(\psi - \phi_0) + C$, multiplet and harmonic laws, guards (`check_mmr_a_response`) | `MMR_FORCE_*`, `MMR_RESPONSE_*`, `MMR_A_RESPONSE_MIN_AMP_DEG` | [force law](physics/response.md#the-force-law) | `mmr/response.py`, `mmr/multiplet.py` |
| 1:1 law | the co-orbital force law, $K/K_\text{theory}$ | `MMR_COORBITAL_*` | [co-orbital force law](physics/response.md#co-orbital-force-law) | `mmr/coorbital.py` |
| L4d | where L4 fails or passes thin on a two-body, non-1:1 angle with a direct strength: the term's averaged force (real planet mass) and $\dot a$, both low-passed at $f\,P_\text{large}$, one scale $s$ per window; lifts L4 (cap `force_law_theory_only`), never lowers it; F reads the large turns (rule `direct_force_law`) | `MMR_DIRECT_FORCE_*`, `MMR_FORCE_R2_*`, `MMR_STRENGTH_CHECK_RATIO`, `MMR_FORCE_NULL_GAIN` | [direct force law](physics/response.md#the-direct-averaged-force-law) | `mmr/response.py` (`direct_force_law`, `period_large`), `mmr/strength.py` (`at_large_period`) |
| R | per episode: the witness that the tree reads L3, L4 and B1 from | — | [where the response is measured](physics/response.md#where-the-response-is-measured) | `mmr/response.py` (`witnesses`) |
| R0, R0σ | the potential: period law on $\sigma$ alone, and $V_E \approx V_F$ with $a$ (`check_mmr_potential`), counted only in the windows where $\sigma$ swings at least 45° (rule `r0_windows`) and compared on the visited arc, without a free constant (rule `potential_on_arc`) | `SEPARATRIX_*` | [energy and potential](physics/response.md#energy-and-potential) | `mmr/zone.py` (`r0_checks`) |
| Z | the separatrix zone and the cycle classes; `sep_mechanism` of the crossings | `MMR_ZONE_*`, `SEPARATRIX_*` | [separatrix zone](physics/geometry.md#the-separatrix-zone) | `mmr/zone.py` |
| B1 | the amplitude of $a$ against the one the libration needs (`check_mmr_a_amplitude`) | `MMR_B1_*` | [amplitude of a](physics/response.md#amplitude-of-a) | `mmr/response.py` |
| K1 | kinematic libration: $\sigma$ follows $p\varpi$ or a secular cycle (`check_mmr_kinematic`) | `MMR_KIN_*` | [kinematic origins](physics/response.md#kinematic-origins) | `mmr/kinematics.py` |
| K2 | kinematic drift of the slow centre under a ripple of episodes (`rec_rule` 3k2) | `MMR_KIN_RIPPLE_SHARE_MAX`, `MMR_KIN_DRIFT_SHIFT_RATIO`, `MMR_KINEMATIC_DRIFT_WORD` | [kinematic origins](physics/response.md#kinematic-origins) | `mmr/decide.py` |
| B5 | a libration slower than the record excluded by the centre's rate | `MMR_B5_*` | [kinematic origins](physics/response.md#kinematic-origins) | `mmr/response.py` (`slow_libration_check`) |
| F6 | an episode that is a forced wave of another argument, removed from the episodes | `MMR_FORCED_WAVE_LINE_ALONE`, `MMR_FORCE_CONTAMINATION` | [runs and episodes](physics/geometry.md#runs-and-episodes) | `mmr/response.py`, `mmr/decide.py` (`forced_wave_word`) |
| F6r | a ripple below 45° whose identity fails, on an unfinished slow swing (the centre turns back once or twice within a turn), is set aside before G2; G5a/G6 read the slow centre (rule `ripple_on_arc`) | `MMR_RIPPLE_ON_ARC`, `MMR_A_RESPONSE_MIN_AMP_DEG` | [steps](physics/status-tree.md#steps-in-order) | `mmr/decide.py` (`ripple_waves`, `_g6_ripple`) |
| F7 | an accepted owner takes an episode of $\sigma$ only when its measured regimes cover enough of it (every accepted episode; the own episodes decide) (rule `owner_per_episode`) and, librating, its $\Psi/u$ turns less than once per cycle (P2, rule `owner_psi_turns`) | `MMR_OWNER_COVER_MIN`, `MMR_OWNER_PSI_TURNS_PER_CYCLE_MAX` | [owner](physics/response.md#another-resonance-owns-the-oscillation) | `mmr/owner.py` |
| owner | another resonance drives $a$: $R^2$ gain, phase, potential, identity routes; second capture, excluded by the sweep and energy or by the secular forcing of $\ddot x$ (F3, rule `second_capture_f3`), or confirmed (both resonances at once, origin `joint`, rule `both_resonances_confirmed`) (`check_mmr_owner`) | `MMR_OWNER_*` | [owner](physics/response.md#another-resonance-owns-the-oscillation) | `mmr/owner.py`, `mmr/owner_potential.py`, `mmr/owner_identity.py` |
| S (scan) | a member of the multiplet librates about a fixed centre (`check_mmr_subresonances`; tests E′, L, A, B, C, D, E); the scan is also triggered by T7 (the theoretical centre of the catalogue member moves over an episode, rule `subresonance_t7`) and runs as a diagnostic under T8 (a moving centre with overlapping bands, an `origin_unknown` verdict; it never accepts a member; rule `subresonance_t8`), and for a two-body angle whose orbits do not cross the theory chooses the member (rule `subresonance_theory_select`) | `MMR_SUBRESONANCE_*` | [subresonances](physics/response.md#subresonances) | `mmr/subresonances.py` |
| O1 | onset: was the $a$-wave already there before the episode? A large ratio objects only to a foreign wave: coherent at the episode's period, in phase with it and not explained by the own law (O1c, O1l; rule `o1_foreign_wave`) | `MMR_O1_*` | [onset](physics/response.md#onset-of-an-episode-o1) | `mmr/response.py` (`onset_group`) |
| F | strength: required against theoretical $\mu_\text{eff}$ (`check_mmr_strength`) | `MMR_STRENGTH_*` | [strength](physics/pendulum.md#how-strong-must-the-resonance-be) | `mmr/strength.py` |

The filter guard (×2) and the folded-short-period rule are on [From the integration to the angle](physics/signal.md)
(`mmr/preprocess.py`, `mmr/sampling.py`). The cutoff:

| ID | meaning | settings | physics | code |
|---|---|---|---|---|
| Pc | the filter cutoff $P_c = \max(1.5\max_j P_{\text{syn},j},\ 1.5P_\text{orb},\ P_\text{floor}/2)$; rule `orbit_cutoff` where the body's own orbital period sets it (`flt_cutoff_binding` = `orbit`) | `MMR_FILTER_SYNODIC_FACTOR`, `MMR_FILTER_ORBIT_FACTOR`, `FILTER_MIN_LIBRATION_CUTOFFS` | [the cutoff](physics/signal.md#the-cutoff-what-is-libration-and-what-is-noise) | `mmr/preprocess.py` (`choose_cutoff`, `cutoff_rules`) |

## The status tree {#status-tree}

`decide.decide_status` (`mmr/decide.py`) is the only place a status is set; `rec_rule` names the step that decided
([The status tree](physics/status-tree.md)).

| step | meaning | physics |
|---|---|---|
| G6 | `insufficient_time` → `uncertain` / `baseline_needed` | [steps](physics/status-tree.md#steps-in-order) |
| C | circulation → `non_resonant` / `circulation`; R0 failing with turns: `no_resonant_potential` | [steps](physics/status-tree.md#steps-in-order) |
| C5 | an aliased circulation: unique branch and far ratio ≥ `MMR_C5_FAR_RATIO_MIN` → `non_resonant` / `alias`, else `invalid` / `alias` | [step resolution](physics/signal.md#what-the-output-step-can-resolve) |
| 1 | an accepted subresonance → the member's status / `subresonance_librates` | [steps](physics/status-tree.md#steps-in-order) |
| 2 | an accepted owner covering the episode → `non_resonant` / `induced_libration` | [steps](physics/status-tree.md#steps-in-order) |
| 2j | 'both resonances at once?': the remainder $x = \Psi/u$ closes cycles of a well of its own → the tree goes on, origin `joint`; one such cycle → `uncertain` / `baseline_needed`, origin `joint`, extend (rule `both_resonances_confirmed`; `MMR_OWNER_CAPTURE_*`, `MMR_OWNER_X_MIN_CYCLES`) | [step 2](physics/status-tree.md#steps-in-order), [owner](physics/response.md#another-resonance-owns-the-oscillation) |
| 3, 3k2, 3b5 | B1 or K1 fails → `kinematic_libration`; K2 → `kinematic_drift`; B5 finds an exact commensurability → `uncertain` / `baseline_needed` | [steps](physics/status-tree.md#steps-in-order) |
| 4 | `drift` without a kinematic cause → `uncertain` / `drift_mechanism_unknown` | [steps](physics/status-tree.md#steps-in-order) |
| 5f | L4 contaminated by a forced line, R0σ not failing → `uncertain` / `origin_unknown` | [step 5](physics/status-tree.md#steps-in-order) |
| 5s, 5s\* | implausible strength → `non_resonant` / `implausible_strength`, its accepted episodes marked forced waves of route `strength` (rule `forced_strength_mark`, `MMR_FORCED_WAVE_STRENGTH`); 5s\*: a three-body ratio the small-divisor envelope could close → `uncertain` | [step 5](physics/status-tree.md#steps-in-order) |
| 5a, 5a\*, 5b, 5e, 5g | a measurement objects (L4 fails; L4 passes but R0 fails; R0 fails; the identity fails on an episode) → `uncertain` / `origin_unknown` (`non_resonant` for a near-separatrix angle) | [step 5](physics/status-tree.md#steps-in-order) |
| 5c, 5d → 6 | the force is measured → by the behaviour, origin `dynamic` | [step 6](physics/status-tree.md#steps-in-order) |
| 5o, 5n | unmeasured branch: O1 or F objects; fewer than two independent cycles → `uncertain` / `origin_unknown` | [unmeasured branch](physics/status-tree.md#steps-in-order) |
| 5x | unmeasured branch: the order-of-magnitude strength $\mu e^q$ ($\mu_i\mu_j e^q$) out of range is a diagnostic, no objection (rule `eq_scale_not_objection`) → `uncertain` / `origin_unknown`, origin not established | [unmeasured branch](physics/status-tree.md#steps-in-order) |
| 5u → 6 | nothing measured → `geometric_libration` / `geometric_episode`, origin `unknown` | [unmeasured branch](physics/status-tree.md#steps-in-order) |
| 6 | by the observed behaviour: `resonant`, `transient`, `near_separatrix` | [step 6](physics/status-tree.md#steps-in-order) |
| 6u | a G2u union that reached `transient` / `dynamic_episode` → `resonant` / `dynamic_libration` (`decide.promote_union`; `rec_rule` 5c/6u or 5d/6u) | [step 6u](physics/status-tree.md#steps-in-order) |
| K3s | after the leaf: a `non_resonant` / `circulation` whose slow scale has the period of a secular cycle and fails its own identity → `kinematic_libration`, detail `secular_cycle` (rule `secular_cycle_kinematic`, `MMR_SECULAR_CYCLE_KINEMATIC`) | [step 3](physics/status-tree.md#steps-in-order) |

## Secular and Lidov-Kozai {#secular-zlk}

Their shorter tree is on [Secular and Lidov-Kozai](physics/secular-zlk.md#geometry-and-status); the G rules G2z, G5s
and G5p above are theirs. The rules that act on the frame, the free elements and the gate:

| ID | meaning | settings | physics | code |
|---|---|---|---|---|
| ZF | a Lidov-Kozai $\omega$, $i$, $\Omega$, the free elements and the $e$–$i$ coupling are measured from the invariable plane by an exact rotation; rule `zlk_invariable_frame` marks a row whose ecliptic and invariable $\omega$ part by a whole turn or more (`zlk_frame_shift_turns`) | `ZLK_ANGLE_FRAME` | [the frame](physics/secular-zlk.md#the-lidov-kozai-angle) | `data/invariable_plane.py`, `zlk/analyse.py` |
| FE | the free elements: every decimation factor is filtered, a remainder no pass of ≤ 10 divides by a last polyphase pass (rule `free_resample_remainder`) | `FREE_ELEMENTS_SAMPLING_YEARS` | [resampling](free-elements.md#resampling) | `secular/free_elements.py` (`decimation_steps`, `_resample`) |
| SZ.G | the free-$\omega$ gate: confirmed, kinematic or no verdict; a co-orbital of a planet whose modes the split does not fit gets no verdict (`coorbital_unfitted_forcing`) whatever its bounded $\omega$, an episodic one included (rule `zlk_coorbital_episodic`) | `FREE_GATE_*`, `ZLK_COORBITAL_HILL_FACTOR` | [the gate](physics/secular-zlk.md#zlk-free-omega), [where it stops](free-elements.md#where-the-method-stops-working) | `secular/decide.py` (`decide_status`) |
| SZ.ei | an episodic $\omega$ without a gate verdict: `transient` only when $e$ and $i$ trade as a ZLK cycle at the deciding episode's period (`zlk_ei_episode_coupling` = `zlk`), else `uncertain` / `gate_unavailable` (−9) (rule `zlk_gate_unconfirmed`) | `ZLK_EPISODIC_NO_VERDICT`, `ZLK_EI_MIN_CYCLES`, `ZLK_EI_PHASE_TOL_DEG`, `ZLK_EI_EPISODE_MIN_COHERENCE`, `ZLK_EI_H_RESIDUAL_MAX` | [the gate](physics/secular-zlk.md#zlk-free-omega) | `secular/decide.py` |

## Confidence and action {#confidence}

Margins, caps, the level rule, calibration and actions are on [Confidence](physics/confidence.md)
(`confidence.py`, `raw_level`; the MMR margins in `mmr/decide.py`).

| margin | class | meaning | settings |
|---|---|---|---|
| L1 | D | turning points of the decisive episode at $P_c/2$, $P_c$, $2P_c$ | `MMR_L1_TP_FULL` |
| L2 | D | distance to the separatrix | `MMR_L2_AMP_*` |
| L3, L4 | S, D | identity, force law | `MMR_IDENTITY_*`, `MMR_FORCE_R2_FULL` |
| L5 | S | one pendulum: $T_0$ over the amplitude regimes | `MMR_L5_*` |
| L6 | D | centre shift over the mean amplitude | `MMR_L6_SHIFT_FULL` |
| L7 | D | coverage gaps of a libration | `MMR_L7_GAP_FULL` |
| L8 | S | step and filter resolve the cycle | `MMR_L8_*` |
| O1 | cap | onset of a claimed capture; failed → cap `onset_failed` | `MMR_O1_*` |
| Z, R0 | D | zone measured; potential with $a$ | — |
| S, W | D | subresonance scan; owner search | `MMR_S_CLOSE_RATIO`, `MMR_W_R2_*` |
| C1–C5 | D/S | a circulation: centre pass, no turning points, smooth rate, rate against the force, alias | `MMR_C1_REV_FULL`, `MMR_C3_MODULATION_MAX`, `MMR_C4_RATIO_FULL`, `MMR_C5_*` |
| K1, B5, F, N | D, S, S, cap | kinematic correlation; slow libration excluded; strength; near-episode | `MMR_K1_FULL`, `MMR_B5_RATIO_FULL`, `MMR_STRENGTH_CHECK_RATIO`, `LIBRATION_NEAR_EPISODE_TP_DEFICIT` |
| K1e | S (physics) | a kinematic libration: the episode's σ follows $p\varpi$, $a$ flat, a secular period; full lifts `all_physics_not_measured`, so a kinematic libration with measured physics can reach high (rule `kinematic_high`) | `MMR_KIN_CORR_MIN`, `MMR_KIN_B1_CLEAR`, `MMR_KIN_PERIOD_TOL` |
| extend | action | length-limited `origin_unknown`: leaves 5e, 5g, 5x wait on a force law with too few windows, 5n on one independent cycle, and the last librating episode reaches the record's end → action `extend`, not `review` (rule `extend_length_limited`) | `MMR_FORCE_MIN_WINDOWS`, `MMR_FORCE_WINDOW_CYCLES`, `MMR_EXTEND_FORCE_WINDOWS` |
| U:&lt;name&gt; | thin | a decision within the thin band of a convention | `CONFIDENCE_THIN_BAND` |
| N, S, F, P, G | — | secular and Lidov-Kozai: turning points, stationary centre, the libration period ≥ `FILTER_MIN_LIBRATION_CUTOFFS` × $P_c$, a circulation at every smoothing (thin below 1.1 turns), the free-$\omega$ gate | `FREE_GATE_MIN_CYCLES`, `CONFIDENCE_THIN_BAND` |

A shared margin ID is read with `resonance_type`. The length to integrate (`extend_to_years`,
`base_extend_to_years`, `base_period_source`) is in [How long to extend](physics/confidence.md#actions).

## Names on the figure {#names}

The right column of the combined figure prints an ID as `short name [id]` and the deciding leaf as
`decided: circulation (C)`, so that it reads without this page; the IDs, `decision_path` and the CSV columns keep the
code's spelling. A prefix appears only where IDs collide: the secular and Lidov-Kozai margins are `SZ.N` … `SZ.G`
(MMR has its own N, S, F), a cap is `cap:<key>`. A path of the tree is named part by part (`5d/6`: force measured →
judged by behaviour). The names live in `resonances/plotting/verdict_vocab.py` (`RULE_NAMES`); an ID without one
prints as `unmapped check`.

| id | name | meaning |
|---|---|---|
| `G0` | record unreadable | unphysical orbit, failed unwrap, $\varpi$ undefined or too coarse a step |
| `G1` | jump of a | the record is cut at a confirmed jump of $a$ |
| `G1c` | circulates after the libration | the centre turns after the last episode |
| `G1e` | libration before the jump | the episodes precede the last jump |
| `G1l` | capture after the jump | a libration only after the jump |
| `G1q` | quasi-satellite well | a 1:1 angle read again at the quasi-satellite scale |
| `G1f` | cycle below the floor | a two-body angle read again at the floor of its own potential |
| `G2` | librating episode | at least three turning points about a fixed centre |
| `G2u` | episodes joined | a union of episodes covers the record |
| `G2a` | step of the action | the libration amplitude changes in a step |
| `G2c` | slow-centre libration | read again at the scale of the slow centre |
| `G3` | stalls and slips at the separatrix | `near_separatrix` pattern |
| `G4` | centre makes a revolution | circulation by a centre pass |
| `G4b` | net revolutions of the centre | circulation by net turns |
| `G4c` | single co-orbital leg | a 1:1 circulation judged on its best single leg: `transient`, `uncertain` or the near-episode |
| `G4h` | one half-swing between passages | not 1:1: a half-swing between passages that follows the resonance's own law: `uncertain` |
| `G5` | drifting centre | `drift` |
| `G5a` | one slow arc | one slow turn back: `insufficient_time` |
| `G5w` | slow swing in the well | two slow turns, the half-swing a swing of this resonance's well: `insufficient_time` |
| `G6` | too few cycles | `insufficient_time` |
| `C` | circulation | `non_resonant`: no own libration |
| `C5 (leaf)` | aliased circulation | the step folds $\sigma$: `non_resonant` when far and on one branch, else `invalid` |
| `R0 (leaf)` | no resonant potential | turning points, but the potential with $a$ fails |
| `1` | a multiplet member librates | the row takes its member's status |
| `2` | another resonance drives a | induced libration (owner) |
| `3` | kinematic libration | B1 or K1 fails: the swing is not in $a$ |
| `3k2` | kinematic drift of the centre | the slow centre follows $\varpi$ (K2) |
| `3b5` | slower libration possible | B5 leaves a slower libration open: `baseline_needed` |
| `4` | unexplained drift | `drift_mechanism_unknown` |
| `5a` | force law fails | L4 fails and R0 does not pass |
| `5a*` | force law fails, no potential | L4 fails, R0 not measured |
| `5b` | force and potential disagree | L4 passes, R0 fails |
| `5c` | potential supports | R0 passes despite L4 |
| `5d` | force measured | the force is measured: on to step 6 |
| `5e` | potential fails | R0 fails, L4 not measured |
| `5f` | force fit contaminated | a forced line may explain L4 |
| `5g` | identity fails | the identity fails on every episode |
| `5n` | too few independent cycles | fewer than two independent cycles |
| `5o` | onset or strength objects | O1 or F objects in the unmeasured branch |
| `5x` | scale out of range | the e^q strength scale fails in the unmeasured branch: no objection, origin not established |
| `5s` | resonance too weak | the libration needs an implausible strength |
| `5s*` | strength open (small divisors) | small divisors may close the gap |
| `5u` | shape only | geometry alone, the force is not measured |
| `6` | judged by behaviour | `resonant`, `transient` or `near_separatrix` |
| `6u` | union covers the record | a G2u union promoted to `resonant` |
| `2j` | both resonances? one cycle of x | 'both resonances at once?' with one cycle of $x = \Psi/u$ in its own well: `uncertain`, extend |
| `K3s` | slow scale is a secular cycle | a circulation whose slow scale is a secular cycle (kinematic) |
| `L1` | survives the filter | the turning points at $P_c/2$, $P_c$, $2P_c$ |
| `L2` | away from the separatrix | the cycles outside the zone |
| `L3` | σ̇ follows a − a* | the identity $\dot\sigma \simeq c\,(a - a_*)$ |
| `L4` | a obeys the force law | $\dot a = K\sin(\sigma - \phi_0)$ on independent halves |
| `L5` | one pendulum | $T_0$ stable over the amplitudes |
| `L6` | fixed centre | the centre shift is small against the amplitude |
| `L7` | covers the record | edge gaps within the coverage convention |
| `L8` | resolved by step and filter | the step and the filter pass the cycle |
| `L8a` | step resolves σ | no folded line in the band |
| `L8b` | filter passes the cycle | the filter keeps the amplitude at $P$ |
| `L8c` | filter guard compared | the cutoff ×2 comparison could be made |
| `B1` | a carries the swing | the observed $a$-wave against the one the libration needs |
| `O1` | no earlier a-wave | the $a$-wave before the episode against inside it |
| `R` | witness episode | the episode the response is read on |
| `R0` | energy fits the potential | the energy fits the potential from $a$ |
| `Z` | separatrix zone measured | the zone and the cycle classes |
| `S` | no member holds the centre | the subresonance scan |
| `W` | no other resonance drives a | the owner search |
| `F` | strength plausible | required against theoretical strength |
| `N` | not short of an episode | not one turning point short of an episode |
| `F6` | forced wave of another angle | an episode removed as a forced wave |
| `F6r` | unconfirmed ripple | a ripple below 45° whose identity does not pass, set aside on an unfinished slow swing |
| `F7` | owner covers the episode | the owner's intervals cover the episode |
| `C1` | centre passes whole revolutions | the largest centre pass |
| `C2` | no reversals at any cutoff | no turning points at the three cutoffs |
| `C3` | steady rate | one sign, no stalls |
| `C4` | too fast for a pendulum | the rate against the separatrix speed |
| `C5` | far from resonance | the rate the mean motions predict, or an alias far from $a_*$ |
| `K1` | slow σ follows pϖ | kinematic correlation |
| `K1e` | episode σ follows pϖ, a flat | K1 on the decisive episode, B1 < `MMR_KIN_B1_CLEAR` and a secular period: the measured physics of a kinematic libration (support) |
| `K2` | slow centre follows ϖ | kinematic drift of the slow centre |
| `B5` | slower libration excluded | the centre rate excludes a slower libration |
| `U` | near its convention | a comparison in the thin band of a convention (`U:<entry>`) |
| `SZ.N` | enough reversals | secular/ZLK turning points |
| `SZ.S` | centres agree | secular/ZLK stationary centre |
| `SZ.F` | slower than the cutoff | the libration period against the filter cutoff |
| `SZ.P` | circulates at every smoothing | a centre revolution at every smoothing |
| `SZ.G` | free-ω gate | the free-$\omega$ gate (Lidov-Kozai) |

The caps ([caps and the other reading](physics/confidence.md#caps-and-the-other-reading)):

| id | name | meaning |
|---|---|---|
| `cap:three_tp_episode` | one closed cycle | three turning points only |
| `cap:exit_suspected` | exit unresolved | the exit may not be an escape |
| `cap:entry_suspected` | entry unresolved | the entry may not be a capture |
| `cap:transient_unexplained` | transition unexplained | no mechanism explains the transitions |
| `cap:one_return_cycle` | one phase-return cycle | one cycle closed by its phase return |
| `cap:force_law_disagrees` | force law against potential | the potential holds, the force law fails |
| `cap:slow_libration_not_excluded` | longer libration possible | B5 thin |
| `cap:planets_missing` | planets not saved | no subresonance scan |
| `cap:owner_search_skipped` | owner search off | no owner search |
| `cap:catalogue_longitudes` | approximate planet longitudes | the owner search on catalogue longitudes |
| `cap:nodal_subresonance_candidate` | nodal member possible | a nodal member may hold the centre |
| `cap:subresonance_tie` | member chosen by order | tied members |
| `cap:subresonance_t6_only` | member scan on theory alone | the scan ran on the theory trigger only |
| `cap:subresonance_theory_unconfirmed` | member theory unconfirmed | test D not passed |
| `cap:subresonance_undetermined` | members indistinguishable | $\varpi_P - \varpi$ librates |
| `cap:jitter_low_e` | pϖ jitter at low e | the $p\varpi$ term wobbles by more than 45° |
| `cap:all_physics_not_measured` | own force unmeasured | none of L4, R0, Z, W, F measured |
| `cap:zone_not_measured` | zone not measured | the separatrix zone not measured |
| `cap:l2_without_zone` | separatrix distance unverified | L2 read from the amplitude alone |
| `cap:onset_failed` | earlier a-wave | O1 failed: a coherent, in-phase a-wave the own law does not explain |
| `cap:episodes_in_separatrix_zone` | episodes in the separatrix zone | the edges may be crossings of the zone |
| `cap:near_separatrix_without_zone` | separatrix pattern without zone | G3 without a measured zone |
| `cap:geometry_only` | geometry only | rule 5u/6, nothing measured |
| `cap:strength_scale_only` | strength scale only | 5u/6 with the order-of-magnitude strength agreeing |
| `cap:implausible_strength` | negative from strength only | rule 5s |
| `cap:episode_before_jump` | episodes before the jump | G1e |
| `cap:episode_force_law_mixed` | force law mixed over episodes | L4 holds on one episode, fails on another |
| `cap:near_episode` | almost an episode | N failed |
| `cap:short_period_aliased` | short periods folded | the step folds short periods |
| `cap:second_capture_unresolved` | both resonances at once? open | the owner does not exclude σ's own trapping ("both resonances at once?" not ruled out) |
| `cap:one_cycle_energy_not_conserved` | one-cycle energy fails | one 1:1 cycle whose energy fails |
| `cap:action_step_provisional` | action step provisional | a G2a cut with one cycle on a side |
| `cap:one_coorbital_leg` | one co-orbital leg | a 1:1 transient on one clean leg, no closed cycle |
| `cap:record_ends_unresolved` | record ends in an arc | a non_resonant circulation whose record ends in a slow arc or stall after circulating (rule `end_arc_extend`) |
| `cap:no_full_physics_support` | no full physical support | no L4/1:1 law, R0 or F passes with its full margin |
| `cap:force_law_theory_only` | force law from theory only | L4 passes only through the direct averaged-force law |
| `cap:slow_scale_provisional` | slow scale, own force untested | read at a rung of the G2c scale ladder with L4 and R0 not measured: only F supports it |
