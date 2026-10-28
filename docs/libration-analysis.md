# Rule index

!!! note "Start with the physics"
    This page lists the rule IDs that appear in the combined figure, in `decision_path`, in `rec_rule` and in the
    output columns, each with a one-line meaning, its settings, the page that explains it and the code that runs it.
    The physics and the formulas are on the pages of [How the status is determined](physics/overview.md); the
    values of the settings are in [Configuration](config.md) and `resonances/.env.dist`.

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
| G1q | a 1:1 angle that stays in the quasi-satellite well: analysed again at the quasi-satellite floor | `MMR_COORBITAL_QS_*` | [co-orbitals](physics/pendulum.md#co-orbitals-11) | `mmr/quasi_satellite.py` |
| G2 | a librating episode: covering → `libration`, else `episodic`; for a 1:1 angle also a union of episodes | `LIBRATION_*` (episodes, coverage) | [runs and episodes](physics/geometry.md#runs-and-episodes), [coverage](physics/geometry.md#coverage-exits-and-entries) | `geometry.py` |
| G2u | a union of librating episodes of a non-1:1 angle: reported `libration` / `regime_change`, read as episodic by the tree (6u) | `LIBRATION_UNION_GAP_PERIODS` | [coverage](physics/geometry.md#coverage-exits-and-entries) | `geometry.py` (`covering_union`) |
| G2a | a step of the libration action within one half-cycle cuts the record as a jump of $a$ | `LIBRATION_ACTION_STEP*` | [action step](physics/geometry.md#a-step-of-the-libration-action) | `geometry.py` (`action_steps`) |
| G2c | no libration at $P_c$ but the centre librates: analysed again at the centre scale | `LIBRATION_CENTRE_CYCLES`, `LIBRATION_CENTRE_SCALE_TP_AT_CENTRE` | [centre scale](physics/geometry.md#the-centre-scale) | `geometry.py` (`centre_scale_angle`), `mmr/analyse.py` |
| G3 | slips (pattern A) or stalls (pattern B) at the top of the well → `near_separatrix` | `LIBRATION_SLIP_*`, `LIBRATION_PATTERN_A_*`, `LIBRATION_STALL_*` | [near the separatrix](physics/geometry.md#near-the-separatrix) | `geometry.py` |
| G4, G4b | the centre makes a revolution, or ≥ 2 net → `circulation` | `LIBRATION_CIRCULATION_NET_REV` | [centre and circulation](physics/geometry.md#the-centre-and-circulation) | `geometry.py` |
| G5, G5a | runs with a moving centre → `drift`; one slow turn back → `insufficient_time` / `arc` | `LIBRATION_DRIFT_MIN_WINDOW_FRACTION` | [centre and circulation](physics/geometry.md#the-centre-and-circulation) | `mmr/decide.py` |
| G6 | turning points but no episode, or none → `insufficient_time` (`few_cycles`, `arc`, `stall`) | `LIBRATION_MIN_TURNING_POINTS` | [centre and circulation](physics/geometry.md#the-centre-and-circulation) | `mmr/decide.py` |

## Response tests {#response}

The checks that ask whether $a$ answers $\sigma$ as the pendulum of this resonance requires
([Is it this resonance?](physics/response.md)).

| ID | meaning | settings | physics | code |
|---|---|---|---|---|
| L3 | the identity $\dot\sigma \simeq c\,(a - a_*)$ (`check_mmr_identity`) | `MMR_IDENTITY_*` | [identity](physics/response.md#the-identity) | `mmr/response.py` |
| L4 | the force law $\dot a = R\sin(\psi - \phi_0) + C$, multiplet and harmonic laws, guards (`check_mmr_a_response`) | `MMR_FORCE_*`, `MMR_RESPONSE_*`, `MMR_A_RESPONSE_MIN_AMP_DEG` | [force law](physics/response.md#the-force-law) | `mmr/response.py`, `mmr/multiplet.py` |
| 1:1 law | the co-orbital force law, $K/K_\text{theory}$ | `MMR_COORBITAL_*` | [co-orbital force law](physics/response.md#co-orbital-force-law) | `mmr/coorbital.py` |
| R | per episode: the witness that the tree reads L3, L4 and B1 from | — | [where the response is measured](physics/response.md#where-the-response-is-measured) | `mmr/response.py` (`witnesses`) |
| R0, R0σ | the potential: period law on $\sigma$ alone, and $V_E \approx V_F$ with $a$ (`check_mmr_potential`) | `SEPARATRIX_*` | [energy and potential](physics/response.md#energy-and-potential) | `mmr/zone.py` (`r0_checks`) |
| Z | the separatrix zone and the cycle classes; `sep_mechanism` of the crossings | `MMR_ZONE_*`, `SEPARATRIX_*` | [separatrix zone](physics/geometry.md#the-separatrix-zone) | `mmr/zone.py` |
| B1 | the amplitude of $a$ against the one the libration needs (`check_mmr_a_amplitude`) | `MMR_B1_*` | [amplitude of a](physics/response.md#amplitude-of-a) | `mmr/response.py` |
| K1 | kinematic libration: $\sigma$ follows $p\varpi$ or a secular cycle (`check_mmr_kinematic`) | `MMR_KIN_*` | [kinematic origins](physics/response.md#kinematic-origins) | `mmr/kinematics.py` |
| K2 | kinematic drift of the slow centre under a ripple of episodes (`rec_rule` 3k2) | `MMR_KIN_RIPPLE_SHARE_MAX`, `MMR_KIN_DRIFT_SHIFT_RATIO`, `MMR_KINEMATIC_DRIFT_WORD` | [kinematic origins](physics/response.md#kinematic-origins) | `mmr/decide.py` |
| B5 | a libration slower than the record excluded by the centre's rate | `MMR_B5_*` | [kinematic origins](physics/response.md#kinematic-origins) | `mmr/response.py` (`slow_libration_check`) |
| F6 | an episode that is a forced wave of another argument, removed from the episodes | `MMR_FORCED_WAVE_LINE_ALONE`, `MMR_FORCE_CONTAMINATION` | [runs and episodes](physics/geometry.md#runs-and-episodes) | `mmr/response.py`, `mmr/decide.py` (`forced_wave_word`) |
| F7 | an accepted owner takes $\sigma$'s episode only when it covers enough of it | `MMR_OWNER_COVER_MIN` | [owner](physics/response.md#another-resonance-owns-the-oscillation) | `mmr/owner.py` |
| owner | another resonance drives $a$: $R^2$ gain, phase, potential, identity routes; second capture (`check_mmr_owner`) | `MMR_OWNER_*` | [owner](physics/response.md#another-resonance-owns-the-oscillation) | `mmr/owner.py`, `mmr/owner_potential.py`, `mmr/owner_identity.py` |
| S (scan) | a member of the multiplet librates about a fixed centre (`check_mmr_subresonances`; tests E′, L, A, B, C, D, E) | `MMR_SUBRESONANCE_*` | [subresonances](physics/response.md#subresonances) | `mmr/subresonances.py` |
| O1 | onset: was the $a$-wave already there before the episode? | `MMR_O1_*` | [onset](physics/response.md#onset-of-an-episode-o1) | `mmr/response.py` (`onset_group`) |
| F | strength: required against theoretical $\mu_\text{eff}$ (`check_mmr_strength`) | `MMR_STRENGTH_*` | [strength](physics/pendulum.md#how-strong-must-the-resonance-be) | `mmr/strength.py` |

The filter guard (×2) and the folded-short-period rule are on [From the integration to the angle](physics/signal.md)
(`mmr/preprocess.py`, `mmr/sampling.py`).

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
| 3, 3k2, 3b5 | B1 or K1 fails → `kinematic_libration`; K2 → `kinematic_drift`; B5 finds an exact commensurability → `uncertain` / `baseline_needed` | [steps](physics/status-tree.md#steps-in-order) |
| 4 | `drift` without a kinematic cause → `uncertain` / `drift_mechanism_unknown` | [steps](physics/status-tree.md#steps-in-order) |
| 5f | L4 contaminated by a forced line, R0σ not failing → `uncertain` / `origin_unknown` | [step 5](physics/status-tree.md#steps-in-order) |
| 5s, 5s\* | implausible strength → `non_resonant` / `implausible_strength`; 5s\*: a three-body ratio the small-divisor envelope could close → `uncertain` | [step 5](physics/status-tree.md#steps-in-order) |
| 5a, 5a\*, 5b, 5e, 5g | a measurement objects (L4 fails; L4 passes but R0 fails; R0 fails; the identity fails on an episode) → `uncertain` / `origin_unknown` (`non_resonant` for a near-separatrix angle) | [step 5](physics/status-tree.md#steps-in-order) |
| 5c, 5d → 6 | the force is measured → by the behaviour, origin `dynamic` | [step 6](physics/status-tree.md#steps-in-order) |
| 5o, 5n | unmeasured branch: O1 or F objects; fewer than two independent cycles → `uncertain` / `origin_unknown` | [unmeasured branch](physics/status-tree.md#steps-in-order) |
| 5u → 6 | nothing measured → `geometric_libration` / `geometric_episode`, origin `unknown` | [unmeasured branch](physics/status-tree.md#steps-in-order) |
| 6 | by the observed behaviour: `resonant`, `transient`, `near_separatrix` | [step 6](physics/status-tree.md#steps-in-order) |
| 6u | a G2u union that reached `transient` / `dynamic_episode` → `resonant` / `dynamic_libration` (`decide.promote_union`; `rec_rule` 5c/6u or 5d/6u) | [step 6u](physics/status-tree.md#steps-in-order) |

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
| U:&lt;name&gt; | thin | a decision within the thin band of a convention | `CONFIDENCE_THIN_BAND` |
| N, S, F, P, G | — | secular and Lidov-Kozai: turning points, stationary centre, the libration period ≥ `FILTER_MIN_LIBRATION_CUTOFFS` × $P_c$, a circulation at every smoothing, the free-$\omega$ gate | `FREE_GATE_MIN_CYCLES` |

A shared margin ID is read with `resonance_type`. The length to integrate (`extend_to_years`,
`base_extend_to_years`, `base_period_source`) is in [How long to extend](physics/confidence.md#actions).
