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
  table `sub_top` (E, W, N, A/W; by E and W only under `sub_strength_caveat`).
- `owner`: catalogue longitudes when a planet's λ is not saved (confidence ≤ medium); `owner_potential`: the owner's
  averaged force predicts ȧ (the potential route); `owner_identity` (route 'identity', W22): a three-body σ needing
  ≥ 10³ × μμe^q follows a two-body φ that librates over its episode with a passing direct strength; not run without
  saved longitudes. A candidate owner is tested by the multiplet law (coherent coefficients), never by a bare frequency
  coincidence.
- `decide`: rule C, an accepted member of the multiplet gives the catalogue row its status.

## Rule glossary (`resonance/analysis/mmr/decide.py` unless stated)

- **F6** `forced_wave_episodes`: an accepted run on a lattice line with a failed identity (or a wrong-sign identity)
  is a forced wave, no episode for the status (`Geometry.forced_waves`, edges re-read by `geometry.refresh_edges`); the
  angle word keeps its loops (`forced_wave_word`). Author 2026-10-19.
- **K2** `slow_centre_is_kinematic`: episodes that ripple a slow centre following λ + p ϖ while every force law fails
  (`force_law_fails`, halves and multiplet) → non_resonant/kinematic_drift, graded as a drift. **K2b**
  `_kinematic_centre_moves`: a kinematic libration whose centre moves > `MMR_KIN_DRIFT_SHIFT_RATIO` × its amplitude
  goes the same way (R-KDRIFT, D10). **G4b** `centre_net_circulates`: net shift ≥ `LIBRATION_CIRCULATION_NET_REV` rev
  is circulation, also K2's word (R-NET, D14).
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
- Three-body strength is known only to a coefficient of 1–100; a computed three-body strength does not exist yet
  (2348 4J-2S-1 is medium only for that reason; see `dev/status.md`).
- Kinematic traps: 285637 (−4ϖ beating, a 3000× too weak), 37087 3J+8S-3 (secular cycle), 101703 41J-14 (driven by
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
