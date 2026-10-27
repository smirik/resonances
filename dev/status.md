# Status on 2026-10-27

Replace this page at each programme step. Latest benchmark run: `v2-w24-tx9b`, 0 disagreements with the author
(`KNOWN_DISAGREEMENTS.csv` empty); episode gate: tier A 45 pass with 5 known failures, tier B 12/12.

## Next steps (agreed order)

1. Code freeze; then the documentation (mkdocs-material, LaTeX formulas, a command that regenerates the documentation
   figures), the agent skill (done 2026-10-27: `resonances/skills/resonances/`, `resonances skill install`,
   references generated from docs/ by `make skill`, docs/agents.md), the benchmark figure tree rebuilt
   (`benchmark/tools/sort_figures.py`). `resonances/data/confidence_cells.json` was recalibrated on `v2-w24-tx9b`
   (2026-10-27: 280 verdicts compared with the author's status, 0 errors, 19 abstentions; the cells count the 173 at
   high or certain, pooled over resonance families: the intervals measure status agreement, not each check); rerun `benchmark/tools/calibrate.py` after any verdict change.
2. The author's final check of the figures.
3. A real co-orbital test.

## Separate tasks

- **Computed three-body strength** (2348 4J-2S-1 medium only because of `strength_scale_only`; would also move 68436
  and 24820 3J-1S-1 and the origin of 48789). Paths: second-order averaging (Morbidelli 2002 ch. 10), an off-resonance
  numerical probe (untested idea), or a family calibration (cheap, data-dependent). Task description:
  `tmp/classifier-v2-2026-10-15/tasks/three-body-strength/TASK.md` (local).
- **2003 TX9 two-regime split (Q1b).** The author accepted uncertain with one episode. A split needs each regime's
  period measured in one shared two-well potential on every cycle; test 7 currently fails on late cycles.

## Deferred episode mechanisms (episode gate `known_fail`)

- E15: 2012 KL9 off-lattice ripple (an external 222-yr wave of a, not on the angle lattice).
- E17 and E30: the K3 stall cut (2013 VA14 60-kyr stall remainder; 385266 forced-wave split); the stall-cut rule waits
  for the author.
- E21: 82831 σ/ψ frame gap.
- E43: 2021 XO3 missing ~300-yr boundary.
- Episode rule (b) (the claimed regular core must pass by itself) is not implemented.

## Known gaps in the code

- No co-orbital applicability guard in `analyse_zlk`/`evaluate_gate`, although the free-element basis excludes
  co-orbitals.
- Secular bounded angles get a dynamic origin directly; an episodic ZLK keeps it when the whole-record gate withholds.
- The secular `FilteredAngle` leaves the MMR regime-cut options (split regimes, return closure) at their defaults; the
  old 88899 ZLK two-island loss (tier D) has not been re-checked.
- Panel (e) draws the single-sine windows even when CS12 selects the harmonic law (ND13: 0.18/0.19 drawn, 0.646
  decided); three ▼-marker overlaps remain.
- The low-memory allocator is macOS-only; Linux was not measured.
- Clones: not resumed after an interruption, not in `simulation.json`; no per-object MPC get-orb fallback lookup.
- MPC `propagate` is checked over 138 days only.
- Found by the docs rewrite (2026-10-27), left as they are under the code freeze (a fix changes columns):
  - The dry-run checks MMR rows only (`simulation/dry_run.py:920` `_body_rows` and `:1170` loop over `body.mmrs`):
    the baseline of a secular or ZLK angle (`SECULAR_LIBRATION_PERIOD_MIN_YR`, the 1.5 cycles, the ZLK floor) is not
    checked before the run.
  - A secular or ZLK verdict with `action` = extend has no `extend_to_years` (NaN in summary.csv): the `Verdict` of
    `resonance/analysis/secular/analyse.py:137` sets only `action` (`:151`, `default_action`); the N8 next check is
    MMR-only (`mmr/decide.py` `extend_years`).
  - Leaf 6u is written as its source leaf in `rec_rule`: `promote_union` (`resonance/analysis/mmr/decide.py:1892`)
    changes the status and reason but not `decision.rule`, so `mmr/analyse.py:1635` writes the dynamic leaf-6 rule that reached the
    transient (`5d/6`, or `<rule>/6` of `decide.py:1628`) for a promoted union.
- Found by the docs review (2026-10-27), left under the code freeze (a fix changes `sep_*` columns and possibly zone
  classes):
  - Fixed 2026-10-27 (W25, `v2-w25-action-well`): the two-harmonic action ratio was read in the deepest well for
    every cycle. Now J/J_s and its slope come from the well the cycle occupies (`pendulum.occupied_well`,
    `CycleEnergy.well_rad`), as the period did; `zone.averaged_classes` compares h of the turning points (1 at the
    averaged well's own barrier) with h_lo over V_E's deepest-well barrier (`pendulum.h_in_barrier_units`), not over
    V_E's global depth. With two harmonics the old ratio of a shallow-well cycle was always too large
    ((J + C)/(J_s + C), C constant). Benchmark: 62 rows with a two-well V_E, 36 with cycles in the shallower well; no
    status, angle_status, confidence, reason, cap, flag, class or mechanism changed; J/J_s changed in 4 rows (17346
    2J-1, 1916 3M-4S-5, 2012 KL9, 2013 VA14), whose shallow-well cycles stay zone because σ_h is comparable to the
    shallow well's depth (or the averaged route decides). Left: the J_lo → h map of the averaged route uses V_E's
    deepest-well shape, not the averaged well's; above the lower barrier J/J_s saturates at 1 (a loop around both
    wells has no action ratio of one well).
  - ZLK `rho_max` is one radius: an episodic/drift ZLK angle turns kinematic on `rho_max < 1`
    (`resonance/analysis/secular/decide.py:145`), and `rho_max` is the median free radius over the minimum of the
    rolling-median forced amplitude (`secular/free_elements.py`, `rho_envelope`, `_beat_window`), not the maximum of the
    instantaneous ratio. On a record that changes regime (exactly the episodic/drift case it decides) the median radius
    can say the free circle never encloses the origin while part of the record does. Untested on such a record.
  - `c1`, `c2`, `h_quad` of a ZLK row are evaluated on the initial ecliptic e, i, ω
    (`simulation/data_manager.py:811`), not in the perturber's plane: diagnostic columns only, not the classical
    integrals (documented in docs/lidov-kozai.md and docs/physics/secular-zlk.md).
  - The e-i coherence flag cannot confirm a retrograde ZLK orbit: the preset always targets 180°
    (`resonance/cross_spectrum.py:64`) and `coherence_analysis` requires a negative correlation (`:298`), while at
    constant signed H a retrograde i varies in phase with e; it also reads the ecliptic i (`resolve_series`, `:84`).
    A flag only (`confirmed_ei`), no status.
  - The three-body catalogue axis applies the body's precession correction once, at the uncorrected axis
    (`mmr/three_body.py:107`), without iterating; only the finder's search window reads `resonant_axis`.
- Moving-centre geometry has two documented possible false refusals with no real-record example yet: the continued
  potential minimum can jump basins when a minimum disappears, and a time-symmetric track cannot beat its reversed
  null (W18).

## Provisional thresholds

- `ZLK_MIN_ANTICORRELATION = 0.5` (set on two objects).
- Conventions, documented as such: `SECULAR_LIBRATION_PERIOD_MIN_YR` (200 kyr), ZLK period floor 2 kyr, TP prominence
  5°, response R² 0.5, `MMR_SHORT_PERIOD_HARMONICS` = 2, `LIBRATION_CIRCULATION_NET_REV` = 2, the N8 factors 2 and 4.
- `MMR_OWNER_POTENTIAL_CONFIRM` (direct averaging as the deciding confirmation of a two-body owner) is off by default.

## Unresolved science

- 591986: the free-element g is 220″/yr against 6.65″/yr in the proper-element catalogue.
- 78085: the captures are clear, yet the unwrapped angle keeps growing (to be analysed later).
- 3040 ZLK over 500 kyr of SABA at dt = 1 is chaotic in its start epoch (resonant, episodic or circulating within a few
  weeks); the slow test is pinned to 2026-06-10.
- 2024 WJ 1E-1: high, the author leans to medium; a "three-turning-point" rule was left for later.
