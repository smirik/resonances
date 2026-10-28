# Status on 2026-10-28

Replace this page at each programme step. Latest benchmark run: `v2-w25-final` (+ `v2-w25-final-secular-zlk`), 0 disagreements with the author
(`KNOWN_DISAGREEMENTS.csv` empty); episode gate: tier A 45 pass with 5 known failures, tier B 12/12.

## Next steps (agreed order)

1. Code freeze; then the documentation (mkdocs-material, LaTeX formulas, a command that regenerates the documentation
   figures), the agent skill (done 2026-10-27: `resonances/skills/resonances/`, `resonances skill install`,
   references generated from docs/ by `make skill`, docs/agents.md), the benchmark figure tree rebuilt
   (`benchmark/tools/sort_figures.py`). `resonances/data/confidence_cells.json` was recalibrated on `v2-w25-final`
   (2026-10-28: 283 verdicts compared with the author's status, 0 errors, 19 abstentions; the cells count the 178 at
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

- Secular bounded angles get a dynamic origin directly; an episodic ZLK keeps it when the whole-record gate withholds.
- The secular `FilteredAngle` leaves the MMR regime-cut options (split regimes, return closure) at their defaults; the
  old 88899 ZLK two-island loss (tier D) has not been re-checked.
- MMR panel (e): the single-sine windows ignore the moving-well shift, and multiplet and raised-cutoff (line-rule) laws
  are still drawn as plain single sines (the harmonic law is printed, not drawn, since W25); three ▼-marker overlaps
  remain.
- `MMRDiagnostics`: the `portrait` with `portrait.y='axis'` and the a0 reference still read the 2000-yr body-level
  filtered a (`cycles` reads the analysis' a since W25; `evolution` documents its filter).
- `mmr/analyse.py` `is_unphysical` (e > 1.3) can no longer fire for an MMR after `domain_problems`: redundant.
- The averaged-route J/J_s (W25) ignores the moving-well kinetic term, as h did, and reads the profile's well count on
  its coarse grid (a very small dimple can be missed). A one-well potential above its top still reports J/J_s = 1
  (`separatrix`); a several-well one reports NaN (`not_measured`).
- The low-memory allocator is macOS-only; Linux was not measured.
- Clones: a restore keeps the clones' stored verdicts even when their series were saved; `restore(recompute_librations=True)`
  rewrites the nominal columns of clones.csv but not the nominal's summary.csv row, so the two can disagree. The resume
  hash includes the planets' initial state and the package versions (strict by design). No triage (decided 2026-10-28:
  a small spread in a does not certify the nominal's verdict, and integration is cheap compared with analysis).
- MPC `propagate` is checked over 138 days only; `MPC_LOOKUP=file_then_api` (get-orb fallback) is off by default and
  names are passed to get-orb unmapped (not verified live).
- ZLK (W25): the e–i coupling phase tolerance is 45° (591986's forced beat sits 32° from in-phase); the 591986 slow-line
  reconstruction reaches r = 0.991 only with a quadratic detrend; `sec_centre_*` of modulated secular rows is covered
  by a synthetic test only (no benchmark row); a node lock to a forced mode (591986: Ω − ϖ5 librates with 1.46 Myr over
  4 Myr) is not flagged (waits for the author).
- The three-body catalogue axis applies the body's precession correction once (`mmr/three_body.py:107`) with the
  small-α g; measured harmless (≤ 4e-7 au iterated, ≤ 2.5e-4 au from the g formula, against a 0.02 au search window;
  the classifier uses its own `exact_axis`).
- Moving-centre geometry has two documented possible false refusals with no real-record example yet: the continued
  potential minimum can jump basins when a minimum disappears, and a time-symmetric track cannot beat its reversed
  null (W18).

## Provisional thresholds

- `ZLK_MIN_ANTICORRELATION = 0.5` (set on two objects); the W25 e–i coupling conventions `ZLK_EI_*`, `ZLK_RHO_WINDOW_CYCLES`,
  `ZLK_COORBITAL_HILL_FACTOR`, `ZLK_CENTRE_*`, `ZLK_EXTEND_*`.
- Conventions, documented as such: `SECULAR_LIBRATION_PERIOD_MIN_YR` (200 kyr), ZLK period floor 2 kyr, TP prominence
  5°, response R² 0.5, `MMR_SHORT_PERIOD_HARMONICS` = 2, `LIBRATION_CIRCULATION_NET_REV` = 2, the N8 factors 2 and 4.
- `MMR_OWNER_POTENTIAL_CONFIRM` (direct averaging as the deciding confirmation of a two-body owner) is off by default.

## Unresolved science

- 591986: the free-element g is 220″/yr (the 6-kyr free precession, seen directly as the period of the osculating ω
  libration) against 6.65″/yr in the proper-element catalogue (definition not resolved). Its ~54-kyr centre wave is the
  g6 − g5 beat of Jupiter's own forced vector (W25); over 4 Myr its node librates against g5 (1.46 Myr).
- 78085: the captures are clear, yet the unwrapped angle keeps growing (to be analysed later).
- 3040 ZLK over 500 kyr of SABA at dt = 1 is chaotic in its start epoch (resonant, episodic or circulating within a few
  weeks); the slow test is pinned to 2026-06-10.
- 2024 WJ 1E-1: high, the author leans to medium; a "three-turning-point" rule was left for later.
