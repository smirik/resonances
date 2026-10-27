# History of the classifier (July – October 2026)

Agreement numbers are "agree / compared" against the author's labels on the benchmark (status, angle_status,
confidence, author_verdict), measured by `benchmark/tools/known_disagreements.py` (earlier by
`agree_author_verdicts.py`). A run name `v2-wNN` is a reclassification of the saved benchmark series in
`benchmark/data/` (git-ignored).

## Before the programme

- **July.** An episode-based rewrite of the classifier (raw angle, coverage-gap episodes, σ–a confirmation) was built
  and reverted: it lost obvious transients, and recall on transients mattered more. The adjudicated ground truth
  (269 rows) was kept; it became the seed of the benchmark.
- **August 11–16.** Lidov-Kozai (ZLK) verification. A cross-spectral e–i coherence gate (591986, 162474) was built,
  then demoted to the `confirmed_ei` flag; the free-omega gate took over the ZLK status. An external specification of
  that gate (v3) was measured and overturned in several points (v4, see [rejected.md](rejected.md)).
- **September 25.** MMR pattern atlas (121 cases): the a↔σ identity, seven stroboscopic alias cases, classes A–X
  (class E: weak arguments that cannot be resolved). It is the physics basis of [physics-notes.md](physics-notes.md).
- **September 30.** An O(N²) `np.correlate` autocorrelation stalled a run for 17 h on long series; replaced by an FFT.
  Batch workers recycled (`max_tasks_per_child=1`), BLAS threads pinned.
- **October 12.** Secular angles had been built from the planets' osculating elements, so every status involving an
  s_j term was wrong. Fixed: the planets' mode longitudes, the invariable plane, reduced angles. Commit fa23598 then
  measured the mode phases in the run (`secular_planet_phases='measured'`).

## The v2 programme

- **Oct 13.** Full package review (127 findings, R-01…R-57; section "verified and correct" became
  [verified.md](verified.md)). Status-model design: status / angle_status / confidence / checks / reason; automatic
  portrait checks (ML) studied and rejected as evidence. Benchmark v2 started; `benchmark/DECISIONS.md` starts.
- **Oct 15.** The author answered the method handbook; SPEC v2 written; an independent review found real errors in
  it and the author accepted all four corrections. W1 (shared analysis modules: verdict, ledger, settings, geometry,
  pendulum, confidence), W2 (subresonances, owner search). Benchmark moved into the repository (`benchmark/`).
- **Oct 16.** W3: v2 wired into `Simulation` (string statuses, `_finalise_verdict`, TTO removed for MMR). W4:
  secular/ZLK on the shared modules; the TTO classifier and `secular_centre.py` deleted. W5: simplification and
  performance. W6–W11: the combined figure, rule C (multiplet), rule R (witness episode), regime prominence.
  Agreement on W3: 171/223 of all author verdicts.
- **Oct 17.** W12 (response cutoff 0.05 → 0.1 cycle), W13 (a closed cycle is mandatory), W15 (owner by the multiplet
  law; alias gate on the λ part of the angle), W16 (physical filter cutoff; Jupiter Trojans fixed; angle_status
  `transient` renamed `episodic`). W17: independent re-investigation of disputed objects; several earlier claims
  corrected (see [rejected.md](rejected.md)).
- **Oct 18.** W18: moving centre — runs tested in ψ = σ − c(t), c(t) the occupied minimum of the averaged potential.
- **Oct 19–21.** Reconciliation of 54 author/package disagreements. W20 in rounds A–F: forced waves (F6), kinematic
  drift (K2), baseline estimate N, subresonance table, `certain` for far circulations, the `short_period_aliased` cap
  measured on 47 fine re-integrations. `v2-w20f`: status 72/83.
- **Oct 23–24.** W21: `KNOWN_DISAGREEMENTS.csv` and the exact benchmark gate, the column cache; the author answered
  N1–N8, then D1–D20 for all 39 remaining disagreements.
- **Oct 25.** W22: every agreed rule implemented; `v2-w22` reaches 0 disagreements (status 94/94, angle_status 87/87,
  confidence 60/60, author_verdict 202/202). A simplification pass removed 370 lines with byte-identical output on
  719 examples.
- **Oct 26.** W23: CS12 multi-harmonic force law, rule (a) at the centre scale, the episode gate (79 cases), the
  low-memory allocator and per-record memo, figure overflow 0. Exactly 5 rows changed against `v2-w22`; 0
  disagreements.
- **Oct 26–27.** W24:
  - reasons `geometric_libration`/`geometric_episode` for leaf 5u/6 (8 rows);
  - output layout (`data/`, `figures/`, `combined/`, presets; default `confidence/status`; `PLOTS=combined`);
  - G2a action-step detector, made robust after an independent review (test 7 `regime_periods`), and R0 reading each
    cycle in its own well (`pendulum.cycle_period`): 2003 TX9 stays uncertain, the author accepted it;
  - MPC as an elements source, as the self-contained `resonances/mpcorb/` subpackage (query by elements, `mpc-search`,
    epoch policy `propagate` by default);
  - clones (virtual asteroids) in `resonances/clones/` and `simulation/ensemble_run.py`;
  - slow tests from 28 to about 6 minutes with pinned dates, `make test-long` for the 10-Myr z1 run, the secular/ZLK
    author gate (14/14) in `make test-benchmark`;
  - a build record for the cached MMR matrices (`Matrix.check_cached`);
  - a computed three-body strength (for 2348 4J-2S-1) studied and split off as a separate task.

  Latest run `v2-w24-tx9b`: 0 disagreements.

## Agreement over the programme

| Run | Date | Status | Angle | Confidence | author_verdict | Note |
|---|---|---|---|---|---|---|
| v2-w3 | Oct 16 | 171/223 (all verdicts) | | | | first wired run |
| v2-w20f | Oct 21 | 72/83 | 68/76 | 51/83 | 219 | after rounds A–F |
| v2-w22 | Oct 25 | 94/94 | 87/87 | 60/60 | 202/202 | 0 disagreements |
| v2-w23 | Oct 26 | 94/94 | 87/87 | 60/60 | 202/202 | 5 rows changed vs w22 |
| v2-w24-g2a-fix | Oct 27 | 104/104 | 88/88 | 61/61 | 202/202 | secular/ZLK author verdicts 14/14 |
| v2-w24-tx9b | Oct 27 | 0 disagreements | | | | TX9 uncertain, accepted |

The denominators change because labels were added and relabelled; a run agreeing with every label is not an accuracy
estimate (see [pitfalls.md](pitfalls.md), statistics).
