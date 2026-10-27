# Rejected approaches (do not retry without new evidence)

Each row was built or measured and rejected. Retrying one needs a new argument or a new measurement that answers the
reason given. Sources in `tmp/` are local and git-ignored.

## Classification strategy

| Approach | Why rejected | When | Source |
|---|---|---|---|
| Episode-based rewrite (raw angle, coverage-gap episodes, σ–a confirmation) | Lost obvious transients; recall on transients matters more | Jul | programme history |
| Separate portrait classifier (rules or CNN/ViT on phase portraits) | The portrait is a function of σ(t) by the a↔σ identity: no independent evidence. ML at most ranks a review queue; transient recall dropped from 0.85 to 0.6 | Oct 13 | `tmp/status-design-2026-10-13/CHECKS-RESEARCH.md` §0 |
| "σ and a share a period", σ–a ellipses, secular portraits as proof of resonance | Automatic by the identity σ̇ = Σk n + k n(a) + p ϖ̇; σ was reconstructed from a within 1–2 % in all 121 atlas cases, forced ripples included | Sep 25 | atlas; `PROPOSAL.md` |
| Turning-point counts ("7 TP") to separate transient from near_separatrix | Not physical; residence times in a chaotic layer are heavy-tailed | Oct 15 | `research/SEPARATRIX.md` §2.6 |
| Fixed 10 % windows, early circulation exits of the old TTO classifier as a fallback | Historical failure modes; the TTO classifier was deleted rather than kept in parallel | Oct 13–16 | review |
| Analytic union of sudden-energy and adiabatic-action separatrix zones as "conservative" | Over-zones at λ ≪ 1, misses the reach near λ = 0.5; the e^q half-range had also been doubled (52356: 0.609 → 0.304) | Oct 15 | `reviews/PHYSICS.md` B1 |
| Minimum label counts per calibration cell as a cap on the shown level (≥ 100 labels) | The author dropped it: show the physically earned level; calibration is reported only | Oct 16 | W8/W9 |
| Rule A "every maximum of σ above every minimum" | Rejects the real moving-centre 2013 AS27 and the capture of 277617; at most diagnostic | Oct 17 | `impl/w17/q3-centre-rule/REPORT-fixed.md` |
| T_obs < T0(K_max) as an argument against resonance | External driving can sustain motion in a well at a faster period | Oct 17 | `impl/w17/q3-23240/REPORT-fixed.md` |
| A fitted pure 2(ϖ − ϖ_J) term in ȧ | A λ-free secular term cannot change a | Oct 17 | same |
| Frozen-potential portrait overlays on by default | Near-separatrix period agreement proves nothing; h errors 0.1–0.2 at low e; (σ, a) stretches the e origin into a line | Oct 17 | `impl/w17/q15-frozen-portrait/REPORT-fixed.md` |
| `island_switch` flag for ZLK | Not robust; 162474 over 1.5 Myr is simply transient | Oct 17 | W16 |
| Episode rules (b) "the claimed regular core must pass by itself" and (c) "carry a fine-scale force failure into the coarse verdict" | (c) fires on a pass that found no libration; (b) deferred | Oct 25 | `w23-work/design/spec/SPEC.md` §1 |

## Force, strength, owner

| Approach | Why rejected | When | Source |
|---|---|---|---|
| μe^q or products of masses as upper bounds (hard rejection or owner acceptance) | Order estimates only; the coefficient reaches ~10². 17886's e^25 ratio 1.6e19 became 1.05–1.30 by direct averaging | Oct 16 | `impl/W3-FIXES.md`, W6 |
| Gallardo's strength for statuses (Gallardo 2014 ρ(σ)) | Reproduced exactly (his Table 1 within 0.3 %), but it is not the resonant coefficient and its normalisation is arbitrary; μ_req/μ_Gallardo varies 7× between resonances (4J-2S-1 1.0–1.1, 5J-2S-2 2.3–2.8, 3J-1S-1 1.7–2.4, 1M+1J-2 0.4–0.6), more than the e^q scale (4×) | Oct 13, measured Oct 26 | `w24-work/gallardo/REPORT.md` |
| "Force law in a clean model" (Sun + four giants) as the three-body theory | A tautology: for a pendulum with σ̇ = c(a − a*), ȧ = −(ω₀²/c) sin σ, so K = ω₀²/c is fixed by the period at any amplitude; any integration where the body librates returns μ_req | Oct 27 | `tasks/three-body-strength/TASK.md` |
| Per-cycle period ∝ e^(−q/2) as a force test | Measured slopes −0.11…−0.27 against −0.5; fixed-slope R² negative in 5/6 rows (osculating e varies through forcing) | Oct 25 | `w23-work/design/2348/REPORT.md` |
| A per-resonance coefficient calibrated on benchmark siblings, or cycle counts, to lift `strength_scale_only` | The verdict would depend on which bodies are in the benchmark; the coefficient must come from theory | Oct 25 | same |
| Tightening co-orbital K/K_th to 0.75–1.25 | Removes real resonances (2016 HF2, WQ3, 2019 YB4, 614689 at 1.29–1.45), keeps the false 2024 WJ at 1.18 | Oct 16 | `impl/W4-REPORT.md` |
| Deleting small turning-point pairs before estimating P_cyc | Selects the slow libration itself, so W_c = 3 P_cyc erases it (2011 KT19: 40 kyr → 120 kyr smoothing); the second centre-turn scale is used instead | Oct 16 | `impl/W5-Q12.md` |
| Owner-grid frequency-coincidence veto | Fires on about half the sample by chance, real 1162/1362 included | Oct 17 | `impl/W12-REPORT.md` |
| Lifting caps for an implausible strength (R-IS, R-SP-IS) | The author changed the 2001 PG14 label; dropped | Oct 24 | DECISIONS D2 |
| A physical exit estimate for 1:1 in the "next check" (N8) | Too complex; the 4T ceiling is used for every resonance | Oct 24 | DECISIONS D18 |
| Tuning 2003 TX9's window lengths or line cap in CS12 | Helped one record; coefficients and independent halves were unstable | Oct 25 | `w23-work/design/cs12/REPORT.md` |
| G2a test 7 as `energy_level` (cycle h against the turning points' energy) | Replaced by `regime_periods`; and the first G2a version lifted TX9 only because the cut made the failed physics unmeasured | Oct 27 | `w24-work/g2a-fix`, `review-g2a` |
| G2a cut that hides a check failed on the whole record | Geometry must not override a failed measurement (W22 rule); `_action_step_guard` withholds such a cut | Oct 27 | same |

## Jumps, filters, sampling

| Approach | Why rejected | When | Source |
|---|---|---|---|
| Jump candidates from cycle-energy changes, merging by peak distance, timing by the steepest a | False jumps (2003 UG22 at 1.87 kyr; 3.5 kyr early; the 2021 BA horseshoe's own turn) | Oct 17 | `impl/W16-REPORT.md` |
| The libration-image TP threshold patch alone | Lost UG22's early episode; accepted only with moving-centre geometry | Oct 18 | `impl/W18-REPORT.md` |
| A universal 1-yr default output step | Still aliases 38/215 angles, too coarse for FAIR at P < 4.2 yr, ~70 GB/body for 10-Myr secular runs; `OUTPUT_STEP=auto` instead | Oct 15 | `research/SAMPLING.md` §9 |
| Lifting `short_period_aliased` by the a-scatter ratio (variant b) | Kept conservative: always capped; (b) only as an opt-in setting | Oct 20–23 | DECISIONS round D, Q1 |
| Certifying an alias from the turn count or rate the predictive unwrap produced | The unwrap inserted those turns; use the distance from resonance and branch evidence | Oct 15 | `reviews/PHYSICS.md` |

## Secular and ZLK

| Approach | Why rejected | When | Source |
|---|---|---|---|
| Amplitude floor in the free-omega gate | Unfixable: the prefit estimates e_free, not e_forced (off by ~100×); do not reintroduce an amplitude criterion | Aug | gate v4 decisions |
| tol = 2/T and unsigned rate matching in the gate | Covers everything at 500 kyr; flips signs (591986). Signed rates, strict tolerance | Aug | same |
| Ridge prefit, amplitude-selected representatives, measured/catalogue hybrid and mandatory FMFT in the gate | Nonlinear free ZLK motion leaks into near-degenerate fitted modes and corrupts the contamination estimate | Aug | same |
| "The cluster representative is a free convention" | False: 3752 and 591986 change verdict; it is a modelling assumption | Aug | `secular/free_elements.py` |
| e–i coherence as the ZLK decision | Its validity band T/(5 n_seg) (25 kyr at 500 kyr) is below the real ZLK periods (27–58 kyr) | Aug 16 | coherence notes |
| Reading the (k, h) eccentricity-vector plot as a ZLK verdict | Does not discriminate | Aug | same |
| Secular non-resonance from the free divisor alone ("−2 if \|divisor\| > 2/T") | 1222 Tina (the ν6 reference) has the same 2.18″/yr signature as 19289 | Oct 13 | pairs run notes |
| Broad order-≤4 secular combination scans for status | The hits were near-commensurabilities with g1/g2/s1/s2, which the model gets wrong without GR | Oct 13 | literature list |
| Planets' osculating ϖ and Ω in secular angles | Wrong for every s_j term (see [physics-notes.md](physics-notes.md)) | Oct 12 | the Oct 12 fix |
| ZLK centre comparison modulo 180° and keeping 162474 resonant across its island switch | The accepted reading is transient over the whole record | Oct 17 | W16 |

## Performance and engineering

| Approach | Why rejected | When | Source |
|---|---|---|---|
| Session-wide filter memo | RSS 1280 → 2084 MB on 88710_c3; per-record memo (`record_memo`) instead | Oct 26 | `w23-work/design/perf/REPORT.md` |
| Looser co-orbital retabulation (e 0.01, i 1°) | Statuses unchanged, but \|ΔK/K_th\| > 1e-3 on 211 rows, \|ΔR²\| > 1e-3 on 152, checks flip both ways; keep 0.005 / 0.5° | Oct 26 | `w23-work/impl/s6/REPORT.md` |
| Changing tolerances, detrend algebra or solvers under a "byte-identical optimisation" | Such a change is not an optimisation; it needs its own benchmark run | Oct 26 | perf report |
| `Verdict.legacy`, a nested parameter object, mandatory calibration (engineering review proposals) | Legacy code removed instead; `AnalysisSettings` through `SimulationConfig`; calibration reported only | Oct 15–16 | `reviews/ENGINEERING.md`, W3 |
| Independent per-element Gaussian clones (Keplerian or cometary diagonal) | The cometary diagonal gets σ_a wrong by 4–889×; sample the full covariance in the fit's own coordinates | Oct 27 | `docs/clones.md`, `w24-work/clones-verify` |
| Linear propagation of an NEA covariance (`.eq1`) over years | Inflates σ_λ by hundreds through encounters; start weak NEAs at the fit epoch (`.eq0`) | Oct 27 | same |
