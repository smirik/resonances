# Dry-run: check a run before integrating it

A run of hours or days can be spoiled by its output step (an aliased angle, a filter that cannot see a short
libration) or by its integrator (a fixed step through close encounters). The dry-run predicts this from the
bodies' elements, before anything is integrated. It changes no verdict. The rules it applies are those of the
analysis ([From the integration to the angle](physics/signal.md): alias, cutoff, shortest libration).

```python
--8<-- "dry_run_examples.py:dry_run"
```

Command line: `resonances dry-run <arguments of find, or of check with --resonances> [--pilot] [--years N]
[--step auto|YR] [--fair] [--out DIR]`. `resonances find` and `resonances check` run the static level before
integrating and write its warnings to the log; with `OUTPUT_STEP=auto` they set the step from it.

## What is checked (per MMR row)

Notation: $\sigma = \sum_j k_j\lambda_j + k\lambda + p\varpi + \sum_j p_j\varpi_j + v\Omega + \sum_j v_j\Omega_j$,
$\Delta t$ the output step, a in au, t in years (2π REBOUND units), rates in rad/yr. For a retrograde angle the body's
terms are $\lambda^* = M + \omega - \Omega$ and $\varpi^* = \omega - \Omega$ ([Resonance matrices](matrix.md)).
The static body precession is the Laplace-Lagrange free rate of a test particle (Murray & Dermott 1999, §7.9),
$\dot\varpi = A$, $\dot\Omega = -A$,

$$
A = \frac n4\sum_j \mu_j\,\alpha_j\bar\alpha_j\,b_{3/2}^{(1)}(\alpha_j), \qquad
b_s^{(m)}(\alpha) = \frac1\pi\int_0^{2\pi}\frac{\cos m\psi\,{\rm d}\psi}{(1 - 2\alpha\cos\psi + \alpha^2)^s},
$$

with $\mu_j = m_j/M_\odot$, $\alpha_j = \min(a, a_j)/\max(a, a_j)$, $\bar\alpha_j = 1$ for an inner planet and
$\alpha_j$ for an outer one. It is omitted (taken as 0) for planet-crossers and retrograde orbits.

| check | formula | columns |
|---|---|---|
| rate of σ | $\nu(a) = \sum_j k_j n_j + k\,n(a) + p\dot\varpi + \sum_j p_j g_j + v\dot\Omega + \sum_j v_j s_j$, $n = 2\pi a^{-3/2}$; $n_j$ of the package model (Murray & Dermott 1999, eq. 8.22); static $\dot\varpi = A$, $\dot\Omega = -A$ (above) | `rate_deg_yr`, `rate_min`, `rate_max` over `a_lo..a_hi` |
| alias | branch $m = \mathrm{round}(\nu\Delta t/2\pi)$, apparent rate $\nu - 2\pi m/\Delta t$; `resolved` if $\max_{a_{lo} \le a \le a_{hi}}\lvert\nu(a)\rvert\Delta t \le (\pi/180°)\,$`MMR_STEP_RESOLVED_DEG`, `marginal` (branch 0 throughout, above that limit), `aliased_unambiguous` (one m ≠ 0: a true circulation), `ambiguous` | `turns_per_step`, `branches`, `verdict`, `apparent_deg_yr`, `expected_angle_status` |
| Nyquist half-width | $\delta a_N = a^{5/2}/(3\lvert k\rvert\Delta t)$, where $\lvert\nu\rvert\Delta t = \pi$ | `nyquist_halfwidth_au` |
| ghosts | the sampled angle stands still at $\nu(a_m) = 2\pi m/\Delta t$ and imitates a resonance there | `ghost_axis_au`, `nearest_ghost_au` |
| safe steps | plain unwrap ηπ/max\|ν\|; predictive unwrap η a^(5/2)/(3\|k\| δa_sp), η = `MMR_STEP_RESOLVED_DEG`/180° | `max_safe_step_plain`, `max_safe_step_pred`, `action` |
| perihelion jitter | \|p\| δe_sp / e > `MMR_JITTER_MAX_RAD` | `jitter`, `jitter_flag` |
| cutoff | the analysis' rule $P_c = \max(f_s P_{\rm syn},\ P_{\rm floor}/f_l)$, $P_{\rm syn}$ the longest synodic period $2\pi/\lvert n - n_j\rvert$ ($2\pi/(n + n_j)$ for a retrograde body) of a planet in the angle, the co-orbital's own planet excluded (its synodic motion is the resonant signal; $P_{\rm syn} \to \infty$ at 1:1), with $f_s$ = `MMR_FILTER_SYNODIC_FACTOR`, $f_l$ = `FILTER_MIN_LIBRATION_CUTOFFS`, independent of $\Delta t$; P_floor = `MMR_LIBRATION_PERIOD_MIN_YR` (150 yr → 75 yr), or for a co-orbital `MMR_COORBITAL_PERIOD_FLOOR_FACTOR` × the tadpole period (Jupiter 73.9 yr → 36.9 yr) | `cutoff_pred_yr`, `cutoff_binding`, `samples_per_cutoff`, `libration_floor_yr` |
| validity | $P_c \le 2\Delta t$: the analysis returns `invalid/undersampled` (Δt ≥ 37.5 yr, a Jupiter Trojan 18.5 yr); resample at the largest of `MMR_RESAMPLE_STEPS_YR` ≤ P_floor / `MMR_RESAMPLE_STEPS_PER_CYCLE` within the alias bounds (Trojan 2.5 yr) | `step_valid_yr`, `expected_status`, `expected_reason`, `resample_step_yr`, `action` |
| short periods | Δt ≥ min(P_orb, P_syn of the angle's planets)/2, or below it a harmonic m ≤ `MMR_SHORT_PERIOD_HARMONICS` (2) of P_orb or P_syn folded slower than the period floor: the analysis flags `short_period_aliased` and caps the confidence at medium (not on a plain circulation, reason `circulation` or `alias` without forced-wave runs: `MMR_SHORT_PERIOD_CAP_CIRCULATION` is off, since a circulation reads no a-based check; the flag stays). The row reads the analysis' own predicate (`sampling.short_period_trigger`) on its own inputs: the static level takes P_orb(a₀), the analysis P_orb(⟨a⟩), so the two agree for equal P_orb, P_syn and floor (the pilot replans with the measured ⟨a⟩; a record whose ⟨a⟩ moves the fold across the floor can still differ). The auto step avoids both cases | `step_short_period_yr`, `folded_slowest_yr`, `short_period_trigger` (nyquist, harmonic), `expected_flags` |
| shortest libration | P_floor / `MMR_RESAMPLE_STEPS_PER_CYCLE` (20 points per cycle: 7.5 yr; Trojan 3.69 yr) | `step_libration_yr` |
| baseline | (`LIBRATION_MIN_TURNING_POINTS`/2) × P_floor + 2 × `FILTER_EDGE_TRIM_CUTOFFS` × P_c | `baseline_min_yr`, `baseline_ok` |
| pilot only | leak of short-period lines into the filtered angle and the filtered ȧ at Δt (decimate, unwrap, filter, compare with the fine record): σ largest deviation, ȧ relative rms error ε; the step is accepted when 2 × leak_σ ≤ the TP prominence of the fine σ_f (a wave of half-amplitude L turns by up to 2L) and ε ≤ √(b/(1−b)) = 0.33, b = `CONFIDENCE_THIN_BAND` (a force law of R² = 1/(1+ε²) stays in the thin band); lines the filter passes, named \|p n_j − q n\| when they are near-commensurabilities | `leak_deg`, `leak_adot_rel`, `leak_tp_prominence_deg`, `pilot_step_ok`, `leak_slips`, `forced_terms` |

`action`: `none` (resolved), `predictive_unwrap` (the analysis' unwrap recovers the angle), `resample` (it cannot,
or the step is too coarse for the filter: integrate again with a smaller step), `drop` (only with `DRYRUN_DROP_ALIASED`: a guaranteed aliased circulation
more than `DRYRUN_DROP_GHOST_FACTOR` × δa_N from a*).

## Per body

`class` (`main`, `nea` for q < `DRYRUN_NEA_Q`, `crosser` when [q, Q] overlaps a planet's [q_j − h R_H, Q_j + h R_H],
h = `DRYRUN_HILL_MARGIN`), `integrator_recommended` and `dt_recommended_yr`: ias15 for crossers and NEAs (adaptive
steps for encounters), otherwise SABA(10,6,4) with dt = min(run dt, P/`DRYRUN_STEPS_PER_ORBIT`,
τ_peri/`DRYRUN_STEPS_PER_PERI`), τ_peri = P√((1−e)³/(1+e)) the effective period at pericentre (Lu, Hernandez &
Rein 2024, eq. 33; the /10 rule is provisional). `fair_step_yr` = `DRYRUN_FAIR_FRACTION` × P; the pilot adds
`moid_<planet>` (a grid screen at the epoch: information, nodes precess), the measured ⟨a⟩, δa_sp, δe_sp, ϖ̇, n from λ
and its wall time. A crosser's a jumps at encounters, so its static alias verdict holds only for a₀ ± `DRYRUN_DA_STATIC`
(it says so).

## The run and `OUTPUT_STEP=auto`

Δt_auto = min(libration, short_period, filter, trend, FAIR, user) (short_period and filter are exclusive bounds), rounded down to `DRYRUN_AUTO_STEPS_YR`, then walked
down `DRYRUN_AUTO_STEPS_YR` to the first step at which no row folds a harmonic slower than its period floor
(`binding_rule` `short_period_harmonic` when it moved, `step_fold_free_yr` in the run; when no step of the grid is fold-free the rounded step stays and its harmonic rows keep `short_period_aliased`: P_orb 10.001 yr gives
short_period 5.0005 yr, but 5 yr folds 2/P_orb to 50 kyr; 2.5 yr folds nothing); Nout = ⌈span/Δt⌉ + 1 (at least 2), so the grid step never exceeds Δt.
`short_period` is the smallest min(P_orb, P_syn)/2 of the rows, `trend` the smallest predictive safe step, `FAIR`
counts only with `DRYRUN_FAIR=required` (or `--fair`), `user` is `DRYRUN_MAX_STEP_YR`. With the default settings
the short periods decide in the main belt (a = 2.5 au: 1 yr; a Hilda: 2.5 yr) and for NEAs (1999 UQ: 0.5 yr), the
libration bound for a Jupiter Trojan (2.5 yr). After a run, each record's own fold-free step is in
`smp_fold_free_step_yr` (the same walk for that row alone on `MMR_RESAMPLE_STEPS_YR`, see output.md), and
`action=resample` names it when the cap `short_period_aliased` holds the level down. `auto` is defined only for runs with MMRs; a purely secular
or ZLK run keeps Nout. A numeric `OUTPUT_STEP` sets Nout = ⌈span/step⌉ + 1; an explicit Nout overrides both. The default
step is unchanged (15.9 yr): a Hilda keeps P_c = 75 yr there and its libration, but every main-belt row is
`short_period_aliased` (confidence ≤ medium). The run warns of each limit the configured step is above; the pilot
lifts the short-period warning when it accepts the step for every flagged row (153 Hilda and the Trojans: leaks
0.1–0.7°, ε 0.1–0.2), and keeps it when not (62243 1M-4J-1: 2.4°, ε 5.9). A row below that step whose harmonic folds slower than
the floor and whose leak the pilot rejects adds the `short_period_harmonic` limit, with the fold-free step to use
instead. The analysis does not read the pilot: the cap stays on every such row except a plain circulation.

The run also gets the recommended integrator (ias15 if any body needs it; `mixed_integrators` then suggests separate
runs), the cost (`rows`, `disk_GB` if every body is saved, `ram_per_worker_GB`, `cpu_hours`; coefficients measured
in 2026-10, the pilot's own speed when present) and `step_spread`.

## Examples

- **1999 UQ `7E+2M-9`, ias15, 20 kyr, Nout 4000 (5.0012 yr).** Static: `aliased_unambiguous`, 0.994 turns per
  step, ghost 1.0941756 au, 1.2·10⁻⁴ au from a₀: the "libration" of the record was a stroboscope. The pilot puts
  ⟨a⟩ 2·10⁻⁶ au from the ghost, gives 71.98°/yr (3999 turns in 20 kyr, as the 1-yr rerun) and finds the real 97-yr
  forced wave 7n_E − 8n (9.6°). At 1 yr the angle is `resolved` (0.2 turns per step).
- **153 Hilda `3J-2`, SABA, 100 kyr.** Class `main`, SABA with dt = 58 d. Pilot: the libration line at 273 yr
  (the 1-yr record: 268 yr).
- **1865 Cerberus.** Crosses Venus, Earth and Mars: ias15; a SABA run is flagged.

On the 294 benchmark MMR records whose metadata has the initial elements, the static
verdict at each record's own step against the verdict measured on the record: 286 agree; every record that the
static level calls not resolved is not resolved; 3 records are `ambiguous` but statically `resolved` — all crossers
whose a moved by 0.03-0.5 au over 20-500 kyr (99942, 2000 YG4, 2020 PP1).
