<!-- Generated from docs/dry-run.md by docs/tools/build_skill.py (make skill). Edit the docs, not this file. -->

# Dry-run: check a run before integrating it

A run of hours or days can be spoiled by its output step (an aliased angle, a filter that cannot see a short
libration) or by its integrator (a fixed step through close encounters). The dry-run predicts this from the
bodies' elements, before anything is integrated. It changes no verdict. The rules it applies are those of the
analysis ([From the integration to the angle](physics/signal.md): alias, cutoff, shortest libration).

```python
import resonances

sim = resonances.Simulation(name='dry-run-demo', integration_years=100_000, output_step='auto')
sim.create_solar_system()
sim.add_body(463, '4J-2S-1', name='463')  # or resonances.find(...) / resonances.check(...)
report = sim.dry_run(level='static')  # elements only: milliseconds per body
report = sim.dry_run(level='pilot')  # a short fine integration per body: 1-3 s each
report.summary()  # tables and warnings
rows, bodies, run = report.rows, report.bodies, report.run  # DataFrame body x MMR, DataFrame of bodies, dict of the run
report.apply()  # set output_step/Nout and the recommended integrator/dt on the config
report.to_csv('out')  # dryrun_rows.csv, dryrun_bodies.csv, dryrun_run.json
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
| alias | branch $m = \mathrm{round}(\nu\Delta t/2\pi)$ and apparent rate $\nu - 2\pi m/\Delta t$ over `a_lo..a_hi`: `resolved`, `marginal`, `aliased_unambiguous` (one m ≠ 0: a true circulation) or `ambiguous`, by [the analysis' rule](physics/signal.md#what-the-output-step-can-resolve) | `turns_per_step`, `branches`, `verdict`, `apparent_deg_yr`, `expected_angle_status` |
| Nyquist half-width | $\delta a_N = a^{5/2}/(3\lvert k\rvert\Delta t)$, where $\lvert\nu\rvert\Delta t = \pi$ | `nyquist_halfwidth_au` |
| ghosts | the sampled angle stands still at $\nu(a_m) = 2\pi m/\Delta t$ and imitates a resonance there | `ghost_axis_au`, `nearest_ghost_au` |
| safe steps | plain unwrap ηπ/max\|ν\|; predictive unwrap η a^(5/2)/(3\|k\| δa_sp), η = `MMR_STEP_RESOLVED_DEG`/180° | `max_safe_step_plain`, `max_safe_step_pred`, `action` |
| perihelion jitter | \|p\| δe_sp / e > `MMR_JITTER_MAX_RAD` | `jitter`, `jitter_flag` |
| cutoff | the analysis' cutoff $P_c$ from the longest synodic period of the angle's planets and the shortest libration it allows, `MMR_LIBRATION_PERIOD_MIN_YR` (a co-orbital: a share of its tadpole period); independent of $\Delta t$ ([rule](physics/signal.md#the-cutoff-what-is-libration-and-what-is-noise)) | `cutoff_pred_yr`, `cutoff_binding`, `samples_per_cutoff`, `libration_floor_yr` |
| validity | $P_c \le 2\Delta t$: the analysis returns `invalid/undersampled` ([rule](physics/signal.md#can-the-record-be-filtered-at-all); Δt ≥ 37.5 yr, a Jupiter Trojan 18.5 yr); the resample step is the largest of `MMR_RESAMPLE_STEPS_YR` that gives `MMR_RESAMPLE_STEPS_PER_CYCLE` points per shortest libration within the alias bounds (Trojan 2.5 yr) | `step_valid_yr`, `expected_status`, `expected_reason`, `resample_step_yr`, `action` |
| short periods | the step folds the orbital or a synodic period of the angle's planets, or one of their harmonics up to `MMR_SHORT_PERIOD_HARMONICS` slower than the period floor: the analysis flags `short_period_aliased` and caps the confidence at medium, except on a plain circulation ([rule](physics/signal.md#short-periods-folded-by-the-step)). The row reads the analysis' own predicate (`sampling.short_period_trigger`) with P_orb(a₀), the analysis with P_orb(⟨a⟩); the pilot replans with the measured ⟨a⟩. The auto step avoids both cases | `step_short_period_yr`, `folded_slowest_yr`, `short_period_trigger` (nyquist, harmonic), `expected_flags` |
| shortest libration | P_floor / `MMR_RESAMPLE_STEPS_PER_CYCLE` (20 points per cycle: 7.5 yr; Trojan 3.69 yr) | `step_libration_yr` |
| baseline | (`LIBRATION_MIN_TURNING_POINTS`/2) × P_floor + 2 × `FILTER_EDGE_TRIM_CUTOFFS` × P_c | `baseline_min_yr`, `baseline_ok` |
| pilot only | leak of short-period lines into the filtered angle and the filtered ȧ at Δt (decimate, unwrap, filter, compare with the fine record): σ largest deviation, ȧ relative rms error ε; the step is accepted when 2 × leak_σ ≤ the TP prominence of the fine σ_f (a wave of half-amplitude L turns by up to 2L) and ε ≤ √(b/(1−b)) = 0.33, b = `CONFIDENCE_THIN_BAND` (a force law of R² = 1/(1+ε²) stays in the thin band); lines the filter passes, named \|p n_j − q n\| when they are near-commensurabilities | `leak_deg`, `leak_adot_rel`, `leak_tp_prominence_deg`, `pilot_step_ok`, `leak_slips`, `forced_terms` |

`action`: `none` (resolved), `predictive_unwrap` (the analysis' unwrap recovers the angle), `resample` (it cannot,
or the step is too coarse for the filter: integrate again with a smaller step), `drop` (only with `DRYRUN_DROP_ALIASED`: a guaranteed aliased circulation
more than `DRYRUN_DROP_GHOST_FACTOR` × δa_N from a*).

## Secular and Lidov-Kozai rows

Advisory and static (also at the pilot level): warnings only, no step and no verdict change. Table `long_period`
(file `dryrun_long_period.csv`), one row per secular or ZLK angle, with the analysis' own rules
([secular signal](physics/secular-zlk.md#signal-what-is-not-the-libration)):

| check | formula | columns |
|---|---|---|
| cutoff | $P_c$ of `choose_cutoff`: the great inequality, the shortest libration and, for a secular angle of two or more modes, its forced waves $2\pi/\lvert f - f_j\rvert$ with the static $g = A$, $s = -A$ (none for crossers and retrograde orbits: `forced_waves` = `unknown`, the analysis' cutoff may then be longer) | `cutoff_pred_yr`, `cutoff_binding`, `forced_waves`, `g_static_arcsec_yr` |
| baseline | N = (`LIBRATION_MIN_TURNING_POINTS`/2) × P_floor + 2 × `FILTER_EDGE_TRIM_CUTOFFS` × P_c, P_floor = max(the type's shortest libration, `FILTER_MIN_LIBRATION_CUTOFFS` × P_c): secular 1.5 × 200 + 2 × 100 = 500 kyr, ZLK 1.5 × 3748 + 2 × 1874 = 9370 yr at the defaults | `libration_floor_yr`, `floor_setting`, `baseline_min_yr` |
| free-ω gate (ZLK, gate on) | `FREE_GATE_MIN_CYCLES` × τ, τ the quadrupole ZLK time $1/\sum_j \tau_j^{-1}$, $\tau_j = \frac{2}{3\pi}\frac{M_\odot}{m_j}\frac{P_j^2}{P}(1-e_j^2)^{3/2}$ over the planets outside the orbit (Antognini 2015, eq. 27; 153 Hilda: τ = 3.8 kyr, 11.3 kyr for the gate). Optimistic: a libration lasts several τ | `zlk_time_yr`, `gate_min_yr` |
| record | span ≥ max(baseline, gate) | `required_yr`, `baseline_ok` |
| step | Δt < P_c/2, else the filter cannot be built (`uncertain / unusable_series`) | `step_valid_yr`, `step_ok` |

A 100-kyr run of a secular angle warns `span 100000 yr < 500000 yr …`: no libration can be confirmed there, and the
verdict will be `uncertain / baseline_needed` with a next check of 400 kyr ([How long to extend](physics/confidence.md#actions)).

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
