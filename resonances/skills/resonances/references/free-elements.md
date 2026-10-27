<!-- Generated from docs/free-elements.md by docs/tools/build_skill.py (make skill). Edit the docs, not this file. -->

# Free elements and the free-omega gate

An osculating element vector is the sum of a **forced** part, imposed by the planets and
turning at *their* frequencies, and a **free** part carrying the object's own secular
motion. Splitting the two answers a question the osculating angle cannot: does ω really
librate, or does it only look bounded because the forced eccentricity offsets the cloud?

For a Lidov-Kozai resonance this is part of the decision: the free-ω gate below is a step of the ZLK analysis, and
its outcome enters the status ([Secular and Lidov-Kozai](physics/secular-zlk.md)). The cross spectra decide nothing
([Cross-spectral coherence](coherence.md)).

## What is computed

Work happens in the non-singular variables (the same ones `secular/proper_angle.py`
builds), because vectors add and angles do not:

$$
z_e(t) = e\,e^{{\rm i}\varpi} \ \ (\text{driven by } g_5 \ldots g_8), \qquad
z_i(t) = \sin\tfrac{i}{2}\,e^{{\rm i}\Omega} \ \ (\text{driven by } s_5 \ldots s_8).
$$

The forced part is removed by a **complex least-squares fit at the known planetary
frequencies** from `resonances/data/const.py`. Fitting known frequencies rather than
searching for them is deliberate: a frequency analysis (NAFF/FMFT) of a real integration
splits the free mode into five or six neighbouring terms — on 4257 the spread is ±0.5 ″/yr
against a Rayleigh resolution of 0.13 — and any of those splinters can be mistaken for a
planetary line and subtracted by accident.

What is left gives

$$
\varpi_{\rm free} = \arg z_{e,\rm free}, \qquad \Omega_{\rm free} = \arg z_{i,\rm free}, \qquad
\omega_{\rm free} = \varpi_{\rm free} - \Omega_{\rm free},
$$

with their least-squares rates `b_varpi`, `b_Omega`, `b_omega` in ″/yr.

## Which modes can be fitted

Not all of them, and the rule is **pairwise**, not per-mode. The Rayleigh width is $1/T$ cycles/yr
($1\,296\,000/T$ ″/yr): two modes closer than that never drift a full turn apart over the window, so no
fit can tell them apart and least squares splits their amplitude arbitrarily. To be conservative, candidates
are grouped by single linkage at twice that width, $2/T$ (two relative turns over the window), and one member of each
group is fitted. The gate's tolerance (below) is one width.

At 500 kyr the two widths are 2.592 and 5.184 ″/yr; the grouping threshold 5.18 ″/yr and the candidate rates 0, 0.67, 3.09, 4.26, 28.25
have neighbour gaps 0.67, 2.42, 1.17, 23.99 — the first three chain together, the last one
breaks:

| baseline | z_e basis | z_i basis |
|---|---|---|
| 100 kyr | one line | one line |
| 300 kyr – 1 Myr | g5, g6 | const, s6 |
| 10 Myr | full | full |

### The representative is a modelling assumption

Which member of a cluster gets fitted is fixed by a documented priority — `g5 > g6 > g7 >
g8 > const` for the eccentricity vector, `const > s6 > s7 > s8` for the inclination one.
That order is not a free convention. Fitting a different member measurably changes the
verdict: on 3752 the gate's outcome moves from *confirmed* to *no verdict*, on 591986 from *kinematic* to *no verdict*. The order encodes what actually
dominates the forcing — Jupiter's g5 for a main-belt eccentricity vector, and the constant
offset between the ecliptic and the invariable plane (the s5 mode, zero by construction)
for the slow inclination modes — and it reproduces the {g5, g6} / {const, s6} basis that
was validated independently against 10 Myr integrations.

What makes it safe rather than merely plausible: **no choice of representative ever turns
a non-librator into a libration**. Errors move the answer towards withholding a verdict,
never towards inventing one. That is asserted directly in the tests.

## The mask

Where |z_free| passes near the origin the argument is undefined, and unwrapping through it
picks up half-turns that look like circulation. Samples below `mask_quantile` × median are
dropped before unwrapping. ϖ_free and Ω_free each live on their own mask; **ω_free lives on
the intersection**, so `|b_omega − (b_varpi − b_Omega)|` is a consistency check with a
tolerance, not an identity. In practice the masks remove 0–8% of a real series.

## The gate

The gate (`omega_free_gate.evaluate_gate`) reads the free elements and returns an outcome:

| condition | outcome |
|---|---|
| no free elements (fewer than 16 samples after decimation, $T \lesssim 8$ kyr, or the split failed) | `unavailable` |
| $T <$ `FREE_GATE_MIN_CYCLES` × $P_{\rm lib}$: the window never held the libration | `short_window` |
| more than half the samples masked: no usable argument | `undefined_phase` |
| $R < 360°$ and `b_varpi` at g5…g8 (or 0, see below) or `b_Omega` at s6…s8 | `planetary_residual` |
| $R < 360°$ otherwise | `confirmed` |
| $R \ge 360°$ and `b_varpi` at g5…g8, `b_Omega` at s6…s8, or `b_omega` at any $g_j - s_k$ ($j = 5…8$, $k = 5…8$, $s_5 = 0$) | `planetary_residual` |
| $R \ge 360°$ otherwise: the free ω winds | `kinematic` |
| `FREE_GATE_ENABLED=False` | `disabled` |
| the osculating ω is not bounded (it circulates): no confirmation asked | `not_applicable` |

"At" means a signed difference below one Rayleigh width, $1\,296\,000/T$ ″/yr. The difference frequencies
$g_j - s_k$ are tested on the winding branch only: $\omega = \varpi - \Omega$, so an unsubtracted term in either
vector shows up in $\omega$ at such a frequency even when neither `b_varpi` nor `b_Omega` matches a mode.
`not_applicable` is not a failed split; the geometric status stands.

$R$ is the span of $\omega_{\rm free}$ trimmed by half a percent at each end (a span is an extreme statistic, and one
surviving glitch ruins a strict maximum). The 360° threshold is topological and **must not be tightened**: inside a
mean-motion resonance the Lidov-Kozai cycle is asymmetric and its centre shifts away from 90°/270°, to 0°/180° for
Hildas and by ±60° for Trojans (Vinogradova 2024).

The status decision (`analysis.secular.decide.decide_status`) reads the outcome:

| angle | outcome | status, reason |
|---|---|---|
| ω librates over the record | `confirmed` (or `disabled`) | `resonant`, `dynamic_libration` |
| | `kinematic` | `non_resonant`, `kinematic_libration` (origin kinematic, `free_omega_winds`) |
| | any other | `uncertain`, `gate_unavailable` |
| episodic, drift, too few cycles | `kinematic` and $\rho_{\max} < 1$ (the median free circle stays inside the forced envelope) | `non_resonant`, `kinematic_libration` |
| | otherwise | the status of the geometry is kept |

`kinematic` and `gate_unavailable` mean different things and must never be merged in a statistic: the first is a
measurement (the libration was examined and found kinematic), the second is the absence of one. For screening,
`uncertain` means "re-check on a longer baseline", so it belongs with the candidates (the `candidates` and
`extended` save/plot modes keep it). The gate has no amplitude test of its own (below); the amplitude ratio $\rho_{\max}$ enters only the decision on an
angle that does not librate over the whole record. With $E_{\rm free} = \operatorname{median}_t |z_{e,\rm free}(t)|$
and $F(t)$ the rolling median of $|z_{e,\rm forced}(t)|$ over a tenth of the shortest beat period between the fitted
modes (no smoothing for a single line),

$$
\rho_{\min} = \frac{E_{\rm free}}{\max_t F(t)}, \qquad \rho_{\max} = \frac{E_{\rm free}}{\min_t F(t)} .
$$

$\rho_{\max} < 1$ means a circle of the median free radius stays inside the forced-amplitude envelope; it is not the
maximum of the instantaneous ratio $|z_{e,\rm free}|/|z_{e,\rm forced}|$.

### Why the same check stands on both branches

A Lidov-Kozai libration lives on the object's *own* rates, so it can only be confirmed when
ϖ_free and Ω_free turn at speeds that are not planetary. A residual sitting at a planetary
rate is either a defect of the subtraction or a genuine linear secular resonance (g near
some g_j, the ν5/ν6 family) where the forced/free split is undefined to begin with. Both
make a verdict unsafe, and telling them apart is not required.

Comparison is **signed** and the tolerance is one Rayleigh width, `1296000/T` ″/yr. Signed
because every s_j is retrograde: a prograde free node is not s8 however close the
magnitudes are. Zero is a target only for the eccentricity vector, and only when the
constant term was clustered away and so left in the residual — an unsubtracted constant is
exactly what makes an angle look confined.

The honest cost: an object whose own g or s sits within one Rayleigh width of a planetary
fundamental is withheld rather than confirmed. That is a real ambiguity, not a gap.

### No amplitude criterion in the gate

An earlier design compared the free eccentricity against an estimated floor of unsubtracted
forcing. That floor cannot be measured: estimating how much forcing sits at a frequency the
baseline cannot resolve requires resolving it. Measured on the reference set, a ridge
prefit of the candidate amplitudes returned |ĉ| of order `e_free` rather than `e_forced` —
two orders of magnitude too large — because the near-degenerate slow cluster amplifies the
projection of the un-modelled free signal. The floor then scaled with the very quantity it
was being compared against, and demoted 3040, 162474 and 3752 outright.

Frequency is the direct observable instead: unsubtracted forcing lives, by construction,
only at known rates.

## Where the method stops working

For a **co-orbital** the forced term is the perturber's own eccentricity vector rather than
a planetary fundamental — Jupiter's, rotated by ±60° for L4/L5 (Milani 1993; Vinogradova
2015) — and the tadpole libration spreads it into a comb of sidebands. The g5…g8 basis
cannot represent that, so for 591986 the kinematic outcome is right at the
level of "the ω libration is not Lidov-Kozai" (Vinogradova 2024 finds no Trojan at all with
a librating ω), but its rates are not measurements. `free_rho = 0.23` is the symptom.
A Trojan-specific forced model is a known limitation, not a bug.

## Reference values

Measured on 500 kyr of mercurius integrations at 100 yr, the baseline the gate is
calibrated for:

| object | outcome | b_varpi | b_Omega | b_omega | R, deg | rho |
|---|---|---|---|---|---|---|
| 4257 | confirmed | −19.08 | −19.05 | −0.02 | 34 | 26.5 |
| 3040 | confirmed | −21.27 | −21.30 | 0.02 | 63 | 9.7 |
| 162474 | confirmed | −15.55 | −15.55 | 0.00 | 55 | 12.1 |
| 3752 | confirmed | −13.81 | −15.60 | 1.79 | 352 | 6.4 |
| 591986 | kinematic | 221.62 | 4.09 | 217.53 | 29896 | 0.23 |

The other five reference objects (373136, 797204, 88899, 555527, 10563) do not librate over the whole 500 kyr record;
there the gate changes the status only on a kinematic outcome with $\rho_{\max} < 1$.

## How long an integration is needed

| baseline | librating and confirmed | 591986 | false positives |
|---|---|---|---|
| 300 kyr | 4257, 3040, 162474, 3752 | kinematic | 88899 librates, gate: no verdict |
| **500 kyr** | **4257, 3040, 162474, 3752** | **kinematic** | **none** |
| 1 Myr | 4257, 3040, 162474 | kinematic | none |
| 10 Myr | 4257 | kinematic | none |

Longer is not better here. Most of these objects lose the lock well before 10 Myr — 3752 at
850 kyr, 162474 at 1.33 Myr, 3040 at 2.83 Myr — so a long baseline answers "does it stay?"
rather than "is it in resonance?". 500 kyr recovers librators that 10 Myr loses while
keeping every false positive out.

Two things a 500 kyr run does **not** buy. The rates `b_varpi`, `b_Omega` cannot be quoted
as numbers: the Rayleigh resolution there is 2.59 ″/yr, and the Lidov-Kozai signature
g − s = 0.09 ″/yr on 4257 sits thirty times below it. The verdict is safe because it comes
from the winding, not the rate; for numbers use 2 Myr (0.65 ″/yr) or 10 Myr (0.13). And the
calibration assumes Lidov-Kozai cycles of 27–58 kyr, as in this set — the rule of thumb is
$T \ge \max(500\ {\rm kyr},\ 10\,P_{\rm ZLK})$, which in practice means a short run to measure the period
and an extension if it turns out long.

## Columns in summary.csv

Written for Lidov-Kozai rows only, since ω = ϖ − Ω is the angle they describe.

The decomposition: `free_basis_e`, `free_basis_i`, `free_e`, `free_e_forced`, `free_rho`,
`free_rho_min`, `free_rho_max`, `free_omega_range_deg`, `free_omega_revolutions`,
`free_b_varpi`, `free_b_Omega`, `free_b_omega`, `free_drift_flag`, `free_mask_e`,
`free_mask_i`, `free_resolution`.

The gate: `free_gate` (the outcome), `free_n_cycles` ($T/P_{\rm lib}$), `free_gate_matches` (the
planetary rates a residual could not be told apart from), and the same as the analysis read them: `zlk_gate`,
`zlk_gate_n_cycles`, `zlk_gate_min_cycles`, `zlk_gate_period_yr`, `zlk_gate_matches`, `zlk_gate_rho_max`. The check
`check_zlk_free_omega` is `pass` (confirmed), `fail` (kinematic) or `not_measured`.

Two notes on reading them. `free_e` / `free_e_forced` come from this fit, while the older
`e_free` / `e_forced` columns come from the (k, h) scatter in `cross_spectrum` and exist for
every resonance type — different estimators, expected to differ. And `free_rho` is
an annotation (the status reads $\rho_{\max}$ only as above): for a large-amplitude librator `e_forced` is inflated by the
same free signal the fit could not represent, so read it in the small-amplitude regime.

`free_drift_flag` marks a span under a full turn that nonetheless carries a reproducible
monotonic ramp — worth a longer integration, but not a demotion.

## Files and figures

`{body}-omega-free.csv` carries the series themselves — the free and forced amplitudes of
both vectors, the three unwrapped angles, and the two masks. Masked samples are NaN rather
than dropped, so the file stays aligned with its times column.

Add `free_omega` to `PLOTS` (or `plots=[...]`) for three figures:

- `…-free-omega-drift.png` — ω accumulated over the baseline, free against osculating. A
  confined band is libration, a ramp is circulation at g − s. Both curves start at zero by
  construction, so the total drift reads as the distance from the bottom-left corner. The
  osculating curve is drawn wide and underneath, so it stays visible as a grey halo where
  the two coincide — which they usually do.
- `…-free-omega-portrait.png` — the Kozai diagram in free variables, free e against free ω,
  coloured by time.
- `…-free-omega-vector.png` — the free eccentricity vector, on **exactly the same axes** as
  the osculating `ecc_vector` figure of the same body. Side by side the two answer what the
  osculating plane alone cannot: whether the ring bounding ω is the object's own or the
  forced offset.

No figure carries numbers. Read `free_gate` and the rates from summary.csv.

## Configuration

The options are listed in [Configuration](config.md#free-elements-and-the-free-gate).

## Using it directly

```python
from resonances.resonance.omega_free_gate import evaluate_gate  # noqa: E402

elements = body.build_free_elements()
gate = evaluate_gate(elements, libration_period_yr, config)  # an outcome: confirmed, kinematic, or no verdict
print(gate.outcome, elements.b_omega)
```

`config` is `sim.config` of the run, `libration_period_yr` the period of the bounded ω in years.

In a run the gate is a step of `analyse_zlk` (`resonances/resonance/analysis/zlk/`), and
`analysis.secular.decide.decide_status` turns its outcome into the status (above).

Or on bare arrays, without a `Body`:

```python
from resonances.secular.free_elements import free_elements  # noqa: E402

elements = free_elements(times_years, ecc, inc, Omega, omega)
```

Times in years, angles in radians.
