# Free elements and the free-omega gate

An osculating element vector is the sum of a **forced** part, imposed by the planets and
turning at *their* frequencies, and a **free** part carrying the object's own secular
motion. Splitting the two answers a question the osculating angle cannot: does ω really
librate, or does it only look bounded because the forced eccentricity offsets the cloud?

For a Lidov-Kozai resonance this is part of the decision: the free-ω gate below is a step of the ZLK analysis, and
its outcome enters the status ([Secular and Lidov-Kozai](physics/secular-zlk.md#zlk-free-omega)). The cross spectra decide nothing
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
| a co-orbital of a planet whose own modes are not fitted: $\lvert\langle a\rangle/a_P - 1\rvert < k\,(m_P/3M_\odot)^{1/3}$ (`ZLK_COORBITAL_HILL_FACTOR` $k = 3$: 0.030 for the Earth), $P$ = Mercury … Mars | `unavailable` (`zlk_gate_detail` = `coorbital_unfitted_forcing`) |
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

The status decision (`analysis.secular.decide.decide_status`) turns the outcome into the status: on a librating ω,
*confirmed* (or `disabled`) keeps `resonant`, *kinematic* gives `non_resonant` / `kinematic_libration` and any
other outcome `uncertain` / `gate_unavailable`; on episodes, a drift or too few cycles only *kinematic* with
$\rho_{\max} < 1$ changes it ($\rho_{\max}$ = the largest ratio of the free to the forced radius, each a rolling
median over about one libration cycle, `FreeElements.rho_local_max`). The table and $\rho_{\max}$:
[Secular and Lidov-Kozai](physics/secular-zlk.md#zlk-free-omega). For screening, `uncertain` means "re-check on a
longer baseline", so the `candidates` and `extended` save/plot modes keep it; `extend_to_years` says how long
(`short_window`: the missing cycles; `planetary_residual`: the baseline that resolves the residual's distance to the
planetary rate, `zlk_gate_d_min_arcsec_yr`), and `undefined_phase` or `unavailable` asks for `review`
([How long to extend](physics/confidence.md#actions)). The gate itself has no amplitude test
(below).

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
a planetary fundamental. For a Jupiter Trojan that vector is Jupiter's, g5 + g6, rotated by
±60° for L4/L5 (Milani 1993; Vinogradova 2015), and the split fits it: on 591986 the fitted
g5 amplitude is 1.02 × Jupiter's own g5 term (g6: 1.12 ×) and the forced direction swings ±arcsin(|c₆|/|c₅|)
= ±22.8° at the g6 − g5 beat, which is the slow wave of the osculating ω's centre. So the
kinematic outcome stands (Vinogradova 2024 finds no Trojan with a librating ω); the tadpole
sidebands are not modelled, and the free rates there are not proper elements.

A co-orbital of **Mercury to Mars** is different: those planets' own modes (g1–g4, s1–s4)
are not candidates of the split, so its forced term is simply not subtracted. Such a body
(inside `ZLK_COORBITAL_HILL_FACTOR` Hill radii of the planet's axis) gets the outcome
`unavailable`, detail `coorbital_unfitted_forcing`: a libration becomes `uncertain` /
`gate_unavailable`, an episodic row keeps its status (2020 PP1, an Earth co-orbital, stays
`transient`). Never `not_applicable`, which would keep the geometric status of a libration.

## Beyond the gate

Columns of a ZLK row that change no status (`analysis.zlk.diagnostics`, `analysis.secular.slow`):

- **The e–i coupling** at the decisive period $P$ ($P_{\rm cyc}$; for a circulating ω the
  $\cos 2\omega$ harmonic $\pi/|\dot\omega|$), for every row whatever its status. One sinusoid
  of $e$ and one of $i$ (from the invariable plane by an exact rotation; the tilt
  $\min(i, 180° - i)$ of a retrograde orbit; an orbit that crosses 90° is `not_measured`) on the
  record less one period at each edge, and $e$, $i$ band-passed to $[(1 \mp 0.2)/P]$ for their
  correlation $r$. Two laws predict $\delta i/\delta e$: constant $\sqrt{1-e^2}\cos i$ (ZLK,
  $-e\cot i/(1-e^2)$, antiphase) and constant $\sqrt{1-e^2}(1-\cos i)$ (the momentum conjugate to Ω
  when the forcing acts on ϖ only, as a forced-eccentricity beat does: $+e\tan(i/2)/(1-e^2)$, in
  phase). `zlk_ei_coupling` is `zlk` or `forced_beat` when the phase is within
  `ZLK_EI_PHASE_TOL_DEG` of the law's sign, $r$ has that sign with $|r| \ge$ `ZLK_EI_MIN_BAND_R`
  and $|\delta i|/|\delta e|$ is within a factor `ZLK_EI_AMPLITUDE_FACTOR` of the law; `none` when
  every number is finite and neither law matches. A band the sampling cannot hold (above the Nyquist
  frequency: `undersampled`), one the filter cannot pass (`band_unmeasured`) or a law that is not
  finite (`law_undefined`) is `not_measured` (`zlk_ei_reason`), never `none`.
  The librators 3040, 162474, 4257, 1373, 1981: `zlk`, phase 177–182°, $r \approx -0.995$, 1.0–1.6 × the
  law. 591986's 6-kyr cycle: `forced_beat`, phase 328°, $\delta i$ 1.2 × the beat law. **Coupling is
  not libration**: 15527 circulates and shows `zlk` coupling at its 2.05-kyr harmonic.
- **Attribution** of a kinematic libration: the fitted mode with the largest $|c_j|$ (named with the
  modes the baseline merged into it) holds ϖ, provided the free circle stays off the origin
  ($\rho_{\max} < 1$, or the strict pin $\max_t|z_{\rm free}| + \sum_{k \ne j}|c_k| < |c_j|$: the
  largest free radius of the record, not its median, since the bound must hold at every sample), the predicted beat
  $2\pi/|b_\varpi - g_j|$ is $P_{\rm cyc}$ within a Rayleigh width, and the free node follows the
  mode ($|b_\Omega - g_j|\,T$ under one turn). 591986: g5 (g5+g7+g8+const), $e_{\rm free}/|c_5| = 0.25$,
  beat 5.96 kyr against $P_{\rm cyc}$ 6.0 kyr, node − g5 = −0.17″/yr (0.06 turn). Otherwise "forcing
  not attributed" with the condition that failed. An episodic $\omega$ that the gate demoted inside its episodes
  (median local $\rho < 1$ in each, the free $\omega$ turning with each cycle) is attributed with
  `zlk_attribution_scope` `episodes`: the node need not follow the mode there (624 Hektor: node − g5 =
  −17.2″/yr, 6.6 turns of the centre $\varpi_5 - \Omega$, which keeps the librations apart). The per-mode amplitudes
  are persisted in `{body}-omega-free.csv`; a file written before them restores no attribution.
- **Node lock** (`zlk_node_lock`, a flag no status reads): the osculating node against the longitude of the fitted
  mode with the nearest rate, $\varpi_j(t) = g_j t + \arg c_j$. Locked when $|b_\Omega - g_j| < 1296000/T$ (one
  Rayleigh width), $\Omega - \varpi_j$ stays within a turn, and, low-passed at the node's longest forced wave
  ($2\pi/|b_\Omega - s_k|$) inside the filter's edge trim, it closes at least `ZLK_NODE_LOCK_MIN_CYCLES` full cycles
  ((turning points − 1)/2): a drift or a slow circulation within the resolution closes none. 591986 over 4 Myr:
  g5, 50–130°, 2 cycles of 1.4 Myr; over 500 kyr the rate matches and $\Omega - \varpi_5$ stays within 52–91°, but
  no cycle closes: `unresolved` (one period of 1.4 Myr does not fit). Columns: the mode, the reason when none
  (`rate_mismatch`, `unbounded`, `unresolved`, …), the candidate, $b_\Omega - g_j$ and the resolution, the range,
  the period and the cycles.
- **The slow line of the centre** of a libration or a drift (`zlk_centre_*`): the strongest
  sinusoid of the centre between its smoothing $W_c$ and half the record, and its origin:
  `forced_beat:<gk>-<gj>` when its period is a beat $2\pi/|g_k - g_j|$ of two fitted forced modes
  with nonzero amplitudes within a Rayleigh width and the phase wave of the fitted forced vector,
  smoothed like the centre, explains it in shape ($r \ge$ `ZLK_CENTRE_MODEL_R_MIN`) and in size
  (`zlk_centre_model_gain`, its swing over the centre's, within a factor
  `ZLK_CENTRE_MODEL_AMP_FACTOR`: correlation alone is blind to size), `node` when the node's own
  wave does, `forced_direction` when $\arg z_{\rm forced} - \Omega$ does (the centre of $\omega$ itself when the
  forced vector holds $\varpi$: 591986 over 4 Myr, a 1.43-Myr wave of the node locked to g5 plus the g6−g5 beat,
  $r = 0.998$; the node alone gives 0.82), else `unexplained`. No per-mode amplitudes (an old file): no forced beat.
  591986: 53.8 kyr, ±21.6°, `forced_beat:g6-g5`, $r = 0.991$, gain 1.00, model ±22.8°. A modulated secular
  angle gets the same line (`sec_centre_*`), named `forced_wave:<mode>` after a forced wave of its
  own modes.
- **The scale guard**: a libration at the centre scale whose period is such a forced beat, whose
  forced phase wave explains the centre (as above), while $\rho_{\max} < 1$ never becomes the
  decisive libration (`sec_centre_rejected`); the gate's period stays the fast one (a clean synthetic
  591986 otherwise decided at 54 kyr instead of 5.95 kyr). It acts only where the gate does: a
  disabled gate, no split, or a co-orbital with unfitted forcing keeps the geometric scale.

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
`free_rho_min`, `free_rho_max` (the median free radius over the largest and the smallest smoothed forced amplitude;
descriptive, the status reads `zlk_gate_rho_max`), `free_omega_range_deg`, `free_omega_revolutions`,
`free_b_varpi`, `free_b_Omega`, `free_b_omega`, `free_drift_flag`, `free_mask_e`,
`free_mask_i`, `free_resolution`.

The gate: `free_gate` (the outcome), `free_n_cycles` ($T/P_{\rm lib}$), `free_gate_matches` (the
planetary rates a residual could not be told apart from), and the same as the analysis read them: `zlk_gate`,
`zlk_gate_n_cycles` ($T/P_{\rm cyc}$: cycles of the osculating ω, not turns of the free one), `zlk_gate_min_cycles`,
`zlk_gate_period_yr`, `zlk_gate_matches`, `zlk_gate_detail` (`coorbital_unfitted_forcing`), `zlk_gate_rho_max` (the
local ratio the status reads), `zlk_gate_rho_global` (the former median ratio, descriptive). The check
`check_zlk_free_omega` is `pass` (confirmed), `fail` (kinematic) or `not_measured`. The columns of
[Beyond the gate](#beyond-the-gate) are listed in [Output](output.md).

Two notes on reading them. `free_e` / `free_e_forced` come from this fit, while the older
`e_free` / `e_forced` columns come from the (k, h) scatter in `cross_spectrum` and exist for
every resonance type — different estimators, expected to differ. And `free_rho` is
an annotation (the status reads only $\rho_{\max}$, [Secular and Lidov-Kozai](physics/secular-zlk.md#zlk-free-omega)): for a large-amplitude librator `e_forced` is inflated by the
same free signal the fit could not represent, so read it in the small-amplitude regime.

`free_drift_flag` marks a span under a full turn that nonetheless carries a reproducible
monotonic ramp — worth a longer integration, but not a demotion.

## Files and figures

`{body}-omega-free.csv` carries the series themselves — the free and forced amplitudes of
both vectors, the three unwrapped angles, and the two masks. Masked samples are NaN rather
than dropped, so the file stays aligned with its times column. The fitted complex amplitude
of every mode is in `amp_<e|i>_<mode>_re` / `_im` on the first row (its value at $t = 0$ of
the `times` column), so a restored run rebuilds the forced vectors with their phases.

Add `free_omega` to `PLOTS` (or `plots=[...]`) for three figures
([How to read them](reading-figures.md#free_omega-lidov-kozai)):

- `…-free-omega-drift.png` — ω accumulated over the baseline, free against osculating;
- `…-free-omega-portrait.png` — the Kozai diagram in free variables, free e against free ω;
- `…-free-omega-vector.png` — the free eccentricity vector, on the same axes as the osculating `ecc_vector` figure
  of the same body: side by side they show whether the ring bounding ω is the object's own or the forced offset.

No figure carries numbers. Read `free_gate` and the rates from summary.csv.

## Configuration

The options are listed in [Configuration](config.md#free-elements-and-the-free-gate).

## Using it directly

```python
--8<-- "free_elements_examples.py:gate"
```

`config` is `sim.config` of the run, `libration_period_yr` the period of the bounded ω in years.

In a run the gate is a step of `analyse_zlk` (`resonances/resonance/analysis/zlk/`), and
`analysis.secular.decide.decide_status` turns its outcome into the status (above).

Or on bare arrays, without a `Body`:

```python
--8<-- "free_elements_examples.py:arrays"
```

Times in years, angles in radians.
