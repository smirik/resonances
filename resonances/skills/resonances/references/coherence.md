<!-- Generated from docs/coherence.md by docs/tools/build_skill.py (make skill). Edit the docs, not this file. -->

# Cross-spectral coherence

Periodograms answer "which periods are present in this series". Cross spectra answer
a different question: **do two series share a period, and with what phase shift?**

That is what distinguishes a real resonance from a coincidence:

- in a mean-motion resonance, the resonant angle $\sigma$ and the semi-major axis share the
  libration frequency, a quarter period apart. From $\dot\sigma = c\,(a - a_*)$ with $c = -3kn/(2a)$, a libration
  $\sigma = A\sin\omega t$ gives $a - a_* = (A\omega/c)\cos\omega t$: $a$ leads $\sigma$ by +90° when $c > 0$
  (the usual prograde angle, $k < 0$) and by −90° when $c < 0$. The preset target below is +90°, a convention for the
  usual sign, not a law ([The resonant pendulum](physics/pendulum.md));
- in a Lidov-Kozai resonance, $e$ and $i$ share the ZLK-cycle frequency **in antiphase**: the body trades
  eccentricity for inclination at constant $H = \sqrt{1-e^2}\cos i$.

The second point is not decorative. The argument of pericentre ω can librate without
any Lidov-Kozai resonance: when the forced eccentricity vector dominates the free one,
the geometry alone keeps ω bounded. Asteroid 591986 is exactly this case — its
osculating ω librates, but its free ω circulates and `e`/`i` do not exchange.

!!! note "No status depends on this page"
    The cross spectra are diagnostics. The e-i criterion gives the flag `confirmed_ei`; what makes 591986
    `non_resonant` is the free-ω gate ([Free elements](free-elements.md)). The reason is the baseline, see
    [below](#why-it-does-not-decide-the-status).

## How it is measured

Welch's method on **non-overlapping** segments, via `scipy.signal.coherence` and
`scipy.signal.csd` with identical parameters. Each of the $n_{\rm seg}$ segments is linearly detrended and windowed
(Hann by default), the two series are Fourier transformed ($X$, $Y$), and the spectra are averaged over segments:
$P_{xx} = \langle |X|^2\rangle$, $P_{yy} = \langle |Y|^2\rangle$, $P_{xy} = \langle X^* Y\rangle$. Then

$$
\gamma^2(f) = \frac{|P_{xy}|^2}{P_{xx}P_{yy}} \in [0, 1], \qquad \varphi(f) = \arg P_{xy}, \qquad
\text{amplitude ratio} = \sqrt{P_{yy}/P_{xx}} .
$$

$\gamma^2$ is the magnitude-squared coherence (1: the two series share that frequency with a fixed phase); a
positive $\varphi$ means the second series leads.

Non-overlapping is deliberate. The significance threshold

$$
\gamma^2_{\rm crit} = 1 - \alpha^{1/(n_{\rm seg} - 1)}
$$

is exact only for statistically independent segments. Measured false-alarm rate on
white noise, `alpha = 0.01`:

| segmentation | `gamma2_crit` | measured fraction of significant bins |
|---|---|---|
| non-overlapping | 0.785 (n=4) | 0.0100 |
| 50% overlap | 0.785 (n=4) | 0.0001 |

With overlap the same threshold is roughly a hundred times too conservative. $\alpha$ is a false-alarm probability
per frequency bin, not for the whole searched spectrum.

### Validity limits

$$
P_{\min} = 4\,\Delta t \quad\text{(a margin over Nyquist)}, \qquad
P_{\max} = \frac{T}{5\,n_{\rm seg}} \quad\text{(at least 5 cycles in every segment)}.
$$

Nothing outside `[P_min, P_max]` is reported. This is the constraint that usually
bites: to see a 50 kyr Lidov-Kozai cycle at `n_seg = 4` you need at least 1 Myr of
integration. A shorter run reports `short_baseline` rather than a verdict.

### Segmentation levels

`P_max` shrinks as `n_seg` grows, while the threshold gets easier. Rather than fixing
one segmentation, several levels are computed (`COHERENCE_N_SEGMENTS`, default
`4,8,16`) and the verdict is taken from the **coarsest level whose validity band is
non-empty** — the one reaching the longest periods with the strictest threshold.
Levels that cannot be computed (fewer than 512 samples per segment, empty band) are
reported as invalid rather than silently dropped.

### Lines, not bins

Contiguous runs of significant bins are grouped into *lines*. A line's period, $\gamma^2$ and amplitude ratio are
those of its strongest bin (largest `|Pxy|`, not largest coherence — a wide
coherent band's most coherent bin is often not the one carrying the power); its phase is the $|P_{xy}|$-weighted
circular mean over all its bins (spread in `phase_std`). Lines
are ranked by integrated cross-amplitude `sum(|Pxy|)`.

**The phase criterion is applied to the top-ranked line only.** Ranking matters: in
the 30 Myr integration of 591986 the `e`–`i` search band holds tens of significant
lines, one of which happens to sit at −166° and would satisfy a naive "is there any
antiphase line" test. The amplitude-dominant line sits at +125° at every segmentation
level, which is the correct answer.

## Phase criteria

A criterion is `(target_deg, tolerance_deg)` per pair, preset per resonance type:

| resonance type | pairs | phase target | search band |
|---|---|---|---|
| `lidov_kozai` | `e-i` | 180° ± 30° | periods > 10 kyr |
| `mmr` | `sigma-a` | 90° ± 30° (for $c > 0$, see above) | full validity band |
| `secular` | `sigma-e` | — | periods > 10 kyr |

Phase sign convention: **positive means the second series leads the first.**

Override per resonance type when creating the simulation; types you do not mention
keep their presets:

```python
import resonances

sim = resonances.check(
    463,
    '4J-2S-1',
    integration_years=2_000,  # 100_000 years for a real verdict
    coherence_pairs={'mmr': ['sigma-a', 'sigma-e', 'sigma-i']},
    coherence_phase_targets={'mmr': {'sigma-a': (90.0, 20.0)}},
)
```

## Why phase alone is not enough

A line's cross-phase is a weighted mean over a few adjacent bins, and those bins lie when several close
lines blend into them. On 591986 the `e`–`i` line near 54 kyr is really a triplet
(53.9 / 54.0 / 54.2 kyr). Truncating one 30 Myr integration to different lengths and
running the same code gives:

| window | decisive line | phase | band-passed r | broadband r |
|---|---|---|---|---|
| 2 Myr | 11.4 kyr | −2.9° | −0.25 | −0.05 |
| 3 Myr | 11.0 kyr | +29.5° | −0.26 | −0.02 |
| **5 Myr** | **54.3 kyr** | **+177.7°** | −0.15 | −0.01 |
| 7.5 Myr | 27.2 kyr | +134.0° | −0.38 | −0.00 |
| 10–30 Myr | 27–56 kyr | +120…+128° | −0.12…−0.18 | ±0.00 |

The phase says "antiphase" in exactly one window out of nine. The correlations say the
same thing in all nine: **e and i never actually trade.** So the flag requires both.

## The e-i exchange flag

Measured after the verdict, for Lidov-Kozai rows. It **changes no status**: it records
`confirmed_ei`, the evidence that eccentricity and inclination really trade, as the check `check_zlk_ei`:

| situation | `confirmed_ei` | `zlk_check` |
|---|---|---|
| antiphase line **and** e/i anticorrelate | `True` | `passed` |
| no antiphase line | `False` | `failed` |
| antiphase line, but no exchange | `False` | `weak_exchange` |
| baseline too short to measure | empty | `short_baseline` |
| `ZLK_COHERENCE_CHECK=False` | empty | `disabled` |
| no `e-i` pair, or no phase target for it | empty | `no_target` |

`no_target` appears when `coherence_pairs` or the phase targets of `lidov_kozai` were overridden without `e-i`;
restoring `coherence_pairs['lidov_kozai'] = ['e-i']` with its target enables the check.

The 180° target is a **prograde** diagnostic. At constant $H$ a retrograde orbit ($\cos i < 0$) has $e$ and $i$
varying *in phase*, and the anticorrelation test does not confirm that branch. The exchange also assumes $i$ measured
from the perturber's plane and $H$ conserved (quadrupole approximation); the package uses the ecliptic $i$ of an
N-body run, so the comparison is approximate. No status depends on it.

`short_baseline` is the important distinction: a check that could not run must not be read
as a negative result, so the reason is recorded and a researcher can filter on the column
instead of guessing.

### Why it does not decide the status

Earlier versions demoted a librating Lidov-Kozai angle on this check alone. The baseline is
what took that away. The validity band tops out at `T / (5 · n_seg)`,
which is **25 kyr on a 500 kyr run at four segments**, while the real Lidov-Kozai lines in
the reference set sit at **27–58 kyr**. On the short integrations this package now targets,
the check would be judging on a line outside its own validity band — it happened to agree
with the free-omega gate on all ten reference objects at 500 kyr, but by luck rather than
by measurement.

The status decision moved to the forced/free split, which has no such band: see
[free elements](free-elements.md). The cross spectra, `coherence.csv` and the figures are
unchanged, and `confirmed_ei` sits next to the status as independent evidence.

> **The `ZLK_MIN_ANTICORRELATION = 0.5` default is provisional.** It sits between the
> only two objects measured so far — 591986 never exceeds 0.38 and is not in resonance,
> 162474 reaches 0.79 and is — but two objects do not calibrate a threshold. The raw
> `r_band` and `r_broadband` values are written to `coherence.csv` for every pair, so a
> larger sample can be collected and the threshold revisited from data.

The correlation condition is deliberately ZLK-only: it encodes `cos(180°) = −1`, and would
be meaningless against the MMR target of 90°, where quadrature puts the correlation at zero
by construction. For MMR and secular resonances the cross spectrum is pure diagnostics.

## Forced and free eccentricity

In the `(k, h) = (e cos ω, e sin ω)` plane the forced term is the offset of the cloud
from the origin and the free term is the radius it traces:

$$
e_{\rm forced} = \sqrt{\langle k\rangle^2 + \langle h\rangle^2}, \qquad
e_{\rm free} = \sqrt{\operatorname{var} k + \operatorname{var} h}.
$$

Note the **sum** of the two variances. A free vector of radius `R` sweeping the circle
uniformly puts $R^2/2$ into each coordinate, so summing recovers `R`; averaging instead
would under-report by a factor √2.

Both go into `summary.csv`. On the 30 Myr integration of 591986 they come out as
`e_forced = 0.0392` and `e_free = 0.0308`, the latter matching the AstDyS `tro.syn`
proper eccentricity `e_p = 0.0308`.

## Output

**`summary.csv`** gains, per body and resonance:
`e_forced`, `e_free`, `confirmed_ei` (True/False/empty), `zlk_coherent` (the same value
under its older name), `zlk_check`, `zlk_line_period`, `zlk_line_phase`,
`zlk_line_gamma2`, `zlk_line_r_band`.

**`coherence.csv`** — one row per detected line, appended per run like `summary.csv`
(written only when `save_summary` is on):

| column | meaning |
|---|---|
| `body`, `resonance`, `pair` | what was analysed |
| `n_seg`, `rank`, `decisive` | segmentation level, rank by cross-amplitude, whether this is the line the criterion used |
| `period`, `gamma2`, `gamma2_crit` | the line and the threshold it cleared |
| `phase_deg`, `phase_std` | cross-phase and its spread across the line's bins |
| `amp_ratio` | `sqrt(Pyy/Pxx)` — how much the second series moves per unit of the first |
| `r_band` | Pearson r of the two series band-passed around this line; filled for the strongest line only |
| `r_broadband` | Pearson r of the two raw series, unfiltered |
| `cross_amplitude`, `n_bins` | integrated `\|Pxy\|` and width of the line |
| `period_min`, `period_max` | validity band of that level |
| `phase_target`, `phase_tolerance` | criterion applied, if any |
| `n_levels_valid`, `n_levels_passed`, `n_levels_agreeing` | how many levels could be computed, how many satisfied the criterion, how many reproduce the decisive line |

`amp_ratio` is a useful independent check. Conservation of $H = \sqrt{1-e^2}\cos i$ gives the signed derivative

$$
\frac{{\rm d}i}{{\rm d}e} = -\frac{e\cos i}{(1 - e^2)\sin i},
$$

so the amplitude ratio $\lvert\delta i\rvert/\lvert\delta e\rvert$ (i in radians) should be the absolute value of
the right-hand side (with $i$ from the perturber's plane; for ecliptic elements this is approximate). The sign is
negative only for a prograde orbit. `amp_ratio` is non-negative, an amplitude ratio, not the signed derivative. A line whose ratio is
far from it is shared but not driven by the Lidov-Kozai mechanism.

## Plots

Add to `plots`:

```python
sim = resonances.check(
    162474,
    'lidov-kozai',
    integration_years=20_000,  # 500 kyr or more for the Lidov-Kozai periods
    plots=['evolution', 'ecc_vector', 'cross_spectrum'],
    plot='all',
)
sim.run()
```

- `ecc_vector` → `{body}-ecc-vector.png` — the `(k, h)` plane coloured by time, in a
  square window centred on the origin. The offset of the cloud from the origin is
  `e_forced`, its radius `e_free`; a cloud that does not enclose the origin keeps ω
  bounded whatever the dynamics. It explains *why*
  ω librates, but does not say whether the libration is a resonance: genuine librators
  and forced-geometry cases both show an offset cloud, and `e_free/e_forced` comes out
  below 1 for both (0.27–0.35 for 3040, 4257 and 162474, 0.79 for 591986).
- `cross_spectrum` → `{body}-{resonance}-coherence-{pair}.png` per pair — three panels
  read together:

  1. **coherence** — a stem per detected line over a faint full spectrum, one colour per
     segmentation level, each with its own dotted threshold, the strongest lines circled
     in red;
  2. **cross-phase** — one marker per detected line, sized by cross-amplitude, filled
     when the consensus is unanimous and hollow otherwise, over the target band;
  3. **band-passed series** — the two quantities filtered around the decisive line, over
     16 of its cycles. This is the panel that settles it: two curves in antiphase are an
     exchange, two curves drifting past each other are not.

  Everything is cropped to the search band — bins outside it cannot be claimed as
  significant, and the short-period forest only distracts. Period and time axes pick
  their own unit (yr / kyr / Myr), so the same figure reads at 1 kyr and at 50 Myr.

### What the shading means

The yellow band marks **where to look**: an interval that every level entitled to vote
clears — you can check it by eye, all the faint curves are above their own dotted
thresholds across it — *and* that carries one of the three strongest lines by
cross-amplitude.

Agreement alone is not selective enough to be worth a highlight. For the `e-i` pair it
mostly measures conservation of the Kozai integral `H = √(1−e²)·cos i`: when H holds, i
is a smooth monotone function of e, so the two are coherent at every frequency that
carries power. The share of the band where all levels agree tracks that almost exactly —
3040 78%, 4257 44%, 88899 24%, 10563 0%, in the same order as how well i is predicted
from e through H (R² = 0.86, 0.64, 0.06, 0). Highlighting all of it would say only "H is
conserved", which is not the question.

A level whose validity band does not reach a period **abstains** rather than objects: it
cannot see the line, so it must not block a consensus. That distinction is why agreement
across levels is reported but not used as a gate condition.

Both plotters are usable directly:

```python
body = sim.bodies[0]
resonance = body.lidov_kozai_resonances[0]  # key 'LK'
resonances.EccentricityVectorPlotter.from_body(body, resonance, sim).plot().save('kh.png')
resonances.CrossSpectrumPlotter.from_body(body, resonance, 'e-i').plot().save('coh.png')
```

## Using the primitives directly

```python
import numpy as np  # noqa: E402

times_yrs = np.asarray(sim.times) / (2 * np.pi)  # a uniform grid, in years
e, inc = body.ecc, body.inc  # a, e and i are not unwrapped; angles must be (np.unwrap)

result = resonances.cross_spectrum(times_yrs, e, inc, n_seg=4, alpha=0.01, period_min=10_000)
if result.valid and result.top_line:
    line = result.top_line
    print(line.period, line.gamma2, line.phase_deg, line.amp_ratio)
    print(resonances.phase_matches(line.phase_deg, 180.0, 30.0))
```

Angles must be `np.unwrap`ed before being passed in; `a`, `e` and `i` must not be. The
grid must be uniform — a non-uniform one returns `valid=False` with a reason rather
than a wrong number.

## Configuration

The options are listed in [Configuration](config.md#cross-spectral-coherence).
