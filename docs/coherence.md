# Cross-Spectral Coherence

Periodograms answer "which periods are present in this series". Cross spectra answer
a different question: **do two series share a period, and with what phase shift?**

That is what distinguishes a real resonance from a coincidence:

- in a mean-motion resonance, the resonant angle σ and the semi-major axis share the
  libration frequency, with `a` a quarter period ahead (`da/dt ~ ∂R/∂σ`);
- in a Lidov-Kozai resonance, `e` and `i` share the ZLK-cycle frequency **in
  antiphase** — the body trades eccentricity for inclination at constant
  `H = sqrt(1-e²)·cos i`.

The second point is not decorative. The argument of pericentre ω can librate without
any Lidov-Kozai resonance: when the forced eccentricity vector dominates the free one,
the geometry alone keeps ω bounded. Asteroid 591986 is exactly this case — its
osculating ω librates, but its free ω circulates and `e`/`i` do not exchange.

> **This page no longer describes a status decision.** The e-i criterion is now
> corroboration (`confirmed_ei`); what demotes 591986 is the free-omega gate in
> [free elements](free-elements.md). The reason is baselines — see
> [below](#why-it-no-longer-decides-the-status).

## How it is measured

Welch's method on **non-overlapping** segments, via `scipy.signal.coherence` and
`scipy.signal.csd` with identical parameters.

Non-overlapping is deliberate. The significance threshold

```
gamma2_crit = 1 - alpha ** (1 / (n_seg - 1))
```

is exact only for statistically independent segments. Measured false-alarm rate on
white noise, `alpha = 0.01`:

| segmentation | `gamma2_crit` | measured fraction of significant bins |
|---|---|---|
| non-overlapping | 0.785 (n=4) | 0.0100 |
| 50% overlap | 0.785 (n=4) | 0.0001 |

With overlap the same threshold is roughly a hundred times too conservative.

### Validity limits

```
P_min = 4 * dt              safety margin over Nyquist
P_max = T / (5 * n_seg)     at least 5 cycles inside every segment
```

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

Contiguous runs of significant bins are grouped into *lines*. Each line is
represented by its strongest bin (largest `|Pxy|`, not largest coherence — a wide
coherent band's most coherent bin is often not the one carrying the power), and lines
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
| `mmr` | `sigma-a` | 90° ± 30° | full validity band |
| `secular` | `sigma-e` | — | periods > 10 kyr |

Phase sign convention: **positive means the second series leads the first.**

Override per resonance type when creating the simulation; types you do not mention
keep their presets:

```python
sim = resonances.check(
    463, '4J-2S-1',
    coherence_pairs={'mmr': ['sigma-a', 'sigma-e', 'sigma-i']},
    coherence_phase_targets={'mmr': {'sigma-a': (90.0, 20.0)}},
)
```

## Why phase alone is not enough

A cross-phase is read off a **single frequency bin**, and a bin lies when several close
lines blend into it. On 591986 the `e`–`i` line near 54 kyr is really a triplet
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
same thing in all nine: **e and i never actually trade.** So the gate requires both.

## The e-i exchange flag

Applied in `Simulation.identify_librations()` after classification. It **changes no
status** — it records `confirmed_ei`, corroboration that eccentricity and inclination
really trade:

| situation | `confirmed_ei` | `zlk_check` |
|---|---|---|
| antiphase line **and** e/i anticorrelate | `True` | `passed` |
| no antiphase line | `False` | `failed` |
| antiphase line, but no exchange | `False` | `weak_exchange` |
| baseline too short to measure | empty | `short_baseline` |
| `ZLK_COHERENCE_CHECK=False` | empty | `disabled` |

`short_baseline` is the important distinction: a check that could not run must not be read
as a negative result, so the reason is recorded and a researcher can filter on the column
instead of guessing.

### Why it no longer decides the status

Until this version the check demoted a librating Lidov-Kozai angle from 2 to −2 on its
own. Baselines are what took that away. The validity band tops out at `T / (5 · n_seg)`,
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

```python
e_forced = hypot(mean(k), mean(h))
e_free   = sqrt(var(k) + var(h))
```

Note the **sum** of the two variances. A free vector of radius `R` sweeping the circle
uniformly puts `R²/2` into each coordinate, so summing recovers `R`; averaging instead
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
(and, like `segments.csv`, written only when `save_summary` is on):

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

`amp_ratio` is a useful independent check. Conservation of `H` predicts
`δi/δe = e·cos i / ((1-e²)·sin i)`; a line whose measured ratio is far from that is
shared but not driven by the Lidov-Kozai mechanism.

## Plots

Add to `plots`:

```python
sim = resonances.check(
    162474, 'lidov-kozai',
    plots=['evolution', 'ecc_vector', 'cross_spectrum'],
    plot='all',
)
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
resonances.EccentricityVectorPlotter.from_body(body, resonance, sim).plot().save('kh.png')

analysis = body.coherence['LK']['e-i']
resonances.CrossSpectrumPlotter.from_analysis(analysis, body_name='162474', resonance_key='LK').plot().save('coh.png')
```

## Using the primitives directly

```python
import numpy as np
import resonances

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

| Parameter | Environment variable | Default | Description |
|---|---|---|---|
| `coherence_enabled` | `COHERENCE_ENABLED` | `True` | Master switch for the whole feature |
| `coherence_skip_non_resonant` | `COHERENCE_SKIP_NON_RESONANT` | `True` | Skip bodies with status 0, which dominate mass surveys |
| `coherence_alpha` | `COHERENCE_ALPHA` | `0.01` | Per-bin false-alarm probability |
| `coherence_n_segments` | `COHERENCE_N_SEGMENTS` | `4,8,16` | Segmentation levels |
| `coherence_window` | `COHERENCE_WINDOW` | `hann` | Welch window |
| `coherence_max_lines` | `COHERENCE_MAX_LINES` | `10` | Lines kept per pair per level in `coherence.csv` |
| `coherence_period_min` | `COHERENCE_PERIOD_MIN` | empty | Extra lower bound, years; empty uses the per-type preset |
| `coherence_period_max` | `COHERENCE_PERIOD_MAX` | empty | Extra upper bound, years |
| `coherence_pairs` | — | presets | kwarg-only, per-resonance-type mapping |
| `coherence_phase_targets` | — | presets | kwarg-only, per-resonance-type mapping |
| `zlk_coherence_check` | `ZLK_COHERENCE_CHECK` | `True` | Whether to measure the e-i exchange flag. It no longer changes any status |
| `zlk_min_anticorrelation` | `ZLK_MIN_ANTICORRELATION` | `0.5` | Minimum \|r\| of band-passed e vs i for the line to count as an exchange. **Provisional** |
