<!-- Generated from docs/plots.md by docs/tools/build_skill.py (make skill). Edit the docs, not this file. -->

# Plots

Which figures a simulation draws is set by `plots` (default `['combined']`, the one-page verdict figure). Which rows
get them is set by `plot` (a filter on the verdict, see [Configuration](config.md#saving-and-plotting)) and `plot_always`.
The default `review` draws every row except a confident plain circulation: the `nonzero` rows and the doubtful
negatives (`non_resonant` at `medium` or `low` confidence, the review queue `review_required`). After the run the log
prints the rows by status × confidence and how many were drawn.

```python
sim = resonances.Simulation(
    plot='nonzero',
    plots=['evolution', 'combined'],
    plot_options={'style': 'screen'},  # optional, see below
)
```

| kind | resonances | file (`{body}-{key}-…`) | what it shows |
|------|------------|-------------------------|---------------|
| `combined` | MMR, secular, ZLK | `combined/…-combined` | the one-page verdict figure: time panels left, the verdict and every number of the decision right |
| `evolution` | all | `{body}-{key}.png` | angle, a, e, i against time, periodograms against period (`PLOT_CONFIG` preset; a low-passed by the body-level filter, `LIBRATION_FILTER_CUTOFF`) |
| `recurrence` | MMR | `-recurrence` | recurrence of the state $(\sigma, \dot\sigma)$ |
| `fair` | prograde two-body MMR | `-fair` | FAIR plane: mean anomaly against the longitude difference to the planet |
| `portrait` | MMR, secular, ZLK | `-portrait` | phase portrait: $\sigma$–$\dot\sigma$ (or $\sigma$–a) for an MMR, $(r\cos\sigma, r\sin\sigma)$ for a secular or ZLK angle |
| `angle` | secular, ZLK | `-angle` | $\sigma(t)$: osculating, filtered, turning points |
| `rate` | secular, ZLK | `-rate` | the rate of the filtered $\sigma$, with the divisor |
| `cycles` | MMR | `-cycles` | oscillation cycles of $\sigma_f$ and of a (the analysis' $a_f$, else the osculating a), stacked on one period |
| `ecc_vector` | all | `{body}-ecc-vector.png` | the $(k, h) = (e\cos\omega, e\sin\omega)$ plane |
| `cross_spectrum` | all | `-coherence-{pair}` | coherence and cross-phase ([Cross-spectral coherence](coherence.md)) |
| `free_omega` | Lidov-Kozai | `-free-omega-drift`, `-portrait`, `-vector` | the free argument of pericentre ([Free elements](free-elements.md)) |
| `clones` | all, a body with clones | `{body}-{key}-clones.png` | the ensemble: every clone's angle by status, status against initial a (with the catalog $a_{\rm res}$ of an MMR), the status shares with Wilson intervals (also drawn with `combined`; [Clones](clones.md)) |

An unknown name is an error; a kind is drawn only for the resonance types it lists. The image format is `image_type`
(`png` or `pdf`). Combined figures go to `plot_path/combined/`, the others to `plot_path/figures/`, both split by
`plot_layout` (`confidence/status` by default: `combined/high/resonant/…`; [Output data](output.md#layout)). None of
these figures feeds a status.

**What every panel, line and icon means: [How to read the figures](reading-figures.md)**, with an example of each
kind. One example for every status: [Gallery](physics/gallery.md).

![The combined figure of (1921) Pala in the 2:1 resonance with Jupiter](https://smirik.github.io/resonances/img/gallery/mmr2__1921__2J-1+0-1.png)

## Colors and units

One palette gives a quantity the same color in every panel of the combined figures, MMR and secular/ZLK alike:

| color | what |
|---|---|
| black | a filtered series |
| gray | the osculating series (a band when the samples are dense) |
| red | the center of the angle and the line $\dot\sigma = 0$ |
| blue | a fitted or model quantity (force law, $\pm 2\omega_0$, $\pm\Delta a$) |
| dark gray | reference marks (turning points, jumps, $h = 1$, $a_*$) |
| dark orange | the warning line at the top of the legend when FAIR (m) is not drawn, with the reason |
| cyan | the catalog resonant axis $a_{\rm res}$ |
| orange, purple | an accepted subresonance, the owner resonance |

The right column is ten blocks (header, subtitle, key facts, why, confidence, checks, evidence, caveats, numbers,
footer; [reading the figures](reading-figures.md#the-right-column)), in two halves with a fixed place for each: header
to checks on the left, evidence to the footer on the right, on every figure. The time axis is labeled under the last
panel and under (b). Its plaque and ◆ take the status color (resonant
#1B7A3A, transient #B35400, near_separatrix #6B3FA0, non_resonant #4A4F57, uncertain #2F5D9E, invalid #B2182B;
`style.STATUS_COLOURS`, also used by the clone figures). The icons are ✓ green, ⚠ amber, ✗ red, ? gray and – light
gray. The action is one chip in the header: gray `no action needed`, amber `glance at (…)`, red `review figure +
numbers`, blue `extend to …` or `resample to …`. A column that does not fit makes the figure taller, by at most
`style.COLUMN_MAX_GROWTH_IN` (6 in); the left panels keep their place at the top. Beyond that the column trims (the
Numbers groups first), and `verdict_column.column_text` gives the complete text of the column. Every time color bar
uses the same map (viridis without its palest yellow). Wrapped angles are labeled in [0°, 360°). Time is in yr, kyr or Myr, whichever suits the span
(the `evolution` figure keeps years).

`plot_options={'combined': {'numbers': 'all'}}` prints every number of the decision in the right column of the
combined figures instead of the ten blocks (`explained`, the default).

The MMR figure has no empty slot: without FAIR the legend takes the place of (m) under one warning line that says why
(`mmr_combined.fair_warning`). Panel (l) draws each accepted episode over its fill interval of (a), the part beyond
its turning points dashed. Panel (e) draws the force law the status read (a witness episode's, rule R, at its own
cutoff; under the direct averaged-force law its low-passed $\dot a$ and $s\times$ the term's force per window,
`mmr_combined.draw_direct_law`). Accepted runs that are no libration of the angle are labeled `#id wave`, `#id forced
(strength)` or `#id ripple` on (l) (`mmr_combined.wave_labels`; on (a) only a counted one is filled and labeled). The ▼ marks of the strip (b′) on (b) keep their x and step clear of the panel letter and the ALIAS plaque
(`mmr_combined.strip_marks`). The MMR header's second line names who makes the cycles by its role (owner, driver,
neighbor with its interval, resonant argument, kinematic term; `verdict_text.cause_line`); the episodes another
resonance induces are hatched `\\\\` on (a) (`mmr_combined.INDUCED_HATCH`, apart from the `////` of an episode before
a jump) and labeled `induced` on (l); `also in …` names the other resonance of a joint capture. Panel (a) of both
the MMR and the secular/ZLK figure draws only what the minimal policy (W29) counts: the counted episodes (a covered
edge filled to the record edge, the filter-trim stretch pale; a counted 1:1 leg labeled `leg`) and, faint with a
dashed outline, the candidates `cand`; no rejected run and no gray strip (`mmr_combined.policy_marks`,
`shade_policy_runs`; the ids come from the analysis' `PolicyResult`, or from the stored verdict's
`counted_episode_ids`/`candidate_episode_ids` when a standalone panel re-runs the analysis without the scans); (l) is
titled `(l) episodes, counted solid` and the numbers count `N accepted, M counted`. A file written before W29 (no
`policy` column) keeps the older drawing: every accepted episode filled, rejected runs and runs the status does not
count as gray strips (`#id < floor`, `#id center turns`, `mmr_combined.uncounted_tags`). A 1:1 circulation judged on one leg or a non-1:1 half-swing between
passages (leaves G4c, G4h), the flag `extend_time`, the 1:1 profile floor and a turn read in the edge trim are said in
the right column ([reading the figures](reading-figures.md#the-right-column)).
One 1:1 episode in parts: (a) brackets its sub-regimes over the panel, named by their well
(`mmr_combined.draw_subregime_brackets`, `run_label_items`; overlapping names merge, `merge_label_items`), and hatches
its inferred residence beyond the fill lighter, `--`, labeled `inferred` (`residence_pieces`, `shade_residences`;
never on (l)); (d) marks the kicks of $a$ at such changes with ▼ `kick` (`draw_kicks`); the header reads `orbit: A →
B` for a sequence of wells; the EPISODES group of the right column lists the inferred stay and the marks (well,
amplitude, kick) under the episode.
Where a slower libration of the center was found (the slow scale, columns `slow_*`), (i) hatches its intervals
(`slow #id`, `mmr_combined.slow_intervals`) and the right column's SLOW SCALE group gives its numbers; when
its cause is a secular cycle, (j) adds the detrended $p\varpi$ labeled with $p$ and the secular period of its column
(`mmr_combined.varpi_overlay`). A kinematic libration's (j) is $\sigma = \lambda$ part $+ p\varpi$ on the scale of
that swing (`mmr_combined.kinematic_scale`).

`plotting.mmr_combined.draw_combined(analysis, series)` draws the MMR figure from an analysis. After a restore without
recomputation the analysis is run again on the saved series and checked against `summary.csv`; a difference is
logged and stamped in red in the header.

## Options

`plot_options` (or `PLOT_OPTIONS` in `.env` as JSON) is merged over the defaults. Unknown keys and invalid values raise
an error.

| option | default | meaning |
|--------|---------|---------|
| `style` | `screen` | standalone diagnostics only: `screen` (11 pt sans-serif, 150 dpi, a title) or `paper` (8 pt STIX serif, 300 dpi, no title; panels 88 mm, rows up to 180 mm). The width depends on the kind. The `combined` verdict figures keep their 24 × 13.5 in layout (taller when the right column needs it) and header in either style (MMR at 120 dpi; secular/ZLK saved at the style's dpi) |
| `raw` | `line` | raw series next to filtered ones: `line`, `markers`, `none` |
| `series` | `auto` | no effect (accepted for old configurations; no figure draws the standalone time series any more) |
| `recurrence.max_points` | `None` | display decimation of the recurrence matrix; `None` = about 8 samples per cycle, within 700–2500 |
| `recurrence.exclude_samples` | `2` | blank band around the diagonal, in decimated steps |
| `portrait.y` | `rate` | the MMR portrait's y axis: `rate` ($\dot\sigma$) or `axis` ($a - a_0$) |
| `cycles.prominence` | `None` | peak prominence in rad; `None` = 5 % of the range of $\sigma$ within [0.005, 0.1] |
| `fair.max_step_fraction` | `0.25` | the standalone `fair` figure warns (and still draws) when the output step exceeds this share of the orbit; panel (m) of the combined figure is not drawn above 1/4.2 of it |
| `combined.osculating_a` | `full` | panel (d): `full` holds the osculating a in view, `clipped` fits the view to the filtered a, `hidden` also hides the osculating a beyond that view |
| `combined.numbers` | `explained` | the right column: `explained` (the verdict in ten blocks: header with the status, level and action chip, subtitle, key facts, why, confidence pills, checks with their margins nested, evidence, caveats, numbers with the DECISIVE lines first, the footer of codes with the W28 tree's verdict) or `all` (every number of the decision) |

## Redrawing after a run

Nothing needs to be integrated again. Restore the run, then draw in any style:

```python
from resonances import SimulationSerializer, MMRPlotter  # noqa: E402
from resonances.plotting import SecularPlotter  # noqa: E402

sim = SimulationSerializer.restore('output/my_run/simulation.json')
body = sim.bodies[0]
plotter = MMRPlotter.from_body(body, body.mmrs[0], sim, options={'style': 'paper'})
plotter.plot('recurrence')  # recurrence, fair, portrait or cycles; the combined figure: plot_body below
plotter.save('recurrence.pdf').close()

secular = SecularPlotter.from_body(body, body.secular_resonances[0], sim)  # secular or ZLK rows: pass `sim`
secular.plot('portrait')
secular.save('portrait.png').close()

sim.config.plot, sim.config.plots = 'all', ['combined']
sim.data_manager.plot_body(body, sim)  # the verdict figures, from a re-run analysis
```

`MMRPlotter.plot` draws the standalone kinds only (`recurrence`, `fair`, `portrait`, `cycles`); the combined MMR
figure is redrawn by `sim.data_manager.plot_body(body, sim)` or `draw_combined(analysis, series)`.
`MMRPlotter.from_body(..., focus=(t0, t1))` restricts cycles, portrait and the recurrence scales to an interval in years.
Only bodies that passed the `save` filter have series on disk. `MMRDiagnostics(...)` builds the same figures from
plain arrays (times in years; `axis_analysis`, the a of the cycles, of the portrait with `portrait.y='axis'` and of
its reference $a_0$, defaults to the osculating `axis`). A backward integration is put in time order internally; the axes show the actual
negative times.

## Secular figures

`SecularPlotter` (`resonances.plotting.secular_plots`; kinds `angle`, `rate`, `portrait`, `combined`) draws the
angle the classifier read (the reduced proper angle by default, see [Secular resonances](secular.md)) next to the
angle of the osculating elements.
In the `combined` figure, panel (d) is the semi-major axis (osculating band, filtered at $P_c$, min – max; for a body
in a planet's co-orbital zone, by $a$ alone, the planet's $a_P$ dashed and the zone's edges $\pm k\,r_H$ dotted,
`ZLK_COORBITAL_HILL_FACTOR` $k$; `secular_plots.coorbital_planet_geometric`, for drawing only). Panel (g) of a secular angle draws the analysis' own response fit (the element and
$\sigma_f$ at the decisive cutoff, the center's when the center scale decided), the one whose $R^2$ it prints
([how to read it](reading-figures.md#the-secular-and-lidov-kozai-figure)). For a Lidov-Kozai angle panel (g) draws
the drift of the free and the osculating $\omega$ in turns with a legend, as the `free_omega` drift figure. The right
column has the ten blocks of the MMR figure; a ZLK row's Evidence asks what holds $\omega$ (the forced vector, the
center's slow wave, the node lock, the e–i coupling;
[Free elements: beyond the gate](free-elements.md#beyond-the-gate)), a secular row's names the period seen on the
angle against the FREQUENCIES rows, or `unidentified`. The numbers of
both types include $a$ in ORBIT and a FREQUENCIES table: the body's rates, the planetary modes (g5, g6, s6 always) and
the beats, each in ″/yr with its period ([how to read it](reading-figures.md#the-secular-and-lidov-kozai-figure)). An induced ZLK
libration (attributed to a forced mode, or a center at a beat of the forced vector) gets one more time panel, (a′)
under (a): $\omega_f$ with the forced direction and the dominant mode against the node
([how to read it](reading-figures.md#the-secular-and-lidov-kozai-figure)); every other row keeps eight. A ZLK
figure reads $\omega$, $i$, $\Omega$ and the free elements in the plane `ZLK_ANGLE_FRAME` names (the invariable plane
by default; the header says which, panel (f) is `i (invariable)`).

## Eccentricity vector

`ecc_vector` draws one figure per body ([how to read it](reading-figures.md#ecc_vector)):

```python
resonances.EccentricityVectorPlotter.from_body(body, resonance, sim).plot().save('kh.png')
resonances.EccentricityVectorPlotter.from_data(times, ecc, omega, body_name='463').plot().save('kh.png')
```

`plot(limit=...)` fixes the half-width, which is how the free-vector figure of [Free elements](free-elements.md) is
drawn on the scale of the osculating one.
