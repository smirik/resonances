<!-- Generated from docs/plots.md by docs/tools/build_skill.py (make skill). Edit the docs, not this file. -->

# Plots

Which figures a simulation draws is set by `plots` (default `['combined']`, the one-page verdict figure). Which rows
get them is set by `plot` (a status filter, see [Configuration](config.md#saving-and-plotting)) and `plot_always`.

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
| `clones` | all, a body with clones | `{body}-{key}-clones.png` | the ensemble: every clone's angle by status, status against initial a (with the catalogue $a_{\rm res}$ of an MMR), the status shares with Wilson intervals (also drawn with `combined`; [Clones](clones.md)) |

An unknown name is an error; a kind is drawn only for the resonance types it lists. The image format is `image_type`
(`png` or `pdf`). Combined figures go to `plot_path/combined/`, the others to `plot_path/figures/`, both split by
`plot_layout` (`confidence/status` by default: `combined/high/resonant/…`; [Output data](output.md#layout)). None of
these figures feeds a status.

**What every panel, line and icon means: [How to read the figures](reading-figures.md)**, with an example of each
kind. One example for every status: [Gallery](physics/gallery.md).

![The combined figure of (1921) Pala in the 2:1 resonance with Jupiter](https://smirik.github.io/resonances/img/gallery/mmr2__1921__2J-1+0-1.png)

## Colours and units

One palette gives a quantity the same colour in every panel of the combined figures, MMR and secular/ZLK alike:

| colour | what |
|---|---|
| black | a filtered series |
| grey | the osculating series (a band when the samples are dense) |
| red | the centre of the angle and the line $\dot\sigma = 0$ |
| blue | a fitted or model quantity (force law, $\pm 2\omega_0$, $\pm\Delta a$) |
| dark grey | reference marks (turning points, jumps, $h = 1$, $a_*$) |
| cyan | the catalogue resonant axis $a_{\rm res}$ |
| orange, purple | an accepted subresonance, the owner resonance |

The plaque of the right column is coloured by the status. Every time colour bar uses the same map (viridis without
its palest yellow). Wrapped angles are labelled in [0°, 360°). Time is in yr, kyr or Myr, whichever suits the span
(the `evolution` figure keeps years).

`plot_options={'combined': {'numbers': 'all'}}` prints every number of the decision in the right column of the
combined figures instead of the verdict explained in words (`explained`, the default).

`plotting.mmr_combined.draw_combined(analysis, series)` draws the MMR figure from an analysis. After a restore without
recomputation the analysis is run again on the saved series and checked against `summary.csv`; a difference is
logged and stamped in red in the header.

## Options

`plot_options` (or `PLOT_OPTIONS` in `.env` as JSON) is merged over the defaults. Unknown keys and invalid values raise
an error.

| option | default | meaning |
|--------|---------|---------|
| `style` | `screen` | standalone diagnostics only: `screen` (11 pt sans-serif, 150 dpi, a title) or `paper` (8 pt STIX serif, 300 dpi, no title; panels 88 mm, rows up to 180 mm). The width depends on the kind. The `combined` verdict figures keep their 24 × 13.5 in layout and header in either style (MMR at 120 dpi; secular/ZLK saved at the style's dpi) |
| `raw` | `line` | raw series next to filtered ones: `line`, `markers`, `none` |
| `series` | `auto` | no effect (accepted for old configurations; no figure draws the standalone time series any more) |
| `recurrence.max_points` | `None` | display decimation of the recurrence matrix; `None` = about 8 samples per cycle, within 700–2500 |
| `recurrence.exclude_samples` | `2` | blank band around the diagonal, in decimated steps |
| `portrait.y` | `rate` | the MMR portrait's y axis: `rate` ($\dot\sigma$) or `axis` ($a - a_0$) |
| `cycles.prominence` | `None` | peak prominence in rad; `None` = 5 % of the range of $\sigma$ within [0.005, 0.1] |
| `fair.max_step_fraction` | `0.25` | the standalone `fair` figure warns (and still draws) when the output step exceeds this share of the orbit; panel (m) of the combined figure is not drawn above 1/4.2 of it |
| `combined.osculating_a` | `full` | panel (d): `full` holds the osculating a in view, `clipped` fits the view to the filtered a, `hidden` also hides the osculating a beyond that view |
| `combined.numbers` | `explained` | the right column: `explained` (the verdict in words, steps, margins, numbers) or `all` (every number of the decision) |

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
plain arrays (times in years; `axis_cycles`, the a of the cycles, defaults to the osculating `axis`). A backward integration is put in time order internally; the axes show the actual
negative times.

## Secular figures

`SecularPlotter` (`resonances.plotting.secular_plots`; kinds `angle`, `rate`, `portrait`, `combined`) draws the
angle the classifier read (the reduced proper angle by default, see [Secular resonances](secular.md)) next to the
angle of the osculating elements.
In the `combined` figure, panel (f) of a secular angle draws the analysis' own response fit (the element and
$\sigma_f$ at the decisive cutoff, the centre's when the centre scale decided), the one whose $R^2$ it prints
([how to read it](reading-figures.md#the-secular-and-lidov-kozai-figure)). For a Lidov-Kozai angle panel (f) draws
the drift of the free and the osculating $\omega$ in turns with a legend, as the `free_omega` drift figure; the right
column adds the measured e–i coupling, the forcing of a kinematic libration and the slow line of the centre
([Free elements: beyond the gate](free-elements.md#beyond-the-gate)).

## Eccentricity vector

`ecc_vector` draws one figure per body ([how to read it](reading-figures.md#ecc_vector)):

```python
resonances.EccentricityVectorPlotter.from_body(body, resonance, sim).plot().save('kh.png')
resonances.EccentricityVectorPlotter.from_data(times, ecc, omega, body_name='463').plot().save('kh.png')
```

`plot(limit=...)` fixes the half-width, which is how the free-vector figure of [Free elements](free-elements.md) is
drawn on the scale of the osculating one.
