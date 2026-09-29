# Plots

Which figures a simulation draws is set by `plots`. Which bodies get them is set by `plot` (a status filter, see [Config](config.md)) and `plot_always`.

```python
sim = resonances.Simulation(
    plot='nonzero',
    plots=['evolution', 'combined'],
    plot_options={'style': 'screen'},  # optional, see below
    save_planets=True,                 # needed by FAIR only
)
```

| Kind | Resonances | File (`{body}-{key}-…`) | What it shows |
|------|------------|-------------------------|---------------|
| `evolution` | all | `{body}-{key}.png` | Angle, a, e, i and periodograms against time (`PLOT_CONFIG` preset) |
| `combined` | MMR | `-combined` | All MMR diagnostics below on one figure |
| `recurrence` | MMR | `-recurrence` | Recurrence of the state (σ, σ̇) |
| `fair` | two-body MMR | `-fair` | FAIR plane: mean anomaly against the longitude difference to the planet |
| `portrait` | MMR | `-portrait` | Phase portrait σ–σ̇ (or σ–a) |
| `cycles` | MMR | `-cycles` | Oscillation cycles of σ and of a, stacked on one period |
| `ecc_vector` | all | `{body}-ecc-vector.png` | The (k, h) = (e cos ω, e sin ω) plane |
| `cross_spectrum` | all | `-coherence-{pair}` | Coherence and cross-phase, see [Cross-spectral coherence](coherence.md) |
| `free_omega` | Lidov-Kozai | `-free-omega-…` | Free argument of pericentre, see [Free elements](free-elements.md) |

An unknown name raises an error. MMR kinds are skipped for other resonances. The image format is `image_type` (`png` by default, or `pdf`).

## MMR diagnostics

Everything is computed from arrays a simulation already saves: the raw, filtered and wrapped angle, a and filtered a, M and λ of the body, plus the planet's λ and a for FAIR. The quantities are diagnostics for reading a figure; they change no status and are not calibrated decision rules. The code is in `resonance/oscillations.py` (math) and `plotting/mmr_plots.py` (drawing).

The combined figure:

- **(a–c) Time series**: the unwrapped σ (raw in grey, filtered in black), the wrapped σ as points, and a − a₀, where a₀ is the median filtered a. The wrapped range is [−π, π] when the angle librates about 0, otherwise [0, 2π]. The red line joins the centres of the cycles (below). It is broken where cycles do not follow each other, or around a cycle more than three times the median period (two maxima bridging circulation); such a centre stays as a lone point.
- **(d) Recurrence**: D(t₁, t₂) = √[(Δσ/s_σ)² + (Δσ̇/s_σ̇)²], taken on the unwrapped angle so that a full circulation never returns. Dark means the state comes back close to itself. The scales are the object's own: s_σ is half the median cycle diameter (half the range of σ when there are no cycles), and s_σ̇ is half the 5–95 % spread of σ̇. **D is therefore comparable within one figure, not between objects**, and must not be thresholded. The matrix is decimated to `max_points` for display, and pairs within `exclude_samples` steps are blank.
- **(e) FAIR** (Forgács-Dajka, Sándor & Érdi 2018): M against λₚ − λ for an inner asteroid, λ − λₚ for an outer one. It is a two-body diagnostic. It is omitted for three-body arguments: a pairwise projection does not test an argument involving two planets.
- **(f) Portrait**: σ against σ̇ over the focus, coloured by time, with the raw points in grey. The unwrapped σ is shifted by whole turns only, so a libration about 180° is drawn about 180°. `portrait.y = 'axis'` plots a − a₀ instead; for an MMR it carries the same information, since σ̇ follows from a.
- **(g–h) Cycles**: a cycle runs from one maximum of the filtered unwrapped σ to the next. Incomplete ends are dropped, and the peak prominence is 5 % of the range of σ, kept within [0.005, 0.1] rad. Each cycle is stretched to [0, 1]. σ keeps its level; a is shown minus the median of each cycle. The colour is the cycle's midpoint time. A cycle is a candidate oscillation, not a confirmed libration: a drifting centre (the stack climbing) or a bridging cycle shows exactly that.

The **focus** is the whole record for now. The API takes `focus=(t0, t1)` in years (`MMRPlotter.from_body(..., focus=...)`): cycles, portrait and recurrence scales then use that interval, and the time series are dimmed outside it.

### FAIR needs planet data

Run with `save_planets=True`. Without planet data FAIR is skipped with a warning (console and log), and the combined figure is drawn without the panel. FAIR also needs the mean anomaly sampled well within one orbit. When the output step exceeds `fair.max_step_fraction` of the orbital period (¼ by default), a warning says the plane is aliased; increase `Nout`.

## Options

`plot_options` (or `PLOT_OPTIONS` in `.env` as JSON) is merged over the defaults. Unknown keys and invalid values raise an error.

| Option | Default | Meaning |
|--------|---------|---------|
| `style` | `screen` | `screen`: 16×9 in, 150 dpi, title with body and resonance. `paper`: 180 mm wide, 8 pt STIX serif, 300 dpi, no title (the caption carries it) |
| `raw` | `line` | Raw series next to filtered ones: `line`, `markers` (plus hollow markers on actual samples), `none` |
| `recurrence.max_points` | `700` | Display decimation of the recurrence matrix |
| `recurrence.exclude_samples` | `2` | Blank band around the diagonal, in decimated steps |
| `portrait.y` | `rate` | `rate` (σ̇, deg per time unit) or `axis` (a − a₀) |
| `cycles.prominence` | `None` | Peak prominence in rad; `None` = 5 % of the range of σ within [0.005, 0.1] |
| `fair.max_step_fraction` | `0.25` | Warn when the output step exceeds this share of the orbit |

Units: σ in rad in time series and stacks, degrees in the portrait. Time is in yr, kyr or Myr, whichever suits the span.

## Redrawing after a simulation

Nothing needs to be integrated again. Restore the run, then draw in any style:

```python
from resonances import SimulationSerializer, MMRPlotter

sim = SimulationSerializer.restore('output/my_run/simulation.json')
body = sim.bodies[0]
plotter = MMRPlotter.from_body(body, body.mmrs[0], sim, options={'style': 'paper'})
plotter.plot_combined().save('fig.pdf')
plotter.plot('recurrence')           # any kind by name, as in `plots`
plotter.save('recurrence.pdf').close()
```

Only bodies that passed the `save` filter have data on disk, and FAIR needs the run to have used `save_planets=True`. `MMRDiagnostics(...)` builds the same figures from plain arrays (times in years). A backward integration (decreasing times) is put in time order internally; the axes show the actual negative times.

## Eccentricity vector

`ecc_vector` draws one figure per body: the (k, h) plane coloured by time. The window is a square centred on the origin and sized by the data, so figures compare side by side and the origin is always in view. That is the whole reading of this plane: a cloud that does not enclose the origin keeps ω bounded whatever the dynamics. The numeric `e_forced` and `e_free` are in `summary.csv`, see [Cross-spectral coherence](coherence.md).

```python
resonances.EccentricityVectorPlotter.from_body(body, resonance, sim).plot().save('kh.png')
resonances.EccentricityVectorPlotter.from_data(times, ecc, omega, body_name='591986').plot().save('kh.png')
```

`plot(limit=...)` fixes the half-width, which is how the free-vector figure of [free elements](free-elements.md) is drawn on the scale of the osculating one.
