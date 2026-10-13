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
| `combined` | MMR, secular | `combined/…-combined` | All diagnostics of the resonance type on one figure |
| `recurrence` | MMR | `-recurrence` | Recurrence of the state (σ, σ̇) |
| `fair` | two-body MMR | `-fair` | FAIR plane: mean anomaly against the longitude difference to the planet |
| `portrait` | MMR, secular | `-portrait` | Phase portrait: σ–σ̇ (or σ–a) for an MMR, (r cos σ, r sin σ) for a secular angle |
| `angle` | secular | `-angle` | σ(t): osculating, smoothed, turning points |
| `rate` | secular | `-rate` | Rate of the smoothed σ, with the divisor |
| `cycles` | MMR | `-cycles` | Oscillation cycles of σ and of a, stacked on one period |
| `ecc_vector` | all | `{body}-ecc-vector.png` | The (k, h) = (e cos ω, e sin ω) plane |
| `cross_spectrum` | all | `-coherence-{pair}` | Coherence and cross-phase, see [Cross-spectral coherence](coherence.md) |
| `free_omega` | Lidov-Kozai | `-free-omega-…` | Free argument of pericentre, see [Free elements](free-elements.md) |

Every time colour bar (MMR, ecc-vector, free-ω) uses the same map, viridis without its palest yellow, and the label `Time (unit)`. An unknown name raises an error. A kind is drawn only for the resonance types it lists; `combined` and `portrait` mean the same kind of figure for an MMR and a secular resonance. The image format is `image_type` (`png` by default, or `pdf`). Combined figures go to their own folder, `plot_path/combined/`, so they can be browsed one after another; with `plot_subfolder_strategy='status'` it is split by status like `plot_path` itself (`combined/resonant/`, `combined/transient/`, …).

## MMR diagnostics

Everything is computed from arrays a simulation already saves: the raw, filtered and wrapped angle, a and filtered a, M and λ of the body, plus the planet's λ and a for FAIR. The quantities are diagnostics for reading a figure; they change no status and are not calibrated decision rules. The code is in `resonance/oscillations.py` (math) and `plotting/mmr_plots.py` (drawing).

The combined figure:

- **(a–c) Time series**: the unwrapped σ (raw in grey, filtered in black), the wrapped σ in [0, 2π) as points, and Δa = a − a₀, where a₀ is the median filtered a. Neither Δa nor a₀ is printed on the figure: give them in the caption (a₀ is `MMRDiagnostics.axis_reference`). The three panels together are exactly as tall as the square panels next to them. When the oscillations are too dense for the panel (a median cycle narrower than four line widths), the raw and filtered series are drawn as min–max bands over about two cycles each, so the band shows the amplitude and its modulation instead of a solid block of ink, and the wrapped points turn translucent. The red line joins the centres of the cycles (below). It is broken where cycles do not follow each other, or around a cycle more than three times the median period (two maxima bridging circulation); such a centre stays as a lone point.
- **(d) Recurrence**: D(t₁, t₂) = √[(Δσ/s_σ)² + (Δσ̇/s_σ̇)²], taken on the unwrapped angle so that a full circulation never returns. Dark means the state comes back close to itself. The scales are the object's own: s_σ is half the median cycle diameter (half the range of σ when there are no cycles), and s_σ̇ is half the 5–95 % spread of σ̇. **D is therefore comparable within one figure, not between objects**, and must not be thresholded. The matrix is decimated for display, by default to about 8 samples per median cycle (within 700–2500): fewer would alias the oscillation into a moiré pattern. Pairs within `exclude_samples` steps are blank.
- **(e) FAIR** (Forgács-Dajka, Sándor & Érdi 2018): M against λₚ − λ for an inner asteroid, λ − λₚ for an outer one. It is a two-body diagnostic. It is omitted for three-body arguments: a pairwise projection does not test an argument involving two planets.
- **(f) Portrait**: σ against σ̇ over the focus, coloured by time, with the raw points in grey. One period of the filter cutoff (1/`LIBRATION_FILTER_CUTOFF` years) is dropped at both ends of the record, where the zero-phase filter is distorted. The unwrapped σ is shifted by whole turns only, so a libration about 180° is drawn about 180°. `portrait.y = 'axis'` plots a − a₀ instead; for an MMR it carries the same information, since σ̇ follows from a.
- **(g–h) Cycles**: a cycle runs from one maximum of the filtered unwrapped σ to the next. Incomplete ends are dropped, and the peak prominence is 5 % of the range of σ, kept within [0.005, 0.1] rad. Each cycle is stretched to [0, 1]. σ keeps its level; a is shown minus the median of each cycle. The colour is the cycle's midpoint time. A cycle is a candidate oscillation, not a confirmed libration: a drifting centre (the stack climbing) or a bridging cycle shows exactly that.

The **focus** is the whole record for now. The API takes `focus=(t0, t1)` in years (`MMRPlotter.from_body(..., focus=...)`): cycles, portrait and recurrence scales then use that interval, and the time series are dimmed outside it.

### FAIR needs planet data

Run with `save_planets=True`. Without planet data FAIR is skipped with a warning (console and log), and the combined figure is drawn without the panel. FAIR also needs the mean anomaly sampled well within one orbit. When the output step exceeds `fair.max_step_fraction` of the orbital period (¼ by default), a warning says the plane is aliased; increase `Nout`.

## Options

`plot_options` (or `PLOT_OPTIONS` in `.env` as JSON) is merged over the defaults. Unknown keys and invalid values raise an error.

| Option | Default | Meaning |
|--------|---------|---------|
| `style` | `screen` | `screen`: 16 in wide, 150 dpi, title `Asteroid <name>, <resonance>` (short notation, e.g. `2J-1`). `paper`: 180 mm wide, 8 pt STIX serif, 300 dpi, no title (the caption carries it) |
| `raw` | `line` | Raw series next to filtered ones: `line`, `markers` (plus hollow markers on actual samples), `none` |
| `series` | `auto` | Time series as `lines`, min–max `envelope` bands, or `auto` (bands only when lines would merge) |
| `recurrence.max_points` | `None` | Display decimation of the recurrence matrix; `None` = ~8 samples per cycle, within 700–2500 |
| `recurrence.exclude_samples` | `2` | Blank band around the diagonal, in decimated steps |
| `portrait.y` | `rate` | `rate` (σ̇, deg per time unit) or `axis` (a − a₀) |
| `cycles.prominence` | `None` | Peak prominence in rad; `None` = 5 % of the range of σ within [0.005, 0.1] |
| `fair.max_step_fraction` | `0.25` | Warn when the output step exceeds this share of the orbit |

Units: σ in rad in time series and stacks, degrees in the portrait.

The filter matters: the filtered σ feeds the cycles, the portrait and the recurrence. `LIBRATION_FILTER_CUTOFF` must lie above the libration frequency. The default 0.0005 (periods under 2000 yr removed) suits slow three-body librations, but erases a Trojan libration of ~150 yr; for that, use e.g. `oscillations_cutoff=0.02`. Time is in yr, kyr or Myr, whichever suits the span.

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

## Secular diagnostics

Four kinds, `angle`, `rate`, `portrait` and `combined` (all three), drawn by `SecularPlotter` (`plotting/secular_plots.py`), which shares `DiagnosticPlotter` (`plotting/base.py`: kinds, styles, titles, the `combined/` folder) with `MMRPlotter` (`plot_options={'style': 'paper'}` for a journal). They show the angle the classifier read — the reduced proper angle by default (see [Secular resonances](secular.md)) — next to the angle of the osculating elements, and change no status.

```python
resonances.SecularPlotter.from_body(body, resonance, sim.config).plot('portrait')
```

**angle — σ(t), degrees.** Grey: the angle from the osculating elements, as for every resonance type (folded mod 360° into the band of the black curve when that one is bounded and the grey one turns). Black: the classified angle smoothed — over `secular_centre_window` of the baseline (the *centre* of the status check), or, when that centre stands still, over a quarter of the libration period the classifier found, so that a fast libration (ν6 of Tina, 600 kyr) is followed. Red circles: turning points — extrema where the black curve changes direction by at least `secular_turning_prominence` degrees, not inflection points. A libration has a curve that oscillates (two turning points per cycle); a drift is a monotonic black ramp of less than a turn, and a single U is half a cycle (both status −6, see below); circulation is a ramp of several turns; a transient case oscillates for a while and runs off.

**rate — dσ/dt, ″/yr.** The rate of the black curve; grey line: zero; dashed: the mean rate of the classified angle (its divisor). A libration swings around zero; a drift runs parallel to zero at the divisor; a passage or transient crosses zero and leaves it.

**portrait — (r cos σ, r sin σ).** r is e when σ contains ϖ, else i measured from the invariable plane, in degrees (ν16 = s − s6 has no ϖ) — the quantities standing for the momentum conjugate to σ (Γ/|p| ≈ e²/2, Z/|q| ≈ i²/2). Grey: osculating; coloured by time: classified. As for the (k, h) plane of ZLK, a cloud that never encloses the origin keeps σ bounded. A libration draws closed loops around the resonance centre, not around the origin: a ring or oval off the origin with an empty middle when the centre is far from the origin compared with the amplitude (Tina in ν6: centre at e ≈ 0.2, σ = 180°), a crescent ("banana") when the libration is wide and the centre close to the origin (z1, z2). A ring around the origin is circulation.

**The status they illustrate** comes from the classifier, plus one check (`resonance/secular_centre.py`): a secular LIBRATION whose centre moves by more than `secular_centre_drift` degrees within a turn and turns back fewer than `secular_min_turning_points` times (3: a cycle and a half) becomes BASELINE_TOO_SHORT (−6) — bounded over the baseline, a longer integration would tell. Examples: 2110 in g − g6 − 2g7 + 2g8 drifts 100° in 10 Myr at 0.04″/yr without turning; 2005 GQ107 makes one U in 10 Myr because its own g drifts through the commensurability; the z1 librator 3701 Purkyne (period ~8 Myr) turns back twice in 10 Myr and needs a longer run, while 1020 Arcadia (~6.7 Myr) turns three times, a cycle and a half, and keeps its 2. A centre that stands still (moves less than `secular_centre_drift`) is a libration faster than the window and keeps its status. It is kept apart from the classifier's PROBABLY_SLOW_CIRCULATION (−4, the trend dominates the oscillation), so that the two never merge in a statistic; the `candidates` and `extended` save/plot modes include it, and its figures go to `baseline-too-short/`. The check leaves a `centre:` comment and the summary.csv columns `centre_check` (`kept`/`lowered`), `centre_drift_deg`, `centre_turning_points`, as the free-omega gate does for Lidov-Kozai. The window is a compromise: forced terms make σ oscillate at the beats 1/|g − f| with the forced frequencies, 40–600 kyr for the fundamentals but arbitrarily long for their combinations, while nonlinear secular librations take 2–10 Myr; T/10 (1 Myr at 10 Myr, close to the 700 kyr low-pass of Carruba et al. 2016 for z2) averages the fundamental beats out and keeps every centre motion slow enough to turn within the baseline.

**Checks** (SABA, 10 Myr, `experiments/2026-09-binary-asteroids/sim_secular_reference.py`): z1 — 3701 Purkyne (period ~8 Myr as in Vokrouhlický et al. 2006), 1020, 363, 339, 633, 21211, 20845, 49, 518; z2 — 689 Zita, 8089 Yukar, 2093, 883, 1249, 915, 298, 254; ν6 — Tina, 16712, 48717; ν5 — 11028; g − 3g6 + 2g5 — 4305: status 2. ν16 (127504) transient. Controls: 51 Nemausa (circulates in z2, Carruba et al. 2021) 0, 3527 McCord −3, (847) Agnia slow circulation, every angle of Ceres and Hygiea 0.
