# Phase Plane Plots

Two families of phase-plane plots are available.

**Phase portraits** visualize the dynamics of resonant angles by plotting the derivative of the resonant angle (σ̇) against the angle itself (σ mod 360°). This representation helps identify libration centers, circulation patterns, and transient behavior.

**Eccentricity-vector plots** show the non-singular components `k = e·cos ω`, `h = e·sin ω` as the `(k, h)` plane coloured by time. The offset of the cloud from the origin is the forced eccentricity, its radius the free one — the pair that tells a genuine Lidov-Kozai libration of ω from one produced by forced-eccentricity geometry.

## Plot Types

When `phase_portrait` is enabled, three plot variants are generated:

1. **Filtered** (`{body_name}-{resonance_key}-filtered.png`) - Uses the low-pass filtered angle, removing high-frequency noise
2. **Unfiltered** (`{body_name}-{resonance_key}-unfiltered.png`) - Uses the raw unwrapped angle with all oscillations
3. **Slow points** (`{body_name}-{resonance_key}-percentile{N}.png`) - Shows only points where |σ̇| is below the `N`-th percentile (unfiltered data)

When `ecc_vector` is enabled, one file is generated per body (not per resonance — `k` and
`h` depend on the body alone): `{body_name}-ecc-vector.png`.

The window is always a square centred on the origin, sized by the data. The box is
therefore identical in every figure while the scale follows the object, so two of them can
be compared side by side; and the origin is always in view, which is the whole reading of
this plane — a cloud that does not enclose it keeps ω bounded whatever the dynamics.

`plot(limit=...)` overrides the half-width. That is how the free-vector figure of
[free elements](free-elements.md) is drawn on exactly the same scale as the osculating one:
the pair side by side is what shows whether the ring bounding ω is the object's own free
circle or the forced offset.

All plots use a time-based colormap (viridis) to show temporal evolution.

## Generating Phase Plots During Simulation

Add `phase_portrait` to the `plots` configuration:

```python
import resonances

sim = resonances.Simulation(
    name='my_simulation',
    tmax=100000,
    plot='all',
    plot_type='save',
    plots=['evolution', 'phase_portrait'],  # Enable both evolution and phase portraits
    phase_portrait_slow_percentile=95,      # Optional: threshold for slow points (default: 95)
)
sim.create_solar_system()
sim.add_body({'a': 2.5, 'e': 0.1, 'inc': 5, 'Omega': 0, 'omega': 0, 'M': 0},
             resonances.ThreeBody('4J-2S-1'))
sim.run()
```

The plots will be saved to the configured `plot_path` with filenames:
- `{body_name}-{resonance_key}-filtered.png`
- `{body_name}-{resonance_key}-unfiltered.png`
- `{body_name}-{resonance_key}-percentile95.png`

## Generating Phase Plots After Simulation

Use `PhasePlotter` directly for more control:

```python
import resonances

# After running a simulation
sim = resonances.Simulation(...)
sim.run()

body = sim.bodies[0]
mmr = body.mmrs[0]

# Create plotter from simulation data
plotter = resonances.PhasePlotter.from_body(body, mmr, sim)

# Generate and save each variant
plotter.plot_phase_portrait_filtered().save('phase_filtered.png')
plotter.plot_phase_portrait_unfiltered().save('phase_unfiltered.png')
plotter.plot_phase_portrait_slow(percentile=95).save('phase_slow.png')

# Always close to free memory
plotter.close()
```

## Creating Phase Plots from Raw Data

If you have time series data from another source:

```python
import numpy as np
import resonances

# Your data (times in years, sigma unwrapped in radians)
times = np.linspace(0, 100000, 10000)
sigma_unwrapped = ...  # Your unwrapped angle data
sigma_filtered = ...   # Optional: filtered version

plotter = resonances.PhasePlotter.from_data(
    times,
    sigma_unwrapped,
    sigma_filtered=sigma_filtered,  # Optional
    body_name='Asteroid',
    resonance_key='4J-2S-1'
)

plotter.plot_phase_portrait_filtered().save('phase.png')
plotter.close()
```

## Configuration

| Parameter | Environment Variable | Default | Description |
|-----------|---------------------|---------|-------------|
| `plots` | `PLOTS` | `['evolution']` | List of plot types to generate |
| `phase_portrait_slow_percentile` | `PHASE_PORTRAIT_SLOW_PERCENTILE` | `95` | Percentile threshold for slow points |

## Eccentricity Vector Plots

```python
sim = resonances.Simulation(plots=['evolution', 'ecc_vector'], plot='all')
```

Produces one figure per body: the `(k, h)` plane coloured by time, in a square
window centred on the origin. The numeric counterparts
`e_forced` and `e_free` are written to `summary.csv`; see
[Cross-Spectral Coherence](coherence.md) for how they are defined and what they mean.

`EccentricityVectorPlotter` can also be used directly:

```python
resonances.EccentricityVectorPlotter.from_body(body, resonance, sim).plot().save('kh.png')

# or from raw arrays (times in years, omega in radians)
resonances.EccentricityVectorPlotter.from_data(times, ecc, omega, body_name='591986').plot().save('kh.png')
```

## Interpreting Phase Portraits

- **Libration**: Points cluster around a fixed center (e.g., 180°) with σ̇ oscillating around zero
- **Circulation**: Points spread across all angles with σ̇ predominantly positive or negative
- **Transient**: Mix of libration and circulation episodes, visible through the time colormap

The slow points plot is particularly useful for identifying libration centers, as it filters out the fast circulation phases and highlights where the system spends most time with low angular velocity.
