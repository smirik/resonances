# Simulations, bodies, resonances

A run has three parts: a `Simulation` (the planets, the integrator and every setting), the bodies added to it, each
with the resonances to check, and `sim.run()`, which integrates, analyses every angle and writes the results.
`resonances.check` and `resonances.find` build such a simulation for you.

## The simulation

```python
--8<-- "core.py:simulation"
```

Every keyword becomes an attribute of `sim.config` (`sim.config.integrator`, `sim.config.save_path`, ...) and can be
changed there before `sim.run()`. A keyword not given takes its value from a `.env` file in the working directory,
else from the package defaults; [Configuration](config.md) lists them all. The ones used most:

| keyword | default | meaning |
|---|---|---|
| `date` | today (source `nasa`); the catalogue epoch (`astdys`, `mpc`) | start of the integration; the planets come from NASA Horizons at this date |
| `source` | `nasa` | where the elements of a body given by number or name come from: `nasa` (Horizons), `astdys`, `mpc` |
| `integration_years` or `tmax` | 100 000 yr | the span; `tmax` is in REBOUND units (see below), negative for a backward run |
| `integrator`, `dt` | `SABA(10,6,4)`, 1.0 | the REBOUND integrator and its step in REBOUND units; `whfast`, `ias15` (adaptive) also work |
| `output_step` or `Nout` | $N_\text{out} = \lfloor\lvert t_\text{max}\rvert/100\rfloor$ points, i.e. a step of about 100 REBOUND units = 15.9 yr (exactly $\lvert t_\text{max}\rvert/[2\pi(N_\text{out}-1)]$ yr) | years between saved points, or the number of points; `'auto'` lets the [dry-run](dry-run.md) choose a step that folds no short-period term |
| `save`, `plot` | `nonzero` | which (body, resonance) rows are saved and plotted, by status (below) |
| `plots` | `['combined']` | figure kinds ([Plots](plots.md)) |
| `name`, `save_path`, `plot_path` | `cache/<name>` | where the output goes ([Output data](output.md)) |
| `plot_layout` | `confidence/status` | subfolders of the figures ([Examples](examples.md#output-layout)) |

**Time units.** REBOUND works in units where $G = 1$, the Sun's mass is 1 and the distance unit is the au, so one
year is $2\pi$ time units. `tmax`, `dt` and `sim.times` are in these units: `tmax = 2π × years`, and `dt = 1.0` is
$1/(2\pi)$ yr $\approx$ 58.1 days. Years are `sim.times / (2 * np.pi)`; `integration_years` and `output_step` are in
years.

## Bodies

```python
--8<-- "core.py:bodies"
```

A body is given by its number or designation (the elements are then taken from `source` at `date`) or by a dict of
osculating heliocentric elements in the J2000 ecliptic: `a` in au, `e`, and `inc`, `Omega`, `omega`, `M` in
radians. An `epoch` key (MJD) is checked against `date`: elements at another epoch are refused, since the planets are
placed at `date`. With `source='mpc'` an MPC row at another epoch is integrated to `date` first
([Configuration](config.md)).

## Resonances

A resonance is given as an object or as a string; strings are parsed by `resonances.create_resonance`.

```python
--8<-- "core.py:objects"
```

| class | notation | angle |
|---|---|---|
| `TwoBody` | `3J-1` (short) or `3J-1+0-2` (full) | $3\lambda_J - \lambda - 2\varpi$ ([Resonance matrices](matrix.md)) |
| `ThreeBody` | `4J-2S-1` or `4J-2S-1+0+0-1` | $4\lambda_J - 2\lambda_S - \lambda - \varpi$ |
| `RetrogradeTwoBody` | `1J-3r0` (p, planet, q, k) | p/−q resonance of an orbit with $i > 90°$ ([Resonance matrices](matrix.md#retrograde-resonances)) |
| `SecularResonance` | `g-g6`, `g-g6+s-s6`, `2(g-g6)+(s-s6)` | combinations of the body's and the planets' secular frequencies ([Secular resonances](secular.md)) |
| `LidovKozaiResonance` | `LK` | the argument of perihelion $\omega$ ([Lidov-Kozai resonances](lidov-kozai.md)) |

The full key (`4J-2S-1+0+0-1`) is the one in the output files and in `body.statuses`. The short form fills the
perihelion coefficients so that the coefficients sum to zero (d'Alembert's rule), putting the remainder on the body's
$\varpi$.

## `check` and `find`

- `resonances.check(asteroids, resonance, **kwargs)` checks the given resonances for the given bodies
  ([Quick start](quick-start.md)).
- `resonances.find(asteroids, planets=None, type=None, **kwargs)` looks for candidates: the MMRs whose resonant
  semi-major axis is within `sigma2 = 0.1` au (two-body) or `sigma3 = 0.02` au (three-body) of the body's $a$
  (`sigma_retro = 0.5` au for $i > 90°$), the secular resonances satisfied by the body's proper frequencies $g$, $s$
  (from the AstDyS synthetic catalogue; `formulas=` gives your own) and the Lidov-Kozai angle. `type='mmr'`, `'secular'` or `'lidov_kozai'` limits the search ([Examples](examples.md#finding-resonances)).

Both return a `Simulation` with the bodies added; call `sim.run()`. Their default `name` is `resonance_find`, so
their output goes to `cache/resonance_find` (with a time stamp when that folder is not empty).

## Running

```python
--8<-- "core.py:run"
```

`sim.run()` first runs the static [dry-run](dry-run.md) (its warnings go to the log), then integrates, analyses
every angle and writes the output. With more than `BATCH_THRESHOLD` (100) bodies, or with
[clones](clones.md), the work is split between worker processes: on macOS and Windows such a script needs an
`if __name__ == '__main__':` guard ([Examples](examples.md#clones)).

## Verdicts

```python
--8<-- "core.py:verdicts"
```

`body.librations[key]` is the full verdict (a `Verdict`): `status`, `angle_status`, `reason`, `libration_origin`,
`confidence`, `action`, `extend_to_years`, `final_regime`, `orbit_type`, the checks and the decision path. Its
values are strings ([How the status is determined](physics/overview.md)). `body.statuses[key]` is the status
alone. The time series are `body.axis`, `body.ecc`, `body.inc`, `body.Omega`, `body.omega`, `body.M`,
`body.longitude`, `body.varpi` and `body.angle(resonance)` (radians).

```python
--8<-- "core.py:summary"
```

## What is saved and plotted

`save` and `plot` select the (body, resonance) rows by their status:

| mode | rows |
|---|---|
| `none` | nothing |
| `all` | everything |
| `resonant` | `resonant`, `transient` |
| `candidates` | `resonant`, `transient`, `near_separatrix`, `uncertain` |
| `extended` | the candidates and a `non_resonant` row with a libration origin (kinematic, induced, subresonance, ...) |
| `nonzero` (default) | everything except a `non_resonant` row without a libration origin |
| `negative` | `near_separatrix`, `uncertain`, `invalid` |

`summary.csv` always holds every row (`save_summary=True`). The files and folders are described in
[Output data](output.md); a saved run can be read back with [`SimulationSerializer.restore`](serialization.md).
