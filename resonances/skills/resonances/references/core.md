<!-- Generated from docs/core.md by docs/tools/build_skill.py (make skill). Edit the docs, not this file. -->

# Simulations, bodies, resonances

A run has three parts: a `Simulation` (the planets, the integrator and every setting), the bodies added to it, each
with the resonances to check, and `sim.run()`, which integrates, analyses every angle and writes the results.
`resonances.check` and `resonances.find` build such a simulation for you.

## The simulation

```python
import numpy as np

import resonances

sim = resonances.Simulation(
    name='core-demo',  # the output goes to SAVE_PATH/core-demo (cache/core-demo by default)
    date='2026-01-01',  # start of the integration; the planets and the elements are taken at this date
    integration_years=2_000,  # 100_000 years for a real verdict of a main-belt MMR
    integrator='SABA(10,6,4)',  # the default; 'whfast' or 'ias15' for close encounters
    dt=1.0,  # REBOUND time units: 1.0 = 1/(2 pi) yr = 58.1 days
    output_step=1.0,  # years between saved points (default: about 100 REBOUND units = 15.9 yr)
    save='all',  # which (body, resonance) rows to save: none, all, resonant, candidates, nonzero, ...
    plot='resonant',  # which rows to plot, by the same status filters
    plots=['combined'],  # the one-page verdict figure
)
sim.create_solar_system()
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
sim.add_body(463, '4J-2S-1', name='463')  # by number, one resonance
sim.add_body('2001 ND13', ['1V-3', 'g-g6'], name='2001 ND13')  # by designation, several resonances
elements = {  # by elements: heliocentric J2000 ecliptic, angles in radians, epoch in MJD (optional)
    'a': 2.395,
    'e': 0.22,
    'inc': np.radians(13.5),
    'Omega': np.radians(97.0),
    'omega': np.radians(180.0),
    'M': np.radians(10.0),
}
sim.add_body(elements, resonances.TwoBody('3J-1'), name='synthetic')
```

A body is given by its number or designation (the elements are then taken from `source` at `date`) or by a dict of
osculating heliocentric elements in the J2000 ecliptic: `a` in au, `e`, and `inc`, `Omega`, `omega`, `M` in
radians. An `epoch` key (MJD) is checked against `date`: elements at another epoch are refused, since the planets are
placed at `date`. With `source='mpc'` an MPC row at another epoch is integrated to `date` first
([Configuration](config.md)).

## Resonances

A resonance is given as an object or as a string; strings are parsed by `resonances.create_resonance`.

```python
mmr2 = resonances.create_mmr('3J-1')  # two-body: 3 lambda_J - lambda - 2 varpi
mmr3 = resonances.create_mmr('4J-2S-1')  # three-body: 4 lambda_J - 2 lambda_S - lambda + 0 + 0 - varpi
retro = resonances.RetrogradeTwoBody('1J-3r0')  # retrograde p/-q with k (orbits with i > 90 deg)
nu6 = resonances.SecularResonance('g-g6')  # linear secular resonance
z1 = resonances.create_resonance('g-g6+s-s6')  # the factory reads any notation
lk = resonances.LidovKozaiResonance()  # the argument of perihelion omega
print(mmr2.to_s(), mmr3.to_s(), retro.to_s(), nu6.to_s(), z1.to_s(), lk.to_s())
print(round(mmr3.resonant_axis, 4))  # nominal catalogue axis a_res [au]; the analysed a* is ctx_a_star_au
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
report = sim.dry_run()  # optional: checks the output step and the integrator before integrating
sim.run()
```

`sim.run()` first runs the static [dry-run](dry-run.md) (its warnings go to the log), then integrates, analyses
every angle and writes the output. With more than `BATCH_THRESHOLD` (100) bodies, or with
[clones](clones.md), the work is split between worker processes: on macOS and Windows such a script needs an
`if __name__ == '__main__':` guard ([Examples](examples.md#clones)).

## Verdicts

```python
for body in sim.bodies:
    for key, verdict in body.librations.items():
        print(body.name, key, verdict.status, verdict.angle_status, verdict.reason, verdict.confidence, verdict.action)
    print(body.statuses)  # {key: status string}

body = sim.bodies[0]
t = sim.times / (2 * np.pi)  # years since the start: REBOUND time is years * 2 pi
angle = body.angle(mmr3)  # the resonant angle, radians in [0, 2 pi)
print(len(t), body.axis[:3], angle[:3])
```

`body.librations[key]` is the full verdict (a `Verdict`): `status`, `angle_status`, `reason`, `libration_origin`,
`confidence`, `action`, `extend_to_years`, `final_regime`, `orbit_type`, the checks and the decision path. Its
values are strings ([How the status is determined](physics/overview.md)). `body.statuses[key]` is the status
alone. The time series are `body.axis`, `body.ecc`, `body.inc`, `body.Omega`, `body.omega`, `body.M`,
`body.longitude`, `body.varpi` and `body.angle(resonance)` (radians).

```python
summary = sim.data_manager.get_simulation_summary(sim.bodies)
print(summary[['name', 'resonance', 'status', 'angle_status', 'confidence', 'reason']])
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
