<!-- Generated from docs/examples.md by docs/tools/build_skill.py (make skill). Edit the docs, not this file. -->

# Examples

Recipes for common tasks. Every block below is part of a script that `make docs-examples` runs with short spans; the comments
say what to use for real work.

## Finding resonances {#finding-resonances}

The candidates near a semi-major axis, from the cached catalogues of resonant axes ([Resonance matrices](matrix.md)):

```python
import resonances

# Candidate MMRs near a = 2.3986 au (463 Lola) with Jupiter and Saturn
for mmr in resonances.find_mmrs(2.3986, planets=['Jupiter', 'Saturn']):
    print(mmr.to_short(), round(mmr.resonant_axis, 4))
```

`find` does the same for each body, adds every candidate and returns the simulation. `type='mmr'` keeps it to MMRs
(by default the secular and Lidov-Kozai candidates are added too):

```python
# find: every candidate MMR of each body (type='mmr'; None also adds the secular and Lidov-Kozai candidates)
sim = resonances.find([463, 490], planets=['Jupiter', 'Saturn'], type='mmr', name='find-demo', save='all', integration_years=2_000)
sim.run()
summary = sim.data_manager.get_simulation_summary(sim.bodies)
print(summary[['name', 'resonance', 'status', 'angle_status', 'reason']])
```

A three-body candidate is accepted within `sigma3 = 0.02` au of the body's $a$, a two-body one within
`sigma2 = 0.1` au; both are arguments of `find` and `find_mmrs`.

## High-order resonances

The default catalogues hold two-body resonances up to order 10 and three-body resonances up to order 6. For higher
orders, raise the limits and give the catalogue a new file before anything reads it (a cached file built with other
limits is read as it is, with a warning):

```python
import resonances

# Higher orders than the default catalogues (2-body order <= 10, 3-body order <= 6): set the limits and a new file
# before anything reads the catalogue (the cached one, built with the old limits, would be read as it is)
resonances.config.set('MATRIX_2BODY_PRIMARY_MAX', 40)  # exclusive: the planet's coefficient runs up to 39
resonances.config.set('MATRIX_2BODY_COEF_MAX', 39)
resonances.config.set('MATRIX_2BODY_ORDER_MAX', 25)
resonances.config.set('MATRIX_2BODY_FILE', 'cache/mmr-2body-order25.csv')

high = resonances.find_mmrs(2.3986, planets=['Mars', 'Jupiter'], sigma=0.005, mmr_type='two-body')
print([mmr.to_short() for mmr in high])
```

The three-body keys are `MATRIX_3BODY_PRIMARY_MAX`, `MATRIX_3BODY_COEF_MAX`, `MATRIX_3BODY_ORDER_MAX` and
`MATRIX_3BODY_FILE`. The same keys can be set in `.env` ([Configuration](config.md)). The strength of a resonance
falls fast with its order, so a high-order libration needs a fine output step and a long record to be confirmed
([Is it this resonance?](physics/response.md)).

## The MPC catalogue

With `source='mpc'` the elements come from the Minor Planet Center's MPCORB (or a subset, `MPC_CATALOG`), cached in
`cache/mpc/` on first use and refreshed only by `resonances mpc-update`. `resonances.mpcorb.query` selects rows by
elements, orbit quality and MPC's orbit-type flags, and its result can be passed to `find`:

```python
# a selection of the MPC catalogue: a DataFrame (a in au, angles in degrees, H, U, MPC's orbit-type flags)
selection = resonances.mpcorb.query(a=(2.3, 2.45), U=(None, 2), numbered=True)
print(selection[['number', 'name', 'a', 'e', 'inc_deg', 'H']])

# elements from MPCORB at its standard epoch (the default date of a source='mpc' run)
sim = resonances.find(selection, source='mpc', planets=['Jupiter', 'Saturn'], type='mmr', name='mpc-demo', integration_years=2_000)
print(sim.config.date, sim.config.mpc_snapshot['sha256'][:12], [body.name for body in sim.bodies])
```

Without a `date`, an MPC run starts at the catalogue's standard epoch, where most rows are given. A row at another
epoch is integrated to the date first (`MPC_EPOCH_POLICY=propagate`) or refused (`strict`).

!!! note
    The tested script reads the 20-row sample `tests/fixtures/mpc/MPCORB_sample.DAT`
    (`MPC_CATALOG=file://...`), so the selection above holds (4) Vesta and (463) Lola. With the default
    `MPC_CATALOG=mpcorb` the same code downloads MPCORB (about 94 MB compressed) on first use.

## Clones

An orbit is known only within its errors. Clones (virtual asteroids) are drawn from the covariance of the orbit
solution; the run reports, per resonance, the share of clones with each status and its 95 % interval
([Clones](clones.md)):

```python
import resonances


def main():
    solution = resonances.orbit_solution(463, source='astdys')  # nominal orbit + covariance (cached in cache/clones/)
    ensemble = solution.sample(n=10, seed=0)  # 10 clones and the nominal; 100-1000 for real shares

    sim = resonances.Simulation(name='clones-demo', date='2026-01-01', integration_years=2_000)
    sim.create_solar_system()
    sim.add_ensemble(ensemble, resonances=['4J-2S-1'])  # clones drawn at another epoch are integrated to the date first
    sim.run()
    shares = sim.clone_summary  # the rows of clones.csv: the share of each status with its 95 % Wilson interval
    print(shares[['resonance', 'n_clones', 'nominal_status', 'modal_status', 'modal_share', 'f_resonant_lo', 'f_resonant_hi']])


if __name__ == '__main__':  # the clones run in worker processes: the guard is required
    main()
```

The clones run in worker processes, so a script needs the `if __name__ == '__main__':` guard on macOS and Windows
(where new processes re-import the script). Short forms: `sim.add_body(463, '4J-2S-1', clones=100)`,
`resonances.check(463, '4J-2S-1', clones=100)`.

## Output layout {#output-layout}

Tables (`summary.csv`, `episodes.csv`, ...) and `simulation.json` stay at the top of `save_path`; the series go to
`data/`, the figures to `figures/` and the one-page verdict figures to `combined/`. Both figure folders are split
into subfolders by a layout, levels from `status`, `confidence`, `type`, `resonance`, `body` joined by `/`
(`confidence/status` by default; the presets: [Output data](output.md#layout)).

```python
sim = resonances.Simulation(
    name='layout-demo',
    date='2026-01-01',
    integration_years=2_000,
    plot='all',
    plots=['combined', 'evolution'],
    plot_layout='type/status',  # figures/<type>/<status>/...
    plot_combined_layout='confidence/status',  # combined/<confidence>/<status>/... (the default)
)
sim.create_solar_system()
sim.add_body(463, ['4J-2S-1', 'g-g6'], name='463')
sim.run()
print(sim.config.plot_path)
```

On this short run both angles are `uncertain`, so the evolution figures go to `figures/mmr/uncertain/` and
`figures/secular/uncertain/`, the verdict figures to `combined/none/uncertain/` (`none` is the confidence folder of
a verdict without a level).

## Restoring a run

A saved run is read back from its `simulation.json`. By default the verdicts come from `summary.csv`; with
`recompute_librations=True` the saved series are analysed again, without integrating:

```python
json_file = f'{sim.config.save_path}/simulation.json'
restored = resonances.SimulationSerializer.restore(json_file)  # the verdicts as they are in summary.csv
print(restored.bodies[0].statuses)

# analyse the saved series again (with the settings of simulation.json; current_settings=True: those of today's .env)
again = resonances.SimulationSerializer.restore(json_file, recompute_librations=True)
print(again.bodies[0].statuses)
```

Only the rows whose series were saved (the `save` mode) can be analysed again. See
[Serialization and restore](serialization.md).

## The integrator

`SABA(10,6,4)` with `dt = 1.0` (58 days) suits main-belt orbits. Orbits that approach a planet need an adaptive
integrator; the [dry-run](dry-run.md) recommends one for each body.

```python
# IAS15 (adaptive) with the Moon as a separate body: for planet-crossing orbits and close encounters
sim = resonances.Simulation(
    name='ias15-demo',
    date='2026-01-01',
    integration_years=500,
    integrator='ias15',
    dt=0.01,  # initial step, REBOUND units (yr / 2 pi); IAS15 adapts it
    solar_system_moon=True,
    output_step=0.2,  # years; a 1:1 or a near-Earth orbit needs a fine output step
)
sim.create_solar_system()
sim.add_body('2002 AA29', '1E-1', name='2002 AA29')  # a co-orbital of the Earth
sim.run()
print(sim.bodies[0].statuses)
```

`solar_system_moon=True` integrates the Moon as a separate body (only with `ias15`); otherwise the Earth and the Moon
are one body at their barycentre. The parameters of REBOUND itself are described in
[Configuration](config.md#integration-options).
