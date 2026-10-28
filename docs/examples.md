# Examples

Recipes for common tasks. Every block below is part of a script that `make docs-examples` runs with short spans; the comments
say what to use for real work.

## Finding resonances {#finding-resonances}

The candidates near a semi-major axis, from the cached catalogues of resonant axes ([Resonance matrices](matrix.md)):

```python
--8<-- "examples.py:find_mmrs"
```

`find` does the same for each body, adds every candidate and returns the simulation. `type='mmr'` keeps it to MMRs
(by default the secular and Lidov-Kozai candidates are added too):

```python
--8<-- "examples.py:find"
```

A three-body candidate is accepted within `sigma3 = 0.02` au of the body's $a$, a two-body one within
`sigma2 = 0.1` au; both are arguments of `find` and `find_mmrs`.

## High-order resonances

The default catalogues hold two-body resonances up to order 10 and three-body resonances up to order 6. For higher
orders, raise the limits and give the catalogue a new file before anything reads it (a cached file built with other
limits is read as it is, with a warning):

```python
--8<-- "high_order.py:high_order"
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
--8<-- "examples.py:mpc"
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
--8<-- "clones_run.py:clones"
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
--8<-- "examples.py:layout"
```

On this short run both angles are `uncertain`, so the evolution figures go to `figures/mmr/uncertain/` and
`figures/secular/uncertain/`, the verdict figures to `combined/none/uncertain/` (`none` is the confidence folder of
a verdict without a level).

## Restoring a run

A saved run is read back from its `simulation.json`. By default the verdicts come from `summary.csv`; with
`recompute_librations=True` the saved series are analysed again, without integrating:

```python
--8<-- "examples.py:restore"
```

Only the rows whose series were saved (the `save` mode) can be analysed again. See
[Serialization and restore](serialization.md).

## The integrator

`SABA(10,6,4)` with `dt = 1.0` (58 days) suits main-belt orbits. Orbits that approach a planet need an adaptive
integrator; the [dry-run](dry-run.md) recommends one for each body.

```python
--8<-- "examples.py:integrator"
```

`solar_system_moon=True` integrates the Moon as a separate body (only with `ias15`); otherwise the Earth and the Moon
are one body at their barycentre. The parameters of REBOUND itself are described in
[Configuration](config.md#integration-options).
