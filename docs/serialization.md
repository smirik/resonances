# Serialization

This page explains how `resonances` saves and restores simulations. The logic lives in `resonances/simulation/serializer.py` (class `SimulationSerializer`).

## What is saved

After a simulation run, the output directory contains:

1. `simulation.json` – the metadata and configuration.
2. `data-<body>.csv` – time series for each body.
3. `data-<body>-periodograms.csv` – periodograms for each body (if available).
4. `summary.csv` – summary table (optional, controlled by `save_summary`).
5. `data-planet-<planet>.csv` – planets time series (optional, controlled by `save_planets`).

The JSON file is the entry point for restore.

## simulation.json structure

The JSON file has three sections:

```json
{
  "config": { ... },
  "simulation": { ... },
  "timing": { ... }
}
```

### config

This is a full snapshot of `SimulationConfig`, including derived values such as `tmax` and `tmax_yrs`.

### simulation

This section stores:

- `number_of_bodies`
- `bodies` – for each body:
  - `name`, `type`, `mass`
  - `initial_data` (orbital elements used for initialization)
  - `resonances` (string representation)
- `data_files` – a manifest of CSV files created in the output directory

### timing

`running_time` contains timestamps (if available), and `running_time_differences` contains durations between stages.

## How restore works

To restore a simulation:

```python
from resonances.simulation import SimulationSerializer

sim = SimulationSerializer.restore("output/my_run/simulation.json")
```

The restore method:

1. Rebuilds `SimulationConfig` from `config`.
2. Recreates `Body` objects and their resonances.
3. Loads all CSV time series into `Body` fields.
4. Loads periodograms (if files exist).
5. Optionally recomputes libration metrics (`recompute_librations=True` by default).

If you need a fast restore without recomputing librations:

```python
sim = SimulationSerializer.restore("output/my_run/simulation.json", recompute_librations=False)
```

## Notes

- `simulation.json` is always written by `DataManager.save_configuration_details`.
- Libration metrics are not stored in JSON or CSV. They are derived and can be recomputed during restore.
- All file paths are relative to the simulation save directory.
