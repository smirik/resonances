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

## Resuming interrupted batch simulations

When running large simulations with batch processing, if the simulation is interrupted (crash, OOM, manual stop), you can resume from where it stopped.

### How it works

During batch processing, a `state.json` file is saved in the output directory after each completed batch. This file tracks:
- Which batches have been completed
- The body names processed in each batch
- Batch configuration (total batches, batch size, etc.)

### How to resume

Simply **run the same script again** with:
1. The **same asteroid list** (same order, same filtering)
2. The **same `save_path`** pointing to the existing output directory

The simulation will automatically:
1. Detect the existing `state.json`
2. Skip already-completed batches
3. Continue from the next pending batch

### Example

If your simulation crashed at batch 537/4368:

```python
# Original script - just run it again
sim = resonances.find(
    asteroids=asteroids,  # Same asteroid list!
    name="my-simulation",
    save_path="/path/to/existing/output",  # Same path!
    plot_path="/path/to/existing/plots",
    # ... other parameters unchanged
)
sim.run(progress=True)
```

### Important notes

1. **Keep the same asteroid list**: Resume works by batch index. If you change the asteroid list, batch indices won't match the saved state.

2. **Point to the existing directory**: If your simulation created a timestamped directory (e.g., `data_2026-01-26_00-06-44`), update `save_path` and `plot_path` to point to those directories.

3. **Check progress**: You can inspect the state file to see how many batches completed:
   ```bash
   cat /path/to/output/state.json | python -c "import sys,json; d=json.load(sys.stdin); print(f\"Completed: {len(d['progress']['completed_batches'])}/{d['batch_config']['total_batches']}\")"
   ```

4. **Check last processed asteroid**: Look at the end of `summary.csv` to see which asteroids were processed last.

## Notes

- `simulation.json` is always written by `DataManager.save_configuration_details`.
- Libration metrics are not stored in JSON or CSV. They are derived and can be recomputed during restore.
- All file paths are relative to the simulation save directory.
