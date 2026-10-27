<!-- Generated from docs/serialization.md by docs/tools/build_skill.py (make skill). Edit the docs, not this file. -->

# Serialization

This page explains how `resonances` saves and restores simulations. The logic lives in `resonances/simulation/serializer.py` (class `SimulationSerializer`).

## What is saved

After a simulation run, the output directory contains (the series in `data/`, `data_folder`; the manifest
`simulation.data_files` of `simulation.json` lists them relative to the run folder, and restore reads them from there, or
from the top for runs saved before the data folder):

1. `simulation.json` – the metadata and configuration.
2. `data/data-<body>.csv` – time series of each body with at least one resonance that passed the `save` mode, and only those resonances' angles.
3. `data/data-<body>-periodograms.csv` – their periodograms (when there are any).
4. `data/<body>-omega-free.csv` – the forced/free series of a saved body with a Lidov-Kozai resonance.
5. `summary.csv`, `coherence.csv`, `episodes.csv` – one row per body and resonance (per coherent line, per run of turning points of any resonance type), for every body whatever its status (`save_summary`). Writing a body again replaces its rows.
   summary.csv: `name, resonance, resonance_type (mmr|secular|lidov_kozai), added_by, parent`, the verdict
   as strings (`status, angle_status, angle_subtype, confidence, reason, flags, action, libration_origin, origin_detail,
   owner, multiplet_status`; `status_code` is the old integer, for comparison with old runs only), the checks `check_mmr_*`/`check_secular_*`/`check_zlk_*`,
   margins and `decision_path`, the numbers the decision read, `settings_hash`, `confidence_cells_sha`,
   `package_commit`, the initial elements `a0, e0, inc0, Omega0, omega0, M0`, the secular/ZLK classifier columns, and
   `comments` last. A derived subresonance row (`added_by=subresonance_check`, `parent` = its catalogue angle) has a
   row of its own. With MMRs also: `mmr_numbers.csv` (every number of the MMR analysis, one row per body × MMR, empty
   = not measured), `episodes.csv` (every run of turning points, accepted or not; `save_episodes`), `candidates.csv`
   (subresonance and owner candidates; `save_candidates`) and `constants.csv` (planets' mean motions and frequencies,
   the great inequality; one per run, `save_constants`). List cells are `;`-separated.
6. `data/data-planet-<planet>.csv` – planets time series: the planets `save_planets` keeps (`all`, `analysis` = those the
   analysis reads, `none`).

An MMR's data columns are `<key>_angle` (the raw angle mod 2π, exact) and `<key>_angle_unwrapped` (the analysis'
predictive unwrap); its filtered angle is not written: the analysis reproduces it with the cutoff of summary.csv.

The columns of every file are listed in [Output files](output.md). The JSON file is the entry point for restore.

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
  - `resonances` (string representation); a derived subresonance row also has `added_by` and `parent`
- `data_files` – the CSV files of the run that are on disk
- `planets_saved` – the planets the run kept
- `free_elements` – the forced/free setup of the run (Lidov-Kozai only)

### timing

`running_time` contains timestamps (if available), and `running_time_differences` contains durations between stages.

## How restore works

```python
from resonances import SimulationSerializer  # noqa: E402,F811

sim = SimulationSerializer.restore("output/my_run/simulation.json")
```

The restore method:

1. Rebuilds `SimulationConfig` from `config`.
2. Recreates `Body` objects and their resonances, derived subresonance rows included (also those known only from
   `summary.csv`, registered before the series are read so that their saved angles load too).
3. Loads the saved time series, periodograms, free-omega series and planets (to the last bit: the numbers are those
   the run analysed).
4. Sets the verdicts from `summary.csv`: `body.statuses[key]` (the status string) and `body.librations[key]` (a
   `Verdict` rebuilt from the row's verdict columns). A summary of an older version with integer statuses is read for
   secular and Lidov-Kozai rows (the old code is translated, `verdict_from_status_code`); its MMR rows are skipped.

A run (and every batch worker) handles one body at a time from its verdicts to its data files and figures, then
releases the arrays of its analyses, so only one body's analyses are in memory; summary.csv and the tables are
written after the last body, from the same rows.

After a restore the numbers of the analysis stay in the CSV files: `body.mmr_analyses` and
`body.classifier_numbers` are empty, and `save_data`/`save_simulation_summary` write and replace only the rows
analysed in memory, so the rows on disk stay.
To analyse again from the saved series:

```python
sim = SimulationSerializer.restore("output/my_run/simulation.json", recompute_librations=True)
```

An MMR is analysed again by `analyse_mmr` on its raw saved angle and elements, with the settings of simulation.json
(`current_settings=True`: the fields of `AnalysisSettings` and `DryRunSettings` are taken from the current
configuration; the free-element and free-gate settings such as `FREE_GATE_ENABLED`, `FREE_GATE_MIN_CYCLES`,
`FREE_GATE_MASK_QUANTILE`, the coherence settings and the secular-angle construction keep the saved run's values); its
derived rows come out of that analysis again. A secular or
Lidov-Kozai angle is analysed again by `analyse_secular`/`analyse_zlk` on its saved unwrapped angle and elements (the
centre scale and the free-omega gate are steps of it). Only the resonances whose series were saved are analysed again
(their periodograms are rebuilt); the others keep their `summary.csv` verdict. A saved derived row whose parent was not
saved (`SAVE=resonant` keeps a resonant subresonance of a non-resonant angle) is analysed alone, at depth one.

## Batch runs

Above `batch_threshold` bodies (`BATCH_ENABLED`), `run()` integrates the bodies in batches, each in its own process,
which writes its files. Every worker keeps the planets of the whole run. The derived rows the workers added are put
on the parent's bodies before simulation.json is written, and the bodies in memory get their verdicts back from
`summary.csv` (none without `save_summary`); `body.mmr_analyses` stays empty. The rows of a batch completed before
an interruption come back the same way on resume (`restore_statuses` adds the derived rows it finds).

## Resuming interrupted batch simulations

If a batch run is interrupted (crash, OOM, manual stop), it can be resumed.

During the run, `simulation_state.json` in the output directory records the completed batches, written after each one (also for batches still running when another failed). It also holds a hash of everything the results depend on: the config except names, paths, plot settings and `n_cores`, which includes `date` and `source`, and the batch layout (body names and resonances of every batch).

To resume, run the same script again with the same bodies, settings and `save_path`. Completed batches are skipped. A batch that was interrupted after writing some of its rows is run again and its rows in `summary.csv`, `episodes.csv`, `coherence.csv`, `mmr_numbers.csv` and `candidates.csv` are replaced, not repeated.

If the hash differs, `run()` raises `ValueError` instead of mixing two runs. The usual cause is the date: without `date=` a `nasa` run starts at today's midnight, so a resume on another day must pass the original date (`date=`). With `batch_size` unset the layout depends on `n_cores`, so keep it too. To start over, delete `simulation_state.json` (and the old files). A `simulation_state.json` of an older version (its hash covered only `tmax`, `Nout`, the integrator, `dt`, `batch_threshold` and `secular_angle_mode`) is still accepted when those match, with a warning.

```python
import resonances

asteroids = [463, 490]  # the original run's list, in the same order
settings = dict(integration_years=1_000, batch_threshold=1, batch_size=1, planets=['Jupiter', 'Saturn'], type='mmr')


def main():
    sim = resonances.find(
        asteroids=asteroids,  # the same list, in the same order
        name="my-simulation",
        date="2026-10-13",  # the date of the interrupted run
        save_path="output/resume",  # the same folder
        plot_path="output/resume-plots",
        **settings,  # the same settings as the interrupted run
    )
    sim.run(progress=True)


if __name__ == '__main__':  # a batch run starts worker processes: the guard is required
    main()
```

These are the settings of the demonstration run; for a real resume use the original run's asteroid list and settings.

Progress so far:

```bash
python -c "import json; d=json.load(open('/path/to/output/simulation_state.json')); print(len(d['progress']['completed_batches']), '/', d['batch_config']['total_batches'])"
```

## Notes

- `simulation.json` is always written by `DataManager.save_configuration_details`.
- All file paths are relative to the simulation save directory.
