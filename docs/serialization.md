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
5. The tables at the top: `summary.csv` (one row per body and resonance, derived subresonance rows included, for every
   body whatever its status: `save_summary`), `episodes.csv`, `coherence.csv` and, with MMRs, `mmr_numbers.csv`,
   `candidates.csv`, `constants.csv`. Writing a body again replaces its rows. Their columns:
   [Output data](output.md#summarycsv).
6. `data/data-planet-<planet>.csv` – planets time series: the planets `save_planets` keeps (`all`, `analysis` = those the
   analysis reads, `none`; with clones, also the planets of the clones' own rows).
7. With clones: `clones.csv` at the top, and `data/ensemble-<body>.csv`/`.json` (the draws) and
   `data/ensemble-<body>-results.npz` (the clones' verdicts and thinned angles) with `SAVE` on ([Clones](clones.md#output)).

The columns of every file, the data files included, are listed in [Output data](output.md). The JSON file is the
entry point for restore.

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
- `data_files` – the CSV files of the run that are on disk (`clones` = clones.csv)
- `planets_saved` – the planets the run kept
- `free_elements` – the forced/free setup of the run (Lidov-Kozai only)
- `ensembles` – with clones: `{"version": 1, "items": [...]}`, per ensemble `parent`, `nominal_in_run`, `resonances`,
  `n_clones`, `n_valid`, `complete` (every clone batch finished), `manifest` and `results` (paths relative to the run
  folder; null while the clones run, or with `SAVE` off), `nominal_offset_sigma`, `source`, `seed`, `clone_verdicts`.
  simulation.json is written after the ordinary bodies (the ensembles not complete) and again after the clones; it is
  replaced atomically. The clones are not in `bodies`.

### timing

`running_time` contains timestamps (if available), and `running_time_differences` contains durations between stages.

## How restore works

```python
--8<-- "saved_run.py:restore"
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
5. With `ensembles`: rebuilds `sim.ensembles` from each manifest and results archive (the clones' names, their
   elements at the date to the bit, the stored verdict rows and thumbnails) and `sim.clone_summary` from clones.csv;
   without a manifest (SAVE off) only `sim.clone_summary`, with a warning. A simulation.json without `ensembles` (older
   runs) restores as before.

A run (and every batch worker) handles one body at a time from its verdicts to its data files and figures, then
releases the arrays of its analyses, so only one body's analyses are in memory; summary.csv and the tables are
written after the last body, from the same rows.

After a restore the numbers of the analysis stay in the CSV files: `body.mmr_analyses` and
`body.classifier_numbers` are empty, and `save_data`/`save_simulation_summary` write and replace only the rows
analysed in memory, so the rows on disk stay.
To analyse again from the saved series:

```python
--8<-- "saved_run.py:recompute"
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
With clones only the nominal is reclassified: the clones keep their stored verdicts, and the nominal columns of
clones.csv are rewritten in the run's folder (the one simulation.json is in) from the nominal's new verdict.

## Batch runs

Above `batch_threshold` bodies (`BATCH_ENABLED`), `run()` integrates the bodies in batches, each in its own process,
which writes its files. Every worker keeps the planets of the whole run. The derived rows the workers added are put
on the parent's bodies before simulation.json is written, and the bodies in memory get their verdicts back from
`summary.csv` (none without `save_summary`); `body.mmr_analyses` stays empty. The rows of a batch completed before
an interruption come back the same way on resume (`restore_statuses` adds the derived rows it finds).

## Resuming interrupted batch simulations

If a batch run is interrupted (crash, OOM, manual stop), it can be resumed.

During the run, `simulation_state.json` in the output directory records the completed batches, written after each one (also for batches still running when another failed). It also holds a hash of everything the results depend on: the config except names, paths, plot settings and `n_cores`, which includes `date` and `source`, the batch layout (body names and resonances of every batch) and every body's initial elements as they were used (exact floats, with the sha256 of an MPC get-orb answer).

To resume, run the same script again with the same bodies, settings and `save_path`. Completed batches are skipped. A batch that was interrupted after writing some of its rows is run again and its rows in `summary.csv`, `episodes.csv`, `coherence.csv`, `mmr_numbers.csv` and `candidates.csv` are replaced, not repeated.

If the hash differs, `run()` raises `ValueError` instead of mixing two runs. The usual cause is the date: without `date=` a `nasa` run starts at today's midnight, so a resume on another day must pass the original date (`date=`). With `batch_size` unset the layout depends on `n_cores`, so keep it too. The elements count as obtained, not as requested: AstDyS and MPC elements come from the cached catalogue and are the same on a rerun, but a refreshed get-orb answer (`MPC_LOOKUP=file_then_api`), a new catalogue or a Horizons re-query (astroquery keeps its answers a week) that returns a new orbit solution makes another run, and the error names the bodies whose elements changed. To start over, delete `simulation_state.json` (and the old files). A `simulation_state.json` of an older version is still accepted with a warning when what it covered matches: only `tmax`, `Nout`, the integrator, `dt`, `batch_threshold` and `secular_angle_mode` (the oldest), or everything above except the elements (the elements are then not checked).

```python
--8<-- "resume_run.py:resume"
```

These are the settings of the demonstration run; for a real resume use the original run's asteroid list and settings.

Progress so far:

```bash
python -c "import json; d=json.load(open('/path/to/output/simulation_state.json')); print(len(d['progress']['completed_batches']), '/', d['batch_config']['total_batches'])"
```

A run with clones is resumed the same way, as a whole: its checkpoint `clones_state/` (the run identity, the clones'
starts, one file per finished clone batch) and `simulation_state.json` stay until the clones are done too; a finished
ordinary part and the finished clone batches are not run again ([Clones](clones.md#resume-and-restore)).

## Notes

- `simulation.json` is always written by `DataManager.save_configuration_details`.
- All file paths are relative to the simulation save directory.
