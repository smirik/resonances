import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from resonances.body import Body
from resonances.logger import logger
from resonances.resonance.analysis.verdict import AngleStatus, ResonanceType, Status, Verdict, verdict_from_status_code
from resonances.resonance.factory import create_resonance, detect_resonance_type
from resonances.data.util import datetime_from_string

# Columns of summary.csv a Verdict is rebuilt from (Verdict.from_flat_dict); older files have only `status`
VERDICT_COLUMNS = ('resonance_type', 'status', 'angle_status')
# Where the manifest looks for a series besides the run's data_folder: the top (runs before the data folder) and the
# default folder (a run resumed across the change, BatchManager.RESUME_DATA_FOLDERS)
MANIFEST_DATA_FOLDERS = ('', 'data')
# The run-wide MMR tables written next to summary.csv (by DataManager), listed in the manifest when present
MMR_TABLES = ('mmr_numbers', 'episodes', 'candidates', 'constants')

if TYPE_CHECKING:
    from resonances.simulation.simulation import Simulation


class SimulationSerializer:
    @classmethod
    def save_simulation_json(cls, config, bodies, simulation) -> None:
        data = cls._build_simulation_json(config, bodies, simulation)
        save_path = Path(config.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        # a temporary file renamed into place: an interruption never leaves half a simulation.json
        temporary = save_path / "simulation.json.tmp"
        with temporary.open("w") as f:
            json.dump(data, f, indent=4)
        os.replace(temporary, save_path / "simulation.json")

    @classmethod
    def restore(cls, json_file_path: str, recompute_librations: bool = False, current_settings: bool = False) -> "Simulation":
        """Rebuild a saved simulation: config, bodies (derived subresonance rows included), saved series, planets
        and verdicts.

        By default the verdicts come from summary.csv: `body.statuses[key]` (a status string) and
        `body.librations[key]` (a `Verdict` rebuilt from the row's verdict columns; numbers that are not verdict
        columns stay in the file). With `recompute_librations=True` the resonances whose series were saved (the
        SAVE mode keeps only some) are analysed again with the settings of simulation.json, through
        `identify_librations`: an MMR by `analyse_mmr` on the raw saved angle and elements (its derived rows are
        produced again by it), a secular/ZLK angle by its classifier and hooks. The others keep their summary.csv
        verdict. `current_settings=True` analyses with the analysis settings of the current .env/.env.dist instead
        of those of simulation.json (everything else still comes from simulation.json).

        A run with clones also gets its ensembles back (`clone_store.restore_ensembles`: `sim.ensembles` with the
        clones' names, starts and stored verdicts, `sim.clone_summary` from clones.csv). Recomputing reclassifies the
        nominals only: the clones keep their stored verdicts, and the nominal columns of clones.csv are rewritten.
        """
        json_path = Path(json_file_path)
        logger.info(f"Restoring simulation from {json_path}")
        with json_path.open("r") as file:
            data = json.load(file)

        config = dict(data.get("config", {}))
        config["_skip_path_verification"] = True
        if isinstance(config.get("date"), str):
            config["date"] = cls._parse_datetime(config["date"])
        cls._drop_removed_plot_kinds(config)
        if current_settings:
            for name in cls._analysis_setting_names():
                config.pop(name, None)

        from resonances.simulation.simulation import Simulation

        logger.info(f"Creating simulation from config: {config['name']}")
        sim = Simulation(**config)
        logger.info(f"Simulation created: {sim.config.name}")

        cls._restore_running_time(sim, data.get("timing", {}))

        bodies_data = data.get("simulation", {}).get("bodies", [])
        for body_data in bodies_data:
            body = cls._build_body_from_json(sim, body_data)
            sim.body_manager.bodies.append(body)

        # the derived rows of summary.csv first: their saved series are read with the others
        cls._add_summary_derived_rows(sim.bodies, json_path.parent / "summary.csv")

        logger.info("Restoring body data")
        simulation_data = data.get("simulation", {})
        files_manifest = simulation_data.get("data_files", {})
        cls._restore_body_data(sim, json_path.parent, files_manifest)
        cls._restore_free_elements(sim, json_path.parent, files_manifest, simulation_data.get("free_elements", {}))

        logger.info("Restoring planets")
        cls._restore_planets(sim, json_path.parent, data.get("simulation", {}).get("data_files", {}))

        restored = cls.restore_statuses(sim.bodies, json_path.parent / "summary.csv")
        if not restored and not recompute_librations:
            logger.warning(f"No statuses restored for {json_path}: summary.csv is missing or has none of these bodies")

        from resonances.simulation import clone_store

        summary_only = clone_store.restore_ensembles(sim, json_path.parent, simulation_data)

        if recompute_librations and sim.bodies:
            saved = {body.name: cls._saved_resonances(body) for body in sim.bodies}
            skipped = sum(len(body.resonances()) - len(saved[body.name]) for body in sim.bodies)
            if skipped:
                logger.info(f"Not reclassified ({skipped} resonances): their series were not saved; statuses from summary.csv")
            logger.info("Identifying librations")
            sim.identify_librations(only=saved)
            sim.build_periodograms(only=saved)  # the spectra of the recomputed angles, as after a run
        if recompute_librations and (sim.ensembles or summary_only):
            clone_store.recompute_clone_rows(sim, json_path.parent, summary_only)

        return sim

    @staticmethod
    def _analysis_setting_names() -> set:
        """The config attributes that are settings of the libration analysis (and its dry-run)."""
        from resonances.resonance.analysis.settings import AnalysisSettings, DryRunSettings, setting_fields

        return {f.name for cls in (AnalysisSettings, DryRunSettings) for f in setting_fields(cls)}

    @classmethod
    def restore_statuses(cls, bodies, summary_csv) -> int:
        """Set the verdicts of the bodies from summary.csv; return how many were set.

        `body.statuses[key]` gets the status string; `body.librations[key]` a `Verdict` rebuilt from the row when the
        file has the verdict columns. A derived row (column `parent` not empty) that the body does not have yet is
        added to it first (`Body.add_derived_resonance`), which also brings back the rows of resumed batches.
        A file of an older version with integer statuses: secular and Lidov-Kozai rows are translated
        (`verdict_from_status_code`); an integer MMR status has no translation and is skipped with a warning.
        """
        frame = cls._summary_frame(summary_csv)
        if frame is None:
            return 0
        has_verdict = all(column in frame.columns for column in VERDICT_COLUMNS)
        by_body = cls._rows_by_body(frame)
        count = skipped_mmr = 0
        for body in bodies:
            rows = by_body.get(str(body.name), [])
            cls._add_derived_rows(body, rows)
            keys = {r.to_s() for r in body.resonances()}
            for row in (row for row in rows if row["resonance"] in keys):
                verdict = cls._verdict_of_row(row, row["resonance"], has_verdict)
                if verdict is None:
                    skipped_mmr += 1
                    continue
                body.statuses[row["resonance"]] = str(verdict.status)
                body.librations[row["resonance"]] = verdict
                count += 1
        if skipped_mmr:
            logger.warning(
                f"{skipped_mmr} MMR rows of {summary_csv} have an integer status (an older version): MMR statuses are strings "
                "now and have no translation; restore with recompute_librations=True to analyse them again"
            )
        return count

    @staticmethod
    def _summary_frame(summary_csv) -> Optional[pd.DataFrame]:
        """summary.csv as strings, or None when it is missing, empty or has no name/resonance/status columns."""
        path = Path(summary_csv)
        if not path.exists() or path.stat().st_size == 0:
            return None
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        return frame if {"name", "resonance", "status"} <= set(frame.columns) else None

    @staticmethod
    def _rows_by_body(frame: pd.DataFrame) -> Dict[str, list]:
        by_body = {}
        for row in frame.to_dict("records"):
            by_body.setdefault(row["name"], []).append(row)
        return by_body

    @classmethod
    def _add_summary_derived_rows(cls, bodies, summary_csv) -> None:
        """Register the derived rows of summary.csv on their bodies (`_add_derived_rows`)."""
        frame = cls._summary_frame(summary_csv)
        if frame is None:
            return
        by_body = cls._rows_by_body(frame)
        for body in bodies:
            cls._add_derived_rows(body, by_body.get(str(body.name), []))

    @staticmethod
    def _add_derived_rows(body: Body, rows) -> None:
        """Add the derived rows of summary.csv (column `parent` not empty) that the body does not have yet."""
        for row in rows:
            parent = str(row.get("parent") or "")
            if parent and row["resonance"] not in {r.to_s() for r in body.resonances()}:
                body.add_derived_resonance(row["resonance"], parent)

    @staticmethod
    def _verdict_of_row(row: Dict[str, Any], key: str, has_verdict: bool) -> Optional[Verdict]:
        """The verdict of one summary.csv row: from its verdict columns, or translated from an integer status
        (secular and Lidov-Kozai only; None for an integer MMR status)."""
        status = str(row["status"]).strip()
        if status in {str(s) for s in Status}:
            if has_verdict and str(row.get("resonance_type") or ""):
                return Verdict.from_flat_dict(row)
            resonance_type = detect_resonance_type(key)
            return Verdict(resonance_type=ResonanceType(resonance_type), status=Status(status), angle_status=AngleStatus.UNDEFINED)
        resonance_type = detect_resonance_type(key)
        if resonance_type == ResonanceType.MMR:
            return None
        return verdict_from_status_code(int(float(status)), resonance_type)

    @staticmethod
    def _saved_resonances(body: Body) -> set:
        """Resonance keys of a restored body whose series the analysis needs were in data-<body>.csv: for an MMR
        the raw wrapped angle (`<key>_angle`), for secular/ZLK the unwrapped angle (`<key>_angle_unwrapped`; the
        analysis filters it itself), each with the body's elements."""
        if body.times is None or body.axis is None:
            return set()
        saved = set()
        for resonance in body.resonances():
            key = resonance.to_s()
            if key in (body.angles if resonance.type == "mmr" else body.angles_unwrapped):
                saved.add(key)
        return saved

    @classmethod
    def _build_simulation_json(cls, config, bodies, simulation) -> Dict[str, Any]:
        config_dict = cls._config_to_dict(config)
        data = {
            "config": config_dict,
            "simulation": {
                "number_of_bodies": len(bodies),
                "bodies": [cls._body_to_dict(body) for body in bodies],
                "time_unit": "years",
                "data_files": cls._data_files_manifest(bodies, config, simulation),
            },
        }
        if simulation is not None and hasattr(simulation, "planets_saved"):
            data["simulation"]["planets_saved"] = list(simulation.planets_saved())
        if simulation is not None and getattr(simulation, "ensembles", None):
            from resonances.simulation import clone_store

            data["simulation"]["ensembles"] = clone_store.ensembles_entry(simulation)

        free = cls._free_elements_setup(bodies, config)
        if free is not None:
            data["simulation"]["free_elements"] = free

        if simulation is not None and hasattr(simulation, "running_time") and simulation.running_time:
            running_time = {k: cls._serialize_value(v) for k, v in simulation.running_time.items()}
            data["timing"] = {"running_time": running_time}

            times_sorted = sorted(
                [(k, v) for k, v in simulation.running_time.items() if v is not None],
                key=lambda x: x[1],
            )
            differences = {}
            for i in range(len(times_sorted) - 1):
                diff = (times_sorted[i + 1][1] - times_sorted[i][1]).total_seconds()
                key = f"{times_sorted[i][0]}_to_{times_sorted[i + 1][0]}"
                differences[key] = diff
            if differences:
                differences["total_time"] = (times_sorted[-1][1] - times_sorted[0][1]).total_seconds()
                data["timing"]["running_time_differences"] = differences

        return data

    @classmethod
    def _free_elements_setup(cls, bodies, config) -> Dict[str, Any]:
        """What the forced/free split was allowed to fit on this baseline.

        Run-level rather than per-body: which planetary modes the window can separate, and
        which of them survived into the fit, follows from the baseline alone. Recorded
        because a verdict read months later is not interpretable without it — the same
        object gives a different basis at 300 kyr and at 10 Myr.
        """
        from resonances.secular import free_elements as fe

        if not getattr(config, 'free_elements_enabled', False):
            return None
        fitted = next((body.free_elements for body in bodies if body.free_elements is not None), None)
        baseline = fitted.baseline if fitted is not None else float(config.tmax_yrs)
        if baseline <= 0:
            return None

        setup = {
            "baseline_years": baseline,
            "resolution_arcsec_per_year": fe.ARCSEC_PER_TURN / baseline,
            "cluster_threshold_arcsec_per_year": 2.0 * fe.ARCSEC_PER_TURN / baseline,
            "frequencies": dict(fe.PLANETARY_FREQUENCIES),
        }
        for kind, priority in (('e', 'e'), ('i', 'i')):
            basis, dropped, clusters = fe.choose_basis(fe.candidate_frequencies(priority), baseline)
            setup[f"clusters_{kind}"] = clusters
            setup[f"basis_{kind}"] = list(basis)
            setup[f"dropped_{kind}"] = dropped
        return setup

    @classmethod
    def _drop_removed_plot_kinds(cls, config: Dict[str, Any]) -> None:
        """Let simulations saved before a plot kind was removed still be restored."""
        from resonances.simulation.config import PLOT_KINDS

        plots = config.get("plots")
        if isinstance(plots, list):
            removed = [kind for kind in plots if kind not in PLOT_KINDS]
            if removed:
                logger.warning(f"Ignoring removed plot kind(s) {', '.join(removed)} (see docs/plots.md)")
                config["plots"] = [kind for kind in plots if kind in PLOT_KINDS]

    @classmethod
    def _config_to_dict(cls, config) -> Dict[str, Any]:
        config_dict = {key: cls._serialize_value(value) for key, value in vars(config).items() if not key.startswith("_")}
        config_dict["tmax"] = cls._serialize_value(config.tmax)
        config_dict["tmax_yrs"] = cls._serialize_value(config.tmax_yrs)
        return config_dict

    @classmethod
    def _body_to_dict(cls, body: Body) -> Dict[str, Any]:
        return {
            "name": body.name,
            "type": body.type,
            "mass": cls._serialize_value(body.mass),
            "initial_data": cls._serialize_value(body.initial_data),
            "resonances": [cls._resonance_to_dict(res, body) for res in body.resonances()],
        }

    @classmethod
    def _resonance_to_dict(cls, resonance, body: Optional[Body] = None) -> Dict[str, Any]:
        """A resonance of a body in simulation.json; a derived row also has `added_by` and `parent`."""
        key = resonance.to_s()
        entry = {
            "type": getattr(resonance, "type", None),
            "key": key,
            "display": key,
        }
        added_by, parent = body.resonance_origin(key) if body is not None else ("", "")
        if parent:
            entry["added_by"] = added_by
            entry["parent"] = parent
        return entry

    @classmethod
    def _data_files_manifest(cls, bodies, config, simulation) -> Dict[str, Any]:
        """The data files of this run that are on disk (the SAVE mode writes only some bodies' files), as paths
        relative to save_path (also for an absolute data folder, so a moved run folder restores): in the data folder
        (`config.data_folder`), else at the top or in data/ (MANIFEST_DATA_FOLDERS: a run resumed across the change)."""
        save_path = Path(config.save_path)
        folders = list(dict.fromkeys([str(getattr(config, 'data_folder', '') or '')] + list(MANIFEST_DATA_FOLDERS)))

        def written(names):
            found = []
            for name in names:
                path = next((save_path / folder / name for folder in folders if (save_path / folder / name).exists()), None)
                if path is not None:
                    found.append(Path(os.path.relpath(path, save_path)).as_posix())
            return found

        manifest = {
            "bodies": written(f"data-{body.name}.csv" for body in bodies),
            "periodograms": written(f"data-{body.name}-periodograms.csv" for body in bodies),
        }
        free_omega = written(f"{body.name}-omega-free.csv" for body in bodies)
        if free_omega:
            manifest["free_omega"] = free_omega
        for name in ("summary", "coherence", "clones") + MMR_TABLES:
            if (save_path / f"{name}.csv").exists():
                manifest[name] = f"{name}.csv"
        if config.save_planets:
            # the planets this run kept: integrated here, or by the workers of a batched run (`planets_saved`)
            planet_names = []
            if simulation is not None and hasattr(simulation, "planets_saved"):
                planet_names = list(simulation.planets_saved())
            elif simulation is not None and getattr(simulation, "integration_engine", None) is not None:
                planet_names = list(simulation.integration_engine.planets_data)
            manifest["planets"] = written(f"data-planet-{planet}.csv" for planet in planet_names)
        return manifest

    @classmethod
    def _serialize_value(cls, value: Any):
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, set):
            return [cls._serialize_value(v) for v in value]
        if isinstance(value, (list, tuple)):
            return [cls._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: cls._serialize_value(v) for k, v in value.items()}
        return value

    @classmethod
    def _restore_running_time(cls, sim: "Simulation", timing_data: Dict[str, Any]) -> None:
        running_time = timing_data.get("running_time", {})
        if not running_time:
            return
        parsed = {key: cls._parse_datetime(value) for key, value in running_time.items()}
        sim.running_time.update(parsed)

    @classmethod
    def _build_body_from_json(cls, sim: "Simulation", body_data: Dict[str, Any]) -> Body:
        body = Body()
        body.initial_data = body_data.get("initial_data", {})
        body.name = body_data.get("name", "")
        body.type = body_data.get("type", body.type)
        body.mass = body_data.get("mass", body.initial_data.get("mass", 0.0))

        mmrs = []
        seculars = []
        lidovs = []

        derived = {}
        for res in body_data.get("resonances", []):
            key = res.get("key")
            if not key:
                continue
            resonance = create_resonance(key)
            if res.get("parent"):
                derived[resonance.to_s()] = res["parent"]
            if resonance.type == "mmr":
                mmrs.append(resonance)
            elif resonance.type == "secular":
                seculars.append(resonance)
            elif resonance.type == "lidov_kozai":
                lidovs.append(resonance)

        body.mmrs = mmrs
        body.secular_resonances = seculars
        body.lidov_kozai_resonances = lidovs
        body.derived_resonances = derived

        for mmr in body.mmrs:
            if mmr.planets_names:
                mmr.index_of_planets = sim.body_manager.get_index_of_planets(mmr.planets_names)

        return body

    @classmethod
    def _restore_body_data(cls, sim: "Simulation", base_dir: Path, files_manifest: Dict[str, Any]) -> None:
        # names as strings: the manifest's file names are, a body may be named by a number. The names are relative
        # to the run folder (data/data-<body>.csv; data-<body>.csv in runs before the data folder)
        if "bodies" in files_manifest:
            body_files_by_name = {Path(fname).stem.removeprefix("data-"): fname for fname in files_manifest["bodies"]}
        else:
            body_files_by_name = {str(body.name): cls._first_existing(base_dir, f"data-{body.name}.csv", sim.config) for body in sim.bodies}

        for body in sim.bodies:
            if str(body.name) not in body_files_by_name:  # not saved under the SAVE mode
                continue
            body_csv = base_dir / body_files_by_name[str(body.name)]
            if not body_csv.exists():
                logger.warning(f"Body data file not found for {body.name}: {body_csv}")
                continue

            # round_trip: the default parser may change the last bit, and the analysis must read what the run read
            df = pd.read_csv(body_csv, index_col=0, float_precision="round_trip")
            cls._populate_body_from_dataframe(sim, body, df)

            periodogram_csv = body_csv.parent / f"data-{body.name}-periodograms.csv"  # next to the body's series
            if periodogram_csv.exists() and periodogram_csv.stat().st_size > 0:
                try:
                    periodograms = pd.read_csv(periodogram_csv)
                except pd.errors.EmptyDataError:  # written empty by versions that saved no-periodogram files
                    continue
                cls._populate_periodograms(body, periodograms)

    @classmethod
    def _populate_body_from_dataframe(cls, sim: "Simulation", body: Body, df: pd.DataFrame) -> None:
        times_years = df["times"].values if "times" in df.columns else None
        if times_years is not None:
            sim.times = np.asarray(times_years) * (2 * np.pi)
            body.times = sim.times
            body.times_years = np.asarray(times_years, dtype=float)

        body.axis = cls._get_column(df, "a")
        body.ecc = cls._get_column(df, "e")
        body.inc = cls._get_column(df, "inc")
        body.Omega = cls._get_column(df, "Omega")
        body.omega = cls._get_column(df, "omega")
        body.M = cls._get_column(df, "M")
        body.longitude = cls._get_column(df, "longitude")
        body.varpi = cls._get_column(df, "varpi")
        body.axis_filtered = cls._get_column(df, "a_filtered")

        for resonance in body.resonances():
            key = resonance.to_s()
            if f"{key}_angle_unwrapped" in df.columns:
                body.angles_unwrapped[key] = cls._get_column(df, f"{key}_angle_unwrapped")
            if f"{key}_angle" in df.columns:
                body.angles[key] = cls._get_column(df, f"{key}_angle")

            # MMR files hold no filtered columns (the analysis reproduces sigma_f); older files may
            filtered_unwrapped = cls._get_column(df, f"{key}_angle_filtered_unwrapped")
            filtered = cls._get_column(df, f"{key}_angle_filtered")
            if filtered_unwrapped is not None:
                body.angles_filtered_unwrapped[key] = filtered_unwrapped
            if filtered is not None:
                body.angles_filtered[key] = filtered

            osculating = cls._get_column(df, f"{key}_angle_osculating")
            proper = cls._get_column(df, f"{key}_angle_proper")
            if osculating is not None:
                body.secular_angles_osculating[key] = osculating
            if proper is not None:
                body.secular_angles_proper[key] = proper

    @classmethod
    def _populate_periodograms(cls, body: Body, df: pd.DataFrame) -> None:
        if "a_frequency" in df.columns and "a_power" in df.columns:
            body.axis_periodogram_frequency = df["a_frequency"].values
            body.axis_periodogram_power = df["a_power"].values

        for resonance in body.resonances():
            key = resonance.to_s()
            freq_col = f"{key}_frequency"
            power_col = f"{key}_power"
            if freq_col in df.columns and power_col in df.columns:
                body.periodogram_frequency[key] = df[freq_col].values
                body.periodogram_power[key] = df[power_col].values

    @classmethod
    def _restore_planets(cls, sim: "Simulation", base_dir: Path, files_manifest: Dict[str, Any]) -> None:
        planet_files = files_manifest.get("planets") or []
        if not planet_files:
            return

        for filename in planet_files:
            planet_csv = base_dir / filename
            if not planet_csv.exists():
                continue
            planet_name = planet_csv.stem.replace("data-planet-", "")
            df = pd.read_csv(planet_csv, float_precision="round_trip")
            # Same shape as after a live run: one array per orbital element.
            sim.integration_engine.planets_data[planet_name] = {column: df[column].to_numpy() for column in df.columns}

    @classmethod
    def _restore_free_elements(cls, sim: "Simulation", base_dir: Path, files_manifest: Dict[str, Any], setup: Dict[str, Any]) -> None:
        """Restore the saved free-omega series needed by the ZLK diagnostic plots."""
        from resonances.secular.free_elements import from_series_frame

        files = files_manifest.get("free_omega") or []
        by_name = {Path(filename).name.removesuffix("-omega-free.csv"): filename for filename in files}
        for body in sim.bodies:
            filename = by_name.get(body.name)
            if not filename:
                continue
            path = base_dir / filename
            if path.exists():
                body.free_elements = from_series_frame(pd.read_csv(path), setup)

    @staticmethod
    def _first_existing(base_dir: Path, filename: str, config) -> str:
        """`filename` in the run's data folder when it is there, else in data/, else at the top (runs before the data
        folder; MANIFEST_DATA_FOLDERS)."""
        for folder in (str(getattr(config, "data_folder", "") or ""), *MANIFEST_DATA_FOLDERS):
            candidate = base_dir / folder / filename
            if candidate.exists():
                return Path(os.path.relpath(candidate, base_dir)).as_posix()
        return filename

    @classmethod
    def _get_column(cls, df: pd.DataFrame, col: str) -> Optional[np.ndarray]:
        if col not in df.columns:
            return None
        return df[col].values

    @classmethod
    def _parse_datetime(cls, value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                try:
                    return datetime_from_string(value)
                except Exception:
                    return None
        return None
