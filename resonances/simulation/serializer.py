import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from resonances.body import Body
from resonances.logger import logger
from resonances.resonance.factory import create_resonance
from resonances.data.util import datetime_from_string

if TYPE_CHECKING:
    from resonances.simulation.simulation import Simulation


class SimulationSerializer:
    @classmethod
    def save_simulation_json(cls, config, bodies, simulation) -> None:
        data = cls._build_simulation_json(config, bodies, simulation)
        save_path = Path(config.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        with (save_path / "simulation.json").open("w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def restore(cls, json_file_path: str, recompute_librations: bool = True) -> "Simulation":
        json_path = Path(json_file_path)
        logger.info(f"Restoring simulation from {json_path}")
        with json_path.open("r") as file:
            data = json.load(file)

        config = dict(data.get("config", {}))
        config["_skip_path_verification"] = True
        if isinstance(config.get("date"), str):
            config["date"] = cls._parse_datetime(config["date"])

        from resonances.simulation.simulation import Simulation

        logger.info(f"Creating simulation from config: {config['name']}")
        sim = Simulation(**config)
        logger.info(f"Simulation created: {sim.config.name}")

        cls._restore_running_time(sim, data.get("timing", {}))

        bodies_data = data.get("simulation", {}).get("bodies", [])
        for body_data in bodies_data:
            body = cls._build_body_from_json(sim, body_data)
            sim.body_manager.bodies.append(body)

        logger.info(f"Restoring body data")
        cls._restore_body_data(sim, json_path.parent, data.get("simulation", {}).get("data_files", {}))

        logger.info(f"Restoring planets")
        cls._restore_planets(sim, json_path.parent, data.get("simulation", {}).get("data_files", {}))

        logger.info(f"Identifying librations")
        try:
            if recompute_librations and sim.bodies:
                sim.identify_librations()
        except Exception as e:
            logger.error(f"Cannot restore librations: {e}")
            raise e

        return sim

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
                key = f"{times_sorted[i][0]}_to_{times_sorted[i+1][0]}"
                differences[key] = diff
            if differences:
                differences["total_time"] = (times_sorted[-1][1] - times_sorted[0][1]).total_seconds()
                data["timing"]["running_time_differences"] = differences

        return data

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
            "resonances": [cls._resonance_to_dict(res) for res in body.resonances()],
        }

    @classmethod
    def _resonance_to_dict(cls, resonance) -> Dict[str, Any]:
        key = resonance.to_s()
        return {
            "type": getattr(resonance, "type", None),
            "key": key,
            "display": resonance.to_s(),
        }

    @classmethod
    def _data_files_manifest(cls, bodies, config, simulation) -> Dict[str, Any]:
        body_files = [f"data-{body.name}.csv" for body in bodies]
        periodogram_files = [f"data-{body.name}-periodograms.csv" for body in bodies]
        manifest = {"bodies": body_files, "periodograms": periodogram_files}
        if config.save_summary:
            manifest["summary"] = "summary.csv"
        if config.save_planets:
            planet_names = []
            if simulation is not None and getattr(simulation, "integration_engine", None) is not None:
                planets_data = simulation.integration_engine.planets_data
                if planets_data:
                    planet_names = list(planets_data.keys())
            manifest["planets"] = [f"data-planet-{planet}.csv" for planet in planet_names]
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

        for res in body_data.get("resonances", []):
            key = res.get("key")
            if not key:
                continue
            resonance = create_resonance(key)
            if resonance.type == "mmr":
                mmrs.append(resonance)
            elif resonance.type == "secular":
                seculars.append(resonance)
            elif resonance.type == "lidov_kozai":
                lidovs.append(resonance)

        body.mmrs = mmrs
        body.secular_resonances = seculars
        body.lidov_kozai_resonances = lidovs

        for mmr in body.mmrs:
            if mmr.planets_names:
                mmr.index_of_planets = sim.body_manager.get_index_of_planets(mmr.planets_names)

        return body

    @classmethod
    def _restore_body_data(cls, sim: "Simulation", base_dir: Path, files_manifest: Dict[str, Any]) -> None:
        body_files = files_manifest.get("bodies", [])
        body_files_by_name = {Path(fname).stem.replace("data-", ""): fname for fname in body_files}

        for body in sim.bodies:
            body_csv = base_dir / body_files_by_name.get(body.name, f"data-{body.name}.csv")
            if not body_csv.exists():
                logger.warning(f"Body data file not found for {body.name}: {body_csv}")
                continue

            df = pd.read_csv(body_csv, index_col=0)
            cls._populate_body_from_dataframe(sim, body, df)

            periodogram_csv = base_dir / f"data-{body.name}-periodograms.csv"
            if periodogram_csv.exists():
                periodograms = pd.read_csv(periodogram_csv)
                cls._populate_periodograms(body, periodograms)

    @classmethod
    def _populate_body_from_dataframe(cls, sim: "Simulation", body: Body, df: pd.DataFrame) -> None:
        times_years = df["times"].values if "times" in df.columns else None
        if times_years is not None:
            sim.times = np.asarray(times_years) * (2 * np.pi)
            body.times = sim.times

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
            df = pd.read_csv(planet_csv)
            sim.integration_engine.planets_data[planet_name] = df.to_dict(orient="records")

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
