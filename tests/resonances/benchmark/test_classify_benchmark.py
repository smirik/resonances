#!/usr/bin/env python3
"""
Benchmark Tests
===============

These tests are excluded from regular test runs due to their long execution time.
Run with: make test-benchmark
"""

import pytest
import pandas as pd
from pathlib import Path

import resonances

BENCHMARK_DIR = Path(__file__).parent


def create_simulation():
    df_exp = pd.read_csv(BENCHMARK_DIR / "dataset.csv")
    expected_statuses = {}
    sim = resonances.Simulation(
        name="test_benchmark",
        integrator="SABA(10,6,4)",
        integration_years=100000,
        save="all",
        plot="all",
        plot_type="save",
        source="astdys",
        plot_config="full",
        plot_subfolder_strategy="status",
        plots=["evolution", "phase_portrait"],
        batch_size=100,
        batch_threshold=100,
    )

    # i = 0
    for _, row in df_exp.iterrows():
        name = str(row["name"])
        resonance = row["resonance"]
        status = int(row["status"])
        expected_statuses[f"{name}:{resonance}"] = status

        sim.add_body(name, resonance, name=name)
        # i += 1
        # if i > 10:
        #     break

    return sim, expected_statuses


def run_simulation() -> tuple[resonances.Simulation, dict]:
    sim, expected_statuses = create_simulation()
    sim.create_solar_system()
    sim.run(progress=True)
    return sim, expected_statuses


@pytest.mark.benchmark
def test_dataset():
    """Stub test to verify benchmark marker works."""
    sim, _expected_statuses = create_simulation()
    assert isinstance(sim, resonances.Simulation)
    assert isinstance(_expected_statuses, dict)
    assert len(_expected_statuses) > 0

    sim, expected_statuses = run_simulation()

    sim = resonances.SimulationSerializer.restore(
        f"{sim.config.save_path}/simulation.json"
    )  # restore the simulation to get all the bodies from batch

    for body in sim.bodies:
        for resonance in body.resonances():
            # 100% accuracy for 2 and 0
            expected_status = expected_statuses[f"{body.name}:{resonance.to_s()}"]
            actual_status = int(body.status(resonance))
            if actual_status != expected_status:
                resonances.logger.warning(f"{body.name}:{resonance.to_s()} {expected_status} {actual_status}")
            if expected_status in [2, -99]:
                assert (
                    actual_status == expected_status
                ), f"Expected status {expected_status} for {body.name}:{resonance.to_s()} but got {actual_status}"
            elif expected_status == -4:
                continue
            elif expected_status == 0:
                if actual_status not in [0, -4]:
                    assert actual_status in [
                        0,
                        -4,
                    ], f"Expected status {expected_status} for {body.name}:{resonance.to_s()} but got {actual_status}"
            else:
                if actual_status not in [
                    1,
                    -1,
                    3,
                    -4,
                    -99,
                ]:
                    assert actual_status in [
                        1,
                        -1,
                        3,
                        -4,
                        -99,
                    ], f"Expected status {expected_status} for {body.name}:{resonance.to_s()} but got {actual_status}"

    # assert False == True
