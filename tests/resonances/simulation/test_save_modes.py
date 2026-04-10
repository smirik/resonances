"""Test that save/plot modes correctly filter which body files are created.

Runs a short real simulation with a mix of resonant and non-resonant
asteroids, then checks that data-*.csv and plot files exist only for
bodies matching the configured save/plot mode.
"""

import os
import glob
import pytest

from tests.tools import run_short_simulation

# 463 is resonant (4J-2S-1), 490 is transient, 1 (Ceres) is non-resonant for LK
TEST_ASTEROIDS = [463, 490, 1]


def _data_files(out_dir):
    """Return set of asteroid names that have data-*.csv files."""
    files = glob.glob(f"{out_dir}/data-*.csv")
    names = set()
    for f in files:
        base = os.path.basename(f)
        if "-periodograms" in base:
            continue
        name = base.replace("data-", "").replace(".csv", "")
        names.add(name)
    return names


def _plot_files(out_dir):
    """Return set of asteroid names that have plot PNG files."""
    files = glob.glob(f"{out_dir}/**/*.png", recursive=True) + glob.glob(f"{out_dir}/*.png")
    names = set()
    for f in files:
        base = os.path.basename(f)
        name = base.split("-")[0]
        if name.isdigit():
            names.add(name)
    return names


@pytest.mark.slow
class TestSaveModes:
    """Test save mode filtering for body data files."""

    def test_save_none(self):
        """save='none' should create no data-*.csv files."""
        out_dir, sim = run_short_simulation(TEST_ASTEROIDS, "none", "none", "save_none")
        assert len(_data_files(out_dir)) == 0

    def test_save_all(self):
        """save='all' should create data files for ALL bodies."""
        out_dir, sim = run_short_simulation(TEST_ASTEROIDS, "all", "none", "save_all")
        saved = _data_files(out_dir)
        for ast in TEST_ASTEROIDS:
            assert str(ast) in saved, f"data-{ast}.csv missing with save='all'"

    def test_save_nonzero(self):
        """save='nonzero' should create data files only for status != 0."""
        out_dir, sim = run_short_simulation(TEST_ASTEROIDS, "nonzero", "none", "save_nonzero")
        saved = _data_files(out_dir)
        summary, _ = sim.data_manager.get_simulation_summary(sim.bodies)

        for _, row in summary.iterrows():
            name = str(row["name"])
            status = int(row["status"])
            if status != 0:
                assert name in saved, f"data-{name}.csv missing (status={status}) with save='nonzero'"
            else:
                assert name not in saved, f"data-{name}.csv exists (status=0) with save='nonzero'"

    def test_save_resonant(self):
        """save='resonant' should create data files only for status > 0."""
        out_dir, sim = run_short_simulation(TEST_ASTEROIDS, "resonant", "none", "save_resonant")
        saved = _data_files(out_dir)
        summary, _ = sim.data_manager.get_simulation_summary(sim.bodies)

        for _, row in summary.iterrows():
            name = str(row["name"])
            status = int(row["status"])
            if status > 0:
                assert name in saved, f"data-{name}.csv missing (status={status}) with save='resonant'"
            else:
                assert name not in saved, f"data-{name}.csv exists (status={status}) with save='resonant'"

    def test_summary_always_has_all_bodies(self):
        """summary.csv should contain ALL bodies regardless of save mode."""
        out_dir, sim = run_short_simulation(TEST_ASTEROIDS, "nonzero", "none", "summary_check")
        summary, _ = sim.data_manager.get_simulation_summary(sim.bodies)
        summary_names = set(summary["name"].astype(str))
        for ast in TEST_ASTEROIDS:
            assert str(ast) in summary_names, f"{ast} missing from summary.csv"


@pytest.mark.slow
class TestPlotModes:
    """Test plot mode filtering for plot files."""

    def test_plot_none(self):
        """plot='none' should create no plot files."""
        out_dir, sim = run_short_simulation(TEST_ASTEROIDS, "none", "none", "plot_none")
        assert len(_plot_files(out_dir)) == 0

    def test_plot_all(self):
        """plot='all' should create plot files for ALL bodies."""
        out_dir, sim = run_short_simulation(TEST_ASTEROIDS, "none", "all", "plot_all")
        plotted = _plot_files(out_dir)
        for ast in TEST_ASTEROIDS:
            assert str(ast) in plotted, f"plot for {ast} missing with plot='all'"

    def test_plot_nonzero(self):
        """plot='nonzero' should create plots only for status != 0."""
        out_dir, sim = run_short_simulation(TEST_ASTEROIDS, "none", "nonzero", "plot_nonzero")
        plotted = _plot_files(out_dir)
        summary, _ = sim.data_manager.get_simulation_summary(sim.bodies)

        for _, row in summary.iterrows():
            name = str(row["name"])
            status = int(row["status"])
            if status != 0:
                assert name in plotted, f"plot for {name} missing (status={status}) with plot='nonzero'"
            else:
                assert name not in plotted, f"plot for {name} exists (status=0) with plot='nonzero'"
