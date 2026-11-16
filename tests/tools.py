import dataclasses
import datetime
import math
import os

import astdys.util
import numpy as np
import resonances

# The output folder of this test process: one per pytest-xdist worker (gw0, gw1, ...) or per session (its pid), so
# that parallel workers and two sessions in one working directory never clean up each other's files (tests/CLAUDE.md)
TEST_CACHE = f"cache/tests/{os.environ.get('PYTEST_XDIST_WORKER') or f'session-{os.getpid()}'}"


def same_values(a, b) -> bool:
    """Equality of result objects with nan equal to nan: dataclasses field by field, mappings, sequences and arrays
    element by element. Since Python 3.13 a dataclass compares its fields with ==, so two equal objects holding nan
    are unequal (dev/pitfalls.md); compare such objects with this, not with ==."""
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        return type(a) is type(b) and all(same_values(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_values(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same_values(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc' and b.dtype.kind in 'fc'))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return bool(a == b)


# 463 Lola
def get_3body_elements_sample():
    return {
        "a": 2.398473292330785,
        "e": 0.22009324739445424,
        "inc": 0.23634522279656767,
        "Omega": 0.63690730765078,
        "omega": 5.753528892504344,
        "M": 6.136002589657356,
        "epoch": 60000.0,
    }


# 624 Hektor
def get_2body_elements_sample():
    return {
        "a": 5.270635994654261,
        "e": 0.02299891948665412,
        "inc": 0.316860843630838,
        "Omega": 5.9827890156476125,
        "omega": 3.141712198994225,
        "M": 5.033788240164378,
        "epoch": 60000.0,
    }


def create_test_simulation_for_solar_system(save=None, plot=None, save_summary=False):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    sim = resonances.Simulation(
        date=astdys.util.convert_mjd_to_datetime(60000),
        save_path=f'{TEST_CACHE}/test_simulation_{timestamp}',
        plot_path=f'{TEST_CACHE}/test_simulation_{timestamp}',
    )
    sim.create_solar_system()

    # create to speedup
    sim.config.tmax = 20
    sim.config.dt = 1
    sim.config.Nout = 10
    sim.config.libration_period_min = 1
    sim.config.integrator = 'whfast'
    sim.config.integration_corrector = None
    sim.config.save_summary = save_summary
    sim.config.save = save
    sim.config.plot = plot

    return sim


def add_test_asteroid_to_simulation(sim: resonances.Simulation):
    elem = get_3body_elements_sample()
    mmr = resonances.create_mmr('4J-2S-1')
    sim.add_body(elem, mmr, name='asteroid')
    return sim


def accepted_episodes(body, key):
    """The accepted libration episodes of an MMR row: its episodes.csv rows from the analysis in memory."""
    return [row for row in body.mmr_analyses[key].episode_rows() if row['accepted']]
