import datetime
import os

import astdys.util
import resonances

# The output folder of this test process: one per pytest-xdist worker (gw0, gw1, ...) or per session (its pid), so
# that parallel workers and two sessions in one working directory never clean up each other's files (tests/CLAUDE.md)
TEST_CACHE = f"cache/tests/{os.environ.get('PYTEST_XDIST_WORKER') or f'session-{os.getpid()}'}"


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
