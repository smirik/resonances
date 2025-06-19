import datetime
import astdys.util
import pytest
import resonances
import astdys


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


@pytest.fixture(autouse=True)
def setup_test_config():
    """Setup test configuration before each test and restore after."""
    original_save_path = resonances.config.get('SAVE_PATH')
    original_plot_path = resonances.config.get('PLOT_PATH')

    resonances.config.set('SAVE_PATH', 'cache/tests')
    resonances.config.set('PLOT_PATH', 'cache/tests')

    yield

    resonances.config.set('SAVE_PATH', original_save_path)
    resonances.config.set('PLOT_PATH', original_plot_path)


def create_test_simulation_for_solar_system(save=None, plot=None, save_summary=False):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    sim = resonances.Simulation(
        date=astdys.util.convert_mjd_to_datetime(60000),
        save_path=f'cache/tests/test_simulation_{timestamp}',
        plot_path=f'cache/tests/test_simulation_{timestamp}',
    )
    sim.create_solar_system()

    # create to speedup
    sim.config.tmax = 20
    sim.config.dt = 1
    sim.config.Nout = 10
    sim.config.libration_period_min = 1
    sim.config.integrator = 'whfast'
    sim.config.integrator_corrector = None
    sim.config.save_summary = save_summary
    sim.config.save = save
    sim.config.plot = plot

    return sim


def add_test_asteroid_to_simulation(sim: resonances.Simulation):
    elem = get_3body_elements_sample()
    mmr = resonances.create_mmr('4J-2S-1')
    sim.add_body(elem, mmr, name='asteroid')
    return sim


def set_fast_integrator():
    resonances.config.set('INTEGRATION_INTEGRATOR', 'SABA(10,6,4)')
    resonances.config.set('INTEGRATION_DT', 5.0)
    resonances.config.set('INTEGRATION_TMAX', 200000)
    resonances.config.set('SAVE_PATH', 'cache/tests')
    resonances.config.set('PLOT_PATH', 'cache/tests')
    resonances.config.set('PLOT_MODE', None)
    resonances.config.set('SAVE_MODE', None)


def reset_fast_integrator():
    resonances.config.set('INTEGRATION_INTEGRATOR', 'SABA(10,6,4)')
    resonances.config.set('INTEGRATION_DT', 1.0)
    resonances.config.set('INTEGRATION_TMAX', 628319)
    resonances.config.set('INTEGRATION_SAFE_MODE', 0)
    resonances.config.set('INTEGRATION_CORRECTOR', 17)
    resonances.config.set('SAVE_PATH', 'cache')
    resonances.config.set('PLOT_PATH', 'cache')
    resonances.config.set('PLOT_MODE', 'nonzero')
    resonances.config.set('SAVE_MODE', 'nonzero')
