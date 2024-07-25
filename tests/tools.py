import astdys.util
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


def create_test_simulation_for_solar_system(save=None, plot=None, save_summary=False):
    sim = resonances.Simulation()

    sim.create_solar_system(date=astdys.util.convert_mjd_to_date(60000.0))

    # create to speedup
    sim.tmax = 20
    sim.dt = 1
    sim.Nout = 10
    sim.libration_period_min = 1
    sim.integrator = 'whfast'
    sim.integrator_corrector = None
    sim.save_path = 'cache/tests'
    sim.plot_path = 'cache/tests'
    sim.save_summary = save_summary
    sim.save = save
    sim.plot = plot

    return sim


def add_test_asteroid_to_simulation(sim):
    elem = get_3body_elements_sample()
    mmr = resonances.create_mmr('4J-2S-1')
    sim.add_body(elem, mmr, name='asteroid')
    return sim
