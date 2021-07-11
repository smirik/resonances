import numpy as np
import rebound
from pathlib import Path

from resonances.resonance.three_body import ThreeBody
from resonances.data.astdys import astdys
from resonances.resonance.libration import libration


def list_of_planets():
    planets = [
        'Sun',
        'Mercury',
        'Venus',
        'Earth',
        'Mars',
        'Jupiter',
        'Saturn',
        'Uranus',
        'Neptune',
        'Pluto',
    ]
    return planets


# @todo additional checks required
def index_of_planets(planets_names):
    planets = list_of_planets()
    if isinstance(planets_names, list):
        arr = []
        for name in planets_names:
            arr.append(planets.index(name))
        return arr

    return planets.index(planets_names)


def create_solar_system():
    solar_file_src = "cache/solar.bin"
    solar_file = Path(solar_file_src)

    if solar_file.exists():
        sim = rebound.Simulation(solar_file_src)
        return sim

    sim = rebound.Simulation()
    sim.add(list_of_planets(), date='2020-12-17 00:00')
    sim.save("cache/solar.bin")
    return sim


def add_asteroid_by_elem(sim: rebound.Simulation, elem):
    sim.add(
        m=0.0,
        a=elem['a'],
        e=elem['e'],
        inc=elem['inc'],
        Omega=elem['Omega'],
        omega=elem['omega'],
        M=elem['M'],
        date=astdys.date,
        primary=sim.particles[0],
    )
    return sim


def add_asteroid_by_num(sim: rebound.Simulation, asteroid_num):
    elem = astdys.search(asteroid_num)
    return add_asteroid_by_elem(elem)


def add_asteroids(sim: rebound.Simulation, asteroids):
    for asteroid in asteroids:
        sim = add_asteroid_by_num(asteroid)
    # os = sim.calculate_orbits(primary=sim.particles[0])
    return sim


def setup(self, sim, Nout, tmax, NBodies, integrator="whfast", dt=1.0):
    self.Nout = Nout
    self.tmax = tmax
    self.NBodies = NBodies

    sim.integrator = integrator
    sim.dt = dt


def integrate(
    sim: rebound.Simulation,
    mmrs,
    tmax=1.0e6,
    Nout=10000,
    integrator='whfast',
    dt=1.0,
):
    num_mmrs = len(mmrs)

    axis, ecc, longitude, varpi, angle = (
        np.zeros((num_mmrs, Nout)),
        np.zeros((num_mmrs, Nout)),
        np.zeros((num_mmrs, Nout)),
        np.zeros((num_mmrs, Nout)),
        np.zeros((num_mmrs, Nout)),
    )

    times = np.linspace(0.0, tmax, Nout)

    sim.integrator = integrator
    sim.dt = dt
    sim.N_active = 10
    sim.move_to_com()
    # sim.integrator = "ias15"
    ps = sim.particles

    for i, time in enumerate(times):
        sim.integrate(time)
        os = sim.calculate_orbits(primary=ps[0])

        for k, mmr in enumerate(mmrs):
            body = os[mmr.index_of_body - 1]  # because Sun is not in os
            axis[k][i], ecc[k][i], longitude[k][i], varpi[k][i] = (
                body.a,
                body.e,
                body.l,
                body.Omega + body.omega,
            )
            angle[k][i] = mmr.angle(os)

    return {"times": times, "axis": axis, "ecc": ecc, "angle": angle}


# @todo validation of mmrs, data
def librations(data, mmrs, Nout):
    status = np.zeros(len(mmrs))
    libration_data = []
    for i, mmr in enumerate(mmrs):
        libration_data.append(libration.libration(data['times'] / (2 * np.pi), data['angle'][i], Nout))
        if libration_data[i]['flag']:
            if libration_data[i]['pure']:
                status[i] = 2
            else:
                status[i] = 1
        else:
            status[i] = 0
    return {'status': status, 'libration_data': libration_data}
