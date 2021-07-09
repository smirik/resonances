import rebound
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def by_mod(num):
    if num > np.pi:
        while num > np.pi:
            num = num - 2 * np.pi
    else:
        while num < -np.pi:
            num = num + 2 * np.pi
    return num


def create_solar_system_file():
    sim = rebound.Simulation()
    sim.add("Sun")
    sim.add("Mercury")
    sim.add("Venus")
    sim.add("Earth")
    sim.add("Mars")
    sim.add("Jupiter")
    sim.add("Saturn")
    sim.add("Uranus")
    sim.add("Neptune")
    sim.add("Pluto")
    # sim.add("99942")
    sim.add("463")
    sim.save("cache/solar.bin")
    sim.status()


def integrate(sim):
    # sim.dt = 0
    Noutputs = 1000
    year = 2.0 * np.pi  # One year in units where G=1
    times = np.linspace(0.0, 10.0 * year, Noutputs)
    x = np.zeros((2, Noutputs))
    y = np.zeros((2, Noutputs))
    z = np.zeros((2, Noutputs))

    sim.integrator = "ias15"
    sim.ri_ias15.min_dt = 1e-9
    sim.ri_ias15.epsilon = 1e-9
    # sim.integrator = "mercurius"  # IAS15 is the default integrator, so we actually don't need this line
    # sim.ri_mercurius.hillfac = 4.0
    sim.move_to_com()  # We always move to the center of momentum frame before an integration
    ps = (
        sim.particles
    )  # ps is now an array of pointers and will change as the simulation runs

    for i, time in enumerate(times):
        sim.integrate(time)
        x[0][i] = ps[3].x  # This stores the data which allows us to plot it later
        y[0][i] = ps[3].y
        z[0][i] = ps[3].z
        x[1][i] = ps[10].x
        y[1][i] = ps[10].y
        z[1][i] = ps[10].z

    distance = np.sqrt(
        np.square(x[0] - x[1]) + np.square(y[0] - y[1]) + np.square(z[0] - z[1])
    )

    # fig = plt.figure(figsize=(12, 5))
    # ax = plt.subplot(111)
    # ax.set_xlabel("time [yrs]")
    # ax.set_ylabel("distance [AU]")
    # plt.plot(times / year, distance)
    # plt.savefig("2.png")
    closeencountertime = times[np.argmin(distance)] / year
    sim.status()
    print(
        "Minimum distance (%f AU or %f km) occured at time: %f years."
        % (
            np.min(distance),
            np.min(distance) * rebound.units.lengths_SI["au"] / 1000.0,
            closeencountertime,
        )
    )


def int_res(sim):
    Nout = 10000
    # tmax = 1.0e6
    tmax = 6.28e5
    NBodies = 3
    IndexOfBodies = [5, 6, 10]

    sim.integrator = "whfast"
    sim.dt = 1.0
    # sim.integrator = "ias15"

    x = np.zeros((NBodies, Nout))
    axis = np.zeros((NBodies, Nout))
    ecc = np.zeros((NBodies, Nout))
    longitude = np.zeros((NBodies, Nout))
    varpi = np.zeros((NBodies, Nout))
    angle = np.zeros(Nout)
    angle2 = np.zeros(Nout)

    times = np.linspace(0.0, tmax, Nout)
    ps = sim.particles

    m1 = 4.0
    m2 = -2.0
    m = -1.0
    p = -1.0

    for i, time in enumerate(times):
        sim.integrate(time)
        os = sim.calculate_orbits(primary=ps[0])
        for j, planetNumber in enumerate(IndexOfBodies):
            body = os[planetNumber - 1]
            axis[j][i], ecc[j][i], longitude[j][i], varpi[j][i] = (
                body.a,
                body.e,
                body.l,
                body.Omega + body.omega,
                # body.pomega,
            )
            # break
        # break
        angle[i] = rebound.mod2pi(
            m1 * longitude[0][i]
            + m2 * longitude[1][i]
            + m * longitude[2][i]
            + p * varpi[2][i]
        )
        # print(by_mod(tmp), rebound.mod2pi(tmp))

    fig = plt.figure(figsize=(12, 5))
    plt.ylim((0, 2 * np.pi))
    plt.plot(times / (2 * np.pi), angle, label="Angle", linestyle="", marker=",")
    plt.savefig("cache/3.png")

    plt.figure(figsize=(12, 5))
    plt.ylim((2.35, 2.45))
    plt.plot(times, axis[2], label="semiaxis", linestyle="", marker=",")
    plt.savefig("cache/5.png")

    Npts = 1000

    periods = np.linspace(3000 / (2 * np.pi), 200000 / (2 * np.pi), Npts)
    ang_freqs = 2 * np.pi / periods
    power = signal.lombscargle(times / (2 * np.pi), angle, ang_freqs)
    fig = plt.figure(figsize=(12, 5))
    plt.plot(periods, np.sqrt(4 * power / Npts))
    plt.savefig("cache/6.png")

    periods = np.linspace(3000 / (2 * np.pi), 200000 / (2 * np.pi), Npts)
    ang_freqs = 2 * np.pi / periods
    power = signal.lombscargle(times / (2 * np.pi), axis[2], ang_freqs)
    fig = plt.figure(figsize=(12, 5))
    plt.plot(periods, np.sqrt(4 * power / Npts))
    plt.savefig("cache/7.png")

    # logPmin = np.log10(100.0)
    # logPmax = np.log10(1.0e5)

    # Ps = np.logspace(logPmin, logPmax, Npts)
    # Ps = np.linspace(1000, 100000)
    # ws = np.asarray([2 * np.pi / P for P in Ps])

    # periodogram = signal.lombscargle(times, angle, ws)

    # fig = plt.figure(figsize=(12, 5))
    # ax = plt.subplot(111)
    # ax.plot(Ps, np.sqrt(4 * periodogram / Nout))
    # # ax.set_xscale("log")
    # ax.set_xlim([10 ** logPmin, 10 ** logPmax])
    # # ax.set_ylim([0, 0.15])
    # ax.set_xlabel("Period (yrs)")
    # ax.set_ylabel("Power")

    # np.savetxt("foo.csv", angle, delimiter=",")
    print("stop")
    sim.status()


def main():
    solar_file_src = "cache/solar.bin"
    solar_file = Path(solar_file_src)

    if solar_file.exists():
        print("It exists")
    else:
        print("there is no file. Creating")
        create_solar_system_file()

    sim = rebound.Simulation(solar_file_src)

    # integrate(sim)
    int_res(sim)
    return 1

    fig, ax_main = rebound.OrbitPlot(
        sim, unitlabel="[AU]", color=True, xlim=[-1.5, 1.5], ylim=[-1.5, 1.5]
    )
    fig.savefig("1.png")
    # fig.show()
    print(fig)
    print(ax_main)

    # for orbit in sim.calculate_orbits():
    #     print(orbit)


def test2():
    sim = rebound.Simulation()
    sim.add(m=1.0)  # Central object
    sim.add(m=1e-3, a=1.0, e=0.1)  # Jupiter mass planet
    sim.add(a=1.4, e=0.1)  # Massless test particle
    sim.integrate(100.0)
    for p in sim.particles:
        print(p.x, p.y, p.z)
    for o in sim.calculate_orbits():
        print(o)
    print("Hello from test2")
