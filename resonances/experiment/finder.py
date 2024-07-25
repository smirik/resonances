import math
import astdys
import resonances


def run(mmr: resonances.MMR, dump=100, max_iterations=1000, plot=False):
    df = astdys.search_by_axis(mmr.resonant_axis)
    asteroids = df['num'].tolist()

    num_particles = len(asteroids)
    num_iterations = int(math.ceil(num_particles / dump))
    for j in range(num_iterations):
        if j > (max_iterations - 1):
            resonances.logger.info(
                'Terminating because the app has reached the limit specified in max_iterations parameter ({}).'.format(max_iterations)
            )
            break

        sim = resonances.Simulation()
        sim.save, sim.save_path, sim.plot, sim.save_only_undetermined = False, 'cache/finder', plot, True
        sim.create_solar_system()

        # Find the number of calculations for a given step. It varies for the last one.
        if j < num_iterations - 1:
            num = dump
        else:
            num = num_particles % dump
            if num == 0:
                num = dump

        for i in range(num):
            key = j * dump + i
            sim.add_body(asteroids[key], mmr, '{}-{}'.format(asteroids[key], key))

        sim.run()
        resonances.logger.info('Iteration {} has finished. Processed from {}. Starting the new one.'.format(j, j * dump))
