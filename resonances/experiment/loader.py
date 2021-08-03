import json
import resonances


def create_simulation_from_json(json_file_src):
    with open(json_file_src, "r") as read_file:
        c_config = json.load(read_file)

    sim = resonances.Simulation()
    sim.setup(
        save=c_config['save'],
        plot=c_config['plot'],
        save_path=c_config['save_path'],
    )
    sim.create_solar_system()

    if 'integration.Nout' in c_config:
        sim.Nout = int(c_config['integration.Nout'])
    if 'integration.tmax' in c_config:
        sim.tmax = int(c_config['integration.tmax'])
    if 'integration.dt' in c_config:
        sim.dt = c_config['integration.dt']

    # @todo need to verify that data are full
    # Add checks for asteroids, resonances, Nout and stop, plot (or default values)
    for asteroid in c_config['asteroids']:
        if 'num' in asteroid['elem']:
            elem_or_num = asteroid['elem']['num']
        else:
            elem_or_num = asteroid['elem']
        # @todo validation
        for resonance in asteroid['resonances']:
            sim.add_body(
                elem_or_num, resonances.create_mmr(resonance['integers'], resonance['bodies']), '{}'.format(asteroid['elem']['label'])
            )

    return sim
