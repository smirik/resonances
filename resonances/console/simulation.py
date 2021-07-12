import json
import time

from resonances.console.console import console as cs
from resonances.simulation import shape
from resonances.resonance.three_body import ThreeBody
from resonances.resonance import integration


def calc_shape():
    start_time = time.time()
    cs.shape()
    with open(cs.args.config, "r") as read_file:
        c_config = json.load(read_file)

    cs.create_output_dir(c_config['save_path'])

    mmr_template = ThreeBody(
        c_config['resonance']['integers'], integration.index_of_planets(c_config['resonance']['bodies']), 10, c_config['label']
    )
    shape.run(
        c_config['elem'],
        c_config['variations'],
        mmr_template,
        c_config['save'],
        c_config['save_path'],
        c_config['plot'],
        c_config['dump'],
    )

    print("--- %s seconds ---" % (time.time() - start_time))
