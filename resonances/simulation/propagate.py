"""Elements carried from their own epoch to the simulation date in the package's model (MPC_EPOCH_POLICY=propagate,
the default).

The bodies are test particles in the planets the run uses (Horizons at the elements' epoch, the Moon when
`solar_system_moon`), integrated with IAS15. No asteroid perturbers, general relativity or non-gravitational forces:
over 138 days this reproduced MPC's own propagation of 433, 719 and 887 at the file's rounding (5e-8 AU, 5e-6 deg;
docs/config.md, "Minor Planet Center"; tests/resonances/test_mpc_slow.py checks 433). Longer spans are not validated;
MPC_PROPAGATE_MAX_DAYS, when set, refuses them.
"""

import copy
import math
from collections import defaultdict
from typing import Dict, List

from resonances.data.const import K
from resonances.data.util import mjd_from_datetime
from resonances.mpcorb import datetime_from_mjd
from resonances.simulation.integration import IntegrationEngine

ELEMENT_KEYS = ('a', 'e', 'inc', 'Omega', 'omega', 'M')


def propagate_elements(elements: List[dict], config) -> List[dict]:
    """Each element dict (angles in radians, 'epoch' MJD) carried to config.date; the other keys are kept and
    'propagated_from_epoch' records the original epoch. Bodies at one epoch share one integration."""
    target = mjd_from_datetime(config.date)
    groups: Dict[float, List[int]] = defaultdict(list)
    for index, elem in enumerate(elements):
        groups[float(elem['epoch'])].append(index)

    out = [dict(elem) for elem in elements]
    for epoch, indices in groups.items():
        at_epoch = copy.copy(config)
        at_epoch.date = datetime_from_mjd(epoch)
        engine = IntegrationEngine(at_epoch)
        engine.create_solar_system()
        sim = engine.sim
        n_massive = sim.N
        for index in indices:
            elem = elements[index]
            sim.add(m=0.0, primary=sim.particles[0], **{key: float(elem[key]) for key in ELEMENT_KEYS})
        sim.N_active = n_massive
        sim.integrator = 'ias15'
        sim.move_to_com()
        sim.integrate(sim.t + (target - epoch) * K)  # one time unit of the simulation (G = 1, AU, Msun) is 1/k days
        for offset, index in enumerate(indices):
            orbit = sim.particles[n_massive + offset].orbit(primary=sim.particles[0])
            out[index].update(
                a=orbit.a,
                e=orbit.e,
                inc=orbit.inc,
                Omega=orbit.Omega % (2 * math.pi),
                omega=orbit.omega % (2 * math.pi),
                M=orbit.M % (2 * math.pi),
                epoch=target,
                propagated_from_epoch=epoch,
            )
    return out
