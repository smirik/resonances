import numbers
from typing import List, Optional, Union

import astdys
import numpy as np
from .config import SimulationConfig
from .propagate import propagate_elements
from resonances.body import Body
from resonances.data.const import SOLAR_SYSTEM_WITH_SUN
from resonances import mpcorb
from resonances.simulation.mpc_source import epoch_refusal, mpc_settings
from resonances.data.util import mjd_from_datetime
from resonances.logger import logger
from resonances.resonance.resonance import Resonance

# Elements whose epoch is within this of config.date (about 9 s) are taken as being at config.date.
EPOCH_TOLERANCE_DAYS = 1e-4


def propagation_warning(row: dict, gap_days: float, date) -> str:
    """The warning of one MPC row propagated to the run date: its label, the gap (date - epoch, days), U, and the
    arc and oppositions when the row has them."""
    quality = [f"U = {row['U'] or 'blank'}"]
    if str(row.get('arc', '')).strip():
        quality.append(f"arc {row['arc']}")
    if str(row.get('n_opp', '')).strip():
        quality.append(f"{row['n_opp']} oppositions")
    return (
        f"{mpcorb.row_label(row)}: MPC epoch MJD {float(row['epoch_mjd']):.5f} is {gap_days:+.1f} days from the run date "
        f"{date}; propagated to it (MPC_EPOCH_POLICY=propagate; {', '.join(quality)})"
    )


class BodyManager:
    """Manages addition and setup of celestial bodies."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.bodies: List[Body] = []
        self.planets = SOLAR_SYSTEM_WITH_SUN

    def get_index_of_planets(self, planets_names):
        """Get indices of planets by name."""
        return [self.planets.index(planet) for planet in planets_names]

    def get_body_elements(self, elem_or_num: Union[int, str, dict]) -> dict:
        """Orbital elements of a body: from config.source by number/name, or the dict itself."""
        if isinstance(elem_or_num, (int, str)):
            if self.config.source == 'astdys':
                elem = astdys.search(elem_or_num)
                if elem is None:
                    raise ValueError(f"Body {elem_or_num} is not in the AstDyS catalog")
                return elem
            elif self.config.source == 'mpc':
                found = self.mpc_elements([elem_or_num])
                if str(elem_or_num) not in found:
                    raise ValueError(f"Body {elem_or_num} is not in the MPC catalogue {self.config.mpc_snapshot['catalog']}")
                return found[str(elem_or_num)]
            else:
                import resonances

                return resonances.horizons.get_body_keplerian_elements(elem_or_num, date=self.config.date)
        elif isinstance(elem_or_num, dict):
            return elem_or_num
        else:
            raise ValueError('You can add body only by its number or all orbital elements')

    def mpc_elements(self, identifiers, refused: Optional[dict] = None) -> dict:
        """{str(identifier): elements at config.date} of the MPC rows found (resonances/mpcorb).

        The snapshot must be the one the config recorded (config.mpc_snapshot). A row at another epoch is integrated
        to config.date with a warning (MPC_EPOCH_POLICY=propagate, the default; refused beyond MPC_PROPAGATE_MAX_DAYS
        when that is set) or refused (strict). A refusal raises ValueError, or, when a `refused` dict is given (a
        find over many bodies), is put there as {str(identifier): message} and the body left out.

        With MPC_LOOKUP=file_then_api an identifier the catalogue does not have (no row; a malformed or ambiguous
        identifier still raises in the lookup) gets its nominal orbit from MPC get-orb (`mpc_getorb`), with the same
        epoch policy; a failed request is a refusal as above.
        """
        settings = mpc_settings(self.config.mpc_catalog)
        snapshot = mpcorb.snapshot(settings=settings)
        recorded = self.config.mpc_snapshot
        if (snapshot['sha256'], snapshot['parser_version']) != (recorded['sha256'], recorded.get('parser_version')):
            raise ValueError(
                f"The cached MPC catalogue {snapshot['catalog']} (sha256 {snapshot['sha256']}, parser "
                f"{snapshot['parser_version']}) is not the one this simulation was configured with ({recorded['sha256']}, "
                f"parser {recorded.get('parser_version')}): it was updated in between"
            )
        rows = mpcorb.lookup(identifiers, snapshot_dict=snapshot, settings=settings)
        date_mjd = mjd_from_datetime(self.config.date)
        policy, max_days = self.config.mpc_epoch_policy, self.config.mpc_propagate_max_days
        found, off_epoch = {}, []
        if getattr(self.config, 'mpc_lookup', 'file') == 'file_then_api':
            missing = [str(i) for i in dict.fromkeys(str(i) for i in identifiers) if i not in rows]
            off_epoch += self._getorb_elements(missing, found, refused, date_mjd, settings.warn_u)
        for key, row in rows.items():
            span = float(row['epoch_mjd']) - date_mjd
            on_date = abs(span) <= EPOCH_TOLERANCE_DAYS
            if not on_date and (policy != 'propagate' or (max_days is not None and abs(span) > max_days)):
                message = epoch_refusal(
                    row,
                    self.config.date,
                    date_mjd,
                    standard_epoch_mjd=snapshot.get('standard_epoch_mjd'),
                    policy=policy,
                    max_days=max_days if policy == 'propagate' else None,
                )
                if refused is None:
                    raise ValueError(message)
                refused[key] = message
                continue
            for warning in mpcorb.quality_warnings(row, warn_u=settings.warn_u):
                logger.warning(warning)
            found[key] = mpcorb.elements_from_row(row, snapshot)
            if not on_date:
                off_epoch.append(key)
                logger.warning(propagation_warning(row, -span, self.config.date))
        if off_epoch:
            for key, elem in zip(off_epoch, propagate_elements([found[key] for key in off_epoch], self.config)):
                found[key] = elem
        return {key: found[key] for key in dict.fromkeys(str(i) for i in identifiers) if key in found}  # the order asked for

    def _getorb_elements(self, identifiers, found: dict, refused: Optional[dict], date_mjd: float, warn_u: int) -> list:
        """MPC_LOOKUP=file_then_api: the get-orb nominal of each identifier the catalogue lacks, put in `found` (or a
        refusal: raised, or put in `refused`); returns those at another epoch than the date, to be propagated."""
        from . import mpc_getorb

        policy, max_days = self.config.mpc_epoch_policy, self.config.mpc_propagate_max_days
        off_epoch = []
        for key in identifiers:
            try:
                elem = mpc_getorb.fetch_elements(key)
                span = elem['epoch'] - date_mjd
                if abs(span) > EPOCH_TOLERANCE_DAYS and (policy != 'propagate' or (max_days is not None and abs(span) > max_days)):
                    why = 'MPC_EPOCH_POLICY=strict' if policy != 'propagate' else f'beyond MPC_PROPAGATE_MAX_DAYS = {max_days:g}'
                    raise ValueError(
                        f"The MPC get-orb elements of {mpc_getorb.label(elem, key)} are at epoch MJD {elem['epoch']:.5f} TT, "
                        f'{span:+.1f} days from the simulation date {self.config.date} ({why})'
                    )
            except ValueError as error:
                message = f"{key} is not in the MPC catalogue {self.config.mpc_snapshot['catalog']}, and get-orb: {error}"
                if refused is None:
                    raise ValueError(message) from None
                refused[key] = message
                continue
            logger.info(f'{key}: not in the MPC catalogue; nominal orbit of {mpc_getorb.label(elem, key)} from MPC get-orb')
            for warning in mpc_getorb.quality_warnings(elem, key, warn_u):
                logger.warning(warning)
            found[key] = elem
            if abs(elem['epoch'] - date_mjd) > EPOCH_TOLERANCE_DAYS:
                off_epoch.append(key)
                logger.warning(
                    f"{mpc_getorb.label(elem, key)}: get-orb epoch MJD {elem['epoch']:.5f} is {date_mjd - elem['epoch']:+.1f} days "
                    f'from the run date {self.config.date}; propagated to it (MPC_EPOCH_POLICY=propagate; U = {elem["U"] or "blank"})'
                )
        return off_epoch

    def add_body(self, elem_or_num, resonance: Union[Resonance, str, list[Resonance], list[str]], name='asteroid'):
        body = Body()

        import resonances

        if isinstance(resonance, list):
            resonances_list = [resonances.create_resonance(res) for res in resonance]
        else:
            resonances_list = [resonances.create_resonance(resonance)]

        elem = self.get_body_elements(elem_or_num)
        self.check_epoch(elem, name)

        body.initial_data = elem
        body.name = name

        mmrs_list = []
        secular_list = []
        lidov_kozai_list = []

        for res in resonances_list:
            if res.type == 'mmr':
                mmrs_list.append(res)
            elif res.type == 'secular':
                secular_list.append(res)
            elif res.type == 'lidov_kozai':
                lidov_kozai_list.append(res)
        body.mmrs = mmrs_list
        body.secular_resonances = secular_list
        body.lidov_kozai_resonances = lidov_kozai_list
        body.mass = elem.get('mass', 0.0)

        for mmr_elem in body.mmrs:
            mmr_elem.index_of_planets = self.get_index_of_planets(mmr_elem.planets_names)
        self.warn_orbit_direction(body, elem)

        self.bodies.append(body)

    @staticmethod
    def warn_orbit_direction(body: Body, elem: dict):
        """Warn when a retrograde MMR is given to a body with i < 90 deg or a prograde MMR to one with i > 90 deg.

        Not an error: an orbit can flip during the integration, and the user may want the angle anyway.
        """
        if 'inc' not in elem:
            return
        retrograde = float(elem['inc']) > np.pi / 2
        for mmr in body.mmrs:
            if mmr.retrograde != retrograde:
                orbit, kind = ('retrograde', 'prograde') if retrograde else ('prograde', 'retrograde')
                logger.warning(
                    f"{body.name} has a {orbit} orbit (i = {np.degrees(float(elem['inc'])):.1f} deg), "
                    f"but {mmr.to_s()} is a {kind} resonance: its angle is meaningful only for {kind} orbits."
                )

    def check_epoch(self, elem: dict, name: str):
        """Raise ValueError if the elements carry an 'epoch' other than config.date.

        The planets are built at config.date and REBOUND takes the Keplerian elements as they are,
        so elements at another epoch put the body at the wrong place on its orbit (126 days moved
        the resonant angle of 463 Lola by 33 deg and turned its libration into circulation).
        'epoch' is a MJD number (AstDyS, MPC, Horizons) or a date; elements without it are not checked.
        """
        epoch = elem.get('epoch')
        if epoch is None:
            return
        epoch_mjd = float(epoch) if isinstance(epoch, numbers.Real) else mjd_from_datetime(epoch)
        date_mjd = mjd_from_datetime(self.config.date)
        if abs(epoch_mjd - date_mjd) > EPOCH_TOLERANCE_DAYS:
            raise ValueError(
                f"The elements of {name} are at epoch MJD {epoch_mjd:.5f}, but the simulation starts at "
                f"{self.config.date} (MJD {date_mjd:.5f}). Set date to the epoch of the elements (with "
                f"source='astdys' or 'mpc', leave date unset: it defaults to the catalog epoch) or take the elements at the date."
            )

    def add_bodies_to_simulation(self, sim):
        """Add all bodies to the REBOUND simulation."""
        if sim is None:
            raise RuntimeError(
                "Solar system or other particles for rebound are not initialized. Call sim.create_solar_system() or add them manually before sim.run()."
            )
        for body in self.bodies:
            self._add_body_to_simulation(body, sim)

    def _add_body_to_simulation(self, body: Body, sim):
        """Add a single body to the REBOUND simulation."""
        body.index_in_simulation = len(sim.particles)
        sim.add(
            m=body.mass,
            a=body.initial_data['a'],
            e=body.initial_data['e'],
            inc=body.initial_data['inc'],
            Omega=body.initial_data['Omega'],
            omega=body.initial_data['omega'],
            M=body.initial_data['M'],
            primary=sim.particles[0],
            hash=body.name,
        )
