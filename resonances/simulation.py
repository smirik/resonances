import numpy as np
import pandas as pd
import rebound
from pathlib import Path
import os
from typing import List, Union

import resonances
import astdys
import datetime

import resonances.horizons


class Simulation:
    def __init__(self, name=None, date: str = None):
        self.name = name
        if self.name is None:
            self.name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.planets = self.list_of_planets()

        self.times = []
        self.bodies: List[resonances.Body] = []
        self.particles = []

        self.bodies_date = resonances.config.get('catalog.date')

        # Libration and filtering settings
        self.oscillations_cutoff = resonances.config.get('libration.oscillation.filter.cutoff')
        self.oscillations_filter_order = resonances.config.get('libration.oscillation.filter.order')

        self.periodogram_frequency_min = resonances.config.get('libration.periodogram.frequency.min')
        self.periodogram_frequency_max = resonances.config.get('libration.periodogram.frequency.max')
        self.periodogram_critical = resonances.config.get('libration.periodogram.critical')
        self.periodogram_soft = resonances.config.get('libration.periodogram.soft')

        self.libration_period_critical = resonances.config.get('libration.period.critical')
        self.libration_monotony_critical = resonances.config.get('libration.monotony.critical')
        self.libration_period_min = resonances.config.get('libration.period.min')

        self.sim = None

        self.Nout = None
        self.date = None
        self.tmax = resonances.config.get('integration.tmax')
        self.integrator = resonances.config.get('integration.integrator')
        self.dt = resonances.config.get('integration.dt')
        if resonances.config.has('integration.integrator.safe_mode'):
            self.integrator_safe_mode = resonances.config.get('integration.integrator.safe_mode')
        else:  # pragma: no cover
            self.integrator_safe_mode = 1

        if resonances.config.has('integration.integrator.corrector'):
            self.integrator_corrector = resonances.config.get('integration.integrator.corrector')
        else:  # pragma: no cover
            self.integrator_corrector = None

        self.save = resonances.config.get('save')
        self.save_path = f"{resonances.config.get('save.path')}/{self.name}"
        self.save_summary = resonances.config.get('save.summary')

        self.plot = resonances.config.get('plot')
        self.plot_type = resonances.config.get('plot.type', 'both')
        self.plot_path = f"{resonances.config.get('plot.path')}/{self.name}/images"

        self.image_type = resonances.config.get('plot.image_type', 'png')
        self.data_source = resonances.config.get('data.source', 'astdys')

    def solar_system_full_filename(self) -> str:
        catalog_file = f"{os.getcwd()}/{resonances.config.get('solar_system_file')}"
        return catalog_file

    def create_solar_system(self, date: str = None):
        solar_file = Path(self.solar_system_full_filename())
        if solar_file.exists():
            self.sim = rebound.Simulation(self.solar_system_full_filename())
        else:  # pragma: no cover
            self.sim = rebound.Simulation()
            if date is not None:
                self.sim.add(self.list_of_planets(), date=date)
            elif self.date is not None:
                self.sim.add(self.list_of_planets(), date=self.date)
            elif self.data_source == 'astdys':
                self.sim.add(self.list_of_planets(), date=f"{astdys.catalog_time()} 00:00")  # date of AstDyS current catalogue
            else:
                self.sim.add(self.list_of_planets())
            self.sim.save(self.solar_system_full_filename())

    def add_body(self, elem_or_num, mmr: Union[str, resonances.MMR, List[resonances.MMR]], name='asteroid', source='nasa'):
        """
        Add a celestial body to the simulation with its corresponding mean motion resonance(s).
        Parameters
        ----------
        elem_or_num : Union[int, str, dict]
            Either an integer/string representing the asteroid's number,
            or a dictionary containing orbital elements with optional mass.
            If dictionary, must contain keys: 'a', 'e', 'inc', 'Omega', 'omega', 'M'
            Optional key: 'mass'
        mmr : Union[str, resonances.MMR, List[resonances.MMR]]
            Mean motion resonance(s) to analyze for this body. Can be:
            - String representation of MMR (e.g. "4J-2S-1")
            - Single MMR object
            - List of MMR objects
            At least one resonance must be provided.
        name : str, optional
            Name identifier for the body. Defaults to 'asteroid'.
        source : str, optional
            Source of orbital elements data. Two options are available: 'nasa' or 'astdys'. Defaults to 'nasa'.
        Raises
        ------
        Exception
            If no resonances are provided or if elem_or_num is invalid type.
        Notes
        -----
        - If elem_or_num is an ID, orbital elements are fetched from NASA catalog
        - If elem_or_num is a dict, it must contain all required orbital elements
        - Added body is stored in self.bodies list
        - For each MMR, planet indices in simulation are calculated and stored

        Examples
        --------
        >>> sim.add_body(1, "4J-2S-1", name="Asteroid 1", source="nasa")  # Add by NASA id
        >>> sim.add_body({"a": 3.2, "e": 0.1, "omega": 0.1, "Omega": 0.1, "M": 0.1}, "3J-1")  # Add by orbital elements
        """
        body = resonances.Body()

        if isinstance(mmr, str):
            mmr = [resonances.create_mmr(mmr)]

        if isinstance(mmr, resonances.MMR):
            mmr = [mmr]

        if len(mmr) == 0:
            raise Exception('You have to provide at least one resonance')

        if isinstance(elem_or_num, int) or (isinstance(elem_or_num, str)):
            if source == 'astdys':
                elem = astdys.search(elem_or_num)
            else:
                elem = resonances.horizons.get_body_keplerian_elements(elem_or_num, self.sim)
        elif isinstance(elem_or_num, dict):
            elem = elem_or_num
            if 'mass' in elem:
                body.mass = elem['mass']
        else:
            raise Exception('You can add body only by its number or all orbital elements')

        body.initial_data = elem
        body.name = name
        body.mmrs = mmr

        for elem in body.mmrs:
            elem.index_of_planets = self.get_index_of_planets(elem.planets_names)
        self.bodies.append(body)

    def add_bodies_to_simulation(self):
        for body in self.bodies:
            self.add_body_to_simulation(body)

    def add_body_to_simulation(self, body: resonances.Body):
        body.index_in_simulation = len(self.sim.particles)
        self.sim.add(
            m=body.mass,
            a=body.initial_data['a'],
            e=body.initial_data['e'],
            inc=body.initial_data['inc'],
            Omega=body.initial_data['Omega'],
            omega=body.initial_data['omega'],
            M=body.initial_data['M'],
            date=self.bodies_date,
            primary=self.sim.particles[0],
        )

    def setup_integrator(self, N_active=10):  # pragma: no cover
        self.sim.integrator = self.integrator
        self.sim.dt = self.dt
        self.sim.N_active = N_active

        if 'whfast' == self.integrator:
            self.sim.ri_whfast.safe_mode = 0
            if self.integrator_corrector is not None:
                self.sim.ri_whfast.corrector = self.integrator_corrector
        elif 'SABA' in self.integrator:
            self.sim.ri_saba.safe_mode = self.integrator_safe_mode

        self.sim.move_to_com()

    def run(self, progress=False):
        self.add_bodies_to_simulation()
        for body in self.bodies:
            body.setup_vars_for_simulation(self.Nout)

        self.times = np.linspace(0.0, self.tmax, self.Nout)
        self.setup_integrator()

        ps = self.sim.particles

        iterations = list(enumerate(self.times))
        if progress:
            import tqdm

            iterations = tqdm.tqdm(iterations, total=len(iterations))

        for i, time in iterations:
            self.sim.integrate(time)
            os = self.sim.calculate_orbits(primary=ps[0])

            for body in self.bodies:
                tmp = os[body.index_in_simulation - 1]  # ? -1 because Sun is not in os
                body.axis[i], body.ecc[i], body.longitude[i], body.varpi[i] = (
                    tmp.a,
                    tmp.e,
                    tmp.l,
                    tmp.Omega + tmp.omega,
                )
                for mmr in body.mmrs:
                    planets = []
                    for index in mmr.index_of_planets:
                        planets.append(os[index - 1])

                    body.angle(mmr)[i] = mmr.calc_angle(os[body.index_in_simulation - 1], planets)

        self.identify_librations()
        self.save_data()

    def save_data(self):
        if self.save_summary:
            self.save_simulation_summary()

        for body in self.bodies:
            for mmr in body.mmrs:
                if self.shall_save_body_in_mmr(body, mmr):
                    self.save_body_in_mmr(body, mmr)
                if self.shall_plot_body_in_mmr(body, mmr):
                    self.plot_body_in_mmr(body, mmr)

    def identify_librations(self):  # pragma: no cover
        for body in self.bodies:
            try:
                resonances.libration.body(self, body)
            except Exception as e:
                resonances.logger.error(f"Error identifying librations for {body.name}: {str(e)}")
                raise e

    def shall_save_body_in_mmr(self, body: resonances.Body, mmr: resonances.MMR):
        return self.process_status(body, mmr, self.save)

    def shall_plot_body_in_mmr(self, body: resonances.Body, mmr: resonances.MMR):
        return self.process_status(body, mmr, self.plot)

    def process_status(self, body: resonances.Body, mmr: resonances.MMR, variable) -> bool:
        if variable is None:
            return False

        if variable == 'all':
            return True

        if (variable == 'resonant') and (body.statuses[mmr.to_s()] > 0):
            return True

        if (variable == 'nonzero') and (body.statuses[mmr.to_s()] != 0):
            return True

        if (variable == 'candidates') and (body.statuses[mmr.to_s()] < 0):
            return True

        return False

    def plot_body_in_mmr(self, body: resonances.Body, mmr: resonances.MMR):
        self.check_or_create_save_path()
        resonances.resonance.plot.body(self, body, mmr, image_type=self.image_type)

    def save_body_in_mmr(self, body: resonances.Body, mmr: resonances.MMR):
        self.check_or_create_save_path()
        df_data = body.mmr_to_dict(mmr, self.times)
        if df_data is not None:
            df = pd.DataFrame(data=df_data)
            df.to_csv('{}/data-{}-{}.csv'.format(self.save_path, body.name, mmr.to_s()))

    def get_simulation_summary(self) -> pd.DataFrame:
        data = []
        for body in self.bodies:
            for mmr in body.mmrs:
                try:
                    s = ', '.join('({:.0f}, {:.0f})'.format(left, right) for left, right in body.periodogram_peaks_overlapping[mmr.to_s()])
                    data.append(
                        [
                            body.name,
                            mmr.to_s(),
                            body.statuses[mmr.to_s()],
                            body.libration_pure[mmr.to_s()],
                            body.libration_metrics[mmr.to_s()]['num_libration_periods'],
                            body.libration_metrics[mmr.to_s()]['max_libration_length'],
                            body.monotony[mmr.to_s()],
                            s,
                            body.initial_data['a'],
                            body.initial_data['e'],
                            body.initial_data['inc'],
                            body.initial_data['Omega'],
                            body.initial_data['omega'],
                            body.initial_data['M'],
                        ]
                    )
                except Exception as e:  # pragma: no cover
                    resonances.logger.error(f"Error getting summary for {body.name} and {mmr.to_s()}: {str(e)}.\n\n{str(body)}")

        df = pd.DataFrame(
            data,
            columns=[
                'name',
                'mmr',
                'status',
                'pure',
                'num_libration_periods',
                'max_libration_length',
                'monotony',
                'overlapping',
                'a',
                'e',
                'inc',
                'Omega',
                'omega',
                'M',
            ],
        )
        return df

    def save_simulation_summary(self) -> pd.DataFrame:
        self.check_or_create_save_path()
        df = self.get_simulation_summary()
        summary_filename = '{}/summary.csv'.format(self.save_path)
        summary_file = Path(summary_filename)
        if summary_file.exists():
            df.to_csv(summary_filename, mode='a', header=False, index=False)
        else:
            df.to_csv(summary_filename, mode='a', header=True, index=False)

        return df

    def check_or_create_save_path(self):
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        Path(self.plot_path).mkdir(parents=True, exist_ok=True)

    def list_of_planets(self):
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

    def get_index_of_planets(self, planets_names):
        arr = []
        for planet in planets_names:
            arr.append(self.planets.index(planet))
        return arr

    @property
    def tmax(self):
        return self.__tmax

    @tmax.setter
    def tmax(self, value):
        self.__tmax = value
        self.tmax_yrs = self.__tmax / (2 * np.pi)
        if self.Nout is None:
            self.Nout = int(self.__tmax / 100)

    @tmax.deleter
    def tmax(self):  # pragma: no cover
        del self.__tmax
