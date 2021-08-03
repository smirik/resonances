import numpy as np
import pandas as pd
import rebound
from pathlib import Path
from typing import List

import resonances
from resonances.data.astdys import astdys
from resonances.resonance import plot


class Simulation:
    def __init__(self):
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
        self.tmax = resonances.config.get('integration.tmax')
        self.integrator = resonances.config.get('integration.integrator')
        self.dt = resonances.config.get('integration.dt')
        if resonances.config.has('integration.integrator.safe_mode'):
            self.integrator_safe_mode = resonances.config.get('integration.integrator.safe_mode')
        else:
            self.integrator_safe_mode = 1

        if resonances.config.has('integration.integrator.corrector'):
            self.integrator_corrector = resonances.config.get('integration.integrator.corrector')
        else:
            self.integrator_corrector = None

        self.save = resonances.config.get('save')
        self.save_path = resonances.config.get('save.path')
        self.save_summary = resonances.config.get('save.summary')
        self.save_additional_data = resonances.config.get('save.additional.data')
        if resonances.config.has('save.only.undetermined'):
            self.save_only_undetermined = resonances.config.get('save.only.undetermined')
        else:
            self.save_only_undetermined = False

        self.save_additional_data = resonances.config.get('save.additional.data')
        self.plot = resonances.config.get('plot')

    def create_solar_system(self):
        solar_file = Path(resonances.config.get('solar_system_file'))

        if solar_file.exists():
            self.sim = rebound.Simulation(resonances.config.get('solar_system_file'))
        else:
            self.sim = rebound.Simulation()
            self.sim.add(self.list_of_planets(), date='2020-12-17 00:00')  # date of AstDyS current catalogue
            self.sim.save(resonances.config.get('solar_system_file'))

    def add_body(self, elem_or_num, mmr: resonances.MMR, name='asteroid'):
        body = resonances.Body()

        if isinstance(elem_or_num, int) or (isinstance(elem_or_num, str)):
            elem = astdys.search(elem_or_num)
        elif isinstance(elem_or_num, dict):
            elem = elem_or_num
            if 'mass' in elem:
                body.mass = elem['mass']
        else:
            raise Exception('You can add body only by its number or all orbital elements')

        body.initial_data = elem
        body.name = name
        body.mmr = mmr
        body.index_of_planets = self.get_index_of_planets(mmr.planets_names)
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

    def setup_integrator(self, N_active=10):
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

    def run(self):
        self.add_bodies_to_simulation()
        for body in self.bodies:
            body.setup_vars_for_simulation(self.Nout)

        self.times = np.linspace(0.0, self.tmax, self.Nout)
        self.setup_integrator()

        ps = self.sim.particles

        for i, time in enumerate(self.times):
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
                planets = []
                for index in body.index_of_planets:
                    planets.append(os[index - 1])

                body.angle[i] = body.mmr.calc_angle(os[body.index_in_simulation - 1], planets)

        self.identify_librations()
        if self.save_summary:
            self.save_simulation_summary()
        if self.save or self.plot or self.save_only_undetermined:
            for body in self.bodies:
                if self.shall_save_body(body):
                    self.save_body(body)
                if self.shall_plot_body(body):
                    self.plot_body(body)

    def identify_librations(self):
        for body in self.bodies:
            resonances.libration.body(self, body)

    def shall_save_body(self, body: resonances.Body):
        if self.save:
            return True
        elif (self.save_only_undetermined) and (body.status < 0):
            return True
        return False

    def shall_plot_body(self, body: resonances.Body):
        if self.plot:
            return True
        elif (self.save_only_undetermined) and (body.status < 0):
            return True
        return False

    def plot_body(self, body: resonances.Body):
        self.check_or_create_save_path()
        plot.body(self, body)

    def save_body(self, body: resonances.Body):
        self.check_or_create_save_path()
        df_data = self.get_body_data(body)
        df = pd.DataFrame(data=df_data)
        df.to_csv('{}/data-{}-{}.csv'.format(self.save_path, body.index_in_simulation, body.name))

    def get_body_data(self, body: resonances.Body):
        df_data = {
            'times': self.times / (2 * np.pi),
            'angle': body.angle,
            'a': body.axis,
            'e': body.ecc,
        }
        if self.save_additional_data and (body.periodogram_power is not None):
            len_diff = len(body.angle) - len(body.periodogram_power)
            df_data['periodogram'] = np.append(body.periodogram_power, np.zeros(len_diff))
            df_data['a_filtered'] = body.axis_filtered
            df_data['a_periodogram'] = np.append(body.axis_periodogram_power, np.zeros(len_diff))
        return df_data

    def get_simulation_summary(self):
        data = []
        for body in self.bodies:
            s = ', '.join('({:.0f}, {:.0f})'.format(left, right) for left, right in body.periodogram_peaks_overlapping)
            data.append(
                [
                    body.name,
                    body.status,
                    body.libration_pure,
                    body.libration_metrics['num_libration_periods'],
                    body.libration_metrics['max_libration_length'],
                    body.monotony,
                    s,
                    body.initial_data['a'],
                    body.initial_data['e'],
                    body.initial_data['inc'],
                    body.initial_data['Omega'],
                    body.initial_data['omega'],
                    body.initial_data['M'],
                ]
            )
        return data

    def save_simulation_summary(self):
        self.check_or_create_save_path()
        data = self.get_simulation_summary()
        df = pd.DataFrame(
            data,
            columns=[
                'name',
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
        df.to_csv('{}/summary.csv'.format(self.save_path), mode='a', header=False)
        return data

    def check_or_create_save_path(self):
        Path(self.save_path).mkdir(parents=True, exist_ok=True)

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

    def setup(self, save=None, save_path=None, save_only_undetermined=None, plot=None, tmax=None, Nout=None):
        if save is not None:
            self.save = save
        if save_only_undetermined:
            self.save_only_undetermined = save_only_undetermined
        if save_path is not None:
            self.save_path = save_path
        if plot is not None:
            self.plot = plot
        if tmax is not None:
            self.tmax = tmax
        if Nout is not None:
            self.Nout = int(Nout)

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
    def tmax(self):
        del self.__tmax
