import datetime
import numpy as np
import pandas as pd
import rebound
from pathlib import Path
import os
from typing import List, Union
import tqdm

import resonances
import astdys

import resonances.data
import resonances.data.util
import resonances.horizons
from resonances.config import config as c


class Simulation:
    def __init__(  # noqa: C901
        self,
        name=None,
        date: Union[str, datetime.datetime] = None,
        source=None,
        tmax=None,
        integrator: str = None,
        integrator_safe_mode: int = None,
        integrator_corrector: int = None,
        dt: float = None,
        save: str = None,
        save_path: str = None,
        save_summary: bool = None,
        plot: str = None,
        plot_path: str = None,
        plot_type: str = None,
        image_type: str = None,
    ):
        self.name = name
        if self.name is None:
            self.name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.Nout = None

        self.source = source
        if self.source is None:
            self.source = c.get('DATA_SOURCE')

        if date is not None:
            self.date = resonances.data.util.datetime_from_string(date)
            if self.source == 'astdys':
                if self.date.strftime("%Y-%m-%d %H:%M:%S") != astdys.datetime().strftime("%Y-%m-%d %H:%M:%S"):
                    resonances.logger.error(
                        "Date specified by the user is not the same as the catalog time, which may cause issues: "
                        f"{self.date.strftime('%Y-%m-%d %H:%M:%S')} != {astdys.catalog_time()}"
                    )
        elif source == 'astdys':
            self.date = astdys.datetime()
        else:
            self.date = datetime.datetime.combine(datetime.datetime.today(), datetime.time.min)

        if tmax is None:
            self.tmax = int(c.get('INTEGRATION_TMAX'))
        else:
            self.tmax = tmax

        self.integrator = integrator
        if self.integrator is None:
            self.integrator = c.get('INTEGRATION_INTEGRATOR')

        self.dt = dt
        if self.dt is None:
            self.dt = float(c.get('INTEGRATION_DT'))

        self.integrator_corrector = integrator_corrector
        if self.integrator_corrector is None:
            self.integrator_corrector = int(c.get('INTEGRATION_CORRECTOR'))

        self.save = save
        if self.save is None:
            self.save = c.get('SAVE_MODE')

        now = datetime.datetime.now()
        self.save_path = save_path
        if self.save_path is None:
            self.save_path = f"{c.get('SAVE_PATH')}/{now.strftime('%Y-%m-%d_%H:%M:%S')}"

        self.save_summary = save_summary
        if self.save_summary is None:
            self.save_summary = bool(c.get('SAVE_SUMMARY'))

        self.plot = plot
        if self.plot is None:
            self.plot = c.get('PLOT_MODE')

        self.plot_type = plot_type
        if self.plot_type is None:
            self.plot_type = c.get('PLOT_TYPE')

        self.plot_path = plot_path
        if self.plot_path is None:
            self.plot_path = f"{c.get('PLOT_PATH')}/{now.strftime('%Y-%m-%d_%H:%M:%S')}"

        self.image_type = image_type
        if self.image_type is None:
            self.image_type = c.get('PLOT_IMAGE_TYPE')

        self.planets = self.list_of_planets()

        self.times = []
        self.bodies: List[resonances.Body] = []
        self.particles = []

        if self.source == 'astdys':
            self.bodies_date = astdys.datetime()
        else:
            self.bodies_date = self.date

        # Libration and filtering settings
        self.oscillations_cutoff = float(resonances.config.get('LIBRATION_FILTER_CUTOFF'))
        self.oscillations_filter_order = int(resonances.config.get('LIBRATION_FILTER_ORDER'))

        self.periodogram_frequency_min = float(resonances.config.get('LIBRATION_FREQ_MIN'))
        self.periodogram_frequency_max = float(resonances.config.get('LIBRATION_FREQ_MAX'))
        self.periodogram_critical = float(resonances.config.get('LIBRATION_CRITICAL'))
        self.periodogram_soft = float(resonances.config.get('LIBRATION_SOFT'))

        self.libration_period_critical = int(resonances.config.get('LIBRATION_PERIOD_CRITICAL'))
        self.libration_monotony_critical = [float(x.strip()) for x in resonances.config.get('LIBRATION_MONOTONY_CRITICAL').split(",")]

        self.libration_period_min = int(resonances.config.get('LIBRATION_PERIOD_MIN'))

        self.sim = None

        if integrator_safe_mode is not None:
            self.integrator_safe_mode = integrator_safe_mode
        else:  # pragma: no cover
            self.integrator_safe_mode = 1

    def solar_system_full_filename(self) -> str:
        timestamp = int(self.date.timestamp())
        catalog_file = f"{os.getcwd()}/{c.get('SOLAR_SYSTEM_FILE')}"
        catalog_file = catalog_file.replace('.bin', f'-{timestamp}.bin')
        return catalog_file

    def create_solar_system(self, force=False):
        """
        Creates or loads the Solar System to rebound Simulation.
        This method either loads an existing Solar System simulation from a file or creates a new one
        if the file doesn't exist or if forced to do so. The simulation includes major planets based
        on the specified date or default configuration.

        Parameters
        ----------
        force : bool, optional
            If True, forces creation of new simulation even if file exists. Defaults to False.
        Returns
        -------
        None
            Updates self.sim with the created/loaded REBOUND simulation.
        Notes
        -----
        - If a saved simulation file exists and force=False, loads from file
        - Otherwise creates new simulation with planets at specified date
        - Saves newly created simulation to file for future use
        """

        solar_file = Path(self.solar_system_full_filename())
        if solar_file.exists() and not force:
            self.sim = rebound.Simulation(self.solar_system_full_filename())
        else:
            self.sim = rebound.Simulation()
            self.sim.add(self.list_of_planets(), date=self.date)
            self.sim.save_to_file(self.solar_system_full_filename())

    def add_body(self, elem_or_num, mmr: Union[str, resonances.MMR, List[resonances.MMR]], name='asteroid'):
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

        if isinstance(mmr, list):
            mmr = resonances.create_mmr(mmr)
        else:
            mmr = [resonances.create_mmr(mmr)]

        elem = self.get_body_elements(elem_or_num)

        if 'mass' in elem:
            body.mass = elem['mass']

        body.initial_data = elem
        body.name = name
        body.mmrs = mmr

        for elem in body.mmrs:
            elem.index_of_planets = self.get_index_of_planets(elem.planets_names)

        self.bodies.append(body)

    def get_body_elements(self, elem_or_num: int) -> dict:
        if isinstance(elem_or_num, int) or (isinstance(elem_or_num, str)):
            if self.source == 'astdys':
                elem = astdys.search(elem_or_num)
            else:
                elem = resonances.horizons.get_body_keplerian_elements(elem_or_num, date=self.date)
        elif isinstance(elem_or_num, dict):
            elem = elem_or_num
        else:
            raise Exception('You can add body only by its number or all orbital elements')
        return elem

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
        if progress:  # pragma: no cover
            iterations = tqdm.tqdm(iterations, total=len(iterations))

        for i, time in iterations:
            self.sim.integrate(time)
            os = self.sim.orbits(primary=ps[0])

            for body in self.bodies:
                tmp = os[body.index_in_simulation - 1]  # ? -1 because Sun is not in os
                body.axis[i], body.ecc[i], body.inc[i], body.Omega[i], body.omega[i], body.M[i], body.longitude[i], body.varpi[i] = (
                    tmp.a,
                    tmp.e,
                    tmp.inc,
                    tmp.Omega,
                    tmp.omega,
                    tmp.M,
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
        self.save_periodogram_data(body, mmr)

    def save_periodogram_data(self, body: resonances.Body, mmr: resonances.MMR):
        """
        Save periodogram data for both resonant angle and semi-major axis.
        Since periodogram data has different scales than time series data,
        it's saved in separate files.
        """
        # Save resonant angle periodogram
        if body.periodogram_frequency.get(mmr.to_s()) is not None and body.periodogram_power.get(mmr.to_s()) is not None:

            self._save_single_periodogram(
                frequency=body.periodogram_frequency[mmr.to_s()],
                power=body.periodogram_power[mmr.to_s()],
                peaks_data=body.periodogram_peaks.get(mmr.to_s()),
                body_name=body.name,
                mmr_str=mmr.to_s(),
                data_type="angle",
            )

        # Save semi-major axis periodogram
        if body.axis_periodogram_frequency is not None and body.axis_periodogram_power is not None:

            self._save_single_periodogram(
                frequency=body.axis_periodogram_frequency,
                power=body.axis_periodogram_power,
                peaks_data=body.axis_periodogram_peaks,
                body_name=body.name,
                mmr_str=mmr.to_s(),
                data_type="axis",
            )

        # Save overlapping peaks information
        if body.periodogram_peaks_overlapping.get(mmr.to_s()):
            overlapping_data = []
            for left, right in body.periodogram_peaks_overlapping[mmr.to_s()]:
                overlapping_data.append({'left_boundary': left, 'right_boundary': right, 'center_period': (left + right) / 2})

            if overlapping_data:
                df_overlapping = pd.DataFrame(overlapping_data)
                df_overlapping.to_csv(
                    '{}/data-{}-{}-periodogram-overlapping.csv'.format(self.save_path, body.name, mmr.to_s()), index=False
                )

    def _save_single_periodogram(self, frequency, power, peaks_data, body_name, mmr_str, data_type):
        """
        Helper method to save a single periodogram (either angle or axis) and its peaks.

        Parameters
        ----------
        frequency : np.ndarray
            Frequency array from periodogram
        power : np.ndarray
            Power array from periodogram
        peaks_data : dict or None
            Dictionary containing peaks information with 'peaks' and 'position' keys
        body_name : str
            Name of the body
        mmr_str : str
            String representation of the MMR
        data_type : str
            Type of data ("angle" or "axis") for filename generation
        """
        periodogram_data = {'frequency': frequency, 'power': power, 'period': 1.0 / frequency}  # Add period for convenience

        # Add peak information if available
        if peaks_data is not None and 'peaks' in peaks_data and peaks_data['peaks'].size > 0:

            peaks = peaks_data['peaks']
            peak_frequencies = frequency[peaks]
            peak_powers = power[peaks]

            # Create arrays to mark peaks (1 for peak, 0 for non-peak)
            is_peak = np.zeros(len(frequency))
            is_peak[peaks] = 1
            periodogram_data['is_peak'] = is_peak

            # Save peak positions separately for easy access
            peak_positions = peaks_data.get('position', [])
            if peak_positions:
                peak_info = []
                for i, (left, right) in enumerate(peak_positions):
                    if i < len(peak_frequencies):
                        peak_info.append(
                            {
                                'peak_frequency': peak_frequencies[i],
                                'peak_power': peak_powers[i],
                                'peak_period': 1.0 / peak_frequencies[i],
                                'left_boundary': left,
                                'right_boundary': right,
                            }
                        )

                if peak_info:
                    peaks_df = pd.DataFrame(peak_info)
                    peaks_df.to_csv(
                        '{}/data-{}-{}-periodogram-{}-peaks.csv'.format(self.save_path, body_name, mmr_str, data_type), index=False
                    )

        # Save main periodogram data
        df_periodogram = pd.DataFrame(periodogram_data)
        df_periodogram.to_csv('{}/data-{}-{}-periodogram-{}.csv'.format(self.save_path, body_name, mmr_str, data_type), index=False)

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
        # Save simulation details
        self.save_configuration_details()

        # Save summary to CSV
        df = self.get_simulation_summary()
        summary_filename = '{}/summary.csv'.format(self.save_path)
        summary_file = Path(summary_filename)
        if summary_file.exists():
            df.to_csv(summary_filename, mode='a', header=False, index=False)
        else:
            df.to_csv(summary_filename, mode='a', header=True, index=False)

        return df

    def save_configuration_details(self):
        self.check_or_create_save_path()

        with open(f"{self.save_path}/simulation.cfg", "w") as f:
            f.write("Simulation Configuration\n")
            f.write("========================\n")
            f.write(f"Name: {self.name}\n")
            f.write(f"Date: {self.date}\n")
            f.write(f"Source: {self.source}\n")
            f.write(f"Number of bodies: {len(self.bodies)}\n")
            f.write("========================\n")

            f.write(f"Tmax: {self.tmax}\n")
            f.write(f"Integrator: {self.integrator}\n")
            f.write(f"Integrator safe mode: {self.integrator_safe_mode}\n")
            f.write(f"Integrator corrector: {self.integrator_corrector}\n")
            f.write(f"dt: {self.dt}\n")
            f.write(f"Save: {self.save}\n")
            f.write(f"Save path: {self.save_path}\n")
            f.write(f"Save summary: {self.save_summary}\n")
            f.write(f"Plot: {self.plot}\n")
            f.write(f"Plot path: {self.plot_path}\n")
            f.write(f"Plot type: {self.plot_type}\n")
            f.write(f"Image type: {self.image_type}\n")

            f.write("\n\n Configuration values (default):\n")
            f.write("========================\n")

            for key, value in c.config.items():
                f.write(f"{key}: {value}\n")

            f.write("\n\n Bodies :\n")
            f.write("========================\n")
            for body in self.bodies:
                f.write(f"\n\n Body: {body.name}, mmrs = {', '.join([mmr.to_short() for mmr in body.mmrs])}\n")

            f.write("\n\n Output Files Description:\n")
            f.write("========================\n")
            f.write("The simulation generates several types of output files:\n\n")
            f.write("1. Time Series Data (data-{body}-{mmr}.csv):\n")
            f.write("   - Contains time-indexed data from 0 to tmax\n")
            f.write("   - Columns: times, angle, a (semi-major axis), e (eccentricity)\n")
            f.write("   - Optional: angle_filtered, a_filtered (if filtering was applied)\n\n")
            f.write("2. Periodogram Data:\n")
            f.write("   - data-{body}-{mmr}-periodogram-angle.csv: Periodogram of resonant angle\n")
            f.write("   - data-{body}-{mmr}-periodogram-axis.csv: Periodogram of semi-major axis\n")
            f.write("   - Columns: frequency, power, period, is_peak (1 for peaks, 0 otherwise)\n\n")
            f.write("3. Peak Information:\n")
            f.write("   - data-{body}-{mmr}-periodogram-angle-peaks.csv: Detailed peak info for resonant angle\n")
            f.write("   - data-{body}-{mmr}-periodogram-axis-peaks.csv: Detailed peak info for semi-major axis\n")
            f.write("   - Columns: peak_frequency, peak_power, peak_period, left_boundary, right_boundary\n\n")
            f.write("4. Overlapping Peaks:\n")
            f.write("   - data-{body}-{mmr}-periodogram-overlapping.csv: Peaks that overlap between angle and axis\n")
            f.write("   - Columns: left_boundary, right_boundary, center_period\n\n")
            f.write("5. Summary (summary.csv):\n")
            f.write("   - One row per body-mmr combination with analysis results\n")
            f.write("========================\n")

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
            self.Nout = abs(int(self.__tmax / 100))  # abs for backward integration case

    @tmax.deleter
    def tmax(self):  # pragma: no cover
        del self.__tmax
