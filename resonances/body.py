import numpy as np

from resonances.resonance import Resonance, MMR, SecularResonance, LidovKozaiResonance
from .logger import logger
from typing import List, Union


class Body:
    def __init__(self, type='particle'):
        self.type = type

        self.initial_data = None
        self.name = ''
        self.mass = 0.0

        # Integration data
        self.times = None
        self.axis = None
        self.ecc = None
        self.inc = None
        self.Omega = None
        self.omega = None
        self.M = None
        self.longitude = None
        self.varpi = None

        # Mean-motion resonances data
        # keys are resonances (4J-2S-1-0-0-1), values are values

        self.mmrs: List[MMR] = []
        self.angles = {}  # For MMR angles
        self.statuses = {}

        # Secular resonances data
        self.secular_resonances: List[SecularResonance] = []
        self.secular_angles = {}  # For secular resonance angles
        self.secular_angles_osculating = {}
        self.secular_angles_proper = {}

        # Lidov-Kozai resonance data
        self.lidov_kozai_resonances: List[LidovKozaiResonance] = []
        self.lidov_kozai_angles = {}

        # Libration and filtering data (shared between MMR and secular)
        self.librations = {}
        self.libration_metrics = {}
        self.libration_status = {}
        self.libration_pure = {}

        self.periodogram_frequency = {}
        self.periodogram_power = {}
        self.periodogram_peaks = {}

        self.angles_filtered = {}
        self.secular_angles_filtered = {}

        self.axis_filtered = None
        self.axis_periodogram_frequency = None
        self.axis_periodogram_power = None
        self.axis_periodogram_peaks = None

        self.eccentricity_periodogram_frequency = None
        self.eccentricity_periodogram_power = None
        self.eccentricity_periodogram_peaks = None

        self.periodogram_peaks_overlapping = {}

        self.monotony = {}

        # Simulation data
        self.index_in_simulation = None
        # self.index_of_planets = None

    def __str__(self):
        s = f'Body(type={self.type}, name={self.name}, mass={self.mass})\n'
        if self.mmrs:
            s += 'MMR Resonances: '
            for mmr in self.mmrs:
                s += mmr.to_s() + ', '
            s += '\n'
        if self.secular_resonances:
            s += 'Secular Resonances: '
            for sec in self.secular_resonances:
                s += sec.to_s() + ', '
        return s

    def mmr_to_dict(self, mmr: MMR, times: np.ndarray):
        try:
            df_data = {
                'times': times / (2 * np.pi),
                'angle': self.angles[mmr.to_s()],
                'a': self.axis,
                'e': self.ecc,
                'inc': self.inc,
                'Omega': self.Omega,
                'omega': self.omega,
                'M': self.M,
                'longitude': self.longitude,
                'varpi': self.varpi,
            }

            if self.angles_filtered.get(mmr.to_s()) is not None:
                df_data['angle_filtered'] = self.angles_filtered[mmr.to_s()]

            if self.axis_filtered is not None:
                df_data['a_filtered'] = self.axis_filtered

        except Exception as e:
            logger.error(f'Error in mmr_to_dict function for body={self.name} and mmr={mmr.to_s()}: {e}')
            return None
        return df_data

    def lidov_kozai_to_dict(self, resonance: LidovKozaiResonance, times: np.ndarray):
        """
        Convert Lidov–Kozai resonance data to dictionary format for saving.
        """
        try:
            df_data = {
                'times': times / (2 * np.pi),
                'angle': self.lidov_kozai_angles[resonance.to_s()],
                'a': self.axis,
                'e': self.ecc,
                'inc': self.inc,
                'Omega': self.Omega,
                'omega': self.omega,
                'M': self.M,
                'longitude': self.longitude,
                'varpi': self.varpi,
            }

            if self.angles_filtered.get(resonance.to_s()) is not None:
                df_data['angle_filtered'] = self.angles_filtered[resonance.to_s()]

            if self.axis_filtered is not None:
                df_data['a_filtered'] = self.axis_filtered

        except Exception as e:
            logger.error(f'Error in lidov_kozai_to_dict for body={self.name} and resonance={resonance.to_s()}: {e}')
            return None
        return df_data

    def secular_to_dict(self, secular: SecularResonance, times: np.ndarray):
        """
        Convert secular resonance data to dictionary format for saving.
        """
        try:
            df_data = {
                'times': times / (2 * np.pi),
                'angle': self.secular_angles[secular.to_s()],
                'a': self.axis,
                'e': self.ecc,
                'inc': self.inc,
                'Omega': self.Omega,
                'omega': self.omega,
                'M': self.M,
                'longitude': self.longitude,
                'varpi': self.varpi,
            }

            if self.secular_angles_filtered.get(secular.to_s()) is not None:
                df_data['angle_filtered'] = self.secular_angles_filtered[secular.to_s()]

            if self.axis_filtered is not None:
                df_data['a_filtered'] = self.axis_filtered

        except Exception as e:
            logger.error(f'Error in secular_to_dict function for body={self.name} and secular={secular.to_s()}: {e}')
            return None
        return df_data

    def setup_vars_for_simulation(self, num):
        self.axis, self.ecc, self.inc, self.Omega, self.omega, self.M, self.longitude, self.varpi = (
            np.zeros(num),
            np.zeros(num),
            np.zeros(num),
            np.zeros(num),
            np.zeros(num),
            np.zeros(num),
            np.zeros(num),
            np.zeros(num),
        )
        # Setup MMR angles
        for mmr in self.mmrs:
            self.angles[mmr.to_s()] = np.zeros(num)
        # Setup secular resonance angles
        for secular in self.secular_resonances:
            arr = np.zeros(num)
            self.secular_angles[secular.to_s()] = arr
            self.secular_angles_osculating[secular.to_s()] = arr
        # Setup Lidov-Kozai resonance angles
        for lidov in self.lidov_kozai_resonances:
            self.lidov_kozai_angles[lidov.to_s()] = np.zeros(num)

    def angle(self, resonance: Resonance) -> np.ndarray:
        """
        Get angle array for any supported resonance.
        """
        try:
            if isinstance(resonance, MMR):
                return self.angles[resonance.to_s()]
            elif isinstance(resonance, SecularResonance):
                return self.secular_angles[resonance.to_s()]
            elif isinstance(resonance, LidovKozaiResonance):
                return self.lidov_kozai_angles[resonance.to_s()]
            else:
                raise ValueError(f"Unknown resonance type: {type(resonance)}")
        except Exception:
            raise Exception('The angle for the resonance {} does not exist in the body {}.'.format(resonance.to_s(), self.name))

    def in_resonance(self, resonance: Union[MMR, SecularResonance, LidovKozaiResonance]):
        """
        Check if body is in resonance (works for both MMR and secular).
        """
        if (self.status(resonance) is not None) and (self.status(resonance) > 0):
            return True
        return False

    def status(self, resonance: Union[MMR, SecularResonance, LidovKozaiResonance]):
        """
        Get resonance status
        """
        return self.statuses[resonance.to_s()]

    def in_pure_resonance(self, resonance: Union[MMR, SecularResonance, LidovKozaiResonance]):
        """
        Check if body is in pure resonance (works for both MMR and secular).
        """
        if (self.status(resonance) is not None) and (2 == self.status(resonance)):
            return True
        return False

    def is_particle(self):
        if 'particle' == self.type:
            return True
        return False
