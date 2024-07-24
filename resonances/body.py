import numpy as np
from .resonance.mmr import MMR
from typing import List


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
        self.longitude = None
        self.varpi = None

        # resonances data
        # keys are resonances (4J-2S-1-0-0-1), values are values

        self.mmrs: List[MMR] = []
        self.angles = {}
        self.statuses = {}

        # Libration and filtering data
        self.librations = {}
        self.libration_metrics = {}
        self.libration_status = {}
        self.libration_pure = {}

        self.periodogram_frequency = {}
        self.periodogram_power = {}
        self.periodogram_peaks = {}

        self.angles_filtered = {}

        self.axis_filtered = None
        self.axis_periodogram_frequency = None
        self.axis_periodogram_power = None
        self.axis_periodogram_peaks = None

        self.periodogram_peaks_overlapping = {}

        self.monotony = {}

        # Simulation data
        self.index_in_simulation = None
        # self.index_of_planets = None

    def __str__(self):
        s = f'Body(type={self.type}, name={self.name}, mass={self.mass})\n Resonances: '
        for mmr in self.mmrs:
            s += mmr.to_s() + ', '
        return s

    def mmr_to_dict(self, mmr: MMR):
        try:
            df_data = {
                'times': self.times / (2 * np.pi),
                'angle': self.angles[mmr.to_s()],
                'a': self.axis,
                'e': self.ecc,
            }
            if self.periodogram_power is not None:
                len_diff = len(self.angles[mmr.to_s()]) - len(self.periodogram_power[mmr.to_s()])
                df_data['periodogram'] = np.append(self.periodogram_power[mmr.to_s()], np.zeros(len_diff))
                df_data['a_filtered'] = self.axis_filtered
                df_data['a_periodogram'] = np.append(self.axis_periodogram_power, np.zeros(len_diff))
        except Exception:
            return None
        return df_data

    def setup_vars_for_simulation(self, num):
        self.axis, self.ecc, self.longitude, self.varpi = (
            np.zeros(num),
            np.zeros(num),
            np.zeros(num),
            np.zeros(num),
        )
        for mmr in self.mmrs:
            self.angles[mmr.to_s()] = np.zeros(num)

    def angle(self, mmr: MMR) -> np.ndarray:
        try:
            return self.angles[mmr.to_s()]
        except Exception:
            raise Exception('The angle for the resonance {} does not exist in the body {}.'.format(mmr.to_s(), self.name))

    def in_resonance(self, mmr: MMR):
        if (self.status(mmr) is not None) and (self.status(mmr) > 0):
            return True
        return False

    def status(self, mmr: MMR):
        return self.statuses[mmr.to_s()]

    def in_pure_resonance(self, mmr: MMR):
        if (self.status(mmr) is not None) and (2 == self.status(mmr)):
            return True
        return False

    def is_particle(self):
        if 'particle' == self.type:
            return True
        return False
