import numpy as np
import rebound


class Body:
    def __init__(self, type='particle'):
        self.type = type

        self.initial_data = None
        self.mmr = None
        self.name = ''
        self.mass = 0.0

        # Integration data
        self.times = None
        self.axis = None
        self.ecc = None
        self.longitude = None
        self.varpi = None
        self.angle = None

        # Libration and filtering data
        self.librations = None
        self.libration_metrics = None
        self.libration_status = None
        self.libration_pure = None

        self.periodogram_frequency = None
        self.periodogram_power = None
        self.periodogram_peaks = None

        self.axis_filtered = None
        self.axis_periodogram_frequency = None
        self.axis_periodogram_power = None
        self.axis_periodogram_peaks = None

        self.periodogram_peaks_overlapping = None

        self.monotony = None

        # Simulation data
        self.index_in_simulation = None
        self.index_of_planets = None

    def calc_angle(self, os):
        body = os[self.index_in_simulation - 1]  # because Sun is not is os
        body1 = os[self.index_of_planets[0] - 1]
        body2 = os[self.index_of_planets[1] - 1]
        angle = rebound.mod2pi(
            self.mmr.coeff[0] * body1.l
            + self.mmr.coeff[1] * body2.l
            + self.mmr.coeff[2] * body.l
            + self.mmr.coeff[3] * (body1.Omega + body1.omega)
            + self.mmr.coeff[4] * (body2.Omega + body2.omega)
            + self.mmr.coeff[5] * (body.Omega + body.omega)
        )
        return angle

    def setup_vars_for_simulation(self, num):
        self.axis, self.ecc, self.longitude, self.varpi, self.angle = (
            np.zeros(num),
            np.zeros(num),
            np.zeros(num),
            np.zeros(num),
            np.zeros(num),
        )

    def in_resonance(self):
        if (self.status is not None) and (self.status > 0):
            return True
        return False

    def in_pure_resonance(self):
        if (self.status is not None) and (2 == self.status):
            return True
        return False

    def is_particle(self):
        if 'particle' == self.type:
            return True
        return False
