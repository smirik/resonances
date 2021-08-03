import numpy as np
import re
from resonances.resonance.mmr import MMR
from resonances.data import const
import rebound


class TwoBody(MMR):
    def __init__(self, coeff, planets_names=None):

        if isinstance(coeff, str):
            coeff, planets_names = self.init_from_short_notation(coeff)

        super().__init__(coeff, planets_names)

        if np.gcd(self.coeff[0], self.coeff[1]) > 1:
            raise Exception('The integers should have gcd equals 1. Given {}.'.format(', '.join(str(e) for e in coeff)))

    def calc_angle(self, body, planets):
        body1 = planets[0]
        angle = rebound.mod2pi(
            self.coeff[0] * body1.l
            + self.coeff[1] * body.l
            + self.coeff[2] * (body1.Omega + body1.omega)
            + self.coeff[3] * (body.Omega + body.omega)
        )
        return angle

    def init_from_short_notation(self, s):
        tmp = re.split('-|\\+', s)
        if 2 != len(tmp):
            raise Exception('You must specify two integers only for a short notation, i.e., 2J-1.')

        first_letter = tmp[0][len(tmp[0]) - 1]
        planets_names = [self.get_planet_name_from_letter(first_letter)]

        coeff1 = int(str.replace(tmp[0], first_letter, ''))
        coeff2 = int(tmp[1])
        symbol2 = s[len(tmp[0])]
        if symbol2 == '-':
            coeff2 = -coeff2
        coeff = [coeff1, coeff2, 0, (0 - coeff1 - coeff2)]
        return coeff, planets_names

    def to_s(self):
        s = '{:d}{:.1}{:+d}{:+d}{:+d}'.format(
            int(self.coeff[0]),
            self.planets_names[0],
            int(self.coeff[1]),
            int(self.coeff[2]),
            int(self.coeff[3]),
        )
        return s

    def to_short(self):
        s = '{:d}{:.1}{:+d}'.format(
            int(self.coeff[0]),
            self.planets_names[0],
            int(self.coeff[1]),
        )
        return s

    def calculate_resonant_axis(self):
        if len(self.planets_names) != 1:
            raise Exception('Cannot calculate resonant axis if the planet is not specified!')

        planet_axis = const.PLANETS_AXIS[self.planets_names[0]]
        axis = planet_axis * (((self.coeff[1] / self.coeff[0]) ** (2.0)) ** (1.0 / 3))

        return axis
