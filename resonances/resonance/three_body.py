import numpy as np
import re
from resonances.resonance.mmr import MMR
import resonances.data.util as util
from resonances.data import const
import rebound


class ThreeBody(MMR):
    def __init__(self, coeff, planets_names=None):
        if isinstance(coeff, str):
            coeff, planets_names = self.init_from_short_notation(coeff)

        super().__init__(coeff, planets_names)

        if np.gcd(np.gcd(self.coeff[0], self.coeff[1]), self.coeff[2]) > 1:
            raise Exception('The integers should have gcd equals 1. Given {}.'.format(', '.join(str(e) for e in coeff)))

    def calc_angle(self, body, planets):
        body1 = planets[0]
        body2 = planets[1]
        angle = rebound.mod2pi(
            self.coeff[0] * body1.l
            + self.coeff[1] * body2.l
            + self.coeff[2] * body.l
            + self.coeff[3] * (body1.Omega + body1.omega)
            + self.coeff[4] * (body2.Omega + body2.omega)
            + self.coeff[5] * (body.Omega + body.omega)
        )
        return angle

    def init_from_short_notation(self, s):
        tmp = re.split('-|\\+', s)
        if 3 != len(tmp):
            raise Exception('You must specify three integers only for a short notation, i.e., 4J-2S-1.')

        first_letter = tmp[0][len(tmp[0]) - 1]
        second_letter = tmp[1][len(tmp[1]) - 1]
        planets_names = [self.get_planet_name_from_letter(first_letter), self.get_planet_name_from_letter(second_letter)]

        coeff1 = int(str.replace(tmp[0], first_letter, ''))
        coeff2 = int(str.replace(tmp[1], second_letter, ''))
        coeff3 = int(tmp[2])
        symbol2 = s[len(tmp[0])]
        symbol3 = s[len(tmp[0]) + len(tmp[1]) + 1]
        if symbol2 == '-':
            coeff2 = -coeff2
        if symbol3 == '-':
            coeff3 = -coeff3
        coeff = [coeff1, coeff2, coeff3, 0, 0, (0 - coeff1 - coeff2 - coeff3)]
        return coeff, planets_names

    def to_s(self):
        s = '{:d}{:.1}{:+d}{:.1}{:+d}{:+d}{:+d}{:+d}'.format(
            int(self.coeff[0]),
            self.planets_names[0],
            int(self.coeff[1]),
            self.planets_names[1],
            int(self.coeff[2]),
            int(self.coeff[3]),
            int(self.coeff[4]),
            int(self.coeff[5]),
        )
        return s

    def to_short(self):
        s = '{:d}{:.1}{:+d}{:.1}{:+d}'.format(
            int(self.coeff[0]),
            self.planets_names[0],
            int(self.coeff[1]),
            self.planets_names[1],
            int(self.coeff[2]),
        )
        return s

    @property
    def resonant_axis(self):
        if self._resonant_axis is None:
            self._resonant_axis = self.calculate_resonant_axis()
        return self._resonant_axis

    @resonant_axis.setter
    def resonant_axis(self, value):
        self._resonant_axis = value

    def calculate_resonant_axis(self):
        if len(self.planets_names) != 2:
            raise Exception('Cannot calculate resonant axis if the planets are not specified!')

        """ "
        a_i - semi-major axis of two planets and the body
        n_i — mean motions
        l_i — longitude of node
        p_i — the names of the planets
        The indexes 1,2 refers to the planets, no index — to the body
        k — gravitational constant (~0.017)
        """
        p1 = self.planets_names[0]
        p2 = self.planets_names[1]

        n1 = util.mean_motion_from_axis(const.PLANETS_AXIS[p1])
        n2 = util.mean_motion_from_axis(const.PLANETS_AXIS[p2])

        l1 = 0
        l2 = 0

        n = (-self.coeff[0] * n1 - self.coeff[1] * n2 - self.coeff[3] * l1 - self.coeff[4] * l2) / self.coeff[2]

        if n < 0:
            raise Exception('Something weird has happened for {}. Mean motion = {}'.format(self.to_short(), n))

        la = 0.0
        for planet in const.SOLAR_SYSTEM:
            ap = const.PLANETS_AXIS[planet]
            mp = const.PLANETS_MASS[planet]
            a = util.axis_from_mean_motion(n)

            if a > ap:
                la = la + (3 * np.pi / 2 * mp / (1.0 + mp) ** 1.5 * (ap**2 / a**3.5)) / const.DAYS_IN_YEAR
            else:
                la = la + (3 * np.pi / 2 * mp * (np.sqrt(a) / ap) ** 3) / const.DAYS_IN_YEAR

        n = (-self.coeff[0] * n1 - self.coeff[1] * n2 - self.coeff[3] * l1 - self.coeff[4] * l2 - self.coeff[5] * la) / self.coeff[2]

        if n < 0:
            raise Exception('Something weird has happened for {}. Mean motion = {}'.format(self.to_short(), n))

        a = util.axis_from_mean_motion(n)

        return a

    # Yes, I know that it is a bad practice to keep commented code.
    # However, this might be useful for some edge cases.
    # def another_calculate_axis(self):
    #     """ "
    #     a_i - semi-major axis of two planets and the body
    #     n_i — mean motions
    #     l_i — longitude of node
    #     p_i — the names of the planets
    #     The indexes 1,2 refers to the planets, no index — to the body
    #     k — gravitational constant (~0.017)
    #     """
    #     p1 = self.planets_names[0]
    #     p2 = self.planets_names[1]

    #     a1 = const.PLANETS_AXIS[p1]
    #     a2 = const.PLANETS_AXIS[p2]

    #     n1 = util.mean_motion_from_axis(a1)
    #     n2 = util.mean_motion_from_axis(a2)

    #     l1 = util.from_s_to_yr(const.PLANETS_LONGITUDE[p1])
    #     l2 = util.from_s_to_yr(const.PLANETS_LONGITUDE[p2])

    #     k = const.K

    #     n = (-self.coeff[0] * n1 - self.coeff[1] * n2 - self.coeff[3] * l1 - self.coeff[4] * l2) / self.coeff[2]

    #     if n < 0:
    #         raise Exception('Something weird has happened for {}. Mean motion = {}'.format(self.to_short(), n))

    #     a = util.axis_from_mean_motion(n)
    #     eps = (a1 - a) / a1
    #     l = k / (2 * np.pi) * np.sqrt(a / a1) * (eps ** 2) * n1

    #     n = (-self.coeff[0] * n1 - self.coeff[1] * n2 - self.coeff[3] * l1 - self.coeff[4] * l2 - self.coeff[5] * l) / self.coeff[2]

    #     if n < 0:
    #         raise Exception('Something weird has happened for {}. Mean motion = {}'.format(self.to_short(), n))

    #     a = util.axis_from_mean_motion(n)
    #     return a
