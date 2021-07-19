import numpy as np
import re
from resonances.resonance.mmr import MMR
import resonances.data.util as util
from resonances.data import const


class ThreeBody(MMR):
    def __init__(self, coeff, planets_names=None, s=None):
        self._resonant_axis = None

        if s is not None:
            self.init_from_short_notation(s)
            return
        if isinstance(coeff, str):
            self.init_from_short_notation(coeff)
            return

        self.coeff = np.array(coeff)
        if sum(self.coeff) != 0:
            raise Exception(
                "Sum of integers in a resonance should follow the D'alambert rule. Given {}, the sum is equal to {}.".format(
                    ', '.join(str(e) for e in coeff), sum(self.coeff)
                )
            )

        if np.gcd(np.gcd(self.coeff[0], self.coeff[1]), self.coeff[2]) > 1:
            raise Exception('The integers should have gcd equals 1. Given {}.'.format(', '.join(str(e) for e in coeff)))

        if planets_names is None:
            self.planets_names = []
        else:
            self.planets_names = planets_names

    def init_from_short_notation(self, s):
        tmp = re.split('-|\\+', s)
        if 3 != len(tmp):
            raise Exception('You must specify three integers only for a short notation, i.e., 4J-2S-1.')

        first_letter = tmp[0][len(tmp[0]) - 1]
        second_letter = tmp[1][len(tmp[1]) - 1]
        self.planets_names = [self.get_planet_name_from_letter(first_letter), self.get_planet_name_from_letter(second_letter)]

        coeff1 = int(str.replace(tmp[0], first_letter, ''))
        coeff2 = int(str.replace(tmp[1], second_letter, ''))
        coeff3 = int(tmp[2])
        symbol2 = s[len(tmp[0])]
        symbol3 = s[len(tmp[0]) + len(tmp[1]) + 1]
        if symbol2 == '-':
            coeff2 = -coeff2
        if symbol3 == '-':
            coeff3 = -coeff3
        self.coeff = [coeff1, coeff2, coeff3, 0, 0, (0 - coeff1 - coeff2 - coeff3)]

    def get_planet_name_from_letter(self, letter):
        if letter == 'R':
            return 'Mercury'
        elif letter == 'V':
            return 'Venus'
        elif letter == 'E':
            return 'Earth'
        elif letter == 'M':
            return 'Mars'
        elif letter == 'J':
            return 'Jupiter'
        elif letter == 'S':
            return 'Saturn'
        elif letter == 'U':
            return 'Uranus'
        elif letter == 'N':
            return 'Neptune'
        raise Exception('Bad notation used. Only the following letter are available: R (for Mercury), V, E, M, J, S, U, N ')

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
                la = la + (3 * np.pi / 2 * mp / (1.0 + mp) ** 1.5 * (ap ** 2 / a ** 3.5)) / const.DAYS_IN_YEAR
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
