import numpy as np
import re
from resonances.resonance.mmr import MMR


class ThreeBody(MMR):
    def __init__(self, coeff=[], planets_names=['Jupiter', 'Saturn'], s=None):
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

    # def angle(self, os):
    #     body = os[self.index_of_body - 1]  # because Sun is not is os
    #     body1 = os[self.index_of_planets[0] - 1]
    #     body2 = os[self.index_of_planets[1] - 1]
    #     angle = rebound.mod2pi(
    #         self.coeff[0] * body1.l
    #         + self.coeff[1] * body2.l
    #         + self.coeff[2] * body.l
    #         + self.coeff[3] * (body1.Omega + body1.omega)
    #         + self.coeff[4] * (body2.Omega + body2.omega)
    #         + self.coeff[5] * (body.Omega + body.omega)
    #     )
    #     return angle

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

    # def index_of_bodies(self):
    #     return self.index_of_planets + [self.index_of_body]

    # def number_of_bodies(self):
    #     return 3
