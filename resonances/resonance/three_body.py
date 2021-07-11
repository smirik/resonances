import rebound.tools
import numpy as np


class ThreeBody:
    def __init__(
        self,
        coeff,
        index_of_planets=[0, 1],
        index_of_body=2,
        body_name='asteroid',
        planets_names=['Jupiter', 'Saturn'],
    ):
        self.coeff = np.array(coeff)
        if sum(self.coeff) != 0:
            raise Exception(
                "Sum of integers in a resonance should follow the D'alambert rule. Given {}, the sum is equal to {}.".format(
                    ', '.join(str(e) for e in coeff), sum(self.coeff)
                )
            )
        self.index_of_planets = index_of_planets
        self.index_of_body = index_of_body
        self.body_name = body_name
        self.planets_names = planets_names

    def angle(self, os):
        body = os[self.index_of_body - 1]  # because Sun is not is os
        body1 = os[self.index_of_planets[0] - 1]
        body2 = os[self.index_of_planets[1] - 1]
        angle = rebound.mod2pi(
            self.coeff[0] * body1.l
            + self.coeff[1] * body2.l
            + self.coeff[2] * body.l
            + self.coeff[3] * (body1.Omega + body1.omega)
            + self.coeff[4] * (body2.Omega + body2.omega)
            + self.coeff[5] * (body.Omega + body.omega)
        )
        return angle

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

    def index_of_bodies(self):
        return self.index_of_planets + [self.index_of_body]

    def number_of_bodies(self):
        return 3
