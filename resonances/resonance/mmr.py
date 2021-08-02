import numpy as np


class MMR:
    def __init__(self, coeff, planets_names=None):
        self.coeff = np.array(coeff)
        if self.coeff[0] <= 0:
            raise Exception("The primary coefficient of the resonance should be greater than 0. Given {}.".format(self.coeff[0]))

        if sum(self.coeff) != 0:
            raise Exception(
                "Sum of integers in a resonance should follow the D'Alembert rule. Given {}, the sum is equal to {}.".format(
                    ', '.join(str(e) for e in coeff), sum(self.coeff)
                )
            )

        if planets_names is None:
            self.planets_names = []
        else:
            self.planets_names = planets_names

        self._resonant_axis = None

    def number_of_bodies(self):
        return len(self.coeff) / 2

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

    @property
    def resonant_axis(self):
        if self._resonant_axis is None:
            self._resonant_axis = self.calculate_resonant_axis()
        return self._resonant_axis

    @resonant_axis.setter
    def resonant_axis(self, value):
        self._resonant_axis = value

    def calculate_resonant_axis(self):
        pass
