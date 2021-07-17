import numpy as np


class MMR:
    def __init__(self, coeff, planets_names=[]):
        self.coeff = np.array(coeff)
        if self.coeff[0] <= 0:
            raise Exception("The primary coefficient of the resonance should be greater than 0. Given {}.".format(self.coeff[0]))

        if sum(self.coeff) != 0:
            raise Exception(
                "Sum of integers in a resonance should follow the D'alambert rule. Given {}, the sum is equal to {}.".format(
                    ', '.join(str(e) for e in coeff), sum(self.coeff)
                )
            )
        self.planets_names = planets_names

    def number_of_bodies(self):
        return len(self.coeff) / 2
