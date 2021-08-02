import re
import resonances


def create_mmr(coeff, planets_names=None):
    if isinstance(coeff, str):
        tmp = re.split('-|\\+', coeff)
        size = len(tmp)
        if 3 == size:
            return resonances.ThreeBody(coeff)
        elif 2 == size:
            return resonances.TwoBody(coeff)
        else:
            raise Exception(
                """Cannot create a resonance because the notation is wrong.
                 It should have either two or three bodies, i.e. 2J-1 or 4M-2J-1. Given {}.""".format(
                    coeff
                )
            )
    elif isinstance(coeff, list):
        size = len(coeff)
        if 6 == size:
            return resonances.ThreeBody(coeff, planets_names)
        elif 4 == size:
            return resonances.TwoBody(coeff, planets_names)
        else:
            raise Exception(
                'Cannot create a resonance because the number of coefficients is wrong. It should be equal to 2 or 3. Given {}.'.format(
                    len(coeff)
                )
            )

    raise Exception(
        'The argument should be either a string if you use the short notation (i.e. 2J-1) or a list containing 2 or 3 elements.'
    )
