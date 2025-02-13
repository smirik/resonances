import re
import resonances


def create_mmr(coeff, planets_names=None):
    """Create Mean Motion Resonance (MMR) object(s) based on the input format.
    This function serves as a factory method for creating MMR objects. It supports multiple input formats
    and provides a universal interface for MMR creation.

    Args:
        coeff: Input that defines the resonance(s). Can be:
            - An MMR instance (returned as-is)
            - A string in format "2J-1" (two-body) or "4M-2J-1" (three-body)
            - A list of coefficients: 4 elements for two-body or 6 elements for three-body resonance
            - A list of strings, each in format "2J-1" or "4M-2J-1"
            - A list of MMR objects (returned as-is)
        planets_names (list, optional): List of planet names when using coefficient list format.
            Defaults to None.
    Returns:
        resonances.MMR or list[resonances.MMR]: A single MMR object or list of MMR objects
    Raises:
        Exception: If the input format is invalid or the number of coefficients doesn't match
            required format (2 or 3 bodies)
    Examples:
        >>> create_mmr("2J-1")  # Creates two-body resonance from string
        >>> create_mmr([1, 2, -3, 4])  # Creates two-body resonance from coefficients
        >>> create_mmr("4M-2J-1")  # Creates three-body resonance from string
        >>> create_mmr(['4J-2S-1', '1J-1'])  # Creates list of resonances from strings
        >>> create_mmr(existing_mmr)  # Returns existing MMR instance as-is
        >>> create_mmr([mmr1, mmr2])  # Returns list of MMR instances as-is
    """

    if isinstance(coeff, resonances.MMR):
        return coeff

    if isinstance(coeff, list):
        if len(coeff) == 0:
            raise Exception('If input is a list, it should contain a string representation of MMRs, MMR objects, or coefficients.')
        if isinstance(coeff[0], resonances.MMR):
            return coeff
        if isinstance(coeff[0], str):
            return [create_mmr(c) for c in coeff]
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

    raise Exception(
        'The argument should be either a string if you use the short notation (i.e. 2J-1) or a list containing 2 or 3 elements.'
    )
