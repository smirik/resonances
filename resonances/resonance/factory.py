import re
from typing import Union

from resonances.mmr.mmr import MMR
from resonances.resonance.resonance import Resonance
from resonances.secular.secular_resonance import SECULAR_RESONANCE_ALIASES, SecularResonance
from resonances.mmr.three_body import ThreeBody
from resonances.mmr.two_body import TwoBody
from resonances.lidov_kozai.lidov_kozai_resonance import LidovKozaiResonance
from resonances.resonance.planets_mappings import planet_name_from_letter


def create_mmr(coeff, planets_names=None):  # noqa: C901
    """Create Mean Motion Resonance (MMR) object(s) based on the input format.
    This function serves as a factory method for creating MMR objects. It supports multiple input formats
    and provides a universal interface for MMR creation.

    Args:
        coeff: Input that defines the resonance(s). Can be:
            - An MMR instance (returned as-is)
            - A string in format "2J-1" (two-body) or "4M-2J-1" (three-body) for MMR
            - A list of coefficients: 4 elements for two-body or 6 elements for three-body resonance
            - A list of strings, each in format "2J-1", "4M-2J-1"
            - A list of MMR objects (returned as-is)
        planets_names (list, optional): List of planet names when using coefficient list format.
            Defaults to None.
    Returns:
        MMR or list: A single MMR object or list of MMR objects
    Raises:
        Exception: If the input format is invalid or the number of coefficients doesn't match
            required format (2 or 3 bodies)
    Examples:
        >>> create_mmr("2J-1")  # Creates two-body MMR from string
        >>> create_mmr([1, 2, -3, 4])  # Creates two-body MMR from coefficients
        >>> create_mmr("4M-2J-1")  # Creates three-body MMR from string
        >>> create_mmr(['4J-2S-1', '1J-1'])  # Creates list of MMRs
        >>> create_mmr(existing_mmr)  # Returns existing MMR instance as-is
        >>> create_mmr([mmr1, mmr2])  # Returns list of MMR instances as-is
    """

    if isinstance(coeff, MMR):
        return coeff

    if isinstance(coeff, list):
        if len(coeff) == 0:
            raise Exception('If input is a list, it should contain a string representation of MMRs, MMR objects, or coefficients.')
        if isinstance(coeff[0], MMR):
            return coeff
        if isinstance(coeff[0], str):
            return [create_mmr(c) for c in coeff]
        size = len(coeff)
        if 6 == size:
            return ThreeBody(coeff, planets_names)
        elif 4 == size:
            return TwoBody(coeff, planets_names)
        else:
            raise Exception(
                'Cannot create a resonance because the number of coefficients is wrong. '
                'It should be equal to 2 or 3. Given {}.'.format(len(coeff))
            )

    if isinstance(coeff, str):
        # Parse as MMR string
        coeff = coeff.replace(" ", "")
        tmp = re.split('-|\\+', coeff)
        size = len(tmp)
        if 3 == size:
            return ThreeBody(coeff)
        elif 2 == size:
            return TwoBody(coeff)
        else:
            tokens = re.findall(r"[A-Za-z]+|[+-]?\d+", coeff)
            letters = [t for t in tokens if t.isalpha()]
            numbers = [int(t) for t in tokens if re.match(r"[+-]?\d+$", t)]
            if len(letters) >= 1 and len(numbers) >= 4:
                if len(letters) == 1 and len(numbers) == 4:
                    return TwoBody(numbers, [planet_name_from_letter(letters[0])])
                if len(letters) >= 2 and len(numbers) >= 6:
                    planets = [planet_name_from_letter(letters[0]), planet_name_from_letter(letters[1])]
                    return ThreeBody(numbers[:6], planets)
            raise Exception(
                """Cannot create a resonance because the notation is wrong.
                 It should have either two or three bodies (i.e. 2J-1 or 4M-2J-1) for MMR. Given {}.""".format(
                    coeff
                )
            )

    raise Exception(
        'The argument should be either a string if you use the short notation (i.e. 2J-1) or a list containing 2 or 3 elements.'
    )


def create_secular_resonance(secular_type):
    """Create Secular Resonance object based on the input format.

    This function serves as a factory method for creating secular resonance objects.

    Args:
        secular_type: Input that defines the secular resonance. Can be:
            - A SecularResonance instance (returned as-is)
            - A string in format "nu6", "nu5", "nu16"
            - A string in mathematical format like "g-g5", "2g-g5-g6", "2(g-g6)+(s-s6)", etc.
            - A list of strings, each representing a secular resonance type
            - A list of SecularResonance objects (returned as-is)
        planets_names (list, optional): List of planet names for custom
            Defaults to None.

    Returns:
        SecularResonance or list: A single SecularResonance object or list of objects

    Raises:
        Exception: If the input format is invalid or the secular resonance type is unknown

    Examples:
        >>> create_secular_resonance("nu6")  # Creates ν₆ secular resonance
        >>> create_secular_resonance("nu5")  # Creates ν₅ secular resonance
        >>> create_secular_resonance("nu16") # Creates ν₁₆ secular resonance
        >>> create_secular_resonance("g-g5") # Creates secular resonance from formula
        >>> create_secular_resonance("2*g-g5-g6") # Creates complex secular resonance
        >>> create_secular_resonance(['nu6', 'nu5'])  # Creates list of secular resonances
        >>> create_secular_resonance(existing_secular)  # Returns existing instance as-is
        >>> create_secular_resonance([sec1, sec2])  # Returns list of instances as-is
    """

    if isinstance(secular_type, SecularResonance):
        return secular_type

    if isinstance(secular_type, str):
        secular_lower = secular_type.lower()
        formula = SECULAR_RESONANCE_ALIASES.get(secular_lower, secular_type)
        return SecularResonance(formula=formula)

    if isinstance(secular_type, list):
        return [create_secular_resonance(s) for s in secular_type]

    raise Exception('The argument should be either a string (i.e. "nu6" or "g-g5") or a SecularResonance object.')


def detect_resonance_type(resonance: Union[Resonance, str]) -> str:
    if isinstance(resonance, Resonance):
        return resonance.type
    elif isinstance(resonance, str):
        normalized = resonance.lower()
        normalized_base = normalized.split(':', 1)[0]
        if normalized_base in ['lk', 'lkr', 'lidov-kozai', 'lidov_kozai', 'lidovkozai', 'lidov', 'kozai']:
            return 'lidov_kozai'
        if (
            (normalized in SECULAR_RESONANCE_ALIASES.keys())
            or normalized.startswith('g')
            or normalized.startswith('s')
            or normalized.startswith('2g')
            or normalized.startswith('2s')
        ):
            return 'secular'
        else:
            return 'mmr'
    raise Exception('The argument should be either a string (i.e. "nu6") or a Resonance object.')


def create_resonance(resonance: Union[Resonance, str]) -> Resonance:
    if isinstance(resonance, Resonance):
        return resonance
    elif isinstance(resonance, str):
        res_type = detect_resonance_type(resonance)
        if res_type == 'mmr':
            return create_mmr(resonance)
        elif res_type == 'secular':
            return create_secular_resonance(resonance)
        elif res_type == 'lidov_kozai':
            identifier = resonance.split(':', 1)[1].strip() if ':' in resonance else resonance.strip()
            if identifier.lower() in ['lk', 'lkr', 'lidov-kozai', 'lidov_kozai', 'lidovkozai', 'kozai']:
                identifier = 'LK'
            return LidovKozaiResonance(identifier=identifier or 'LK')
        else:
            raise Exception(f'Unknown resonance type: {res_type}')
    raise Exception('The argument should be either a valid secular resonance (i.e. "nu6") or a Resonance object.')
