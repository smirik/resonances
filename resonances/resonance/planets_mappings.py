PLANET_LETTER_TO_NAME = {
    "R": "Mercury",
    "V": "Venus",
    "E": "Earth",
    "M": "Mars",
    "J": "Jupiter",
    "S": "Saturn",
    "U": "Uranus",
    "N": "Neptune",
}

PLANET_NAME_TO_LETTER = {
    "Mercury": "R",
    "Venus": "V",
    "Earth": "E",
    "Mars": "M",
    "Jupiter": "J",
    "Saturn": "S",
    "Uranus": "U",
    "Neptune": "N",
}


def planet_name_from_letter(letter: str) -> str:
    return PLANET_LETTER_TO_NAME.get(letter.upper(), letter)


def planet_letter_from_name(planet_name: str) -> str:
    return PLANET_NAME_TO_LETTER.get(planet_name, planet_name[0])
