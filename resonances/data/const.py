import rebound

# PLANETS_LONGITUDE = {
#     "Mercury": 0.0,
#     "Venus": 7.452,
#     "Earth": 17.368,
#     "Mars": 17.916,
#     "Jupiter": 4.257,
#     "Saturn": 28.243,
#     "Uranus": 0.0,
#     "Neptune": 0.0,
# }

PLANETS_AXIS = {
    "Mercury": 0.38709843,
    "Venus": 0.72332102,
    "Earth": 1.00000018,
    "Mars": 1.52371243,
    "Jupiter": 5.20248019,
    "Saturn": 9.54149883,
    "Uranus": 19.18797948,
    "Neptune": 30.06952752,
    "Pluto": 39.48686035,
}

masses = rebound.units.masses_SI

PLANETS_MASS = {
    "Mercury": masses['mmercury'] / masses['msun'],
    "Venus": masses['mvenus'] / masses['msun'],
    "Earth": masses['mearth'] / masses['msun'],
    "Mars": masses['mmars'] / masses['msun'],
    "Jupiter": masses['mjupiter'] / masses['msun'],
    "Saturn": masses['msaturn'] / masses['msun'],
    "Uranus": masses['muranus'] / masses['msun'],
    "Neptune": masses['mneptune'] / masses['msun'],
    "Pluto": masses['mpluto'] / masses['msun'],
}

K = 0.0172020989484
DAYS_IN_YEAR = 365.2422

SOLAR_SYSTEM = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
