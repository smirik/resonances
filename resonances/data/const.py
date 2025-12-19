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

PLANETARY_FREQUENCIES = {
    'g5': 4.25749319,
    'g6': 28.24552984,
    'g7': 3.08675577,
    'g8': 0.67255084,
    's5': 0.0,
    's6': -26.34496354,
    's7': -2.99266093,
    's8': -0.69251386,
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

SECULAR_FORMULAS = [
    # --- Linear secular resonances ---
    "g-g5",
    "g-g6",
    "g-g7",
    "s-s6",
    "s-s7",
    # --- Kozai-type ---
    "2g-2s",
    # --- Quadratic / degree-4 nonlinear (classical set) ---
    "g+s-s7-g5",
    "g+s-s7-g6",
    "g+s-s6-g5",
    "g+s-s6-g6",
    "g-g5+s7-s6",
    "g-g5-s7+s6",
    "g-g6+s7-s6",
    "g-g6-s7+s6",
    "-g+s+g5-s7",
    "-g+s+g6-s7",
    "-g+s+g5-s6",
    "-g+s+g6-s6",
    "g-s+g5-s7",
    "g-s+g5-s6",
    "g-s+g6-s7",
    "g-s+g6-s6",
    "2g-s-s7",
    "2g-s-s6",
    "-g+2s-g5",
    "-g+2s-g6",
    "2g-2s7",
    "2g-2s6",
    "2g-s7-s6",
    "g+g5-2s7",
    "g+g6-2s7",
    "g+g5-2s6",
    "g+g6-2s6",
    "g+g5-s7-s6",
    "g+g6-s7-s6",
    "s-2s7+s6",
    "s+s7-2s6",
    "2s-s7-s6",
    "s+g5-g6-s7",
    "s-g5+g6-s7",
    "s+g5-g6-s6",
    "s-g5+g6-s6",
    "2s-2g5",
    "2s-2g6",
    "2s-g5-g6",
    "s-2g5+s7",
    "s-2g5+s6",
    "s-2g6+s7",
    "s-2g6+s6",
    "s-g5-g6+s7",
    "s-g5-g6+s6",
    "2g-2g5",
    "2g-2g6",
    "2s-2s7",
    "2s-2s6",
    # --- g-type nonlinear (Huaman et al. 2017; Carruba et al. 2024) ---
    "g-2g6+g5",
    "g-3g6+2g5",
    "2g-3g5+g6",
    # --- z-series (Milani & Knežević notation) ---
    "g-g6+s-s6",  # z1
    "2(g-g6)+(s-s6)",  # z2
    "3(g-g6)+(s-s6)",  # z3
    "4(g-g6)+(s-s6)",  # z4
    # --- Higher-order forced terms explicitly discussed in literature ---
    "g-2g6+g7",
    "g-3g6+2g5",
    "g+g5-g6-g7",
    "g-g5-g6+g7",
    "g+g5-2g6-s6+s7",
]
