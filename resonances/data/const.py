import datetime
import math

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

# Fundamental secular frequencies of the planets, arcsec/yr: g_j of the eccentricity vectors, s_j of
# the inclination vectors, j = 1..8 for Mercury..Neptune. s5 is identically zero (conservation of
# the angular momentum; its "mode" is the invariable plane).
#
# g5..g8, s6..s8 are the AstDyS / Knezevic & Milani (2000) values the package always used.
# g1..g4, s1..s4 are measured on the package's own model (Newtonian, Earth = Earth-Moon barycentre,
# Horizons 2026-09-29) by NAFF over 20 Myr: experiments/2026-10-planetary-frequencies/measure.py.
# They differ from La2004 (Laskar et al. 2004, A&A 428, 261) mostly through general relativity,
# which the model lacks: g1 by -0.38 (Mercury's 43"/century), g2, g3, g4 by ~-0.05..-0.12.
# The inner-planet frequencies are chaotic and drift by ~0.01-0.1 over 10 Myr; s2 in particular
# is split into several lines and is the least certain.
PLANETARY_FREQUENCIES = {
    'g1': 5.2131,
    'g2': 7.3353,
    'g3': 17.3164,
    'g4': 17.8603,
    'g5': 4.25749319,
    'g6': 28.24552984,
    'g7': 3.08675577,
    'g8': 0.67255084,
    's1': -5.5055,
    's2': -6.8001,
    's3': -18.8610,
    's4': -17.7386,
    's5': 0.0,
    's6': -26.34496354,
    's7': -2.99266093,
    's8': -0.69251386,
}

# Phases of the modes at J2000, degrees: the argument of the g_j (s_j) term in planet j's own
# e*exp(i*varpi) (sin(i/2)*exp(i*Omega)), ecliptic J2000, same measurement as above. A secular
# angle uses g_j*t + phase_j for planet j, not the planet's osculating longitude: the osculating
# varpi of Venus, Earth, Mars, Uranus and Neptune and the osculating (ecliptic) Omega of every
# planet do not turn at their own mode (see docs/secular.md).
PLANETARY_PHASES = {
    'g1': 90.4,
    'g2': 195.9,
    'g3': 294.3,
    'g4': 335.0,
    'g5': 30.6,
    'g6': 127.6,
    'g7': 120.9,
    'g8': 76.1,
    's1': 22.0,
    's2': 287.1,
    's3': 243.3,
    's4': 280.9,
    's5': 0.0,
    's6': 127.3,
    's7': 320.9,
    's8': 204.9,
}

# The invariable plane w.r.t. the ecliptic J2000, degrees (Souami & Souchay 2012, A&A 543, A133:
# i = 1.5787, Omega = 107.5822; the package's model gives 1.5788, 107.575). It is the constant
# term of every inclination vector measured from the ecliptic; secular angles measure the body's
# Omega from it, so that a low-inclination orbit still turns at its own s.
INVARIABLE_PLANE = {'inc': 1.5787, 'Omega': 107.5822}

J2000 = datetime.datetime(2000, 1, 1, 12)

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
# Days in the model's (Gaussian) year, the unit of every time of the package (G M_sun = 4 pi^2 au^3/yr^2, REBOUND time
# / 2 pi): 2 pi / k = 365.2569 d, not the Julian 365.25 d nor the tropical DAYS_IN_YEAR
GAUSSIAN_YEAR_DAYS = 2 * math.pi / K
DAYS_IN_YEAR = 365.2422

# Mean mean motions of the planets in the package's own model, deg per Gaussian year (2 pi REBOUND
# units = 365.2569 d): slopes of the turn-counted mean longitudes over 100 kyr of a SABA(10,6,4) run
# (benchmark bench2-A, 2026-10; a 20-kyr ias15 run at another epoch agrees within 0.0016 deg/yr).
# Kepler on PLANETS_AXIS errs by up to 0.014 deg/yr (Venus), enough to flip the alias branch near an
# exact stroboscopic commensurability (1999 UQ at a 5.0012-yr step). The MMR analysis uses these
# when a planet's longitudes are not saved; never 2 pi / P with DAYS_IN_YEAR.
PLANET_N_DEG_PER_GAUSS_YR = {
    "Mercury": 1494.756681,
    "Venus": 585.189239,
    "Earth": 360.000505,
    "Mars": 191.406585,
    "Jupiter": 30.349646,
    "Saturn": 12.221327,
    "Uranus": 4.284750,
    "Neptune": 2.184902,
}

# How far the catalogue line lambda_j0 + n_j t (n_j above) strays from a planet's integrated mean longitude. The MMR
# owner search uses that line when a planet's lambda is not saved (analysis/mmr/owner.py, catalogue longitudes) and
# bounds its phase error over a span T by min(SPREAD T, 2 SLOW_MAX + OFFSET T). Measured on the package model:
# ias15, 200 kyr, step 1 yr, JD 2461297.5 (benchmark run bench2-R-ias15-200kyr-1yr; script lam_linear.py of the
# classifier-v2 W7 check, 2026-10-16), rounded up:
#   SPREAD   max |slope of lambda_j over a 20-kyr window - n_j| [deg / Gaussian yr];
#   OFFSET   |slope over the 200 kyr - n_j| [deg / Gaussian yr] (n_j is SABA's; ias15 runs this far off on average);
#   SLOW_MAX max |lambda_j - its least-squares line| after a 100-yr running mean [deg].
# Mercury has no entry and is never catalogued: in ias15 its rate is 1.5e-3 deg/yr off the table (31 deg in 20 kyr).
PLANET_N_SPREAD_DEG_PER_GAUSS_YR = {
    "Venus": 2.7e-5,
    "Earth": 2.4e-5,
    "Mars": 3.7e-5,
    "Jupiter": 1.5e-4,
    "Saturn": 3.7e-4,
    "Uranus": 2.8e-5,
    "Neptune": 1.2e-5,
}
PLANET_N_OFFSET_DEG_PER_GAUSS_YR = {
    "Venus": 1.1e-5,
    "Earth": 1.1e-5,
    "Mars": 6.1e-6,
    "Jupiter": 4.1e-6,
    "Saturn": 8.5e-6,
    "Uranus": 7.8e-6,
    "Neptune": 3.6e-6,
}
PLANET_LAMBDA_SLOW_MAX_DEG = {
    "Venus": 0.60,
    "Earth": 0.37,
    "Mars": 0.72,
    "Jupiter": 1.91,
    "Saturn": 4.69,
    "Uranus": 1.29,
    "Neptune": 0.88,
}

# J2000 mean eccentricities (Standish), only for the orbit-crossing test when a planet is not saved.
PLANETS_ECCENTRICITY = {
    "Mercury": 0.2056,
    "Venus": 0.0068,
    "Earth": 0.0167,
    "Mars": 0.0934,
    "Jupiter": 0.0484,
    "Saturn": 0.0539,
    "Uranus": 0.0473,
    "Neptune": 0.0086,
}

# Long-period near-commensurabilities between the planets: the integer coefficients of the mean
# longitudes and, where the member matters, of the perihelia (rate g_j of the planet's own mode).
# Period = 2 pi / |sum k_j n_j + sum p_j g_j|; with PLANET_N_DEG_PER_GAUSS_YR: great inequality
# 938 yr, Venus-Earth 13:8 239 yr, Saturn-Uranus 1:3 569 yr, Jupiter-Uranus 1:7 1010 yr,
# Uranus-Neptune 1:2 4.2 kyr. Only the great inequality is subtracted from MMR angles (below).
PLANET_NEAR_COMMENSURABILITIES = {
    "great_inequality": {"mean_longitudes": {"Jupiter": 2, "Saturn": -5}, "perihelia": {"Saturn": 3}},
    "venus_earth_8_13": {"mean_longitudes": {"Venus": 8, "Earth": -13}, "perihelia": {}},
    "saturn_uranus_1_3": {"mean_longitudes": {"Saturn": 1, "Uranus": -3}, "perihelia": {}},
    "jupiter_uranus_1_7": {"mean_longitudes": {"Jupiter": 1, "Uranus": -7}, "perihelia": {}},
    "uranus_neptune_1_2": {"mean_longitudes": {"Uranus": 1, "Neptune": -2}, "perihelia": {}},
}

# The great inequality 2 lambda_J - 5 lambda_S + 3 varpi_S (Jupiter-Saturn 5:2): period and the
# half-amplitudes of the wave it makes in the planets' mean longitudes, measured on the package's
# planets (band-pass 300-3000 yr of their mean longitudes). A record's own value can differ: the
# other members of the 5:2 multiplet (923, 907, 892 yr) beat with it (20 kyr: 0.23 and 0.57 deg).
GREAT_INEQUALITY = {
    "period_yr": 937.0,
    "dlambda_jupiter_deg": 0.34,
    "dlambda_saturn_deg": 0.83,
}

SOLAR_SYSTEM = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
SOLAR_SYSTEM_WITH_SUN = ['Sun'] + SOLAR_SYSTEM + ['Pluto']

# The Moon as a separate body (option `solar_system_moon`). Earth is then the geocentre, not the Earth-Moon
# barycentre, and the Moon is appended after Pluto, so every planet keeps its particle index.
MOON = 'Moon'
SOLAR_SYSTEM_WITH_MOON = SOLAR_SYSTEM_WITH_SUN + [MOON]
# Horizons IDs used instead of the names in that system: 'Earth' alone resolves to body 3, the barycentre.
HORIZONS_IDS_WITH_MOON = {'Earth': '399', MOON: '301'}

# One entry per distinct angle (z1 'g-g6+s-s6' is also 'g+s-s6-g6'); for a complete per-body
# search beyond this list see `resonances.secular.combinations.near_resonant`.
SECULAR_FORMULAS = [
    # --- Linear secular resonances: nu_j = g-g_j, nu_1j = s-s_j for every planet (no s5 = 0) ---
    "g-g1",
    "g-g2",
    "g-g3",
    "g-g4",
    "g-g5",
    "g-g6",
    "g-g7",
    "g-g8",
    "s-s1",
    "s-s2",
    "s-s3",
    "s-s4",
    "s-s6",
    "s-s7",
    "s-s8",
    # --- Kozai-type ---
    "2g-2s",
    # --- Quadratic / degree-4 nonlinear (classical set) ---
    "g+s-s7-g5",
    "g+s-s7-g6",
    "g+s-s6-g5",
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
    "g+g5-g6-g7",
    "g-g5-g6+g7",
    "g+g5-2g6-s6+s7",
]
