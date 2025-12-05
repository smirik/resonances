import numpy as np
from numpy import i0 as bessi0  # modified Bessel I0

from resonances.matrix.secular_resonances import load_planetary_frequencies


def non_singular_elements(eccentricity, inclination, Omega, varpi):
    """
    Convert orbital elements to non-singular variables (k, h, q, p).
    """
    k = eccentricity * np.cos(varpi)
    h = eccentricity * np.sin(varpi)
    q = np.sin(inclination / 2.0) * np.cos(Omega)
    p = np.sin(inclination / 2.0) * np.sin(Omega)
    return k, h, q, p


def quinn_fir_coeffs(M=80, x0=0.024, beta=10.0):
    """
    Reproduce the FIR filter from Quinn et al. (1991) / SWIFT (Broz).
    """
    d = np.zeros(2 * M + 1, dtype=float)
    d[M] = 2.0 * x0 * bessi0(beta)
    for m in range(1, M + 1):
        coeff = np.sin(2.0 * np.pi * m * x0) / (np.pi * m)
        coeff *= bessi0(beta * np.sqrt(1.0 - (m * m) / (M * M)))
        d[M + m] = coeff
        d[M - m] = coeff
    d /= d.sum()
    return d


def apply_fir(signal, coeffs):
    """
    Apply symmetric FIR filter via convolution, keeping array length.
    """
    return np.convolve(signal, coeffs, mode='same')


def build_planetary_longitudes(times_years, planet_names, planetary_freqs):
    """
    Build simple linear secular longitudes for planets using configured frequencies.
    Phases are set so that varpi/Omega are zero at t=0; a later shift aligns with existing_angle.
    """
    varpi_planets = []
    Omega_planets = []
    for planet in planet_names:
        g = _planetary_frequency(planet, 'varpi', planetary_freqs)
        s = _planetary_frequency(planet, 'Omega', planetary_freqs)
        varpi_planets.append((g * times_years) % (2.0 * np.pi))
        Omega_planets.append((s * times_years) % (2.0 * np.pi))
    return varpi_planets, Omega_planets


def _planetary_frequency(planet_name, kind, freq_map):
    """
    Map planet name and longitude type to configured frequencies.
    """
    mapping = {
        'varpi': {'Mars': 'g4', 'Jupiter': 'g5', 'Saturn': 'g6', 'Uranus': 'g7', 'Neptune': 'g8'},
        'Omega': {'Mars': 's4', 'Jupiter': 's5', 'Saturn': 's6', 'Uranus': 's7', 'Neptune': 's8'},
    }
    if kind not in mapping or planet_name not in mapping[kind]:
        raise ValueError(f"Unsupported planet for secular reconstruction: {planet_name}")
    label = mapping[kind][planet_name]
    if label not in freq_map:
        raise ValueError(f"Missing planetary frequency {label} in configuration")
    # freq_map values are in arcsec/yr -> convert to rad/yr
    return np.deg2rad(freq_map[label] / 3600.0)


def _build_secular_longitudes(times_years, freq, phase):
    return (freq * times_years + phase) % (2.0 * np.pi)


def build_proper_angle_series(times, body, resonance, planetary_freqs=None, existing_angle=None, cutoff_period_years=50_000.0):
    """
    Construct a filtered secular critical angle time series for a given resonance.
    Keeps multi-Myr structure by low-pass filtering non-singular elements
    using the Quinn FIR filter (as in SWIFT).
    """
    if planetary_freqs is None:
        planetary_freqs = load_planetary_frequencies()

    times = np.asarray(times)
    times_years = times / (2.0 * np.pi)

    varpi = body.Omega + body.omega
    k, h, q, p = non_singular_elements(body.ecc, body.inc, body.Omega, varpi)

    dt_years = np.mean(np.diff(times_years))
    x0 = dt_years / cutoff_period_years  # normalized cutoff freq = dt / T_c
    coeffs = quinn_fir_coeffs(x0=x0)
    k_f = apply_fir(k, coeffs)
    h_f = apply_fir(h, coeffs)
    q_f = apply_fir(q, coeffs)
    p_f = apply_fir(p, coeffs)

    varpi_mean = np.mod(np.arctan2(h_f, k_f), 2.0 * np.pi)
    Omega_mean = np.mod(np.arctan2(p_f, q_f), 2.0 * np.pi)

    varpi_planets, Omega_planets = build_planetary_longitudes(times_years, resonance.planet_names, planetary_freqs)

    angle = np.zeros_like(times, dtype=float)

    if 'varpi' in resonance.coeffs:
        angle += resonance.coeffs['varpi'][0] * varpi_mean
        for coeff, planet_long in zip(resonance.coeffs['varpi'][1:], varpi_planets):
            angle += coeff * planet_long

    if 'Omega' in resonance.coeffs:
        angle += resonance.coeffs['Omega'][0] * Omega_mean
        for coeff, planet_long in zip(resonance.coeffs['Omega'][1:], Omega_planets):
            angle += coeff * planet_long

    angle = np.mod(angle, 2.0 * np.pi)

    if existing_angle is not None and len(existing_angle) > 0:
        offset = (existing_angle[0] - angle[0]) % (2.0 * np.pi)
        angle = (angle + offset) % (2.0 * np.pi)

    return angle
