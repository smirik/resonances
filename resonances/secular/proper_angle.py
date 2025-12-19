import numpy as np
from numpy import i0 as bessi0  # modified Bessel I0
from scipy.signal import firwin, butter, filtfilt
from resonances.data.const import PLANETARY_FREQUENCIES


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


def _design_lowpass_filter(
    kind: str,
    dt_years: float,
    cutoff_period_years: float,
    M: int = 80,
    beta: float = 10.0,
    butter_order: int = 4,
):
    """
    Design a low-pass filter used to smooth non-singular elements.

    Parameters
    ----------
    kind : {'quinn', 'firwin', 'butter'}
        Filter family:
        - 'quinn'  : original Quinn FIR (as in SWIFT).
        - 'firwin' : FIR via scipy.signal.firwin (Kaiser window).
        - 'butter' : Butterworth IIR + filtfilt (zero-phase).
    dt_years : float
        Mean time step in years.
    cutoff_period_years : float
        Cutoff period T_c (years). Frequencies with periods << T_c are suppressed.
    M : int
        Half-length of FIR kernel (2*M+1 taps) for FIR filters.
    beta : float
        Kaiser window beta for FIR filters.
    butter_order : int
        Order of the Butterworth filter.

    Returns
    -------
    filt : ndarray or (b, a)
        If kind_out == 'fir': filt is FIR kernel (impulse response).
        If kind_out == 'iir': filt is (b, a) IIR coefficients.
    kind_out : {'fir', 'iir'}
        Filter implementation type.
    """
    if cutoff_period_years <= 0:
        # Treat non-positive cutoff as "no filtering" (identity FIR).
        return np.array([1.0], dtype=float), 'fir'

    # normalized cutoff frequency: x0 = f_c / f_s = dt / T_c
    x0 = dt_years / cutoff_period_years

    if kind == 'quinn':
        coeffs = quinn_fir_coeffs(M=M, x0=x0, beta=beta)
        return coeffs, 'fir'

    elif kind == 'firwin':
        numtaps = 2 * M + 1
        # firwin cutoff normalized to Nyquist (0..1), where 1 = f_N = f_s/2.
        # Our x0 = f_c / f_s -> cutoff_firwin = f_c / f_N = 2 * x0.
        cutoff = 2.0 * x0
        if not (0.0 < cutoff < 1.0):
            raise ValueError(
                f"Invalid normalized cutoff for firwin: 2*x0 = {cutoff:.4g} " "(must be in (0,1)). Check dt_years / cutoff_period_years."
            )

        coeffs = firwin(
            numtaps=numtaps,
            cutoff=cutoff,
            window=("kaiser", beta),
            pass_zero="lowpass",
            scale=True,  # unity gain at DC (approximately)
        )
        # Ensure exact unity gain at DC (sum of taps = 1)
        coeffs /= coeffs.sum()
        return coeffs, 'fir'

    elif kind == 'butter':
        # Butterworth also uses normalized cutoff to Nyquist (0..1)
        Wn = 2.0 * x0
        if not (0.0 < Wn < 1.0):
            raise ValueError(
                f"Invalid normalized cutoff for butter: Wn = {Wn:.4g} " "(must be in (0,1)). Check dt_years / cutoff_period_years."
            )
        b, a = butter(butter_order, Wn, btype="lowpass", analog=False)
        return (b, a), 'iir'

    else:
        raise ValueError(f"Unknown filter kind '{kind}'. Use 'quinn', 'firwin' or 'butter'.")


def apply_filter(signal, filt, fkind: str, skip=True):
    """
    Apply a pre-designed low-pass filter to a 1D signal.

    Parameters
    ----------
    signal : array-like
        Input time series.
    filt : ndarray or (b, a)
        - If fkind == 'fir': FIR kernel coefficients (impulse response).
        - If fkind == 'iir': (b, a) coefficients of an IIR filter.
    fkind : {'fir', 'iir'}
        Filter implementation type ('fir' or 'iir').

    Returns
    -------
    filtered : ndarray
        Filtered time series (same length as signal).
    """
    signal = np.asarray(signal)
    if skip:
        return signal

    if fkind == 'fir':
        coeffs = np.asarray(filt)
        return np.convolve(signal, coeffs, mode='same')

    if fkind == 'iir':
        b, a = filt
        # filtfilt -> zero-phase, same length, symmetric extension at boundaries
        return filtfilt(b, a, signal)

    raise ValueError(f"Unknown filter implementation kind '{fkind}'")


# def apply_fir(signal, coeffs):
#     """
#     Apply symmetric FIR filter via convolution, keeping array length.
#     """
#     return np.convolve(signal, coeffs, mode='same')


def build_planetary_longitudes(times_years, planets_names, planetary_freqs):
    """
    Build simple linear secular longitudes for planets using configured frequencies.
    Phases are set so that varpi/Omega are zero at t=0; a later shift aligns with existing_angle.
    """
    varpi_planets = []
    Omega_planets = []
    for planet in planets_names:
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


def _frequency_label_to_rad_per_year(label: str, freq_map: dict) -> float:
    if label not in freq_map:
        raise ValueError(f"Missing planetary frequency {label} in configuration")
    # freq_map values are in arcsec/yr -> convert to rad/yr
    return np.deg2rad(freq_map[label] / 3600.0)


def build_planetary_longitudes_by_index(times_years, indices, planetary_freqs):
    """
    Build simple linear secular longitudes for planets using configured frequencies.

    Parameters
    ----------
    times_years : array-like
        Time array in years.
    indices : iterable[int]
        Secular indices (e.g. 5, 6, 7, 8).
    planetary_freqs : dict
        Map of planetary frequencies (g_i, s_i) in arcsec/yr.

    Returns
    -------
    varpi_planets : dict[int, ndarray]
    Omega_planets : dict[int, ndarray]
    """
    varpi_planets: dict[int, np.ndarray] = {}
    Omega_planets: dict[int, np.ndarray] = {}

    for idx in indices:
        g = _frequency_label_to_rad_per_year(f"g{idx}", planetary_freqs)
        s = _frequency_label_to_rad_per_year(f"s{idx}", planetary_freqs)
        varpi_planets[idx] = (g * times_years) % (2.0 * np.pi)
        Omega_planets[idx] = (s * times_years) % (2.0 * np.pi)

    return varpi_planets, Omega_planets


def _build_angle_series_from_terms(*, times_years, varpi_mean, Omega_mean, terms, planetary_freqs) -> np.ndarray:
    indices = sorted({t.index for t in terms if t.index is not None})
    varpi_planets, Omega_planets = build_planetary_longitudes_by_index(times_years, indices, planetary_freqs)

    angle = np.zeros_like(times_years, dtype=float)
    for term in terms:
        if term.mode == "g":
            series = varpi_mean if term.index is None else varpi_planets[term.index]
        elif term.mode == "s":
            series = Omega_mean if term.index is None else Omega_planets[term.index]
        else:
            raise ValueError(f"Unsupported secular term mode: {term.mode!r}")
        angle += term.coefficient * series

    return angle


def _build_angle_series_legacy(*, times_years, varpi_mean, Omega_mean, resonance, planetary_freqs) -> np.ndarray:
    varpi_planets, Omega_planets = build_planetary_longitudes(
        times_years,
        resonance.planets_names,
        planetary_freqs,
    )

    angle = np.zeros_like(times_years, dtype=float)

    # Build linear combination of varpi's
    if "varpi" in resonance.coeffs:
        angle += resonance.coeffs["varpi"][0] * varpi_mean
        for coeff, planet_long in zip(resonance.coeffs["varpi"][1:], varpi_planets):
            angle += coeff * planet_long

    # Build linear combination of Omega's
    if "Omega" in resonance.coeffs:
        angle += resonance.coeffs["Omega"][0] * Omega_mean
        for coeff, planet_long in zip(resonance.coeffs["Omega"][1:], Omega_planets):
            angle += coeff * planet_long

    return angle


def build_proper_angle_series(
    times,
    body,
    resonance,
    existing_angle=None,
    cutoff_period_years=50_000.0,
    planetary_freqs=None,
    filter_kind: str = 'quinn',
    M: int = 80,
    beta: float = 10.0,
    butter_order: int = 4,
):
    """
    Convenience wrapper: use body.* attributes.
    """
    return calc_proper_angle_series(
        times=times,
        omega=body.omega,
        Omega=body.Omega,
        ecc=body.ecc,
        inc=body.inc,
        resonance=resonance,
        existing_angle=existing_angle,
        cutoff_period_years=cutoff_period_years,
        planetary_freqs=planetary_freqs,
        filter_kind=filter_kind,
        M=M,
        beta=beta,
        butter_order=butter_order,
    )


def calc_proper_angle_series(
    times,
    omega,
    Omega,
    ecc,
    inc,
    resonance,
    existing_angle=None,
    cutoff_period_years=50_000.0,
    planetary_freqs=None,
    filter_kind: str = 'quinn',
    M: int = 80,
    beta: float = 10.0,
    butter_order: int = 4,
):
    """
    Construct a filtered secular critical angle time series for a given resonance.

    The method:
    - converts osculating elements to non-singular (k, h, q, p),
    - applies a chosen low-pass filter (Quinn FIR / firwin FIR / Butterworth IIR),
      with cutoff defined by `cutoff_period_years`,
    - reconstructs secular (mean) longitudes varpi_mean, Omega_mean,
    - builds the critical angle using resonance.coeffs and planetary secular frequencies,
    - optionally aligns phase with an existing angle time series.

    Parameters
    ----------
    times : array-like
        Times in "radians" such that 2π corresponds to 1 year.
    omega, Omega, ecc, inc : array-like
        Osculating elements at given times (angles in radians).
    resonance : object
        Either:
        - a modern secular resonance with `resonance.formula.terms` (from `SecularResonanceFormula`), or
        - a legacy secular resonance with:
          - `planets_names`: list of planet names (for forced terms),
          - `coeffs`: dict with keys 'varpi' and/or 'Omega', each a list of coefficients.
    existing_angle : array-like or None
        If provided, phase of the resulting series is shifted so that angle[0] matches existing_angle[0].
    cutoff_period_years : float
        Cutoff period T_c: variations with periods << T_c are filtered out.
    planetary_freqs : dict or None
        Map of planetary frequencies (g_i, s_i) in arcsec/yr.
        If None, loaded via config.
    filter_kind : {'quinn', 'firwin', 'butter'}
        Which low-pass filter family to use.
    M, beta, butter_order : see _design_lowpass_filter.

    Returns
    -------
    angle : ndarray
        Time series of the secular critical angle in [0, 2π).
    """
    if planetary_freqs is None:
        planetary_freqs = PLANETARY_FREQUENCIES

    times = np.asarray(times)
    times_years = times / (2.0 * np.pi)

    varpi = Omega + omega
    k, h, q, p = non_singular_elements(ecc, inc, Omega, varpi)

    dt_years = np.mean(np.diff(times_years))

    # Design and apply low-pass filter to (k, h, q, p)
    filt, fkind = _design_lowpass_filter(
        kind=filter_kind,
        dt_years=dt_years,
        cutoff_period_years=cutoff_period_years,
        M=M,
        beta=beta,
        butter_order=butter_order,
    )

    k_f = apply_filter(k, filt, fkind, skip=True)
    h_f = apply_filter(h, filt, fkind, skip=True)
    q_f = apply_filter(q, filt, fkind, skip=True)
    p_f = apply_filter(p, filt, fkind, skip=True)

    # Reconstruct secular longitudes of the body
    varpi_mean = np.mod(np.arctan2(h_f, k_f), 2.0 * np.pi)
    Omega_mean = np.mod(np.arctan2(p_f, q_f), 2.0 * np.pi)

    # New secular resonance model: parse from a SecularResonanceFormula-like object.
    if hasattr(resonance, "formula") and hasattr(resonance.formula, "terms"):
        angle = _build_angle_series_from_terms(
            times_years=times_years,
            varpi_mean=varpi_mean,
            Omega_mean=Omega_mean,
            terms=resonance.formula.terms,
            planetary_freqs=planetary_freqs,
        )

    # Backward-compatible model: resonance.coeffs / resonance.planets_names (legacy API).
    else:
        angle = _build_angle_series_legacy(
            times_years=times_years,
            varpi_mean=varpi_mean,
            Omega_mean=Omega_mean,
            resonance=resonance,
            planetary_freqs=planetary_freqs,
        )

    angle = np.mod(angle, 2.0 * np.pi)

    # Optional phase alignment with an existing time series
    if existing_angle is not None and len(existing_angle) > 0:
        offset = (existing_angle[0] - angle[0]) % (2.0 * np.pi)
        angle = (angle + offset) % (2.0 * np.pi)

    return angle
