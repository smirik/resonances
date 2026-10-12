"""The proper secular angle: the body's varpi and Omega against the planets' linear mode longitudes.

See docs/secular.md. The angle is built from the osculating elements, optionally reduced: the forced
terms at the fundamentals that are not in the angle are subtracted from the body's vectors first.
"""

import numpy as np

from resonances.data.const import PLANETARY_FREQUENCIES, PLANETARY_PHASES


def non_singular_elements(eccentricity, inclination, Omega, varpi):
    """
    Convert orbital elements to non-singular variables (k, h, q, p).
    """
    k = eccentricity * np.cos(varpi)
    h = eccentricity * np.sin(varpi)
    q = np.sin(inclination / 2.0) * np.cos(Omega)
    p = np.sin(inclination / 2.0) * np.sin(Omega)
    return k, h, q, p


def _frequency_label_to_rad_per_year(label: str, freq_map: dict) -> float:
    if label not in freq_map:
        raise ValueError(f"Missing planetary frequency {label} in configuration")
    # freq_map values are in arcsec/yr -> convert to rad/yr
    return np.deg2rad(freq_map[label] / 3600.0)


def build_planetary_longitudes_by_index(times_years, indices, planetary_freqs, planetary_phases=None, epoch_years=0.0):
    """
    Build linear secular longitudes of the planetary modes: g_j * t + phase_j.

    Parameters
    ----------
    times_years : array-like
        Time array in years, 0 at the start of the integration.
    indices : iterable[int]
        Secular indices (1..8 for Mercury..Neptune).
    planetary_freqs : dict
        Map of planetary frequencies (g_i, s_i) in arcsec/yr.
    planetary_phases : dict or None
        Phases of the modes at J2000, degrees. None puts every phase at zero at t = 0.
    epoch_years : float
        Years from J2000 to t = 0; only used with `planetary_phases`.

    Returns
    -------
    varpi_planets : dict[int, ndarray]
    Omega_planets : dict[int, ndarray]
    """
    varpi_planets: dict[int, np.ndarray] = {}
    Omega_planets: dict[int, np.ndarray] = {}
    t = np.asarray(times_years) + (epoch_years if planetary_phases is not None else 0.0)

    for idx in indices:
        g = _frequency_label_to_rad_per_year(f"g{idx}", planetary_freqs)
        s = _frequency_label_to_rad_per_year(f"s{idx}", planetary_freqs)
        phase_g = np.deg2rad(planetary_phases[f"g{idx}"]) if planetary_phases is not None else 0.0
        phase_s = np.deg2rad(planetary_phases[f"s{idx}"]) if planetary_phases is not None else 0.0
        varpi_planets[idx] = (g * t + phase_g) % (2.0 * np.pi)
        Omega_planets[idx] = (s * t + phase_s) % (2.0 * np.pi)

    return varpi_planets, Omega_planets


def removed_modes(terms, baseline_years: float):
    """The planetary modes to subtract from the body's vectors for this angle: the free-element basis of
    this baseline (`free_elements.choose_basis`) without the modes that occur in the angle.

    The forced terms at the angle's own frequencies (g6 and s6 for z1, g6 for nu6) are part of the
    resonance and stay; those at the other fundamentals only shake the angle. 'const' — the invariable
    plane, s5 = 0 — is never in an angle, so it always goes. Returns ({label: arcsec/yr} for e, for i).
    """
    from resonances.secular import free_elements as fe

    present = {f'{t.mode}{t.index}' for t in terms if t.index is not None}
    basis_e, _, _ = fe.choose_basis(fe.candidate_frequencies('e'), baseline_years)
    basis_i, _, _ = fe.choose_basis(fe.candidate_frequencies('i'), baseline_years)
    return {k: v for k, v in basis_e.items() if k not in present}, {k: v for k, v in basis_i.items() if k not in present}


def reduced_vectors(times_years, ecc, varpi, inc, Omega, terms, vectors=('e', 'i')):
    """e exp(i varpi) and sin(i/2) exp(i Omega) without the forced terms of `removed_modes`, fitted by
    least squares at their known frequencies (`free_elements.remove_forced`). Only the `vectors`
    asked for are fitted; the other comes back as None."""
    from resonances.secular.free_elements import remove_forced

    times_years = np.asarray(times_years, dtype=float)
    modes_e, modes_i = removed_modes(terms, times_years[-1] - times_years[0])
    z_e = remove_forced(np.asarray(ecc) * np.exp(1j * np.asarray(varpi)), times_years, modes_e)[0] if 'e' in vectors else None
    z_i = remove_forced(np.sin(np.asarray(inc) / 2.0) * np.exp(1j * np.asarray(Omega)), times_years, modes_i)[0] if 'i' in vectors else None
    return z_e, z_i


def invariable_plane_vector(plane=None) -> complex:
    """sin(i/2) exp(i Omega) of the invariable plane w.r.t. the elements' reference plane (`INVARIABLE_PLANE`)."""
    if plane is None:
        from resonances.data.const import INVARIABLE_PLANE as plane
    return np.sin(np.deg2rad(plane['inc']) / 2.0) * np.exp(1j * np.deg2rad(plane['Omega']))


def inclination_from_vector(z_i):
    """Inclination (rad) of an inclination vector sin(i/2) exp(i Omega)."""
    return 2.0 * np.arcsin(np.clip(np.abs(z_i), 0.0, 1.0))


def inclination_from_invariable_plane(inc, Omega, plane=None):
    """Inclination (rad) measured from the invariable plane, the plane a secular angle's Omega is taken from."""
    return inclination_from_vector(np.sin(np.asarray(inc) / 2.0) * np.exp(1j * np.asarray(Omega)) - invariable_plane_vector(plane))


def _build_angle_series_from_terms(
    *, times_years, varpi_mean, Omega_mean, terms, planetary_freqs, planetary_phases=None, epoch_years=0.0
) -> np.ndarray:
    indices = sorted({t.index for t in terms if t.index is not None})
    varpi_planets, Omega_planets = build_planetary_longitudes_by_index(
        times_years, indices, planetary_freqs, planetary_phases=planetary_phases, epoch_years=epoch_years
    )

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


def build_proper_angle_series(
    times,
    body,
    resonance,
    planetary_freqs=None,
    planetary_phases=None,
    epoch_years: float = 0.0,
    invariable_plane=None,
    remove_forced: bool = False,
):
    """`calc_proper_angle_series` with the elements of `body`."""
    return calc_proper_angle_series(
        times=times,
        omega=body.omega,
        Omega=body.Omega,
        ecc=body.ecc,
        inc=body.inc,
        resonance=resonance,
        planetary_freqs=planetary_freqs,
        planetary_phases=planetary_phases,
        epoch_years=epoch_years,
        invariable_plane=invariable_plane,
        remove_forced=remove_forced,
    )


def calc_proper_angle_series(
    times,
    omega,
    Omega,
    ecc,
    inc,
    resonance,
    planetary_freqs=None,
    planetary_phases=None,
    epoch_years: float = 0.0,
    invariable_plane=None,
    remove_forced: bool = False,
):
    """
    The secular critical angle of `resonance`, in [0, 2pi).

    The body's varpi and Omega come from its vectors (k, h) and (q, p): the osculating ones with the
    invariable plane removed from (q, p) when `invariable_plane` is given, or, with `remove_forced`,
    the reduced ones (`reduced_vectors`; the invariable plane is one of the removed terms). The
    planets enter through their mode longitudes g_j*t + phase_j.

    Parameters
    ----------
    times : array-like
        Times in "radians" such that 2π corresponds to 1 year.
    omega, Omega, ecc, inc : array-like
        Osculating elements at given times (angles in radians).
    resonance : object
        A secular resonance with `resonance.formula.terms` (from `SecularResonanceFormula`).
    planetary_freqs, planetary_phases : dict or None
        Frequencies (arcsec/yr) and phases at J2000 (degrees) of the modes; `PLANETARY_FREQUENCIES`
        and `PLANETARY_PHASES` when None.
    epoch_years : float
        Years from J2000 to times = 0.
    invariable_plane : dict or None
        {'inc': deg, 'Omega': deg} of the invariable plane w.r.t. the reference plane of the
        elements (`INVARIABLE_PLANE` for the ecliptic J2000): measured from the ecliptic, a
        low-inclination orbit's Omega librates around the invariable node instead of turning at s.
    remove_forced : bool
        Build the angle from `reduced_vectors` (the "reduced" angle).
    """
    planetary_freqs = PLANETARY_FREQUENCIES if planetary_freqs is None else planetary_freqs
    planetary_phases = PLANETARY_PHASES if planetary_phases is None else planetary_phases
    times_years = np.asarray(times) / (2.0 * np.pi)

    varpi = Omega + omega
    k, h, q, p = non_singular_elements(ecc, inc, Omega, varpi)
    if remove_forced:
        z_e, z_i = reduced_vectors(times_years, ecc, varpi, inc, Omega, resonance.formula.terms)
        k, h, q, p = z_e.real, z_e.imag, z_i.real, z_i.imag
    elif invariable_plane is not None:
        plane = invariable_plane_vector(invariable_plane)
        q, p = q - plane.real, p - plane.imag

    angle = _build_angle_series_from_terms(
        times_years=times_years,
        varpi_mean=np.arctan2(h, k),
        Omega_mean=np.arctan2(p, q),
        terms=resonance.formula.terms,
        planetary_freqs=planetary_freqs,
        planetary_phases=planetary_phases,
        epoch_years=epoch_years,
    )
    return np.mod(angle, 2.0 * np.pi)
