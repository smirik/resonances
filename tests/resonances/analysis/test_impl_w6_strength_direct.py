"""Two-body strength by direct averaging (analysis.mmr.averaged_potential, analysis.mmr.strength) and the free/forced
split on the cutoff series (analysis.mmr.kinematics.short_period_vector).

Independent truths:
- 1:1 tadpole, e = i = 0: <R> = mu / a (1 / (2 |sin(theta/2)|) - cos theta) (M&D 1999 eq. 3.179), <R>''(60 deg) = 9/4 by
  hand (omega^2 = 27/4 mu n^2, M&D §3.9);
- 2:1 interior, circular coplanar planet: the first harmonic of <R> is f27(alpha) e / 2 with f27 = (1/2)(-4 - alpha D)
  b_1/2^(2)(alpha), the Laplace coefficient integrated here with scipy (M&D Table 8.5: f27 = -1.19049);
- the planar first harmonic of `multiplet._first_harmonic` (a separate 2-D Fourier code) for an eccentric planet;
- the W6 skeptic's curvature numbers (verify-c-confidence/direct_curvature.txt, Codex 1.0015) from median elements and a
  circular ecliptic planet: required/theory 15875 1.00, 153 1.13, 149349 1.00, 119979 0.906 (Codex; centre 75 deg, an
  asymmetric well: the skeptic's +-8 deg quadratic gave 0.93, the +-2 deg fit is nearer to the second derivative), 500878 0.74;
- 2005 GQ107 40M-57 at the snapshot elements, brute force with the skeptic's own Kepler code: 2.78, 0.945, 1.11, the
  theoretical centre 95, 174.5, 153 deg following the measured 98.7, 172.6, 150 deg;
- 37087 3J+8S-3: e_forced 0.0286, g 231.28"/yr from the unsmoothed vector (verify-c-episodes, skeptic).
"""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

from resonances.resonance.analysis.mmr import averaged_potential as ap
from resonances.resonance.analysis.mmr import io, kinematics as kin, multiplet
from resonances.resonance.analysis.mmr import strength as st
from resonances.resonance.analysis.mmr.coorbital import Orbit
from resonances.resonance.analysis.mmr.models import MMRSeries, ResonanceKind, ResonanceSpec, Regime
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue

SETTINGS = settings_from_config()
RULE = st.quadrature(SETTINGS)
ARCSEC = np.pi / 180 / 3600
CIRCLE = Orbit(0.0, 0.0, 0.0, 0.0)


def _laplace(j: int, alpha: float) -> float:
    """b_1/2^(j)(alpha) = (1/pi) int_0^2pi cos(j psi) / sqrt(1 - 2 alpha cos psi + alpha^2) d psi."""
    value, _ = integrate.quad(lambda x: np.cos(j * x) / np.sqrt(1 - 2 * alpha * np.cos(x) + alpha**2), 0, 2 * np.pi, epsabs=1e-13)
    return value / np.pi


def test_tadpole_curvature_is_nine_quarters():
    body = ap.Ellipse(1.0, CIRCLE)
    result = ap.centre_curvature(body, ap.Ellipse(1.0, CIRCLE), 1, -1, 0.0, np.radians(60.0), RULE)
    assert result.measured
    assert result.curvature == pytest.approx(9 / 4, rel=2e-3)
    assert np.degrees(result.centre_rad) == pytest.approx(60.0, abs=0.3)
    # the other tadpole: the well nearest to the measured centre is the one taken
    other = ap.centre_curvature(body, ap.Ellipse(1.0, CIRCLE), 1, -1, 0.0, np.radians(290.0), RULE)
    assert np.degrees(other.centre_rad) == pytest.approx(300.0, abs=0.3)


def test_two_to_one_first_harmonic_is_the_laplace_coefficient():
    alpha = 0.5 ** (2 / 3)
    h = 1e-5
    derivative = (_laplace(2, alpha + h) - _laplace(2, alpha - h)) / (2 * h)
    f27 = 0.5 * (-4 * _laplace(2, alpha) - alpha * derivative)
    assert f27 == pytest.approx(-1.19049, abs=1e-5)  # M&D Table 8.5
    for e in (0.001, 0.01):
        body = ap.Ellipse(alpha, Orbit(e, 0.0, 0.0, 0.0))
        sigma, profile = ap.averaged_profile(body, ap.Ellipse(1.0, CIRCLE), -1, 2, 0.0, 720)  # sigma = 2 lambda_P - lambda - varpi
        first = np.mean(profile * np.exp(-1j * sigma))
        assert first.real == pytest.approx(f27 * e / 2, rel=2e-4 + 2 * e)  # + e^3 terms of the same harmonic
        # f27 < 0: the stable centre is sigma = 0, the curvature -> |f27| e as e -> 0 (the e^2 cos 2 sigma term is O(e))
        result = ap.centre_curvature(body, ap.Ellipse(1.0, CIRCLE), -1, 2, 0.0, 0.3, RULE)
        assert result.centre_rad == pytest.approx(0.0, abs=1e-6) or result.centre_rad == pytest.approx(2 * np.pi, abs=1e-6)
        assert result.curvature == pytest.approx(abs(f27) * e, rel=6 * e)


def test_planar_first_harmonic_matches_the_multiplet_code_with_an_eccentric_planet():
    # 3:1 with an eccentric planet, beta = varpi_P - varpi = 1 rad: sigma_0 = 3 lambda_P - lambda (+ 2 varpi, varpi = 0)
    a, e, planet_e, beta = 0.48, 0.15, 0.05, 1.0
    body = ap.Ellipse(a, Orbit(e, 0.0, 0.0, 0.0))
    planet = ap.Ellipse(1.0, Orbit(planet_e, 0.0, beta, 0.0))
    sigma, profile = ap.averaged_profile(body, planet, -1, 3, 0.0, 720)
    mine = np.mean(profile * np.exp(-1j * sigma))
    theirs = multiplet._first_harmonic(3, -1, a, 1.0, e, planet_e, 1.0, np.array([beta]))[0]
    assert mine == pytest.approx(theirs, rel=1e-9)


def test_quadrature_not_converged_on_a_crossing_orbit_with_too_few_nodes():
    # an exterior 3:2 whose perihelion (0.975) is inside the planet's orbit: the conjunction peak needs many nodes
    body = ap.Ellipse(1.3, Orbit(0.25, 0.0, 0.0, 0.0))
    coarse = ap.Quadrature(16, 16, 8, np.radians(10.0), 1e-3)
    assert ap.centre_curvature(body, ap.Ellipse(1.0, CIRCLE), 3, -2, 0.0, np.pi, coarse).reason in (
        ap.REASON_NOT_CONVERGED,
        ap.REASON_NO_CENTRE,
    )
    fine = ap.centre_curvature(body, ap.Ellipse(1.0, CIRCLE), 3, -2, 0.0, np.pi, RULE)
    assert fine.measured and fine.min_distance_au > 0


def test_range_rules():
    assert st.nearest_to_one(0.2, 8.0) == 1.0 and st.nearest_to_one(12.0, 40.0) == 12.0 and st.nearest_to_one(0.01, 0.05) == 0.05
    limit = SETTINGS.mmr_strength_check_ratio
    assert st.strength_check(True, st.nearest_to_one(5.0, 26.0), SETTINGS) == CheckValue.PASS  # overlaps [0.1, 10]
    assert st.strength_check(True, st.nearest_to_one(limit * 1.1, 1e5), SETTINGS) == CheckValue.FAIL
    assert st.strength_check(False, 1.0, SETTINGS) == CheckValue.NOT_MEASURED
    # direct averaging: implausible both ways
    assert st.implausible_strength(True, 1e-3, SETTINGS) and st.implausible_strength(True, 1e3, SETTINGS)
    direct = st.Strength(1, 1, True, 1, 1, 1, 1, False, CheckValue.PASS, method=st.METHOD_DIRECT)
    scale = st.Strength(1, 1, True, 1, 1, 1, 1, False, CheckValue.PASS, method=st.METHOD_SCALE)
    assert st.physics_margins(('L4', 'F'), direct) == ('L4', 'F')
    assert st.physics_margins(('L4', 'F'), scale) == ('L4',) and st.physics_margins(('L4', 'F'), None) == ('L4',)


def test_direct_applicable():
    assert st.direct_applicable(ResonanceSpec.from_key('3J-2+0-1'))
    assert st.direct_applicable(ResonanceSpec.from_key('1J-3r0'))
    assert not st.direct_applicable(ResonanceSpec.from_key('1E-1+0+0'))  # the co-orbital law
    assert not st.direct_applicable(ResonanceSpec.from_key('1J-1r1'))
    assert not st.direct_applicable(ResonanceSpec.from_key('4J-2S-1'))  # three-body: the e^q scale
    assert not st.direct_applicable(None)


# --- real records (no integration; skipped when the git-ignored benchmark data is absent) ---------------------------

# required mu_eff of benchmark v2 (mmr_numbers.csv str_mu_eff_req) and the skeptic's required/theory
SKEPTIC = {
    'mmr2__15875__2N-3+0+1': (4.9055985624883405e-06, 1.00, 180.0),
    'mmr2__153__3J-2+0-1': (7.329828896702384e-05, 1.13, 0.0),
    'mmr2__149349__3N-5+0+2': (3.3796003185488747e-06, 1.00, 180.0),
    'mmr2__119979__1N-2+0+1': (2.1741668772904193e-05, 0.906, 75.0),  # Codex; the skeptic's +-8 deg fit gave 0.93
    'mmr2__500878__11N-18+0+7': (6.53056534809887e-08, 0.74, 213.0),
}
SEMI_AXES = {'J': 5.2026, 'N': 30.1104}
MASSES = {'J': 9.5459e-4, 'N': 5.1503e-5}


def _needs(example: str):
    if not (io.example_path(example) / 'series.csv.gz').exists():
        pytest.skip('benchmark data not available')


@pytest.mark.parametrize('example', sorted(SKEPTIC))
def test_curvature_reproduces_the_skeptic_numbers(example):
    _needs(example)
    required, expected, centre = SKEPTIC[example]
    spec = ResonanceSpec.from_key(example.split('__')[2])
    letter = example.split('__')[2].lstrip('0123456789')[0]
    frame = pd.read_csv(io.example_path(example) / 'series.csv.gz', usecols=['a', 'e', 'inc', 'Omega', 'omega'])
    a, e, inc = frame.a.median(), frame.e.median(), frame.inc.median()
    node, omega = np.median(np.unwrap(frame.Omega)), np.median(np.unwrap(frame.omega))
    phase = (spec.body_coefficient + spec.body_varpi_coefficient) * (node + omega)
    planet = ap.Ellipse(SEMI_AXES[letter], CIRCLE)
    result = ap.centre_curvature(
        ap.Ellipse(a, Orbit(e, inc, omega, node)),
        planet,
        spec.body_coefficient,
        spec.planet_coefficients[0],
        phase,
        np.radians(centre),
        RULE,
    )
    assert result.measured
    assert required / (MASSES[letter] * a * result.curvature) == pytest.approx(expected, abs=0.02)
    assert float(ap.circular_distance(result.centre_rad, np.radians(centre))) < np.radians(2.0)


def test_record_strength_15875_and_2005_gq107():
    for example in ('mmr2__15875__2N-3+0+1', 'mmr2__2005_GQ107__40M-57+0+17'):
        _needs(example)
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    strength = analyse_mmr(io.load_series('mmr2__15875__2N-3+0+1'), SETTINGS).strength
    assert strength.method == st.METHOD_DIRECT and strength.planet_elements == st.PLANET_SAVED
    assert strength.check == CheckValue.PASS and 0.95 < strength.ratio_min <= strength.ratio_max < 1.1
    # Mars's eccentricity through varpi - varpi_P: the theory at the snapshots is the brute force of the skeptic's code
    analysis = analyse_mmr(io.load_series('mmr2__2005_GQ107__40M-57+0+17'), SETTINGS)
    strength = analysis.strength
    # the deciding episode is the stationary core since W9 (geometry R1 trims the slow last half-cycle at 40.2 kyr): its
    # snapshots gave 0.937-2.36 at P_c = 12 yr (20 x 0.6 yr) and give 0.859-2.18 at the physical P_c = 75 yr (W16, the
    # episode's bounds move with the cutoff; 0.945-2.78 on the untrimmed run 31.7-40.2 kyr, W6)
    assert (strength.ratio_min, strength.ratio_max) == pytest.approx((0.859, 2.181), rel=0.01)
    assert strength.check == CheckValue.PASS and strength.n_evaluations == SETTINGS.mmr_strength_snapshots
    assert strength.n_consistent == strength.n_evaluations  # every swing inside the well (W9)
    assert strength.centre_offset_max_deg < 15.0  # the theory's centre follows the measured one (11.9 deg) as varpi - varpi_P turns


def test_record_without_saved_planet_scans_its_phase():
    example = 'mmr2__2004_TK14__31M-34+0+3'
    _needs(example)
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    series = io.load_series(example)
    # the run's planets now come with the record (W9 planets link); without Mars's series the phase is scanned
    series.planets.pop('Mars', None)
    strength = analyse_mmr(series, SETTINGS).strength
    assert strength.planet_elements == st.PLANET_PHASE_SCAN
    assert strength.n_evaluations == SETTINGS.mmr_strength_snapshots * SETTINGS.mmr_strength_unsaved_planet_phases
    assert strength.ratio_min < strength.ratio_max  # the unknown phase widens the range


# --- kinematics: the free/forced split on the cutoff series -------------------------------------------------------


def _series(t, e, varpi) -> MMRSeries:
    n = len(t)
    zeros = np.zeros(n)
    spec = ResonanceSpec('3J-1+0-2', ResonanceKind.TWO_BODY, ('Jupiter',), (3,), -1, -2, (0,))
    return MMRSeries(t, np.full(n, 2.5), e, zeros, zeros, varpi, varpi, zeros, spec, Regime(1.0, t[-1], n))


def test_short_period_vector_keeps_the_e_cycle():
    g5 = 4.257 * ARCSEC
    g = 231.28 * ARCSEC  # 37087: 2 pi / (g - g5) = 5.7 kyr
    t = np.arange(0.0, 100000.0, 1.0)
    z = 0.06 * np.exp(1j * g * t) + 0.0286 * np.exp(1j * (g5 * t + 0.4)) + 0.004 * np.exp(1j * 2 * np.pi * t / 12.0)
    e, varpi = kin.short_period_vector(_series(t, np.abs(z), np.angle(z)), 200.0, SETTINGS)
    inside = (t > 1000.0) & (t < 99000.0)
    e_free, e_forced, g_free = kin.free_forced_split(t[inside], e[inside], varpi[inside], g5)
    assert (e_free, e_forced) == pytest.approx((0.06, 0.0286), rel=2e-3)
    assert g_free / ARCSEC == pytest.approx(231.28, rel=1e-3)


def test_kinematics_37087_forced_term():
    example = 'mmr3__37087__3J+8S-3+0+0-8'
    _needs(example)
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    kinematics = analyse_mmr(io.load_series(example), SETTINGS).kinematics
    assert kinematics.e_forced == pytest.approx(0.0286, rel=0.1)
    g = kinematics.modulation_period_yr
    assert 2 * np.pi / g == pytest.approx((231.28 - 4.257) * ARCSEC, rel=0.02)
