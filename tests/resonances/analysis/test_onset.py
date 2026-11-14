"""O1 narrowed to a coherent foreign a-wave (B4, W28; cards g03-4 A, g08-3 A) and the q-2 b ceiling (no full physical
support -> <= medium).

Synthetic first: a pendulum captured from circulation near its separatrix (the old ratio test fails it, the narrowed
one must not), a capture after encounter kicks, and sigma riding an external wave of a with continuous phase (the one
case O1 must still object to). The pendulum is sigma_ddot = -w0^2 sin sigma - gamma sigma_dot (damping brings it through
the separatrix); a follows from the identity sigma_dot = C (a - a*)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.confidence import Cap
from resonances.resonance.analysis.ledger import Ledger, Margin
from resonances.resonance.analysis.mmr import coorbital, decide
from resonances.resonance.analysis.mmr import response as R
from resonances.resonance.analysis.mmr.models import EpisodePhysics
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import LibrationOrigin, MarginClass, MarginValue, Reason, Status
from tests.resonances.analysis.builders import make

SETTINGS = AnalysisSettings()
C = -0.5  # identity coefficient [rad/yr per unit of a]
W0 = 2 * np.pi / 100.0  # small-amplitude libration frequency: P0 = 100 yr
T = np.arange(0.0, 8000.0, 1.0)


# --- synthetic records ---------------------------------------------------------------------------


def pendulum(gamma: float, sigma0: float, rate0: float, t: np.ndarray) -> tuple:
    def rhs(_, y):
        return [y[1], -(W0**2) * np.sin(y[0]) - gamma * y[1]]

    sol = solve_ivp(rhs, (t[0], t[-1]), [sigma0, rate0], t_eval=t, rtol=1e-10, atol=1e-12)
    return sol.y[0], sol.y[1]


def near_separatrix_capture(eps: float, gamma: float):
    """Circulation at energy (1 + eps) x the separatrix's, damped through it into libration."""
    sigma, rate = pendulum(gamma, 0.0, W0 * math.sqrt(2 * (2 + eps)), T)
    return T, sigma, rate / C


def capture_after_kicks():
    """sigma circulates far from the resonance while encounters kick a (random steps every 25 yr, no resonant force);
    a last kick at 2000 yr leaves it inside the pendulum's well, where it librates."""
    rng = np.random.default_rng(7)
    t = T[T < 6000]
    before = t < 2000
    steps = np.zeros(len(t))
    steps[np.arange(25, 2000, 25)] = rng.normal(0.0, 0.03, len(np.arange(25, 2000, 25)))
    a = 0.25 + np.cumsum(steps)  # a - a*, far from 0: sigma circulates at C (a - a*)
    sigma = np.cumsum(C * a)
    s_after, rate_after = pendulum(0.0, sigma[before][-1], 0.6 * W0, t[~before])
    sigma[~before], a[~before] = s_after, rate_after / C
    return t, sigma, a


def external_wave(drift: float = 0.45):
    """a carries a wave at 100 yr throughout (continuous phase); before 1000 yr the resonance's a* sits `drift` away,
    so sigma circulates through the wave; after, sigma rides it (a kinematic libration of 69 deg)."""
    t = T[T < 4000]
    a = 0.15 * np.sin(2 * np.pi * t / 100.0)
    a_star = np.where(t < 1000.0, -drift, 0.0)
    return t, np.cumsum(C * (a - a_star)), a


def onset_physics(t: np.ndarray, sigma: np.ndarray, a: np.ndarray) -> EpisodePhysics:
    """O1's numbers of the record's first libration episode, measured as the package does: the band [P/2, 2P] of a,
    the 2P before the first turning point, the episode's own law fitted on the episode (K, phi0) and read fixed on the
    background."""
    rate = np.gradient(sigma, t)
    turns = t[np.flatnonzero(np.sign(rate[1:]) != np.sign(rate[:-1]))]
    start, end = float(turns[0]), float(turns[-1])
    period = float(np.median(turns[2:] - turns[:-2]))
    background = (start - 2 * period, start)
    before, inside, _ = R.onset_bands(t, a, period, start, end, SETTINGS.filter_order)
    band = filtering.band_pass(t, a, period, R.ONSET_BAND_FACTOR, SETTINGS.filter_order)
    coherence, gap = R.onset_coherence(t, band, period, background, (start, min(end, start + 2 * period)))
    adot = np.gradient(a, t)
    episode = (t >= start) & (t <= end)
    sin_part, cos_part, _, _ = R.sine_coefficients(sigma[episode], adot[episode])
    force, phi0 = math.hypot(sin_part, cos_part), math.atan2(-cos_part, sin_part)
    own_bg, own_ep = R.onset_own_law(t, np.sin(sigma - phi0), adot, force, background, (start, end))
    return make(
        EpisodePhysics,
        episode_id=0,
        a_band_std_before_au=before,
        a_band_std_in_au=inside,
        onset_coherence_r2=coherence,
        onset_phase_gap_deg=gap,
        onset_own_law_r2=own_bg,
        onset_own_law_r2_episode=own_ep,
    )


def o1(physics: EpisodePhysics) -> MarginValue:
    return decide._o1(physics, None, Ledger.from_settings(SETTINGS), SETTINGS)


# --- the measurement -----------------------------------------------------------------------------


def test_sinusoid_at_recovers_r2_and_phase():
    t = np.arange(0.0, 500.0, 1.0)
    r2, phase = R.sinusoid_at(t, 3.0 * np.cos(2 * np.pi * t / 80.0 - 0.7) + 1.0, 80.0)
    assert r2 == pytest.approx(1.0, abs=1e-12)
    assert phase == pytest.approx(0.7, abs=1e-9)


def test_phase_gap_is_the_background_wave_extrapolated_into_the_episode():
    t = np.arange(0.0, 1000.0, 1.0)
    shift = np.radians(100.0)
    band = np.where(t < 500.0, np.cos(2 * np.pi * t / 50.0), np.cos(2 * np.pi * t / 50.0 - shift))
    r2, gap = R.onset_coherence(t, band, 50.0, (300.0, 499.5), (500.0, 600.0))
    assert r2 == pytest.approx(1.0, abs=1e-9)
    assert gap == pytest.approx(100.0, abs=1e-6)
    # a background shorter than one period is not measured (as onset_bands)
    assert np.isnan(R.onset_coherence(t, band, 50.0, (460.0, 500.0), (500.0, 600.0))[0])


@pytest.mark.parametrize('offset_deg, expected', [(0.0, 1.0), (60.0, 0.0), (90.0, -1.0)])
def test_own_law_is_read_with_the_episodes_k_and_phi0_not_refitted(offset_deg, expected):
    """For sigma uniform over a turn, a_dot = K sin(sigma - phi0 - d) read with the law K sin(sigma - phi0) leaves
    var = K^2 (1 - cos d) of var K^2 / 2: R^2 = 1 - 2 (1 - cos d) (1, 0, -1 at 0, 60, 90 deg); a refit would give 1."""
    sigma = np.linspace(0.0, 40 * np.pi, 20001)[:-1]
    adot = 2.0 * np.sin(sigma - 0.3 - np.radians(offset_deg)) + 0.1
    assert coorbital.fixed_scale_r2(adot, np.sin(sigma - 0.3), 2.0) == pytest.approx(expected, abs=1e-6)  # onset_own_law's
    assert R.sine_coefficients(sigma, adot)[3] == pytest.approx(1.0, abs=1e-9)


# --- the decision on synthetic records -----------------------------------------------------------


def test_capture_from_circulation_near_the_separatrix_is_not_an_objection():
    physics = onset_physics(*near_separatrix_capture(0.3, 6e-4))
    assert decide.onset_ratio_fails(physics, SETTINGS)  # the old ratio test objected to this capture
    assert physics.onset_own_law_r2 / physics.onset_own_law_r2_episode > SETTINGS.mmr_o1_own_law_r2_fraction
    assert o1(physics).margin == Margin.THIN


def test_coherent_in_phase_background_explained_by_the_own_law_is_not_an_objection():
    physics = onset_physics(*near_separatrix_capture(0.1, 2e-4))
    assert physics.onset_coherence_r2 >= SETTINGS.mmr_o1_coherence_r2_min
    assert physics.onset_phase_gap_deg <= SETTINGS.mmr_o1_phase_gap_max_deg
    assert decide.onset_ratio_fails(physics, SETTINGS)
    assert physics.onset_own_law_r2 == pytest.approx(1.0, abs=0.02)  # the same pendulum before and after the capture
    assert o1(physics).margin == Margin.THIN


def test_capture_after_kicks_is_not_an_objection():
    physics = onset_physics(*capture_after_kicks())
    assert physics.onset_coherence_r2 < SETTINGS.mmr_o1_coherence_r2_min
    assert o1(physics).margin != Margin.FAIL


def test_sigma_on_an_external_in_phase_wave_is_an_objection():
    physics = onset_physics(*external_wave())
    assert decide.onset_ratio(physics) == pytest.approx(1.0, abs=0.05)
    assert physics.onset_coherence_r2 > 0.99 and physics.onset_phase_gap_deg < 5.0
    assert physics.onset_own_law_r2_episode > 0.9 and physics.onset_own_law_r2 < 0.0
    assert o1(physics).margin == Margin.FAIL


# --- the predicate on given numbers --------------------------------------------------------------


def physics_of(coherence=0.9, gap=20.0, own=0.1, own_episode=0.8, before=1.0, inside=1.0) -> EpisodePhysics:
    return make(
        EpisodePhysics,
        episode_id=0,
        a_band_std_before_au=before,
        a_band_std_in_au=inside,
        onset_coherence_r2=coherence,
        onset_phase_gap_deg=gap,
        onset_own_law_r2=own,
        onset_own_law_r2_episode=own_episode,
    )


@pytest.mark.parametrize(
    'fields, expected',
    [
        ({}, Margin.FAIL),  # coherent, in phase, not explained: a foreign wave
        ({'own': float('nan')}, Margin.FAIL),  # an own law that cannot be read explains nothing
        ({'own': 0.6}, Margin.THIN),  # 0.6 >= 0.5 x 0.8: the episode's own force
        ({'coherence': 0.3}, Margin.THIN),  # kicks, not a wave (2020 CB: 0.02)
        ({'coherence': float('nan')}, Margin.THIN),  # coherence not measured is no evidence of a wave
        ({'gap': 146.0}, Margin.THIN),  # a capture starts its own phase (2024 JZ64)
        ({'before': 0.3}, Margin.THIN),  # the ratio itself only thin
        ({'before': 0.1}, Margin.FULL),
    ],
)
def test_o1_fails_only_for_a_foreign_wave(fields, expected):
    assert o1(physics_of(**fields)).margin == expected


# --- q-2 b: high needs one physical support with its full margin --------------------------------


def margin(value: Margin) -> MarginValue:
    return MarginValue(value, 0.5, MarginClass.DECISIVE)


def transient(origin=LibrationOrigin.DYNAMIC, status=Status.TRANSIENT):
    return make(decide.StatusDecision, status=status, reason=Reason.DYNAMIC_EPISODE, origin=origin)


PHYSICS = ('L4', 'R0', 'Z', 'W')


@pytest.mark.parametrize(
    'margins, decision, settings, capped',
    [
        ({'L4': Margin.THIN, 'R0': Margin.THIN, 'Z': Margin.FULL}, transient(), SETTINGS, True),  # 2020 CB
        ({'L4': Margin.FULL, 'R0': Margin.THIN}, transient(), SETTINGS, False),
        ({'L4': Margin.THIN, 'R0': Margin.FULL}, transient(), SETTINGS, False),
        ({'L4': Margin.NOT_MEASURED, 'R0': Margin.NOT_MEASURED}, transient(), SETTINGS, False),  # all_physics_not_measured
        ({'L4': Margin.THIN, 'R0': Margin.THIN}, transient(LibrationOrigin.KINEMATIC, Status.NON_RESONANT), SETTINGS, False),
        ({'L4': Margin.THIN, 'R0': Margin.THIN}, transient(), AnalysisSettings(mmr_high_needs_full_support=False), False),
    ],
)
def test_high_needs_one_full_physical_support(margins, decision, settings, capped):
    caps = decide.full_support_caps(decision, {k: margin(v) for k, v in margins.items()}, PHYSICS, settings)
    assert caps == ([str(Cap.NO_FULL_PHYSICS_SUPPORT)] if capped else [])
