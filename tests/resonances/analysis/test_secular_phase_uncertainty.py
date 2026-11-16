"""W28 D5 (card g09-3): a secular libration under the linear mode phases must swing more than the phases can stray.

Under SECULAR_PLANET_PHASES=linear the angle carries sum_j k_j (phi_j(t) - const), the planets' mode phases against
the linear law; its swing over a record of length T is at most dphi(T) = sum_j |k_j| range_j(T). A libration whose
swing (2 x the largest half-amplitude) is within SECULAR_PHASE_UNCERTAINTY_FACTOR x dphi is not resolved from it:
insufficient_time / phase_unresolved, action review (rerun with measured phases). 5026 g+g2-g4-g6: half-amplitude
8-11 deg on 10 Myr, dphi(10 Myr) = 34 deg (g2 2.7 + g4 30.3 + g6 0.9).

Expected numbers, computed by hand:
- a mode phase phi(t) = 5 deg sin(2 pi t / 50 Myr): peak to peak 10 deg; the 5 Myr least-squares fit averages it over
  its window, sinc(pi 5/50) = 0.984, so 9.84 deg; over a 1 Myr window its largest change is the slope's,
  5 deg x 2 pi / 50 x 1 Myr = 0.628 deg (x 0.984 = 0.618);
- a table with range(g4) = 10, 20, 40 deg at 1, 2, 4 Myr: 15 deg at 1.5 Myr, 10 deg at 0.5 Myr (the shortest's value),
  ceil(9/4) = 3 x 40 = 120 deg at 9 Myr;
- the card's case: a swing of 8 deg against dphi = 19 deg is unresolved; 2 x 47 = 94 deg is resolved.
"""

import math

import numpy as np
import pytest

from resonances.data.const import PLANETARY_FREQUENCIES
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.rules import Rule
from resonances.resonance.analysis.secular import LongPeriodSeries
from resonances.resonance.analysis.secular import decide
from resonances.resonance.analysis.secular.analyse import analyse_long_period
from resonances.resonance.analysis.secular.extend import CAUSE_PHASE_UNRESOLVED
from resonances.resonance.analysis.secular.models import AngleTerm
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import Action, AngleStatus, AngleSubtype, Reason, ResonanceType, Status
from resonances.secular.phase_uncertainty import PhaseUncertaintyTable, load_table, mode_phase_ranges
from resonances.secular.planet_modes import RAD_PER_ARCSEC

SETTINGS = settings_from_config()
G4_TERMS = (AngleTerm('g', None, 1), AngleTerm('g', 4, -1))


def _libration(half_amp_deg: float, terms=G4_TERMS, measured=()) -> LongPeriodSeries:
    """A secular angle over 10 Myr librating with a 1.25 Myr period and the given half-amplitude."""
    t = np.arange(0.0, 10_000_001.0, 2000.0)
    one = np.ones_like(t)
    angle = 1.0 + np.radians(half_amp_deg) * np.sin(2 * np.pi * t / 1.25e6)
    return LongPeriodSeries(
        ResonanceType.SECULAR,
        'g-g4',
        'synthetic',
        t,
        angle,
        2.2 * one,
        0.1 * one,
        0.1 * one,
        terms=terms,
        planet_frequencies=PLANETARY_FREQUENCIES,
        measured_modes=measured,
    )


# --- the table builder --------------------------------------------------------------------------------------------


def test_the_builder_reads_back_a_known_sine():
    """Mars' vector carries g4 with a phase 5 deg sin(2 pi t / 50 Myr) and a weaker g3 line (the closest mode, a
    2.4 Myr beat): the 5 Myr fit separates them and the range over a 30 Myr window is the sine's 10 deg (9.84 after the
    window's averaging); over 1 Myr it is the slope's 0.618 deg."""
    t = np.arange(0.0, 60e6 + 1.0, 20_000.0)
    phi = np.radians(5.0) * np.sin(2 * np.pi * t / 50e6)
    g4 = PLANETARY_FREQUENCIES['g4'] * RAD_PER_ARCSEC
    g3 = PLANETARY_FREQUENCIES['g3'] * RAD_PER_ARCSEC
    z = 0.07 * np.exp(1j * (g4 * t + 0.3 + phi)) + 0.03 * np.exp(1j * (g3 * t + 2.0))
    ranges = mode_phase_ranges(t, {'g4': z}, [1e6, 30e6], step_years=100_000.0)
    one, thirty = np.degrees(ranges['g4'])
    assert thirty == pytest.approx(10.0, abs=0.5)
    assert one == pytest.approx(0.618, abs=0.05)


def test_the_table_interpolates_and_bounds_beyond_its_grid():
    table = PhaseUncertaintyTable((1e6, 2e6, 4e6), {'g4': (10.0, 20.0, 40.0), 'g2': (1.0, 2.0, 3.0)})
    assert math.degrees(table.range_rad('g4', 1.5e6)) == pytest.approx(15.0)
    assert math.degrees(table.range_rad('g4', 0.5e6)) == pytest.approx(10.0)
    assert math.degrees(table.range_rad('g4', 9e6)) == pytest.approx(120.0)
    assert table.range_rad('s5', 3e6) == 0.0
    # dphi of g + g2 - 2 g4 at 2 Myr: 1 x 2 + 2 x 20 = 42 deg; with g4 measured only g2 is left
    assert math.degrees(table.delta_phi_rad({'g2': 1, 'g4': -2}, 2e6)) == pytest.approx(42.0)
    assert math.degrees(table.delta_phi_rad({'g2': 1, 'g4': -2}, 2e6, measured=('g4',))) == pytest.approx(2.0)


def test_the_packaged_table_covers_every_mode_and_never_shrinks():
    table = load_table()
    assert set(table.ranges_deg) == {f'g{j}' for j in range(1, 9)} | {f's{j}' for j in (1, 2, 3, 4, 6, 7, 8)}
    for label, values in table.ranges_deg.items():
        assert np.all(np.diff(values) >= 0), label
    assert table.provenance['script'].endswith('phase_uncertainty.py')


# --- the decision -------------------------------------------------------------------------------------------------


def _geometry_and_angle(half_amp_deg: float):
    """The geometry and angle decision of a synthetic libration, the phase check off (factor 0)."""
    analysis = analyse_long_period(_libration(half_amp_deg), settings_from_config(secular_phase_uncertainty_factor=0.0))
    angle = decide.AngleDecision(analysis.verdict.angle_status, analysis.verdict.angle_subtype, analysis.decisive_episode_id)
    return analysis.geometry, angle


@pytest.mark.parametrize('half_amp_deg, unresolved', [(4.0, True), (47.0, False)])
def test_the_swing_is_compared_with_the_phase_uncertainty(half_amp_deg, unresolved):
    """The card's numbers: dphi = 19 deg; a swing of 8 deg is unresolved, 94 deg is a libration."""
    geometry, angle = _geometry_and_angle(half_amp_deg)
    assert angle.angle_status == AngleStatus.LIBRATION
    assert math.degrees(decide.phase_swing_rad(angle, geometry)) == pytest.approx(2 * half_amp_deg, rel=0.02)
    ledger = Ledger.from_settings(SETTINGS)
    checked = decide.phase_check(angle, geometry, math.radians(19.0), SETTINGS, ledger)
    entry = next(entry for entry in ledger if entry.name == 'phase_uncertainty')
    assert entry.value == pytest.approx(2 * half_amp_deg / 19.0, rel=0.02)
    if unresolved:
        assert (checked.angle_status, checked.subtype) == (AngleStatus.INSUFFICIENT_TIME, AngleSubtype.PHASE_UNRESOLVED)
        assert Rule.PHASE_UNRESOLVED in checked.rules_fired
        assert checked.decisive_episode_id == angle.decisive_episode_id
    else:
        assert checked is angle


def test_a_small_swing_under_linear_phases_is_uncertain_with_a_review():
    """The whole analysis with the packaged table: g4 over 10 Myr strays 30.3 deg, more than the 8 deg swing."""
    analysis = analyse_long_period(_libration(4.0), SETTINGS)
    verdict = analysis.verdict
    assert (verdict.status, verdict.angle_status, verdict.angle_subtype) == (
        Status.UNCERTAIN,
        AngleStatus.INSUFFICIENT_TIME,
        AngleSubtype.PHASE_UNRESOLVED,
    )
    assert verdict.reason == Reason.BASELINE_NEEDED and verdict.action == Action.REVIEW
    numbers = analysis.numbers_row()
    assert numbers['base_extend_cause'] == CAUSE_PHASE_UNRESOLVED
    assert numbers['sec_phase_uncertainty_deg'] == pytest.approx(load_table().ranges_deg['g4'][6])  # T = 10 Myr
    assert numbers['sec_phase_swing_deg'] == pytest.approx(8.0, rel=0.02)
    assert str(Rule.PHASE_UNRESOLVED) in verdict.rules_fired


def test_a_wide_swing_stays_a_libration():
    verdict = analyse_long_period(_libration(47.0), SETTINGS).verdict
    assert (verdict.status, verdict.angle_status) == (Status.RESONANT, AngleStatus.LIBRATION)
    assert str(Rule.PHASE_UNRESOLVED) not in verdict.rules_fired


def test_measured_phases_switch_the_check_off():
    analysis = analyse_long_period(_libration(4.0, measured=('g4',)), SETTINGS)
    assert (analysis.verdict.status, analysis.verdict.angle_status) == (Status.RESONANT, AngleStatus.LIBRATION)
    assert analysis.numbers_row()['sec_phase_uncertainty_deg'] == 0.0
    assert analysis.numbers_row()['sec_measured_modes'] == 'g4'


def test_outer_modes_stray_too_little_to_reach_a_z1_librator():
    """z1 g - g6 + s - s6 over 10 Myr: dphi = g6 + s6 from the table, about 10 deg; the narrowest z1/z2 libration of
    the benchmark swings 2 x 25.7 = 51 deg (52773), the narrowest nu6 66 deg (73415)."""
    terms = (AngleTerm('g', None, 1), AngleTerm('g', 6, -1), AngleTerm('s', None, 1), AngleTerm('s', 6, -1))
    dphi = math.degrees(decide.phase_uncertainty_rad(_libration(25.7, terms=terms)))
    table = load_table()
    assert dphi == pytest.approx(table.ranges_deg['g6'][6] + table.ranges_deg['s6'][6])
    assert dphi < 0.5 * 51.0


def test_no_phase_uncertainty_for_a_zlk_angle():
    series = _libration(4.0)
    series.resonance_type = ResonanceType.LIDOV_KOZAI
    assert math.isnan(decide.phase_uncertainty_rad(series))
