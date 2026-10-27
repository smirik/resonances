"""Independent tests of W8 (confidence without a calibration gate, the orbit type, the subresonance degree from the
catalogue, the explained right column). Expected values are derived here, not read from the code:

- confidence: the rule of SPEC §7.2 as written in the module docstring (certain/high/medium/low from full, thin, failed
  support, caps), the Wilson 95 % interval recomputed from (n_examples, n_errors);
- orbit type: θ = λ − λ_P of a co-orbital from the σ turning points (σ = λ_P − λ for 1J-1, so θ = −σ): L4 leads by
  +60°, a horseshoe passes 180° only, a quasi-satellite passes 0° only; interior/exterior from n/n_P = |k_P|/|k|;
- subresonances: every slow-coefficient vector with sum −Σk and Σ|c| ≤ max(|q|, MATRIX_*_ORDER_MAX) by brute force;
- panel: synthetic records built from the identity σ = 3λ_J − 2λ − ϖ (a from n by Kepler), so the turning points,
  periods and coverage are known by construction.
"""

import itertools
import math
import re
from types import SimpleNamespace

import numpy as np
import pytest

from resonances.plotting.verdict_text import explain_long_period, explain_mmr
from resonances.resonance.analysis.confidence import (
    Cap,
    CellRecord,
    ConfidenceCells,
    apply_caps,
    assess,
    load_cells,
)
from resonances.resonance.analysis.geometry import Episode, EpisodeShape, RejectReason
from resonances.resonance.analysis.ledger import Margin
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.models import MMRSeries, PlanetSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.mmr.subresonances import degree_max, enumerate_subresonances
from resonances.resonance.analysis.orbit_type import coorbital_text, long_period_orbit_type, mmr_orbit_type
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    Confidence,
    MarginClass,
    MarginValue,
    ResonanceType,
    Status,
    Verdict,
)
from tests.resonances.analysis.builders import make_geometry

TWO_PI = 2.0 * np.pi
SETTINGS = AnalysisSettings()
PHYSICS = ('L4', 'Z', 'R0')


# ----------------------------------------------------------------------------------------------- confidence


def mv(margin, margin_class=MarginClass.DECISIVE):
    return MarginValue(margin=margin, value=1.0, margin_class=margin_class)


FULL, THIN, FAIL, NM, NA = Margin.FULL, Margin.THIN, Margin.FAIL, Margin.NOT_MEASURED, Margin.NOT_APPLICABLE
S, C = MarginClass.SUPPORT, MarginClass.CAP

# (margins, caps, ledger_thin, expected raw, expected shown) — by the rule of SPEC §7.2
CONFIDENCE_CASES = {
    'all_full': ({'L4': mv(FULL), 'Z': mv(FULL), 'R0': mv(FULL), 'B1': mv(FULL, S)}, [], [], 'certain', 'certain'),
    'one_thin': ({'L4': mv(FULL), 'Z': mv(THIN), 'R0': mv(FULL)}, [], [], 'high', 'high'),
    'many_thin': ({'L4': mv(THIN), 'Z': mv(THIN), 'R0': mv(THIN), 'L2': mv(THIN)}, [], [], 'high', 'high'),
    'not_measured_counts_thin': ({'L4': mv(FULL), 'Z': mv(FULL), 'R0': mv(NM)}, [], [], 'high', 'high'),
    'one_failed_support': ({'L4': mv(FULL), 'Z': mv(FULL), 'B1': mv(FAIL, S)}, [], [], 'medium', 'medium'),
    'two_failed_support': ({'L4': mv(FULL), 'B1': mv(FAIL, S), 'L6': mv(FAIL, S)}, [], [], 'low', 'low'),
    'failed_cap_margin_not_counted': ({'L4': mv(FULL), 'Z': mv(FULL), 'N': mv(FAIL, C)}, [], [], 'certain', 'certain'),
    'not_applicable_ignored': ({'L4': mv(FULL), 'Z': mv(NA), 'R0': mv(NA)}, [], [], 'certain', 'certain'),
    'cap_high': ({'L4': mv(FULL), 'Z': mv(FULL)}, [Cap.THREE_TP_EPISODE], [], 'certain', 'high'),
    'cap_low': ({'L4': mv(FULL), 'Z': mv(FULL)}, [Cap.ONSET_FAILED], [], 'certain', 'low'),
    'cap_never_raises': ({'B1': mv(FAIL, S), 'L6': mv(FAIL, S)}, [Cap.THREE_TP_EPISODE], [], 'low', 'low'),
    'two_caps_lowest_wins': ({'L4': mv(FULL)}, [Cap.ZONE_NOT_MEASURED, Cap.EPISODES_IN_SEPARATRIX_ZONE], [], 'certain', 'medium'),
    'all_physics_not_measured': ({'L4': mv(NM), 'Z': mv(NM), 'R0': mv(NM)}, [], [], 'high', 'medium'),
    'one_physics_measured': ({'L4': mv(NM), 'Z': mv(FULL), 'R0': mv(NM)}, [], [], 'high', 'high'),
    'ledger_u_thin': ({'L4': mv(FULL), 'Z': mv(FULL)}, [], ['U:libration_min_tp'], 'high', 'high'),
}

CELL = ('resonant', 'libration', 'dynamic_libration')


@pytest.mark.parametrize('case', sorted(CONFIDENCE_CASES))
def test_confidence_level_from_margins_and_caps(case):
    margins, caps, ledger_thin, raw, shown = CONFIDENCE_CASES[case]
    # a calibration cell with no author labels at all: it must not hold the level down (no gate)
    cells = ConfidenceCells(records={CELL: CellRecord(0, 0)})
    result = assess(Status.RESONANT, 'libration', 'dynamic_libration', margins, caps, ledger_thin, PHYSICS, cells)
    assert str(result.confidence_raw) == raw
    assert str(result.confidence) == shown
    assert result.cell == CellRecord(0, 0)


def test_all_physics_not_measured_adds_its_cap():
    margins, caps, ledger_thin, _, _ = CONFIDENCE_CASES['all_physics_not_measured']
    result = assess(Status.RESONANT, 'libration', 'dynamic_libration', margins, caps, ledger_thin, PHYSICS, ConfidenceCells())
    assert str(Cap.ALL_PHYSICS_NOT_MEASURED) in result.caps
    assert result.cell is None  # no record for the cell: nothing attached, the level is still shown


def test_ledger_thin_item_is_listed():
    margins, caps, ledger_thin, _, _ = CONFIDENCE_CASES['ledger_u_thin']
    result = assess(Status.RESONANT, 'libration', 'dynamic_libration', margins, caps, ledger_thin, PHYSICS, ConfidenceCells())
    assert 'U:libration_min_tp' in result.thin_margins


@pytest.mark.parametrize('cell', [CellRecord(0, 0), CellRecord(1, 1), CellRecord(3, 3)])
def test_certain_shown_whatever_the_cell(cell):
    """Author 16b C2: the real levels now, certain included, whatever the labels say."""
    margins = {'L4': mv(FULL), 'Z': mv(FULL), 'R0': mv(FULL)}
    result = assess(Status.RESONANT, 'libration', 'dynamic_libration', margins, [], [], PHYSICS, ConfidenceCells({CELL: cell}))
    assert result.confidence == Confidence.CERTAIN
    assert result.cell == cell


@pytest.mark.parametrize('status', [Status.UNCERTAIN, Status.INVALID])
def test_uncertain_and_invalid_have_no_level_and_no_cell(status):
    cells = ConfidenceCells({(str(status), 'drift', 'x'): CellRecord(10, 0)})
    result = assess(status, 'drift', 'x', {'L4': mv(FULL)}, [], [], PHYSICS, cells)
    assert result.confidence == Confidence.NONE and result.confidence_raw == Confidence.NONE
    assert result.cell is None


def test_apply_caps_ceiling_table():
    # by hand: high ceiling 3TP and zone; medium for the rest; low for onset_failed
    assert apply_caps(Confidence.CERTAIN, ['three_tp_episode']) == Confidence.HIGH
    assert apply_caps(Confidence.CERTAIN, ['zone_not_measured']) == Confidence.HIGH
    assert apply_caps(Confidence.CERTAIN, ['near_episode']) == Confidence.MEDIUM
    assert apply_caps(Confidence.HIGH, ['onset_failed']) == Confidence.LOW
    assert apply_caps(Confidence.NONE, ['onset_failed']) == Confidence.NONE


def wilson(n, k, z=1.959963984540054):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def test_cells_file_wilson_intervals():
    cells = load_cells()
    assert cells.records, 'the package cells file holds no cell'
    checked = 0
    for key, record in cells.records.items():
        assert 0 <= record.n_errors <= record.n_examples
        if record.n_examples == 0:
            continue
        low, high = wilson(record.n_examples, record.n_examples - record.n_errors)
        assert record.accuracy_low_95 == pytest.approx(low, abs=2e-3), key
        assert record.accuracy_high_95 == pytest.approx(high, abs=2e-3), key
        checked += 1
    assert checked > 0


def test_wilson_hand_values():
    # by hand: 24/24 -> 24 / (24 + z²) = 0.862; 2 of 4 -> 0.5 ± 0.350
    assert wilson(24, 24)[0] == pytest.approx(0.862, abs=1e-3)
    low, high = wilson(4, 2)
    assert (low, high) == pytest.approx((0.150, 0.850), abs=1e-3)


def test_verdict_columns_orbit_type_and_cell():
    verdict = Verdict(
        resonance_type=ResonanceType.MMR,
        status=Status.RESONANT,
        angle_status=AngleStatus.LIBRATION,
        confidence=Confidence.CERTAIN,
        confidence_raw=Confidence.CERTAIN,
        orbit_type='compound: horseshoe → quasi-satellite',
        cell_n_examples=7,
        cell_n_errors=1,
        cell_accuracy_low_95=0.49,
        cell_accuracy_high_95=0.97,
    )
    row = verdict.to_flat_dict()
    names = list(row)
    assert names[names.index('final_regime_start_yr') + 1] == 'orbit_type'
    i = names.index('confidence_raw')
    cell_columns = ['cell_n_examples', 'cell_n_errors', 'cell_accuracy_low_95', 'cell_accuracy_high_95']
    assert names[i + 1 : i + 5] == cell_columns
    back = Verdict.from_flat_dict(row)
    assert back.orbit_type == 'compound: horseshoe → quasi-satellite'
    assert (back.cell_n_examples, back.cell_n_errors) == (7, 1)
    assert back.cell_accuracy_low_95 == pytest.approx(0.49)
    assert back.cell_accuracy_high_95 == pytest.approx(0.97)


def test_cell_record_verdict_fields_nan_to_none():
    assert CellRecord(5, 2).verdict_fields() == {
        'cell_n_examples': 5,
        'cell_n_errors': 2,
        'cell_accuracy_low_95': None,
        'cell_accuracy_high_95': None,
    }


# ----------------------------------------------------------------------------------------------- orbit type


def episode(episode_id, sigma_tp_deg, t0=0.0, period=100.0, accepted=True, centre_deg=None, shift_over_amp=0.0):
    """An accepted run with the given σ turning points (unwrapped, degrees)."""
    values = np.radians(np.asarray(sigma_tp_deg, dtype=float))
    times = t0 + period / 2 * np.arange(len(values))
    if centre_deg is None:
        centre_deg = np.degrees(np.mean(values))
    amp = abs(values[0] - values[1]) / 2
    return Episode(
        episode_id=episode_id,
        tp_times_yr=times,
        tp_values_rad=values,
        accepted=accepted,
        shape=EpisodeShape.STATIONARY if accepted else EpisodeShape.NONE,
        reject_reason=RejectReason.NONE if accepted else RejectReason.TOO_FEW_TP,
        period_med_yr=period,
        period_min_yr=period,
        period_max_yr=period,
        half_amp_min_rad=amp,
        half_amp_med_rad=amp,
        half_amp_max_rad=amp,
        centre_mod_2pi_rad=float(np.mod(np.radians(centre_deg), TWO_PI)),
        centre_shift_rad=0.0,
        centre_shift_over_amp=shift_over_amp,
        bands_overlap=True,
        centre_returns=0,
        coverage_start_gap_p=0.0,
        coverage_end_gap_p=0.0,
    )


def geometry(*runs):
    return make_geometry(runs=list(runs))


def zigzag(low, high, n=8):
    return [low if i % 2 == 0 else high for i in range(n)]


COORBITAL = ResonanceSpec.from_key('1J-1+0+0')  # σ = λ_J − λ, θ = λ − λ_J = −σ


# θ ranges (deg) -> σ = −θ; regimes by the definition of θ = λ − λ_P
COORBITAL_CASES = {
    'L4': ([zigzag(-40, -90)], 'tadpole L4'),  # θ 40..90: leads Jupiter
    'L5': ([zigzag(40, 90)], 'tadpole L5'),  # θ −90..−40 = 270..320
    'horseshoe': ([zigzag(-25, -335)], 'horseshoe'),  # θ 25..335: passes 180 only
    'quasi_satellite': ([zigzag(-20, 20)], 'quasi-satellite'),  # θ −20..20: passes 0 only
    'compound_in_one_run': ([zigzag(30, -200)], 'compound (horseshoe ↔ quasi-satellite)'),  # θ −30..200
    'horseshoe_then_qs': ([zigzag(-25, -335), zigzag(-20, 20)], 'compound: horseshoe → quasi-satellite'),
    'two_horseshoes_merge': ([zigzag(-25, -335), zigzag(-30, -330)], 'horseshoe'),
}


@pytest.mark.parametrize('case', sorted(COORBITAL_CASES))
def test_coorbital_orbit_type_from_turning_points(case):
    tps, expected = COORBITAL_CASES[case]
    runs = [episode(i, values, t0=1000.0 * i) for i, values in enumerate(tps)]
    assert mmr_orbit_type(COORBITAL, geometry(*runs), None, None, SETTINGS) == expected


def test_coorbital_law_regimes_take_priority():
    runs = [episode(0, zigzag(-40, -90))]  # the TPs alone would say L4
    law = SimpleNamespace(regime_sequence=('horseshoe', 'quasi_satellite', 'horseshoe'))
    text = mmr_orbit_type(COORBITAL, geometry(*runs), None, law, SETTINGS)
    assert text == 'compound: horseshoe → quasi-satellite → horseshoe'


def test_coorbital_text_names():
    assert coorbital_text(('tadpole_L5',)) == 'tadpole L5'
    assert coorbital_text(()) == ''
    assert coorbital_text(('quasi_satellite', 'horseshoe')) == 'compound: quasi-satellite → horseshoe'


def test_retrograde_coorbital():
    spec = ResonanceSpec.from_key('1J-1r0')
    runs = [episode(0, zigzag(-30, 30), centre_deg=0.0)]
    assert mmr_orbit_type(spec, geometry(*runs), None, None, SETTINGS) == 'retrograde co-orbital, libration about 0°'


def test_no_accepted_episode_gives_empty_type():
    runs = [episode(0, zigzag(-40, -90), accepted=False)]
    assert mmr_orbit_type(COORBITAL, geometry(*runs), None, None, SETTINGS) == ''
    assert mmr_orbit_type(COORBITAL, None, None, None, SETTINGS) == ''


# non-co-orbital: (key, [(centre, shift/amp)], decisive, expected); tolerance 30° (MMR_SUBRESONANCE_CENTRE_TOL_DEG)
LIBRATION_CASES = {
    # Hilda 3:2: n/n_J = 3/2 > 1 -> inside Jupiter's orbit
    'hilda': ('3J-2+0-1', [(180.0, 0.0)], None, 'interior, symmetric libration about 180°'),
    # plutino 2:3 with Neptune: n/n_N = 2/3 -> outside
    'plutino': ('2N-3+0+1', [(205.0, 0.0)], None, 'exterior, symmetric libration about 180°'),
    'asymmetric': ('1N-2+0+1', [(215.0, 0.0)], None, 'exterior, asymmetric libration about 215°'),
    'asymmetric_mirror': ('1N-2+0+1', [(72.0, 0.0), (288.0, 0.0)], 0, 'exterior, asymmetric libration about 72° and 288°'),
    'moving_centre': ('3J-2+0-1', [(0.0, 0.8)], None, 'interior, symmetric libration about 0°, moving centre'),
    'three_body_no_side': ('5J-2S-2+0+0-1', [(180.0, 0.0)], None, 'symmetric libration about 180°'),
    # retrograde 1/-3: n/n_J = 1/3 -> outside
    'retrograde': ('1J-3r0', [(180.0, 0.0)], None, 'retrograde, exterior, symmetric libration about 180°'),
    'decisive_not_longest': ('3J-2+0-1', [(0.0, 0.0), (180.0, 0.0)], 1, 'interior, symmetric libration about 180°'),
}


@pytest.mark.parametrize('case', sorted(LIBRATION_CASES))
def test_libration_orbit_type(case):
    key, centres, decisive, expected = LIBRATION_CASES[case]
    runs = []
    for i, (centre, shift) in enumerate(centres):
        n = 12 if i == 0 else 6  # the first is the longest
        runs.append(episode(i, zigzag(centre - 20, centre + 20, n), t0=5000.0 * i, centre_deg=centre, shift_over_amp=shift))
    spec = ResonanceSpec.from_key(key)
    assert mmr_orbit_type(spec, geometry(*runs), decisive, None, SETTINGS) == expected


def test_long_period_orbit_type():
    runs = [episode(0, zigzag(150, 210), centre_deg=180.0)]
    assert long_period_orbit_type(ResonanceType.SECULAR, geometry(*runs), None) == 'σ libration about 180°'
    runs = [episode(0, zigzag(60, 120), centre_deg=90.0)]
    assert long_period_orbit_type(ResonanceType.LIDOV_KOZAI, geometry(*runs), None) == 'ω libration about 90°'


# ----------------------------------------------------------------------------------------------- subresonances


def q_signed(spec):
    return -(sum(spec.planet_coefficients) + spec.body_coefficient)


def own_vector(spec, nodal):
    n = len(spec.planets)
    planar = (spec.body_varpi_coefficient,) + tuple(spec.planet_varpi_coefficients)
    if not nodal:
        return planar
    nodes = tuple(spec.planet_node_coefficients) or (0,) * n
    return planar + (spec.body_node_coefficient,) + nodes


def brute_force(spec, d_max, nodal):
    """Every integer vector of the slow coefficients with signed sum q and degree Σ|c| ≤ d_max; nodes even."""
    n = len(spec.planets)
    size = 2 * (n + 1) if nodal else n + 1
    q = q_signed(spec)
    own = own_vector(spec, nodal)
    found = []
    for head in itertools.product(range(-d_max, d_max + 1), repeat=size - 1):
        vector = head + (q - sum(head),)  # the last coefficient closes the signed sum
        if sum(abs(c) for c in vector) > d_max:
            continue
        if nodal and sum(vector[n + 1 :]) % 2:
            continue
        if vector == own:
            continue
        found.append(vector)
    return found


def test_degree_max_from_catalogue_order():
    assert SETTINGS.matrix_2body_order_max == 10 and SETTINGS.matrix_3body_order_max == 6  # .env.dist
    assert degree_max(ResonanceSpec.from_key('3J-2+0-1'), SETTINGS) == 10
    assert degree_max(ResonanceSpec.from_key('5J-2S-2+0+0-1'), SETTINGS) == 6
    assert degree_max(ResonanceSpec.from_key('13M-24+0+11'), SETTINGS) == 11  # |q| = 11 beyond the catalogue


# hand counts (planar): 3J-2: u + u_J = −1, |u| + |u_J| ≤ 10 -> u in −5..4 (10) minus own = 9;
# 13M-24: u + u_M = 11, |u| + |u_M| ≤ 11 -> u in 0..11 (12) minus own = 11
HAND_COUNTS = {'3J-2+0-1': 9, '13M-24+0+11': 11}


@pytest.mark.parametrize('key', sorted(HAND_COUNTS))
def test_subresonance_hand_count(key):
    spec = ResonanceSpec.from_key(key)
    members = enumerate_subresonances(spec, degree_max(spec, SETTINGS))
    assert len(members) == HAND_COUNTS[key] == len(brute_force(spec, degree_max(spec, SETTINGS), False))


@pytest.mark.parametrize(
    'key, nodal',
    [('3J-2+0-1', False), ('3J-2+0-1', True), ('4J-2+0-2', True), ('5J-2S-2+0+0-1', False), ('5J-2S-2+0+0-1', True)],
)
def test_subresonance_count_brute_force(key, nodal):
    spec = ResonanceSpec.from_key(key)
    d = degree_max(spec, SETTINGS)
    expected = brute_force(spec, d, nodal)
    members = enumerate_subresonances(spec, d, nodal=nodal)
    assert len(members) == len(expected)
    n = len(spec.planets)
    vectors = set()
    for member, j, is_nodal in members:
        vector = own_vector(member, nodal)
        vectors.add(vector)
        assert sum(abs(c) for c in vector) <= d
        assert member.planet_coefficients == spec.planet_coefficients and member.body_coefficient == spec.body_coefficient
        # j: u_P of a two-body member, the sum of the planets' varpi coefficients of a three-body one
        assert j == sum(member.planet_varpi_coefficients)
        assert is_nodal == any(vector[n + 1 :])
    assert vectors == set(expected)


def test_parity_degree_max_adds_nothing():
    # degree has the parity of q: q = −1 (odd), so degree_max 10 lists what 9 lists
    spec = ResonanceSpec.from_key('3J-2+0-1')
    assert len(enumerate_subresonances(spec, 10)) == len(enumerate_subresonances(spec, 9))


@pytest.mark.parametrize('key', ['1J-1+0+0', '1J-1r0', '1J-3r0'])
def test_no_members_for_coorbital_and_retrograde(key):
    assert enumerate_subresonances(ResonanceSpec.from_key(key), 10, nodal=True) == []


# ----------------------------------------------------------------------------------------------- panel


SPAN, STEP, P_LIB, P_CIRC = 20000.0, 1.0, 250.0, 300.0
N_J = TWO_PI / 11.862


def hilda_record(sigma, t, ecc=None, name='syn'):
    """A 3J-2+0-1 record whose σ is `sigma`: λ = (3λ_J − ϖ − σ) / 2, a = (2π/n)^(2/3) (au, yr, M_sun)."""
    lam_j = 0.3 + N_J * t
    varpi = 1.0 + TWO_PI / 2.5e5 * t
    lam = (3 * lam_j - varpi - sigma) / 2
    a = (TWO_PI / np.gradient(lam, t)) ** (2 / 3)
    e = np.full_like(t, 0.15) if ecc is None else ecc
    node = 2.0 - TWO_PI / 3e5 * t
    ones = np.ones_like(t)
    jupiter = PlanetSeries('Jupiter', t, 5.2026 * ones, 0.048 * ones, 0.0228 * ones, 1.75 * ones, 0.25 * ones, np.mod(lam_j, TWO_PI))
    return MMRSeries(
        t,
        a,
        e,
        0.1 * ones,
        np.mod(node, TWO_PI),
        np.mod(varpi - node, TWO_PI),
        np.mod(varpi, TWO_PI),
        np.mod(sigma, TWO_PI),
        ResonanceSpec.from_key('3J-2+0-1'),
        Regime(t[1] - t[0], t[-1] - t[0], len(t), 'ias15'),
        {'Jupiter': jupiter},
        np.mod(lam, TWO_PI),
        np.mod(lam - varpi, TWO_PI),
        name=name,
    )


def windows_sigma(t, windows, amp_deg):
    """Circulation at 2π/P_CIRC outside `windows`, a libration of period P_LIB about the frozen phase inside."""
    inside = np.zeros_like(t, dtype=bool)
    osc = np.zeros_like(t)
    for t0, t1 in windows:
        m = (t >= t0) & (t < t1)
        inside |= m
        osc[m] = np.radians(amp_deg) * np.sin(TWO_PI * (t[m] - t0) / P_LIB)
    return np.cumsum(np.where(inside, 0.0, TWO_PI / P_CIRC)) * (t[1] - t[0]) + osc


def pendulum_near_separatrix(t, mean=1.03, swing=0.06, period=1500.0, gain=0.02):
    """A pendulum (ω0 = 2π/250 yr) held at an energy around the separatrix (`mean` ± `swing` of the separatrix energy,
    period `period`): it slips over the top every few cycles, or librates a few cycles between slips."""
    w0 = TWO_PI / P_LIB
    target = mean + swing * np.sin(TWO_PI * t / period)  # energy / separatrix energy
    sig, vel = np.zeros_like(t), np.zeros_like(t)
    vel[0] = np.sqrt(2 * target[0] * 2 * w0**2)
    for i in range(1, len(t)):
        vel[i] = vel[i - 1] - w0**2 * np.sin(sig[i - 1] - np.pi) * STEP
        sig[i] = sig[i - 1] + vel[i] * STEP
        energy = 0.5 * vel[i] ** 2 + w0**2 * (1 - np.cos(sig[i] - np.pi))
        if energy > 0:
            vel[i] *= 1 + gain * (np.sqrt(target[i] * 2 * w0**2 / energy) - 1)
    return sig


def _records():
    t = np.arange(0.0, SPAN + STEP / 2, STEP)
    short_t = np.arange(0.0, 6000.0 + STEP / 2, STEP)
    # a Hilda's 3λ_J − 2λ − ϖ librates about 0°, where the directly averaged potential of Jupiter has its minimum: a
    # libration about 180° lies outside that well and fails the strength check, which the branch with L4 and R0 not
    # measured reads as an objection (W18); the windows start where the circulation's phase is 0 mod 2π (t0 / P_CIRC
    # an integer, counting circulation time only)
    return {
        'libration': hilda_record(np.radians(40) * np.sin(TWO_PI * t / P_LIB), t),
        'circulation': hilda_record(TWO_PI * t / P_CIRC, t),
        'one_episode': hilda_record(windows_sigma(t, [(6000, 14000)], 40), t),
        'two_episodes': hilda_record(windows_sigma(t, [(1800, 6800), (11300, 16300)], 40), t),
        'short_episode_70': hilda_record(windows_sigma(t, [(8100, 9400)], 70), t),
        'near_separatrix': hilda_record(pendulum_near_separatrix(t), t),
        # dips below the separatrix for ~5 cycles every 3 kyr: short librating episodes
        'pendulum_episodes': hilda_record(pendulum_near_separatrix(t, 1.0, 0.15, 3000.0, 0.01), t),
        'too_short': hilda_record(np.pi + np.radians(40) * np.sin(TWO_PI * short_t / 5000.0), short_t),
        'drift': hilda_record(np.pi + 1.5 * np.sin(TWO_PI * t / 80000.0) + np.radians(20) * np.sin(TWO_PI * t / P_LIB), t),
        'unphysical': hilda_record(np.pi + 0.5 * np.sin(TWO_PI * t / P_LIB), t, ecc=np.where(t > 10000, 1.2, 0.15)),
    }


# what a physicist expects of each record (status by construction)
EXPECTED_STATUS = {
    'libration': 'resonant',
    'circulation': 'non_resonant',
    'one_episode': 'transient',
    'two_episodes': 'transient',
    'short_episode_70': 'transient',
    # W20 (closure by phase return): every dip of the energy below the separatrix (0.97 of it, every 1.5 kyr) is one
    # cycle seen through two turning points, closed by its return: episodes, so the record is transient, not pattern A
    'near_separatrix': 'transient',
    'pendulum_episodes': 'transient',
    'too_short': 'uncertain',
    'drift': 'uncertain',
    'unphysical': 'invalid',
}

# the status as a word opening its sentence and on the plaque
STATUS_WORD = {
    'resonant': 'Resonant',
    'transient': 'Transient',
    'near_separatrix': 'Near-separatrix',
    'non_resonant': 'Non-resonant',
    'uncertain': 'Uncertain',
    'invalid': 'Invalid',
}


@pytest.fixture(scope='module')
def explained():
    out = {}
    for name, record in _records().items():
        analysis = analyse_mmr(record, SETTINGS)
        episodes = analysis.episode_rows()
        text = explain_mmr(analysis, SETTINGS, {**analysis.summary_row(), **analysis.numbers_row()}, episodes)
        out[name] = (analysis, episodes, text)
    return out


def all_text(text):
    parts = list(text.sentences) + [text.badge, text.subtitle, text.arithmetic] + list(text.info)
    parts += [f'{s.title} {s.outcome}' for s in text.steps]
    parts += [f'{c.name} {c.detail} {c.need} {c.note}' for c in text.checks]
    return '\n'.join(parts)


def accepted(episodes):
    return [row for row in episodes if str(row['accepted']).lower() in ('true', '1')]


def kyr(value):
    """`years` as printed for a time ≥ 1 kyr: three significant digits."""
    return f'{float(f"{value / 1000:.3g}"):g} kyr'


@pytest.mark.parametrize('case', sorted(EXPECTED_STATUS))
def test_synthetic_status(explained, case):
    analysis, _, _ = explained[case]
    assert str(analysis.verdict.status) == EXPECTED_STATUS[case]


@pytest.mark.parametrize('case', sorted(EXPECTED_STATUS))
def test_status_words_printed_once(explained, case):
    analysis, _, text = explained[case]
    status = str(analysis.verdict.status)
    head = STATUS_WORD[status].upper()
    assert text.badge.startswith(head)
    assert text.badge.count(head) == 1
    # the plaque is the only place of the status in capitals
    assert head not in '\n'.join(text.sentences + [text.subtitle])
    # 'Status: …' opens the why-sentence
    assert any(s.startswith(STATUS_WORD[status] + ':') for s in text.sentences), text.sentences
    level = str(analysis.verdict.confidence)
    if status in ('uncertain', 'invalid'):
        assert level == ''
        assert not any(word in text.badge.replace(head, '', 1) for word in ('CERTAIN', 'HIGH', 'MEDIUM', 'LOW'))
    else:
        assert text.badge.endswith('· ' + level.upper())
        assert any(s.startswith(level.capitalize()) for s in text.sentences)


@pytest.mark.parametrize('case', sorted(EXPECTED_STATUS))
def test_whole_record_only_for_libration(explained, case):
    analysis, _, text = explained[case]
    says = any('librates over the whole record' in s for s in text.sentences)
    assert says == (str(analysis.verdict.angle_status) == 'libration')


def test_one_episode_covering_the_record_never_says_episode(explained):
    analysis, episodes, text = explained['libration']
    rows = accepted(episodes)
    assert len(rows) == 1
    assert 'episode' not in all_text(text).lower()
    # TPs at 62.5 + 125 k yr inside the window [P_c, 20000 - P_c], P_c = 150 / 2 = 75 yr: 187.5 ... 19812.5, i.e.
    # (19812.5 - 187.5) / 125 + 1 = 158, by construction; the text prints the episode row's numbers
    assert int(rows[0]['n_tp']) == 158 and float(rows[0]['period_med_yr']) == pytest.approx(P_LIB, abs=1.0)
    assert '158 turning points, period 250 yr' in text.sentences[0]


def test_one_episode_covering_the_record_edge_gap(explained):
    _, episodes, text = explained['libration']
    row = accepted(episodes)[0]
    gap = max(float(row['t_start_yr']), SPAN - float(row['t_end_yr'])) / float(row['period_med_yr'])
    detail = next(c.detail for c in text.checks if 'covers the record' in c.name)
    assert re.search(r'larger edge gap ([\d.]+) ', detail).group(1) == f'{gap:.3g}'


def test_single_episode_not_covering(explained):
    _, episodes, text = explained['one_episode']
    rows = accepted(episodes)
    assert len(rows) == 1
    row = rows[0]
    t0, t1, period = float(row['t_start_yr']), float(row['t_end_yr']), float(row['period_med_yr'])
    sentence = text.sentences[0]
    assert 'only' in sentence and 'episode' not in '\n'.join(text.sentences).lower()
    assert f'({round(100 * (t1 - t0) / SPAN)}% of the record)' in sentence
    # edge gaps in libration periods: start t0/P, end (span − t1)/P
    assert f'misses the start by {t0 / period:.2g} and the end by {(SPAN - t1) / period:.2g}' in sentence
    assert f'from {kyr(t1)} to the end σ circulates' in sentence  # plain words, not the enum (W20 round F)
    assert text.ends == 'ends: circulation' and 'ends:' not in text.badge


def test_single_episode_from_x_to_y(explained):
    _, episodes, text = explained['one_episode']
    row = accepted(episodes)[0]
    assert f'librates from {kyr(float(row["t_start_yr"]))} to {kyr(float(row["t_end_yr"]))} only' in text.sentences[0]


def test_two_episodes_are_counted(explained):
    _, episodes, text = explained['two_episodes']
    rows = accepted(episodes)
    assert len(rows) == 2
    covered = sum(float(r['t_end_yr']) - float(r['t_start_yr']) for r in rows) / SPAN
    assert f'2 separate episodes covering {round(100 * covered)}% of the record' in text.sentences[0]


def cycle_claims(text):
    """(duration_yr, cycles, P_cyc_yr) of every 'lasts X = N cycles … (cycles of P_cyc Y' in the text."""
    pattern = r'lasts ([\d.]+) (yr|kyr) = ([\d.]+) cycles .*?cycles of P_cyc ([\d.]+) (yr|kyr)'
    out = []
    for m in re.finditer(pattern, '\n'.join(text.sentences)):
        scale = {'yr': 1.0, 'kyr': 1000.0}
        out.append((float(m.group(1)) * scale[m.group(2)], float(m.group(3)), float(m.group(4)) * scale[m.group(5)]))
    return out


@pytest.mark.parametrize('case', ['pendulum_episodes', 'short_episode_70'])
def test_cycles_equal_duration_over_period(explained, case):
    analysis, episodes, text = explained[case]
    claims = cycle_claims(text)
    assert claims, text.sentences
    longest = max(float(r['t_end_yr']) - float(r['t_start_yr']) for r in accepted(episodes))
    period = float(accepted(episodes)[0]['period_med_yr'])
    for duration, cycles, p_cyc in claims:
        assert duration == pytest.approx(longest, rel=6e-3)  # printed with 3 significant digits
        assert p_cyc == pytest.approx(period, rel=6e-3)
        assert cycles == pytest.approx(longest / p_cyc, abs=0.051 if cycles != round(cycles) else 0.5)


@pytest.mark.parametrize(
    'case',
    [
        'pendulum_episodes',
        'short_episode_70',
    ],
)
def test_printed_cycle_comparison_holds(explained, case):
    _, _, text = explained[case]
    found = re.findall(r'([\d.]+) cycles < ([\d.]+)', '\n'.join(text.sentences))
    assert found
    for left, right in found:
        assert float(left) < float(right), f'{left} cycles < {right}'


def test_synthetic_libration_orbit_type(explained):
    analysis, _, _ = explained['libration']
    # Hilda-like 3J-2: interior, σ about 0° by construction
    assert analysis.verdict.orbit_type == 'interior, symmetric libration about 0°'
    assert analysis.summary_row()['orbit_type'] == analysis.verdict.orbit_type


def test_synthetic_no_episode_no_orbit_type(explained):
    for case in ('circulation',):
        assert explained[case][0].verdict.orbit_type == ''


def test_summary_cell_columns_match_cells_file(explained):
    analysis, _, _ = explained['libration']
    row = analysis.summary_row()
    record = load_cells().records.get((row['status'], row['angle_status'], row['reason']))
    for column in ('cell_n_examples', 'cell_n_errors', 'cell_accuracy_low_95', 'cell_accuracy_high_95'):
        assert column in row
    if record is not None:
        assert int(row['cell_n_examples']) == record.n_examples
        assert int(row['cell_n_errors']) == record.n_errors


# ------------------------------------------------------------------------- secular (explain_long_period)


@pytest.fixture(scope='module')
def secular_explained():
    from resonances.resonance.analysis.secular.analyse import analyse_secular
    from resonances.resonance.analysis.secular.models import LongPeriodSeries

    t = np.arange(0.0, 10e6 + 50.0, 100.0)
    ones = np.ones_like(t)
    angles = {
        'libration': np.pi + np.radians(50) * np.sin(TWO_PI * t / 1.0e6),
        'circulation': TWO_PI * t / 0.7e6,
    }
    out = {}
    for name, angle in angles.items():
        series = LongPeriodSeries(ResonanceType.SECULAR, 'g-g6', f'syn_{name}', t, angle, 2.3 * ones, 0.1 * ones, 0.1 * ones)
        analysis = analyse_secular(series, SETTINGS)
        episodes = analysis.episode_rows()
        out[name] = (analysis, explain_long_period(analysis, SETTINGS, episodes))
    return out


def test_secular_libration_text(secular_explained):
    analysis, text = secular_explained['libration']
    assert str(analysis.verdict.status) == 'resonant'
    assert text.badge.startswith('RESONANT') and text.badge.count('RESONANT') == 1
    assert any('over the whole record' in s for s in text.sentences)
    assert 'episode' not in '\n'.join(text.sentences).lower()
    assert analysis.verdict.orbit_type == 'σ libration about 180°'


def test_secular_circulation_text(secular_explained):
    analysis, text = secular_explained['circulation']
    assert str(analysis.verdict.status) == 'non_resonant'
    assert text.badge.startswith('NON-RESONANT')
    assert not any('over the whole record' in s for s in text.sentences)
    assert any(s.startswith('Non-resonant') for s in text.sentences)
