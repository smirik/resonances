"""W8 confidence package: the level shown without a minimum of labels (author 2026-10-16, C2), the orbit type of the
figure's header, the subresonance degree from the catalogue's order (C14).

Independent truths: the classical libration centres (Hildas 3:2 about 0°, Plutinos 2:3 about 180°, Murray & Dermott
1999, §8.7 and fig. 8.13; Twotinos 1:2 asymmetric at about ±75°, Chiang & Jordan 2002, AJ 124, 3430), the co-orbital
regimes of theta = lambda - lambda_P (tadpoles about ±60°, horseshoes over 180°, quasi-satellites about 0°; M&D §3.9),
and known objects: 121514 (1999 UJ7) is the L4 Mars Trojan (Scholl, Marzari & Tricarico 2005, Icarus 175, 397),
2002 AA29 a horseshoe that becomes a quasi-satellite around 2600 AD (Connors et al. 2002, MAPS 37, 1435)."""

import numpy as np
import pytest

from resonances.resonance.analysis import orbit_type as ot
from resonances.resonance.analysis.mmr import coorbital, subresonances as sr
from resonances.resonance.analysis.mmr.models import ResonanceSpec
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, ResonanceType, Status, Verdict
from tests.resonances.analysis.builders import make_episode, make_geometry

S = settings_from_config()


def run(episode_id, centre_deg, start=0.0, end=1000.0, shift=0.1, accepted=True, tp_deg=None):
    tp = np.radians(tp_deg if tp_deg is not None else [centre_deg - 20, centre_deg + 20, centre_deg - 20])
    return make_episode(
        episode_id=episode_id,
        accepted=accepted,
        centre_mod_2pi_rad=np.radians(centre_deg % 360),
        centre_shift_over_amp=shift,
        tp_times_yr=np.linspace(start, end, len(tp)),
        tp_values_rad=tp,
    )


def geometry(*runs):
    return make_geometry(runs=list(runs))


# --- co-orbital regimes ---------------------------------------------------------------------------------


def _oscillation(centre_deg, half_amp_deg, n=400):
    return np.radians(centre_deg + half_amp_deg * np.sin(np.linspace(0, 6 * np.pi, n)))


@pytest.mark.parametrize(
    'centre, amp, expected',
    [
        (60, 20, coorbital.REGIME_TADPOLE_L4),
        (-60, 20, coorbital.REGIME_TADPOLE_L5),
        (300, 20, coorbital.REGIME_TADPOLE_L5),
        (180, 150, coorbital.REGIME_HORSESHOE),
        (0, 40, coorbital.REGIME_QUASI_SATELLITE),
        (90, 120, coorbital.REGIME_COMPOUND),  # 0..210 deg passes both 0 and 180
    ],
)
def test_piece_regimes_of_one_piece(centre, amp, expected):
    theta = _oscillation(centre, amp)
    assert coorbital.piece_regimes(theta, np.zeros(len(theta), dtype=int)) == (expected,)


def test_piece_regimes_in_time_order_with_repeats_merged():
    hs, qs = _oscillation(180, 150), _oscillation(0, 40)
    theta = np.concatenate([hs, hs, qs])
    labels = np.repeat([0, 1, 2], len(hs))
    assert coorbital.piece_regimes(theta, labels) == (coorbital.REGIME_HORSESHOE, coorbital.REGIME_QUASI_SATELLITE)
    # the column keeps its old meaning: pieces of different regimes are compound
    assert coorbital.regime(theta, labels)[0] == coorbital.REGIME_COMPOUND


@pytest.mark.parametrize(
    'sequence, text',
    [
        (('tadpole_L4',), 'tadpole L4'),
        (('horseshoe', 'quasi_satellite'), 'compound: horseshoe → quasi-satellite'),
        (('compound',), 'compound (horseshoe ↔ quasi-satellite)'),
        ((), ''),
    ],
)
def test_coorbital_text(sequence, text):
    assert ot.coorbital_text(sequence) == text


def test_coorbital_without_a_law_reads_the_turning_points():
    # 1J-1: sigma = lambda_J - lambda, theta = -sigma; sigma TPs at 50..70 deg -> theta about -60: L5 (as 617 Patroclus)
    spec = ResonanceSpec.from_key('1J-1+0+0')
    trojan = run(0, 60, tp_deg=[50, 70, 50, 70])
    assert ot.mmr_orbit_type(spec, geometry(trojan), 0, None, S) == 'tadpole L5'


# --- other MMR angles -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    'key, centres, expected',
    [
        ('3J-2+0-1', [2], 'interior, symmetric libration about 0°'),  # Hilda
        ('2N-3+0+1', [178], 'exterior, symmetric libration about 180°'),  # Plutino
        ('1N-2+0+1', [75, 285], 'exterior, asymmetric libration about 75° and 285°'),  # Twotino, both lobes seen
        ('1N-2+0+1', [75], 'exterior, asymmetric libration about 75°'),
        ('4J-2S-1+0+0-1', [180], 'symmetric libration about 180°'),  # three-body: no side
    ],
)
def test_mmr_orbit_type_by_the_centre(key, centres, expected):
    runs = [run(i, c, start=100.0 * i, end=100.0 * i + 50) for i, c in enumerate(centres)]
    assert ot.mmr_orbit_type(ResonanceSpec.from_key(key), geometry(*runs), 0, None, S) == expected


def test_the_decisive_episode_is_described_and_a_moving_centre_said():
    spec = ResonanceSpec.from_key('4M-7+0+3')
    long, short = run(0, 100, end=5000.0), run(1, 3, start=6000.0, end=6500.0, shift=S.mmr_l6_shift_full * 2)
    assert ot.mmr_orbit_type(spec, geometry(long, short), 1, None, S) == 'exterior, symmetric libration about 0°, moving centre'
    # without a decisive episode the longest accepted one
    assert ot.mmr_orbit_type(spec, geometry(long, short), None, None, S) == 'exterior, asymmetric libration about 100°'


@pytest.mark.parametrize(
    'word, expected',
    [
        ('episodic', 'exterior, asymmetric libration about 100°'),
        ('libration', 'exterior, asymmetric libration about 100°'),
        ('circulation', 'exterior'),  # W20 round C: no libration named in a circulation's header
        ('drift', 'exterior'),
    ],
)
def test_a_word_without_libration_names_no_libration(word, expected):
    spec = ResonanceSpec.from_key('4M-7+0+3')
    assert ot.mmr_orbit_type(spec, geometry(run(0, 100, end=5000.0)), 0, None, S, word) == expected
    assert ot.mmr_orbit_type(ResonanceSpec.from_key('4J-2S-1+0+0-1'), geometry(run(0, 180)), 0, None, S, 'drift') == ''


def test_no_accepted_episode_no_type():
    spec = ResonanceSpec.from_key('3J-1+0-2')
    assert ot.mmr_orbit_type(spec, geometry(run(0, 0, accepted=False)), 0, None, S) == ''
    assert ot.mmr_orbit_type(spec, None, None, None, S) == ''


def test_retrograde_marker():
    assert ot.mmr_orbit_type(ResonanceSpec.from_key('1J-3r0'), geometry(run(0, 180)), 0, None, S).startswith('retrograde, ')
    assert ot.mmr_orbit_type(ResonanceSpec.from_key('1J-1r0'), geometry(run(0, 0)), 0, None, S) == (
        'retrograde co-orbital, libration about 0°'
    )


@pytest.mark.parametrize(
    'rtype, centre, expected',
    [(ResonanceType.LIDOV_KOZAI, 92, 'ω libration about 92°'), (ResonanceType.SECULAR, 181, 'σ libration about 181°')],
)
def test_long_period_orbit_type(rtype, centre, expected):
    assert ot.long_period_orbit_type(rtype, geometry(run(0, centre)), 0) == expected


# --- the verdict's columns --------------------------------------------------------------------------------


def test_verdict_round_trip_of_orbit_type_and_cell():
    v = Verdict(
        resonance_type=ResonanceType.MMR,
        status=Status.RESONANT,
        angle_status=AngleStatus.LIBRATION,
        orbit_type='tadpole L4',
        cell_n_examples=24,
        cell_n_errors=0,
        cell_accuracy_low_95=0.862,
        cell_accuracy_high_95=1.0,
    )
    row = v.to_flat_dict()
    assert (row['orbit_type'], row['cell_n_examples'], row['cell_n_errors']) == ('tadpole L4', 24, 0)
    back = Verdict.from_flat_dict({k: ('' if value is None else str(value)) for k, value in row.items()})
    assert (back.orbit_type, back.cell_n_examples, back.cell_n_errors, back.cell_accuracy_low_95) == ('tadpole L4', 24, 0, 0.862)


# --- C14: the subresonance degree from the catalogue ------------------------------------------------------


@pytest.mark.parametrize(
    'key, expected',
    [
        ('3J-1S-1+0+0-1', 6),  # three-body: MATRIX_3BODY_ORDER_MAX
        ('4M-7+0+3', 10),  # two-body: MATRIX_2BODY_ORDER_MAX
        ('13M-24+0+11', 11),  # beyond the catalogue: its own order
        ('1E-10S-1+0+0+10', 10),
    ],
)
def test_degree_max_is_the_catalogue_order(key, expected):
    assert S.matrix_2body_order_max == 10 and S.matrix_3body_order_max == 6
    assert sr.degree_max(ResonanceSpec.from_key(key), S) == expected


def test_the_24820_member_is_scanned_by_the_decision():
    # C14: 3J-1S-1+2-2-1 (degree 5) is now a decision member of 3J-1S-1+0+0-1, not only listed
    spec = ResonanceSpec.from_key('3J-1S-1+0+0-1')
    keys = [m.key for m, _, _ in sr.enumerate_subresonances(spec, sr.degree_max(spec, S))]
    assert '3J-1S-1+2-2-1' in keys and max(sr.member_degree(ResonanceSpec.from_key(k)) for k in keys) == 5


# --- real records -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    'example_id, expected',
    [
        ('coorbital__121514__1M-1+0+0__20kyr', 'tadpole L4'),
        ('coorbital__2002_AA29__1E-1+0+0', 'compound (horseshoe ↔ quasi-satellite)'),
    ],
)
def test_real_coorbitals(example_id, expected):
    from resonances.resonance.analysis.mmr import io
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    analysis = analyse_mmr(io.load_series(example_id), S)
    assert analysis.verdict.orbit_type == expected
    assert analysis.summary_row()['orbit_type'] == expected
    # the cell of resonant/libration/dynamic_libration is reported from the package file
    assert analysis.verdict.cell_n_examples is not None and analysis.verdict.cell_n_examples > 0
