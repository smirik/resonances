"""Independent tests of W8 confidence: the level without a calibration gate, the calibration columns, the orbit type,
the subresonance degree from the catalogue, and the explanation text of the right column.

Expected values are derived here, not read from the code: the level rule and the caps from docs/libration-analysis.md
§7, the Wilson interval from its formula (Wilson 1927), the orbit types from the regimes of synthetic angles, the
member counts by brute force over the d'Alembert rule, and the explanation from synthetic pendulum records whose
episodes and periods are known by construction.
"""

import dataclasses
import hashlib
import itertools
import json
import math
import re

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from resonances.resonance.analysis import confidence as conf
from resonances.resonance.analysis import orbit_type as ot
from resonances.resonance.analysis.geometry import Episode, EpisodeShape, Geometry, RejectReason
from resonances.resonance.analysis.ledger import Margin
from resonances.resonance.analysis.mmr import coorbital
from resonances.resonance.analysis.mmr import subresonances as sub
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.models import MMRSeries, PlanetSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    CheckValue,
    Confidence,
    Flag,
    MarginClass,
    MarginValue,
    ResonanceType,
    Status,
    Verdict,
)
from resonances.plotting.verdict_text import MMRFacts, explain_mmr, force_windows_text

TWO_PI = 2 * np.pi
Z95 = 1.959963984540054


def wilson(successes, n, z=Z95):
    """Wilson score interval (Wilson 1927), written out here."""
    p = successes / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    denominator = 1 + z * z / n
    return (centre - half) / denominator, (centre + half) / denominator


@pytest.fixture(scope='module')
def settings():
    return AnalysisSettings()


# ---------------------------------------------------------------------------------------------------------------
# 1. Confidence: synthetic margins and caps -> level, no calibration gate
# ---------------------------------------------------------------------------------------------------------------

PHYSICS = ('L4', 'R0', 'Z')


def margins(**states):
    """A full set of MMR margins, each FULL unless overridden: name=(Margin, MarginClass)."""
    base = {
        'L1': (Margin.FULL, MarginClass.DECISIVE),
        'L2': (Margin.FULL, MarginClass.DECISIVE),
        'L3': (Margin.FULL, MarginClass.SUPPORT),
        'L4': (Margin.FULL, MarginClass.DECISIVE),
        'L5': (Margin.FULL, MarginClass.SUPPORT),
        'L6': (Margin.FULL, MarginClass.DECISIVE),
        'R0': (Margin.FULL, MarginClass.DECISIVE),
        'Z': (Margin.FULL, MarginClass.DECISIVE),
        'F': (Margin.FULL, MarginClass.SUPPORT),
    }
    base.update(states)
    return {k: MarginValue(margin=m, value=1.0, margin_class=c) for k, (m, c) in base.items()}


class TestRawLevel:
    def test_every_margin_full_is_certain(self):
        level, thin, failed, none_physics = conf.raw_level(margins(), PHYSICS)
        assert level == Confidence.CERTAIN
        assert thin == [] and failed == [] and none_physics is False

    def test_one_thin_is_high(self):
        level, thin, failed, _ = conf.raw_level(margins(L2=(Margin.THIN, MarginClass.DECISIVE)), PHYSICS)
        assert level == Confidence.HIGH
        assert 'L2' in thin and failed == []

    def test_unmeasured_decisive_counts_as_thin(self):
        level, _, failed, _ = conf.raw_level(margins(L4=(Margin.NOT_MEASURED, MarginClass.DECISIVE)), PHYSICS)
        assert level == Confidence.HIGH
        assert failed == []

    def test_not_applicable_does_not_lower(self):
        level, _, _, _ = conf.raw_level(margins(L6=(Margin.NOT_APPLICABLE, MarginClass.DECISIVE)), PHYSICS)
        assert level == Confidence.CERTAIN

    def test_one_failed_support_is_medium(self):
        level, _, failed, _ = conf.raw_level(margins(F=(Margin.FAIL, MarginClass.SUPPORT)), PHYSICS)
        assert level == Confidence.MEDIUM
        assert failed == ['F']

    def test_two_failed_supports_are_low(self):
        m = margins(F=(Margin.FAIL, MarginClass.SUPPORT), L5=(Margin.FAIL, MarginClass.SUPPORT))
        level, _, failed, _ = conf.raw_level(m, PHYSICS)
        assert level == Confidence.LOW
        assert sorted(failed) == ['F', 'L5']

    def test_failed_cap_margin_is_left_to_its_cap(self):
        m = margins(O1=(Margin.FAIL, MarginClass.CAP))
        level, _, failed, _ = conf.raw_level(m, PHYSICS)
        assert level == Confidence.CERTAIN
        assert 'O1' not in failed

    def test_no_physics_measured_is_flagged(self):
        m = margins(**{k: (Margin.NOT_MEASURED, MarginClass.DECISIVE) for k in PHYSICS})
        assert conf.raw_level(m, PHYSICS)[3] is True


class TestCaps:
    @pytest.mark.parametrize(
        'level, caps, expected',
        [
            (Confidence.CERTAIN, [], Confidence.CERTAIN),
            (Confidence.CERTAIN, ['three_tp_episode'], Confidence.HIGH),
            (Confidence.HIGH, ['three_tp_episode'], Confidence.HIGH),
            (Confidence.CERTAIN, ['exit_suspected'], Confidence.MEDIUM),
            (Confidence.CERTAIN, ['force_law_disagrees'], Confidence.MEDIUM),
            (Confidence.HIGH, ['l2_without_zone'], Confidence.MEDIUM),
            (Confidence.CERTAIN, ['onset_failed'], Confidence.LOW),
            (Confidence.CERTAIN, ['three_tp_episode', 'onset_failed', 'exit_suspected'], Confidence.LOW),
            (Confidence.LOW, ['three_tp_episode'], Confidence.LOW),
            (Confidence.NONE, ['onset_failed'], Confidence.NONE),
        ],
    )
    def test_apply_caps(self, level, caps, expected):
        assert conf.apply_caps(level, caps) == expected


def empty_cell_file(tmp_path, status='resonant', angle='libration', reason='dynamic_libration', n=0, err=0):
    cell = {'status': status, 'angle_status': angle, 'reason': reason, 'n_examples': n, 'n_errors': err}
    if n:
        cell['accuracy_95'] = list(wilson(n - err, n))
    path = tmp_path / 'cells.json'
    path.write_text(json.dumps({'version': 1, 'cells': [cell]}))
    return path


class TestAssessWithoutGate:
    def test_certain_shown_with_no_labels_in_the_cell(self, tmp_path):
        cells = conf.load_cells(empty_cell_file(tmp_path, n=0))
        a = conf.assess(Status.RESONANT, 'libration', 'dynamic_libration', margins(), [], [], PHYSICS, cells)
        assert a.confidence == Confidence.CERTAIN
        assert a.confidence_raw == Confidence.CERTAIN
        assert not hasattr(a, 'flags')

    def test_certain_shown_with_an_empty_cells_file(self, tmp_path):
        cells = conf.load_cells(tmp_path / 'missing.json')
        assert len(cells.records) == 0
        a = conf.assess(Status.NON_RESONANT, 'circulation', 'circulation', margins(), [], [], PHYSICS, cells)
        assert a.confidence == Confidence.CERTAIN

    def test_high_shown_with_errors_in_the_cell(self, tmp_path):
        # 2 errors in 4: the old gate (max errors) would have withheld 'high'; the level must not care.
        cells = conf.load_cells(empty_cell_file(tmp_path, 'transient', 'episodic', 'dynamic_episode', 4, 2))
        m = margins(L2=(Margin.THIN, MarginClass.DECISIVE))
        a = conf.assess(Status.TRANSIENT, 'episodic', 'dynamic_episode', m, [], [], PHYSICS, cells)
        assert a.confidence == Confidence.HIGH
        assert a.cell is not None and a.cell.n_examples == 4 and a.cell.n_errors == 2
        low, high = wilson(2, 4)
        assert a.cell.accuracy_low_95 == pytest.approx(low, abs=1e-9)
        assert a.cell.accuracy_high_95 == pytest.approx(high, abs=1e-9)

    def test_ledger_thin_item_gives_high(self, tmp_path):
        cells = conf.load_cells(empty_cell_file(tmp_path))
        a = conf.assess(Status.RESONANT, 'libration', 'dynamic_libration', margins(), [], ['U:some_threshold'], PHYSICS, cells)
        assert a.confidence == Confidence.HIGH
        assert 'U:some_threshold' in a.thin_margins

    def test_caps_lower_the_shown_level_only(self, tmp_path):
        cells = conf.load_cells(empty_cell_file(tmp_path))
        a = conf.assess(Status.RESONANT, 'libration', 'dynamic_libration', margins(), ['exit_suspected'], [], PHYSICS, cells)
        assert a.confidence_raw == Confidence.CERTAIN
        assert a.confidence == Confidence.MEDIUM

    def test_no_physics_measured_at_most_medium(self, tmp_path):
        cells = conf.load_cells(empty_cell_file(tmp_path))
        m = margins(**{k: (Margin.NOT_MEASURED, MarginClass.DECISIVE) for k in PHYSICS})
        a = conf.assess(Status.RESONANT, 'libration', 'dynamic_libration', m, [], [], PHYSICS, cells)
        assert a.confidence in (Confidence.MEDIUM, Confidence.LOW)

    @pytest.mark.parametrize('status', [Status.UNCERTAIN, Status.INVALID])
    def test_uncertain_and_invalid_have_no_level_and_no_cell(self, tmp_path, status):
        cells = conf.load_cells(empty_cell_file(tmp_path, status=str(status), angle='drift', reason='x', n=10))
        a = conf.assess(status, 'drift', 'x', margins(), [], [], PHYSICS, cells)
        assert a.confidence == Confidence.NONE and a.confidence_raw == Confidence.NONE
        assert a.cell is None
        assert a.cell_fields() == {}

    def test_cell_fields_are_the_verdict_columns(self, tmp_path):
        cells = conf.load_cells(empty_cell_file(tmp_path, n=24, err=0))
        a = conf.assess(Status.RESONANT, 'libration', 'dynamic_libration', margins(), [], [], PHYSICS, cells)
        f = a.cell_fields()
        assert set(f) == {'cell_n_examples', 'cell_n_errors', 'cell_accuracy_low_95', 'cell_accuracy_high_95'}
        assert f['cell_n_examples'] == 24 and f['cell_n_errors'] == 0
        assert f['cell_accuracy_low_95'] == pytest.approx(0.8620, abs=1e-4)  # Wilson by hand: 24/24 -> 1/(1 + z^2/24)
        assert f['cell_accuracy_high_95'] == pytest.approx(1.0, abs=1e-9)


class TestCellsFile:
    def test_sha_is_that_of_the_bytes(self, tmp_path):
        path = empty_cell_file(tmp_path, n=5, err=1)
        assert conf.load_cells(path).sha == hashlib.sha256(path.read_bytes()).hexdigest()

    def test_package_cells_wilson_intervals(self):
        raw = json.loads(conf.PACKAGE_CELLS.read_text())
        loaded = conf.load_cells(None)
        assert loaded.sha == hashlib.sha256(conf.PACKAGE_CELLS.read_bytes()).hexdigest()
        assert len(loaded.records) == len(raw['cells'])
        for cell in raw['cells']:
            n, err = cell['n_examples'], cell['n_errors']
            rec = loaded.records[(cell['status'], cell['angle_status'], cell['reason'])]
            assert (rec.n_examples, rec.n_errors) == (n, err)
            if n:
                low, high = wilson(n - err, n)
                assert rec.accuracy_low_95 == pytest.approx(low, abs=6e-4), cell
                assert rec.accuracy_high_95 == pytest.approx(high, abs=6e-4), cell

    def test_package_cells_have_no_threshold_fields(self):
        raw = json.loads(conf.PACKAGE_CELLS.read_text())
        assert 'min_examples' not in raw and 'high_cells' not in raw


class TestRemovedGate:
    def test_pending_flags_gone(self):
        values = {str(f) for f in Flag}
        assert 'certain_pending_calibration' not in values
        assert 'high_pending_calibration' not in values

    @pytest.mark.parametrize(
        'name',
        [
            'confidence_certain_min_examples',
            'confidence_high_min_examples',
            'confidence_high_max_errors',
            'mmr_subresonance_order_extra',
            'mmr_subresonance_list_order_extra',
        ],
    )
    def test_removed_settings(self, settings, name):
        assert not hasattr(settings, name)

    def test_removed_env_keys(self):
        text = (conf.PACKAGE_CELLS.parents[1] / '.env.dist').read_text()
        keys = {line.split('=', 1)[0].strip() for line in text.splitlines() if '=' in line and not line.startswith('#')}
        for key in (
            'CONFIDENCE_CERTAIN_MIN_EXAMPLES',
            'CONFIDENCE_HIGH_MIN_EXAMPLES',
            'CONFIDENCE_HIGH_MAX_ERRORS',
            'MMR_SUBRESONANCE_ORDER_EXTRA',
            'MMR_SUBRESONANCE_LIST_ORDER_EXTRA',
        ):
            assert key not in keys


class TestVerdictColumns:
    def make(self):
        return Verdict(
            resonance_type=ResonanceType.MMR,
            status=Status.RESONANT,
            angle_status=AngleStatus.LIBRATION,
            confidence=Confidence.CERTAIN,
            confidence_raw=Confidence.CERTAIN,
            reason='dynamic_libration',
            orbit_type='tadpole L4',
            cell_n_examples=24,
            cell_n_errors=0,
            cell_accuracy_low_95=0.862,
            cell_accuracy_high_95=1.0,
        )

    def test_columns_and_order(self):
        keys = list(self.make().to_flat_dict())
        assert keys.index('orbit_type') == keys.index('final_regime_start_yr') + 1
        i = keys.index('confidence_raw')
        assert keys[i + 1 : i + 5] == [
            'cell_n_examples',
            'cell_n_errors',
            'cell_accuracy_low_95',
            'cell_accuracy_high_95',
        ]

    def test_round_trip(self):
        v = self.make()
        back = Verdict.from_flat_dict(v.to_flat_dict())
        assert back.orbit_type == 'tadpole L4'
        assert back.confidence == Confidence.CERTAIN
        assert (back.cell_n_examples, back.cell_n_errors) == (24, 0)
        assert back.cell_accuracy_low_95 == pytest.approx(0.862)
        assert back.cell_accuracy_high_95 == pytest.approx(1.0)

    def test_empty_cell_columns(self):
        v = dataclasses.replace(
            self.make(), cell_n_examples=None, cell_n_errors=None, cell_accuracy_low_95=None, cell_accuracy_high_95=None
        )
        d = v.to_flat_dict()
        assert d['cell_n_examples'] is None and d['cell_accuracy_low_95'] is None


# ---------------------------------------------------------------------------------------------------------------
# 2. Orbit type
# ---------------------------------------------------------------------------------------------------------------


def episode(eid, centre_deg, amp_deg=40.0, t0=0.0, n_tp=9, period=300.0, shift_over_amp=0.0, accepted=True):
    times = t0 + np.arange(n_tp) * period / 2
    signs = np.where(np.arange(n_tp) % 2 == 0, 1.0, -1.0)
    values = np.radians(centre_deg + signs * amp_deg)
    amp = math.radians(amp_deg)
    return Episode(
        episode_id=eid,
        tp_times_yr=times,
        tp_values_rad=values,
        accepted=accepted,
        shape=EpisodeShape.STATIONARY if accepted else EpisodeShape.NONE,
        reject_reason=RejectReason.NONE if accepted else RejectReason.CENTRE_MOVES,
        period_med_yr=period,
        period_min_yr=period,
        period_max_yr=period,
        half_amp_min_rad=amp,
        half_amp_med_rad=amp,
        half_amp_max_rad=amp,
        centre_mod_2pi_rad=math.radians(centre_deg) % TWO_PI,
        centre_shift_rad=shift_over_amp * amp,
        centre_shift_over_amp=shift_over_amp,
        bands_overlap=shift_over_amp < 1,
        centre_returns=0,
        coverage_start_gap_p=0.1,
        coverage_end_gap_p=0.1,
    )


def geometry(runs, span=20000.0):
    return Geometry(
        window=(0.0, span),
        regime_start_yr=0.0,
        turning_points=None,
        n_tp_cut_half=None,
        n_tp_cut_double=None,
        period_cycle_yr=300.0,
        period_min_yr=300.0,
        period_max_yr=300.0,
        centre=None,
        net_rev_filtered=0.0,
        net_rev_raw=0.0,
        band_first_rad=0.0,
        band_last_rad=0.0,
        bands_overlap=True,
        wiggle_half_amp_rad=0.0,
        wiggle_period_yr=0.0,
        runs=list(runs),
    )


def law(sequence):
    return coorbital.CoorbitalLaw(
        planet='Earth',
        angle_sign=1,
        n_tables=1,
        n_blocks=1,
        r2=0.9,
        r2_h1=0.9,
        r2_h2=0.9,
        scale=1.0,
        scale_h1=1.0,
        scale_h2=1.0,
        energy_r2=0.9,
        energy_scale=1.0,
        harmonic_r2=0.5,
        regime=sequence[0] if len(sequence) == 1 else coorbital.REGIME_COMPOUND,
        theta_min_deg=-170.0,
        theta_max_deg=170.0,
        force_check=CheckValue.PASS,
        force_thin=False,
        energy_check=CheckValue.PASS,
        potential_check=CheckValue.PASS,
        regime_sequence=tuple(sequence),
    )


def pieces(*specs, n=2000):
    """theta (rad, unwrapped) and labels from pieces (centre_deg, amp_deg): theta = centre + amp sin."""
    thetas, labels = [], []
    for i, (c, a) in enumerate(specs):
        x = np.linspace(0, 6 * TWO_PI, n)
        thetas.append(np.radians(c + a * np.sin(x)))
        labels.append(np.full(n, i))
    return np.concatenate(thetas), np.concatenate(labels)


class TestPieceRegimes:
    def test_each_regime(self):
        theta, labels = pieces((60, 20), (-60, 20), (180, 150), (0, 40), (180, 200))
        assert coorbital.piece_regimes(theta, labels) == (
            coorbital.REGIME_TADPOLE_L4,
            coorbital.REGIME_TADPOLE_L5,
            coorbital.REGIME_HORSESHOE,
            coorbital.REGIME_QUASI_SATELLITE,
            coorbital.REGIME_COMPOUND,
        )

    def test_neighbours_merged(self):
        theta, labels = pieces((180, 150), (180, 140), (0, 40), (0, 30))
        assert coorbital.piece_regimes(theta, labels) == (
            coorbital.REGIME_HORSESHOE,
            coorbital.REGIME_QUASI_SATELLITE,
        )

    def test_horseshoe_shifted_by_turns(self):
        # the same horseshoe two turns of theta further: the regime reads the wrapped angle
        theta, labels = pieces((180 + 720, 150))
        assert coorbital.piece_regimes(theta, labels) == (coorbital.REGIME_HORSESHOE,)


class TestCoorbitalText:
    @pytest.mark.parametrize(
        'sequence, text',
        [
            (('tadpole_L4',), 'tadpole L4'),
            (('tadpole_L5',), 'tadpole L5'),
            (('horseshoe',), 'horseshoe'),
            (('quasi_satellite',), 'quasi-satellite'),
            (('horseshoe', 'quasi_satellite'), 'compound: horseshoe → quasi-satellite'),
            (('quasi_satellite', 'horseshoe'), 'compound: quasi-satellite → horseshoe'),
            (('compound',), 'compound (horseshoe ↔ quasi-satellite)'),
        ],
    )
    def test_text(self, sequence, text):
        assert ot.coorbital_text(sequence) == text

    @pytest.mark.parametrize(
        'sequence, text',
        [
            (('tadpole_L4',), 'tadpole L4'),
            (('tadpole_L5',), 'tadpole L5'),
            (('horseshoe', 'quasi_satellite'), 'compound: horseshoe → quasi-satellite'),
            (('compound',), 'compound (horseshoe ↔ quasi-satellite)'),
        ],
    )
    def test_mmr_orbit_type_of_a_1to1(self, settings, sequence, text):
        spec = ResonanceSpec.from_key('1E-1')
        geo = geometry([episode(0, 180.0, 150.0)])
        assert ot.mmr_orbit_type(spec, geo, 0, law(sequence), settings) == text

    def test_from_the_force_law_pieces(self, settings):
        theta, labels = pieces((180, 150), (0, 40))
        sequence = coorbital.piece_regimes(theta, labels)
        geo = geometry([episode(0, 180.0, 150.0), episode(1, 0.0, 40.0, t0=10000.0)])
        assert ot.mmr_orbit_type(ResonanceSpec.from_key('1E-1'), geo, 0, law(sequence), settings) == (
            'compound: horseshoe → quasi-satellite'
        )

    def test_no_accepted_episode_gives_empty(self, settings):
        geo = geometry([episode(0, 60.0, accepted=False)])
        assert ot.mmr_orbit_type(ResonanceSpec.from_key('1E-1'), geo, None, law(('tadpole_L4',)), settings) == ''
        assert ot.mmr_orbit_type(ResonanceSpec.from_key('3J-1'), None, None, None, settings) == ''


class TestLibrationText:
    TOL = 30.0
    SHIFT = 0.5

    @pytest.mark.parametrize(
        'centre, text',
        [
            (0.0, 'symmetric libration about 0°'),
            (350.0, 'symmetric libration about 0°'),  # 10° from 0 across the wrap
            (29.0, 'symmetric libration about 0°'),
            (180.0, 'symmetric libration about 180°'),
            (151.0, 'symmetric libration about 180°'),
            (209.0, 'symmetric libration about 180°'),
            (31.0, 'asymmetric libration about 31°'),
            (72.0, 'asymmetric libration about 72°'),
            (288.0, 'asymmetric libration about 288°'),
        ],
    )
    def test_centre(self, centre, text):
        ep = episode(0, centre)
        assert ot.libration_text(ep, [ep], self.TOL, self.SHIFT) == text

    def test_mirror_centre_of_another_episode(self):
        a, b = episode(0, 72.0), episode(1, 288.0, t0=10000.0)
        assert ot.libration_text(a, [a, b], self.TOL, self.SHIFT) == 'asymmetric libration about 72° and 288°'

    def test_moving_centre(self):
        ep = episode(0, 180.0, shift_over_amp=0.8)
        assert ot.libration_text(ep, [ep], self.TOL, self.SHIFT) == 'symmetric libration about 180°, moving centre'
        still = episode(0, 180.0, shift_over_amp=0.3)
        assert ot.libration_text(still, [still], self.TOL, self.SHIFT) == 'symmetric libration about 180°'


class TestMMROrbitType:
    def test_interior_and_exterior(self, settings):
        geo = geometry([episode(0, 180.0)])
        assert ot.mmr_orbit_type(ResonanceSpec.from_key('3J-1'), geo, 0, None, settings) == ('interior, symmetric libration about 180°')
        assert ot.mmr_orbit_type(ResonanceSpec.from_key('2N-3'), geo, 0, None, settings) == ('exterior, symmetric libration about 180°')

    def test_three_body_has_no_side(self, settings):
        geo = geometry([episode(0, 0.0)])
        assert ot.mmr_orbit_type(ResonanceSpec.from_key('4J-2S-1'), geo, 0, None, settings) == ('symmetric libration about 0°')

    def test_retrograde(self, settings):
        geo = geometry([episode(0, 180.0)])
        text = ot.mmr_orbit_type(ResonanceSpec.from_key('1J-3r0'), geo, 0, None, settings)
        assert text.startswith('retrograde, ')
        assert text.endswith('symmetric libration about 180°')

    def test_decisive_or_longest_episode(self, settings):
        short = episode(0, 72.0, n_tp=5)
        long_ = episode(1, 180.0, n_tp=31, t0=5000.0)
        geo = geometry([short, long_])
        spec = ResonanceSpec.from_key('3J-1')
        assert ot.mmr_orbit_type(spec, geo, 1, None, settings) == 'interior, symmetric libration about 180°'
        assert ot.mmr_orbit_type(spec, geo, None, None, settings) == 'interior, symmetric libration about 180°'
        assert ot.mmr_orbit_type(spec, geo, 0, None, settings) == 'interior, asymmetric libration about 72°'


class TestLongPeriodOrbitType:
    def test_secular_and_zlk(self):
        assert ot.long_period_orbit_type(ResonanceType.SECULAR, geometry([episode(0, 180.0)]), 0) == ('σ libration about 180°')
        assert ot.long_period_orbit_type(ResonanceType.LIDOV_KOZAI, geometry([episode(0, 90.0)]), 0) == ('ω libration about 90°')

    def test_empty_without_an_accepted_episode(self):
        assert ot.long_period_orbit_type(ResonanceType.SECULAR, geometry([episode(0, 180.0, accepted=False)]), 0) == ''
        assert ot.long_period_orbit_type(ResonanceType.LIDOV_KOZAI, None, None) == ''


# ---------------------------------------------------------------------------------------------------------------
# 3. Subresonance enumeration up to the catalogue's order, by brute force
# ---------------------------------------------------------------------------------------------------------------


def brute_force(n_planets, target, catalogue, degree_max, nodal):
    """Every slow vector (varpi, varpi_P.., Omega, Omega_P..) with sum = target, sum |c| in [|target|, degree_max],
    node coefficients summing to an even number (zero nodes unless nodal), the catalogue vector excluded."""
    n = 1 + n_planets
    found = set()
    bound = range(-degree_max, degree_max + 1)
    for planar in itertools.product(bound, repeat=n):
        d_planar = sum(abs(c) for c in planar)
        if d_planar > degree_max:
            continue
        node_sets = [(0,) * n]
        if nodal:
            budget = degree_max - d_planar
            node_sets = [v for v in itertools.product(range(-budget, budget + 1), repeat=n) if sum(abs(c) for c in v) <= budget]
        for nodes in node_sets:
            vec = tuple(planar) + tuple(nodes)
            if sum(vec) != target or sum(nodes) % 2:
                continue
            degree = sum(abs(c) for c in vec)
            if degree < abs(target) or degree > degree_max:
                continue
            if vec != catalogue:
                found.add(vec)
    return found


def enumerated(spec, degree_max, nodal=False):
    return {sub.slow_coefficients(m) for m, _, _ in sub.enumerate_subresonances(spec, degree_max, nodal=nodal)}


class TestDegreeMax:
    @pytest.mark.parametrize(
        'key, expected',
        [('4M-7', 10), ('3J-1', 10), ('13M-24+0+11', 11), ('3J-1S-1+0+0-1', 6), ('9J-1S-1', 7)],
    )
    def test_catalogue_order(self, settings, key, expected):
        assert settings.matrix_2body_order_max == 10 and settings.matrix_3body_order_max == 6
        assert sub.degree_max(ResonanceSpec.from_key(key), settings) == expected


class TestEnumeration:
    @pytest.mark.parametrize('key', ['4M-7', '3J-1', '2N-3', '13M-24+0+11'])
    def test_two_body_planar(self, settings, key):
        spec = ResonanceSpec.from_key(key)
        target = -(sum(spec.planet_coefficients) + spec.body_coefficient)
        dmax = sub.degree_max(spec, settings)
        expected = brute_force(1, target, sub.slow_coefficients(spec), dmax, nodal=False)
        assert enumerated(spec, dmax) == expected

    def test_counts_by_hand(self, settings):
        # 4M-7: u + u_P = 3, |u| + |u_P| <= 10 -> u in [-3, 6], 10 vectors, minus (3, 0): 9 (12218: list 7 -> 9)
        assert len(enumerated(ResonanceSpec.from_key('4M-7'), 10)) == 9
        # 13M-24: q = 11 beyond 10, only degree 11: u in [0, 11], 12 vectors minus the catalogue: 11
        assert len(enumerated(ResonanceSpec.from_key('13M-24+0+11'), sub.degree_max(ResonanceSpec.from_key('13M-24+0+11'), settings))) == 11
        # degree_max below |q| still gives the degree-|q| members
        assert len(enumerated(ResonanceSpec.from_key('13M-24+0+11'), 3)) == 11

    def test_two_body_nodal(self):
        spec = ResonanceSpec.from_key('4M-7')
        expected = brute_force(1, 3, sub.slow_coefficients(spec), 7, nodal=True)
        assert enumerated(spec, 7, nodal=True) == expected

    def test_three_body_planar_to_six(self, settings):
        spec = ResonanceSpec.from_key('3J-1S-1+0+0-1')
        dmax = sub.degree_max(spec, settings)
        expected = brute_force(2, -1, sub.slow_coefficients(spec), dmax, nodal=False)
        got = enumerated(spec, dmax)
        assert got == expected
        # 24820: 3J-1S-1+2-2-1 (varpi -1, varpi_J +2, varpi_S -2; degree 5) is a member now
        assert (-1, 2, -2, 0, 0, 0) in got
        keys = {sub.member_key(spec, v) for v in got}
        assert '3J-1S-1+2-2-1' in keys

    def test_three_body_nodal_to_four(self):
        spec = ResonanceSpec.from_key('3J-1S-1+0+0-1')
        expected = brute_force(2, -1, sub.slow_coefficients(spec), 4, nodal=True)
        assert enumerated(spec, 4, nodal=True) == expected

    def test_sorted_by_degree_and_valid(self, settings):
        spec = ResonanceSpec.from_key('3J-1S-1+0+0-1')
        members = sub.enumerate_subresonances(spec, 6)
        degrees = [sub.member_degree(m) for m, _, _ in members]
        assert degrees == sorted(degrees)
        assert all(d % 2 == 1 for d in degrees)  # parity of q = -1
        for m, _, nodal in members:
            assert nodal is False
            assert sum(m.planet_coefficients) + m.body_coefficient == sum(spec.planet_coefficients) + spec.body_coefficient
            assert m.planet_coefficients == spec.planet_coefficients and m.body_coefficient == spec.body_coefficient

    def test_parity_of_degree_max(self):
        spec = ResonanceSpec.from_key('4M-7')  # q = 3: odd degrees only
        assert enumerated(spec, 6) == enumerated(spec, 5)

    @pytest.mark.parametrize('key', ['1E-1', '1J-3r0'])
    def test_no_members_for_coorbital_and_retrograde(self, key):
        assert sub.enumerate_subresonances(ResonanceSpec.from_key(key), 10) == []


# ---------------------------------------------------------------------------------------------------------------
# 4. The explanation of synthetic verdicts of every status
# ---------------------------------------------------------------------------------------------------------------

A_JUPITER = 5.2026
W0 = TWO_PI / 300.0  # small-amplitude libration period 300 yr


def pendulum(t, sigma0, sigmadot0, eps=None):
    """sigma'' = -w0^2 sin sigma (+ eps(t) sigma'): a slowly damped/anti-damped pendulum is captured and escapes."""

    def rhs(tt, y):
        extra = eps(tt) * y[1] if eps else 0.0
        return [y[1], -(W0**2) * np.sin(y[0]) + extra]

    sol = solve_ivp(rhs, (t[0], t[-1]), [sigma0, sigmadot0], t_eval=t, rtol=1e-10, atol=1e-12, method='DOP853')
    return sol.y[0], sol.y[1]


def mmr_series(t, sigma, sigmadot, key='3J-1', e=0.15, name='syn'):
    """A 3J-1 record consistent with sigma: lambda and a from sigma = 3 lambda_J - lambda - 2 varpi (varpi fixed)."""
    spec = ResonanceSpec.from_key(key)
    k_p, k = spec.planet_coefficients[0], spec.body_coefficient
    p, p_p = spec.body_varpi_coefficient, spec.planet_varpi_coefficients[0]
    n_j = TWO_PI * A_JUPITER**-1.5
    lam_j = n_j * t + 0.3
    varpi = np.full_like(t, 1.0)
    varpi_j = 0.25
    lam = (sigma - k_p * lam_j - p * varpi - p_p * varpi_j) / k
    n = (sigmadot - k_p * n_j) / k
    a = (TWO_PI / n) ** (2.0 / 3.0)
    size = len(t)
    jupiter = PlanetSeries(
        'Jupiter',
        t,
        np.full(size, A_JUPITER),
        np.full(size, 0.048),
        np.full(size, 0.022),
        np.full(size, 1.75),
        np.full(size, varpi_j),
        np.mod(lam_j, TWO_PI),
    )
    return MMRSeries(
        t_yr=t,
        a=a,
        e=np.full(size, e),
        inc=np.full(size, 0.05),
        Omega=np.full(size, 0.5),
        omega=varpi - 0.5,
        varpi=varpi,
        angle_wrapped=np.mod(sigma, TWO_PI),
        resonance=spec,
        regime=Regime(step_yr=t[1] - t[0], span_yr=t[-1] - t[0], n_rows=size, planets_saved=('Jupiter',)),
        planets={'Jupiter': jupiter},
        mean_longitude=np.mod(lam, TWO_PI),
        mean_anomaly=np.mod(lam - varpi, TWO_PI),
        name=name,
    )


AMP60 = 2 * W0 * np.sin(np.radians(60) / 2)  # sigma_dot at sigma = 0 for a half-amplitude of 60 deg


def _two_forcing(tt):
    return -8e-4 if tt < 5000 else (8e-4 if tt < 10000 else -8e-4)


def _deep_forcing(tt):
    return -8e-4 if tt < 10000 else 8e-4


FORCING = {'transient_two': _two_forcing, 'transient_deep': _deep_forcing}


def _record(kind):
    t = np.arange(0.0, 20000.0, 1.0)
    e = 0.15
    if kind == 'resonant':
        s, sd = pendulum(t, 0.0, AMP60)
    elif kind == 'resonant_180':
        s, sd = pendulum(t, 0.0, AMP60)
        s = s + np.pi  # the same pendulum about pi (sigma -> sigma + pi changes no rate)
    elif kind == 'transient_one':
        s, sd = pendulum(t, 0.0, 2.3 * W0, lambda tt: -4e-4 if tt < 10000 else 4e-4)
    elif kind in FORCING:
        # transient_two: captured, escapes, captured again (`test_the_episodes_are_the_pendulum_s_librations`).
        # transient_deep: the transient_two of W8, captured, librates down to 2.8-deg swings at 10 kyr, escapes; before
        # W18 the TP prominence counted the libration's own 300-yr swing as noise (9.2 deg) and cut it in two there
        s, sd = pendulum(t, 0.0, 2.3 * W0, FORCING[kind])
    elif kind == 'near_separatrix':
        s, sd = pendulum(t, 0.0, 2 * W0 * np.sqrt(1.02))  # energy 2 % above the separatrix
    elif kind == 'non_resonant':
        s, sd = 0.3 + TWO_PI * t / 700.0, np.full_like(t, TWO_PI / 700.0)
    elif kind == 'uncertain':
        t = np.arange(0.0, 300.0, 1.0)  # one cycle of a 322-yr libration
        s, sd = pendulum(t, 0.0, AMP60)
    elif kind == 'invalid':
        s, sd = pendulum(t, 0.0, AMP60)
        e = 1.2
    elif kind.startswith('short'):
        t = np.arange(0.0, float(kind[5:]), 1.0)
        s, sd = pendulum(t, 0.0, AMP60)
    else:
        raise ValueError(kind)
    return mmr_series(t, s, sd, e=e, name=kind)


_CACHE = {}


def analysed(kind, settings):
    if kind not in _CACHE:
        an = analyse_mmr(_record(kind), settings)
        row = {**an.summary_row(), **an.numbers_row()}
        episodes = an.episode_rows()
        _CACHE[kind] = (an, row, episodes, explain_mmr(an, settings, row, episodes))
    return _CACHE[kind]


def all_text(vt):
    parts = [vt.badge, vt.subtitle, vt.arithmetic, *vt.sentences, *vt.info]
    parts += [f'{s.title} {s.outcome}' for s in vt.steps]
    parts += [f'{c.name} {c.detail} {c.need} {c.counts} {c.note} {c.short}' for c in vt.checks]
    return '\n'.join(str(p) for p in parts)


HEADS = {
    'resonant': 'Resonant',
    'transient': 'Transient',
    'near_separatrix': 'Near-separatrix',
    'non_resonant': 'Non-resonant',
    'uncertain': 'Uncertain',
    'invalid': 'Invalid',
}
BADGES = {k: v.upper() for k, v in HEADS.items()}

STATUS_CASES = [
    ('resonant', Status.RESONANT),
    ('transient_one', Status.TRANSIENT),
    ('transient_two', Status.TRANSIENT),
    ('near_separatrix', Status.NEAR_SEPARATRIX),
    ('non_resonant', Status.NON_RESONANT),
    ('uncertain', Status.UNCERTAIN),
    ('invalid', Status.INVALID),
]


class TestSyntheticVerdicts:
    """The synthetic records reach the statuses they were built for (else the text tests below test nothing)."""

    @pytest.mark.parametrize('kind, status', STATUS_CASES)
    def test_status(self, settings, kind, status):
        an = analysed(kind, settings)[0]
        assert an.verdict.status == status

    def test_episode_counts(self, settings):
        assert len([r for r in analysed('resonant', settings)[2] if r['accepted']]) == 1
        assert len([r for r in analysed('transient_one', settings)[2] if r['accepted']]) == 1
        assert len([r for r in analysed('transient_two', settings)[2] if r['accepted']]) == 2
        assert len([r for r in analysed('transient_deep', settings)[2] if r['accepted']]) == 1

    @pytest.mark.parametrize('kind, n_librations', [('transient_two', 2), ('transient_deep', 1)])
    def test_the_episodes_are_the_pendulum_s_librations(self, settings, kind, n_librations):
        """The reference is the pendulum's own energy H = sigma_dot^2 / 2 - w0^2 cos sigma (libration: H < w0^2), not the
        classifier: each stretch of libration longer than one small-amplitude period is one accepted episode."""
        t = np.arange(0.0, 20000.0, 1.0)
        sigma, rate = pendulum(t, 0.0, 2.3 * W0, FORCING[kind])
        librating = rate**2 / 2 - W0**2 * np.cos(sigma) < W0**2
        edges = np.flatnonzero(np.diff(librating.astype(int)))
        stretches = np.split(t, edges + 1)
        flags = np.split(librating, edges + 1)
        long_ones = [s for s, f in zip(stretches, flags) if f[0] and s[-1] - s[0] > TWO_PI / W0]
        assert len(long_ones) == n_librations
        accepted = [r for r in analysed(kind, settings)[2] if r['accepted']]
        assert len(accepted) == n_librations
        for stretch, row in zip(long_ones, accepted):
            assert stretch[0] <= float(row['t_start_yr']) and float(row['t_end_yr']) <= stretch[-1]

    @pytest.mark.parametrize('kind, status', STATUS_CASES)
    def test_no_pending_flags(self, settings, kind, status):
        assert not [f for f in analysed(kind, settings)[0].verdict.flags if 'pending' in f]


class TestSyntheticOrbitType:
    def test_resonant_about_zero(self, settings):
        assert analysed('resonant', settings)[0].verdict.orbit_type == 'interior, symmetric libration about 0°'

    def test_resonant_about_180(self, settings):
        assert analysed('resonant_180', settings)[0].verdict.orbit_type == 'interior, symmetric libration about 180°'

    @pytest.mark.parametrize('kind', ['non_resonant', 'near_separatrix', 'uncertain', 'invalid'])
    def test_empty_without_libration(self, settings, kind):
        assert analysed(kind, settings)[0].verdict.orbit_type == ''

    def test_summary_column(self, settings):
        _, row, _, _ = analysed('resonant', settings)
        assert row['orbit_type'] == 'interior, symmetric libration about 0°'


class TestCalibrationInTheVerdict:
    def test_resonant_cell(self, settings):
        an, row, _, vt = analysed('resonant', settings)
        raw = json.loads(conf.PACKAGE_CELLS.read_text())
        cell = next(
            c for c in raw['cells'] if (c['status'], c['angle_status'], c['reason']) == ('resonant', 'libration', 'dynamic_libration')
        )
        assert row['cell_n_examples'] == cell['n_examples']
        assert row['cell_n_errors'] == cell['n_errors']
        low, high = wilson(cell['n_examples'] - cell['n_errors'], cell['n_examples'])
        assert row['cell_accuracy_low_95'] == pytest.approx(low, abs=6e-4)
        assert row['cell_accuracy_high_95'] == pytest.approx(high, abs=6e-4)
        text = ' '.join(vt.sentences)
        assert f"{cell['n_examples']} author-labelled verdicts, {cell['n_errors']} wrong" in text
        assert f'{round(100 * low)}%–{round(100 * high)}%' in text

    @pytest.mark.parametrize('kind', ['uncertain', 'invalid'])
    def test_no_cell_without_a_level(self, settings, kind):
        _, row, _, vt = analysed(kind, settings)
        assert row['confidence'] == ''
        assert row['cell_n_examples'] is None
        assert 'author-labelled' not in all_text(vt)


class TestExplanationWords:
    @pytest.mark.parametrize('kind, status', STATUS_CASES)
    def test_badge_names_the_status_once(self, settings, kind, status):
        an, _, _, vt = analysed(kind, settings)
        badge = vt.badge
        word = BADGES[str(status)]
        assert badge.startswith(word)
        # once in capitals; the angle_status follows in lower case even when it repeats the status (author, 10-17)
        assert badge.count(word) == 1
        assert badge.split(' · ')[1].split(' (')[0] == str(an.verdict.angle_status)
        for other, other_word in BADGES.items():
            if other != str(status) and other_word not in word:
                assert not re.search(rf'(?<![A-Z-]){re.escape(other_word)}', badge), badge

    @pytest.mark.parametrize('kind, status', STATUS_CASES)
    def test_status_sentence(self, settings, kind, status):
        _, _, _, vt = analysed(kind, settings)
        head = HEADS[str(status)]
        assert any(s.startswith(head + ':') for s in vt.sentences), vt.sentences

    @pytest.mark.parametrize('kind, status', STATUS_CASES)
    def test_confidence_word(self, settings, kind, status):
        an, _, _, vt = analysed(kind, settings)
        level = str(an.verdict.confidence)
        if status in (Status.UNCERTAIN, Status.INVALID):
            assert level == ''
            for word in ('CERTAIN', 'HIGH', 'MEDIUM', 'LOW'):
                assert not re.search(rf'\b{word}\b', vt.badge), vt.badge
        else:
            assert vt.badge.endswith(level.upper())
            assert any(s.startswith(level.capitalize()) for s in vt.sentences), vt.sentences

    def test_transient_ends_in_circulation(self, settings):
        an, _, _, vt = analysed('transient_one', settings)
        assert an.verdict.final_regime == 'circulation'
        assert vt.ends == 'ends: circulation' and 'ends:' not in vt.badge
        assert 'to the end σ circulates' in vt.sentences[0]  # the regime in plain words (W20 round F)

    def test_uncertain_asks_for_a_longer_record(self, settings):
        _, _, _, vt = analysed('uncertain', settings)
        assert 'too short' in ' '.join(vt.sentences)
        assert any(s.startswith('Action: extend') for s in vt.sentences)


class TestEpisodeWording:
    @pytest.mark.parametrize('kind', ['resonant', 'short800', 'short1000'])
    def test_no_episode_when_one_covers_the_record(self, settings, kind):
        an, _, episodes, vt = analysed(kind, settings)
        accepted = [r for r in episodes if r['accepted']]
        assert an.verdict.status == Status.RESONANT and len(accepted) == 1
        assert 'whole record' in vt.sentences[0]
        assert 'episode' not in all_text(vt).lower()

    def test_one_partial_episode_is_not_several(self, settings):
        _, _, _, vt = analysed('transient_one', settings)
        assert 'separate episodes' not in ' '.join(vt.sentences)
        assert 'whole record' not in vt.sentences[0]

    def test_two_episodes_are_named(self, settings):
        _, _, _, vt = analysed('transient_two', settings)
        assert '2 separate episodes' in vt.sentences[0]


# 'of one force window' when the stretch is shorter than one window; else against the span of the windows the test
# needs ('spanned by 2 force windows of 3.5 cycles (step 0.5 window)', integration W9: TEST-DISPUTES.md)
CYCLES = re.compile(
    r'lasts ([\d.]+) (yr|kyr|Myr) = ([\d.]+) cycles < ([\d.]+) '
    r'(?:of one force window|spanned by \d+ force windows of [\d.]+ cycles \(step [\d.]+ window\)) '
    r'\(cycles of P_cyc ([\d.]+) (yr|kyr)\)'
)
UNIT = {'yr': 1.0, 'kyr': 1e3, 'Myr': 1e6}


class TestCyclesArithmetic:
    """'lasts T = N cycles < 3.5 of one force window (cycles of P_cyc P)': N = T / P_cyc, T the libration's duration."""

    @pytest.mark.parametrize('kind', ['short800', 'short1000', 'short1150'])
    def test_cycles_equal_duration_over_period(self, settings, kind):
        an, row, episodes, vt = analysed(kind, settings)
        accepted = [r for r in episodes if r['accepted']]
        longest = max(accepted, key=lambda r: r['t_end_yr'] - r['t_start_yr'])
        duration = longest['t_end_yr'] - longest['t_start_yr']
        period = row['geo_period_cycle_yr']
        text = ' '.join(vt.sentences)
        match = CYCLES.search(text)
        assert match, text
        t_printed = float(match.group(1)) * UNIT[match.group(2)]
        n_printed = float(match.group(3))
        p_printed = float(match.group(5)) * UNIT[match.group(6)]
        assert t_printed == pytest.approx(duration, rel=5e-3)
        assert p_printed == pytest.approx(period, rel=5e-3)
        assert n_printed == pytest.approx(duration / period, abs=0.051)
        assert float(match.group(4)) == settings.mmr_force_window_cycles
        assert duration / period < settings.mmr_force_window_cycles
        assert force_windows_text(MMRFacts(an, settings, row, episodes)) in text

    def test_printed_comparison_holds_as_printed(self, settings):
        # 1127 yr / 322 yr = 3.5 cycles: one window fits, two (5.25 cycles with the half-window step) do not
        _, _, _, vt = analysed('short1400', settings)
        match = CYCLES.search(' '.join(vt.sentences))
        assert match
        assert float(match.group(3)) < float(match.group(4)), match.group(0)
