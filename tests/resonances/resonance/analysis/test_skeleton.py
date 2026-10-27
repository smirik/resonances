"""Smoke test of the v2 analysis skeleton: every module imports, the verdict status codes follow the
tables of SPEC §2.2 and §5.7, the ledger classifies margins, and the settings agree with .env.dist."""

import ast
import importlib
import math
from dataclasses import fields
from pathlib import Path

import pytest
from dotenv import dotenv_values

from resonances.resonance.analysis.ledger import Ledger, Margin, Op
from resonances.resonance.analysis.settings import AnalysisSettings, DryRunSettings, format_value, settings_from_config, settings_hash
from resonances.resonance.analysis.verdict import (
    Action,
    AngleStatus,
    CheckValue,
    Confidence,
    Flag,
    LibrationOrigin,
    Reason,
    ResonanceType,
    Status,
    Verdict,
    default_action,
    verdict_from_status_code,
)
from resonances.simulation.config import SimulationConfig

MODULES = [
    'resonances.resonance.analysis',
    'resonances.resonance.analysis.verdict',
    'resonances.resonance.analysis.ledger',
    'resonances.resonance.analysis.settings',
    'resonances.resonance.analysis.filtering',
    'resonances.resonance.analysis.geometry',
    'resonances.resonance.analysis.pendulum',
    'resonances.resonance.analysis.confidence',
    'resonances.resonance.analysis.mmr',
    'resonances.resonance.analysis.mmr.models',
    'resonances.resonance.analysis.mmr.context',
    'resonances.resonance.analysis.mmr.sampling',
    'resonances.resonance.analysis.mmr.preprocess',
    'resonances.resonance.analysis.mmr.response',
    'resonances.resonance.analysis.mmr.kinematics',
    'resonances.resonance.analysis.mmr.subresonances',
    'resonances.resonance.analysis.mmr.multiplet',
    'resonances.resonance.analysis.mmr.owner',
    'resonances.resonance.analysis.mmr.strength',
    'resonances.resonance.analysis.mmr.zone',
    'resonances.resonance.analysis.mmr.decide',
    'resonances.resonance.analysis.mmr.analyse',
    'resonances.resonance.analysis.mmr.io',
    'resonances.plotting.mmr_combined',
]


@pytest.mark.parametrize('name', MODULES)
def test_every_module_imports(name):
    importlib.import_module(name)


def test_pure_modules_do_not_import_body_or_simulation():
    root = Path(importlib.import_module('resonances.resonance.analysis').__file__).parent
    for path in root.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or '']
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            for name in names:
                assert not name.startswith(('resonances.body', 'resonances.simulation', 'resonances.mmr')), f'{path.name} imports {name}'


# SPEC §2.2: (status, reason) -> code for MMR
MMR_TABLE = [
    (Status.RESONANT, AngleStatus.LIBRATION, Reason.DYNAMIC_LIBRATION, 2),
    (Status.TRANSIENT, AngleStatus.EPISODIC, Reason.DYNAMIC_EPISODE, 1),
    (Status.NEAR_SEPARATRIX, AngleStatus.EPISODIC, Reason.SEPARATRIX_ZONE, -3),
    (Status.NON_RESONANT, AngleStatus.CIRCULATION, Reason.CIRCULATION, 0),
    (Status.NON_RESONANT, AngleStatus.CIRCULATION, Reason.ALIAS, 0),
    (Status.NON_RESONANT, AngleStatus.LIBRATION, Reason.KINEMATIC_LIBRATION, 0),
    (Status.NON_RESONANT, AngleStatus.DRIFT, Reason.KINEMATIC_DRIFT, 0),
    (Status.NON_RESONANT, AngleStatus.EPISODIC, Reason.INDUCED_LIBRATION, 0),
    (Status.NON_RESONANT, AngleStatus.CIRCULATION, Reason.SUBRESONANCE_LIBRATES, 0),
    (Status.UNCERTAIN, AngleStatus.INSUFFICIENT_TIME, Reason.BASELINE_NEEDED, -6),
    (Status.UNCERTAIN, AngleStatus.LIBRATION, Reason.BASELINE_NEEDED, -6),
    (Status.UNCERTAIN, AngleStatus.LIBRATION, Reason.ORIGIN_UNKNOWN, -9),
    (Status.UNCERTAIN, AngleStatus.DRIFT, Reason.DRIFT_MECHANISM_UNKNOWN, -9),
    (Status.INVALID, AngleStatus.UNDEFINED, Reason.ALIAS, -99),
    (Status.INVALID, AngleStatus.UNDEFINED, Reason.UNPHYSICAL, -99),
]


@pytest.mark.parametrize('status, angle_status, reason, code', MMR_TABLE)
def test_mmr_status_code_table(status, angle_status, reason, code):
    verdict = Verdict(ResonanceType.MMR, status, angle_status, reason=reason)
    assert verdict.status_code == code


# SPEC §5.7: old integer -> (status, angle_status, reason, action)
LEGACY_TABLE = [
    (2, Status.RESONANT, AngleStatus.LIBRATION, Reason.DYNAMIC_LIBRATION, Action.NONE),
    (1, Status.TRANSIENT, AngleStatus.EPISODIC, Reason.DYNAMIC_EPISODE, Action.NONE),
    (0, Status.NON_RESONANT, AngleStatus.CIRCULATION, Reason.CIRCULATION, Action.NONE),
    (-2, Status.NON_RESONANT, AngleStatus.LIBRATION, Reason.KINEMATIC_LIBRATION, Action.NONE),
    (-3, Status.NEAR_SEPARATRIX, AngleStatus.NEAR_SEPARATRIX, Reason.SEPARATRIX_ZONE, Action.SPOT_CHECK),
    (-4, Status.UNCERTAIN, AngleStatus.INSUFFICIENT_TIME, Reason.BASELINE_NEEDED, Action.EXTEND),
    (-5, Status.NEAR_SEPARATRIX, AngleStatus.NEAR_SEPARATRIX, Reason.SEPARATRIX_ZONE, Action.REVIEW),
    (-6, Status.UNCERTAIN, AngleStatus.DRIFT, Reason.BASELINE_NEEDED, Action.EXTEND),
    (-9, Status.UNCERTAIN, AngleStatus.UNDEFINED, Reason.GATE_UNAVAILABLE, Action.EXTEND),
    (-99, Status.INVALID, AngleStatus.UNDEFINED, Reason.UNPHYSICAL, Action.REVIEW),
]


@pytest.mark.parametrize('code, status, angle_status, reason, action', LEGACY_TABLE)
def test_zlk_verdict_from_status_code_and_back(code, status, angle_status, reason, action):
    verdict = verdict_from_status_code(code, ResonanceType.LIDOV_KOZAI, confidence='high')
    assert (verdict.status, verdict.angle_status, verdict.reason, verdict.action) == (status, angle_status, reason, action)
    assert verdict.status_code == code
    if status in (Status.UNCERTAIN, Status.INVALID):
        assert verdict.confidence == Confidence.NONE
    else:
        assert Flag.CONFIDENCE_UNCALIBRATED in verdict.flags


def test_secular_status_codes_round_trip_and_differences():
    for code in (2, 1, 0, -2, -3, -4, -5, -6, -9, -99):
        assert verdict_from_status_code(code, 'secular', confidence='medium').status_code == code
    assert verdict_from_status_code(-9, 'secular').reason == Reason.UNUSABLE_SERIES
    assert verdict_from_status_code(-5, 'secular', confidence='medium').confidence == Confidence.LOW
    assert verdict_from_status_code(-2, 'lidov_kozai').libration_origin == LibrationOrigin.KINEMATIC
    assert verdict_from_status_code(2, 'secular', confidence='certain').confidence == Confidence.HIGH
    with pytest.raises(ValueError):
        verdict_from_status_code(-1, 'secular')
    with pytest.raises(ValueError):
        verdict_from_status_code(2, 'mmr')


def test_default_action():
    assert default_action(Status.INVALID, Reason.ALIAS, Confidence.NONE) == Action.RESAMPLE
    assert default_action(Status.INVALID, Reason.UNPHYSICAL, Confidence.NONE) == Action.REVIEW
    assert default_action(Status.UNCERTAIN, Reason.ORIGIN_UNKNOWN, Confidence.NONE) == Action.REVIEW
    assert default_action(Status.UNCERTAIN, Reason.BASELINE_NEEDED, Confidence.NONE) == Action.EXTEND
    assert default_action(Status.RESONANT, Reason.DYNAMIC_LIBRATION, Confidence.HIGH) == Action.NONE
    assert default_action(Status.RESONANT, Reason.DYNAMIC_LIBRATION, Confidence.MEDIUM) == Action.SPOT_CHECK
    assert default_action(Status.TRANSIENT, Reason.DYNAMIC_EPISODE, Confidence.LOW) == Action.REVIEW


def test_verdict_flat_dict_round_trip():
    ledger = Ledger(thin_band=0.1, conventions={'mmr_force_r2_full'})
    ledger.compare('force_r2', 0.72, Op.GE, 0.7, '', 'rsp_force_R2_h1', setting='mmr_force_r2_full')
    verdict = Verdict(
        ResonanceType.MMR,
        Status.NON_RESONANT,
        AngleStatus.CIRCULATION,
        confidence=Confidence.HIGH,
        confidence_raw=Confidence.HIGH,
        reason=Reason.SUBRESONANCE_LIBRATES,
        flags=['subresonance_composite'],
        libration_origin=LibrationOrigin.SUBRESONANCE,
        origin_detail='composite_multiplet',
        owner='9M-16+2+5',
        multiplet_status=Status.TRANSIENT,
        checks={'check_mmr_subresonances': CheckValue.FAIL},
        decision_path=list(ledger),
    )
    row = verdict.to_flat_dict()
    assert row['status'] == 'non_resonant' and row['status_code'] == 0 and row['owner'] == '9M-16+2+5'
    assert row['check_mmr_subresonances'] == 'fail' and row['check_zlk_ei'] == ''
    assert row['decision_path'] == 'force_r2' and row['flags'] == 'subresonance_composite'
    back = Verdict.from_flat_dict(row)
    assert back.to_flat_dict() == row


def test_ledger_margins():
    # thin band 0.1: 0.72 vs 0.7 is 2.9% away (thin), 0.95 vs 0.7 is 36% away (full)
    ledger = Ledger(thin_band=0.1, conventions={'mmr_force_r2_full', 'mmr_a_response_min_amp_deg'})
    near = ledger.compare('force_r2', 0.72, '>=', 0.7, '', 'rsp_force_R2_h1', setting='mmr_force_r2_full')
    far = ledger.compare('force_r2_h2', 0.95, '>=', 0.7, '', 'rsp_force_R2_h2', setting='mmr_force_r2_full')
    below = ledger.compare('amplitude', 42.0, '>=', 45.0, 'deg', 'geo_amp_deg', setting='mmr_a_response_min_amp_deg')
    physics = ledger.compare('centre_pass', 6.4, '>=', 2 * math.pi, 'rad', 'geo_centre_max_pass_rad')
    missing = ledger.compare('identity', float('nan'), '>=', 0.9, '', 'rsp_identity_corr', setting='mmr_identity_corr_min')
    assert near and near.margin == Margin.THIN and near.relative_distance == pytest.approx(0.02 / 0.7)
    assert far and far.margin == Margin.FULL
    assert not below and below.margin == Margin.THIN  # closeness, whatever the outcome: 42/45 is 6.7% away
    assert physics and physics.margin == Margin.FULL and not physics.convention  # derived thresholds never make U
    assert not missing and missing.result is None and missing.margin == Margin.NOT_MEASURED
    assert ledger.thin_margins() == ['U:force_r2', 'U:amplitude']
    assert ledger.decision_path() == 'force_r2;force_r2_h2;amplitude;centre_pass;identity'
    assert ledger.columns()[0] == 'rsp_force_R2_h1' and len(ledger) == 5
    assert near.describe().endswith('~') and far.mark == '✓' and below.mark == '✗' and missing.mark == '–'


def test_ledger_from_settings_and_zero_threshold():
    ledger = Ledger.from_settings(settings_from_config())
    assert ledger.thin_band == 0.1 and 'mmr_l1_tp_full' in ledger.conventions and 'filter_min_steps' not in ledger.conventions
    entry = ledger.compare('rate_sign', 0.001, '>', 0.0, 'rad/yr', 'ctx_rate_measured_deg_yr', setting='mmr_l1_tp_full')
    assert entry and entry.relative_distance is None and entry.margin == Margin.FULL


def test_settings_agree_with_env_dist_and_validate():
    env = dotenv_values(Path(importlib.import_module('resonances').__file__).parent / '.env.dist')
    for cls in (AnalysisSettings, DryRunSettings):
        for f in fields(cls):
            assert env.get(f.metadata['env']) == format_value(f.default), f.metadata['env']
    assert 'SECULAR_MIN_TURNING_POINTS' not in env
    config = SimulationConfig(save=None, plot=None, save_summary=False, mmr_force_r2_full=0.75)
    assert config.libration_min_turning_points == 3
    assert config.analysis_settings.mmr_force_r2_full == 0.75
    assert settings_hash(config.analysis_settings) != settings_hash(settings_from_config())
    assert settings_from_config({'MMR_OWNER_PLANETS': 'Mars,Jupiter'}).mmr_owner_planets == ('Mars', 'Jupiter')
    with pytest.raises(ValueError):
        settings_from_config(mmr_force_r2_fail=0.8)  # fail threshold above the full one
    with pytest.raises(ValueError):
        settings_from_config(mmr_zone_method='analytic')
    with pytest.raises(ValueError):
        settings_from_config(no_such_setting=1)
