"""Regression of the free-omega gate against four real objects.

Fixtures are the first 500 kyr of mercurius integrations, thinned to a 100 yr cadence and
gzipped (`tests/fixtures/free-omega/`). They are committed because the integrations they
come from live outside the repository, and without them nothing here is reproducible.

Three of the four are Lidov-Kozai librators whose free omega stays confined. The fourth,
591986, is the case the whole gate exists for: a Jupiter Trojan whose *osculating* omega is
locked for 30 Myr while the free one leaves confinement after 6 kyr, because what bounds
the angle is the forced eccentricity vector, not a resonance. Vinogradova (2024) studied
60 L4/L5 Trojans and found none with a librating omega, which is the external check on
this verdict.

For a co-orbital the g5..g8 basis does not strictly apply at all — the forced term is
Jupiter's own vector rotated by 60 degrees (Milani 1993; Vinogradova 2015) — so 591986 is
here as a false-positive guard, not as a measurement of its rates. `free_rho` and the free
fraction are what show the decomposition is on thin ice for it.
"""

import gzip
import pathlib

import numpy as np
import pandas as pd
import pytest

from resonances.resonance.analysis.secular import LongPeriodSeries
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, ResonanceType, Status
from resonances.resonance.analysis.zlk import analyse_zlk
from resonances.resonance.omega_free_gate import GateOutcome
from resonances.secular import free_elements as fe
from resonances.simulation.config import SimulationConfig

FIXTURES = pathlib.Path(__file__).parents[2] / 'fixtures' / 'free-omega'
BASELINE = 5e5


def _load(name):
    """The fixture as the ZLK analysis reads it (omega unwrapped, elements, years) and the run's config."""
    config = SimulationConfig(save=None, plot=None, save_summary=False, integration_years=BASELINE)
    with gzip.open(FIXTURES / f'{name}.csv.gz', 'rt') as handle:
        data = pd.read_csv(handle)
    series = LongPeriodSeries(
        resonance_type=ResonanceType.LIDOV_KOZAI,
        key='LK',
        name=name,
        t_yr=data['times'].values,
        angle=np.unwrap(data['LK_angle_unwrapped'].values),
        a=data['a'].values,
        e=data['e'].values,
        inc=data['inc'].values,
        varpi=data['Omega'].values + data['omega'].values,
        Omega=data['Omega'].values,
    )
    return series, data, config


def _elements(series, data, config, representatives=None):
    return fe.free_elements(
        series.t_yr,
        series.e,
        series.inc,
        data['Omega'].values,
        data['omega'].values,
        sampling_years=config.free_elements_sampling_years,
        mask_quantile=config.free_gate_mask_quantile,
        representatives=representatives,
    )


def _gate(name):
    series, data, config = _load(name)
    elements = _elements(series, data, config)
    analysis = analyse_zlk(series, settings_from_config(config), free_elements=elements, gate_config=config)
    return analysis, elements


@pytest.mark.parametrize('name', ['4257', '3040', '162474'])
def test_a_real_librator_is_confirmed(name):
    analysis, elements = _gate(name)

    assert analysis.verdict.angle_status == AngleStatus.LIBRATION
    assert analysis.verdict.status == Status.RESONANT
    assert analysis.gate.outcome == GateOutcome.CONFIRMED
    # varpi and Omega precess together, which is what a Lidov-Kozai lock is.
    assert elements.b_omega == pytest.approx(0.0, abs=0.1)
    assert np.degrees(elements.omega_range) < 90.0
    assert elements.rho > 1.0


def test_the_trojan_is_demoted():
    analysis, elements = _gate('591986')

    assert analysis.verdict.angle_status == AngleStatus.LIBRATION  # the osculating angle does librate
    assert analysis.verdict.status == Status.NON_RESONANT
    assert analysis.verdict.reason == 'kinematic_libration'
    assert analysis.gate.outcome == GateOutcome.KINEMATIC
    assert analysis.gate.matches == ()
    # The free omega winds at about 220 arcsec/yr, nowhere near any planetary combination.
    assert elements.b_omega == pytest.approx(220.0, abs=10.0)
    # And the free circle does not enclose the origin: the angle is bounded by geometry.
    assert elements.rho < 1.0


def test_every_object_uses_the_hand_validated_basis():
    # 500 kyr merges the slow modes, and the priority convention resolves each cluster to
    # the basis that was checked independently against 10 Myr integrations.
    for name in ('4257', '3040', '162474', '591986'):
        elements = _gate(name)[1]

        assert list(elements.basis_e) == ['g5', 'g6']
        assert list(elements.basis_i) == ['s6', 'const']


@pytest.mark.parametrize('name', ['4257', '3040', '162474', '591986'])
def test_no_representative_choice_turns_a_non_librator_into_a_libration(name):
    # The cluster representative is a modelling assumption, not a free convention: fitting
    # a different member of the same cluster measurably changes the verdict on some objects.
    # What must hold is that it can only ever move the answer towards withholding, never
    # towards a false confirmation — so the priority order is a safe default, not a guess
    # that could invent a resonance.
    from itertools import product

    series, data, config = _load(name)
    settings = settings_from_config(config)
    expected = Status.RESONANT if name != '591986' else Status.NON_RESONANT

    clusters = fe.cluster(fe.candidate_frequencies('e'), BASELINE) + fe.cluster(fe.candidate_frequencies('i'), BASELINE)
    for choice in product(*clusters):
        elements = _elements(series, data, config, representatives=choice)
        status = analyse_zlk(series, settings, free_elements=elements, gate_config=config).verdict.status

        assert status in (expected, Status.UNCERTAIN), f'{name} fitted {choice} -> {status}'
