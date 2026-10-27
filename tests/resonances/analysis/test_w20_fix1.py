"""W20 FIX1: the confirmed findings of the Codex review of part A that have no better home (the edge, return, owner,
strength and cap fixes are tested next to their rules), and the suspect-step mask of the unwrap (258640).

Expected values are worked out by hand in each test:
- the separatrix half-width in a of a snapshot: 2 omega0 / c with omega0 = |k| n sqrt(3 mu) and c = 1.5 |k| n / a is
  (4/3) a sqrt(3 mu), whatever k and n;
- a profile <D>(phi) on L nodes per turn evaluates |k| L^2 kernels (L values of phi, |k| L nodes each);
- the wobble of `test_w20_undefined_phase` with a loop radius r = 0.0045: varpi's own scatter r / e = 0.1 < pi/4
  (defined everywhere), while p varpi changes per 0.6-yr step by up to 2 x 27 x arcsin(0.1) x sin(pi 0.6 / 4.9) =
  2.03 rad = 116 deg > MMR_UNWRAP_SUSPECT_RESID_DEG (90): fast steps of the slow elements, none of them suspect.
"""

import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.mmr import averaged_potential, coorbital, io, owner_potential, quasi_satellite
from resonances.resonance.analysis.mmr import strength as st
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.coorbital import Orbit
from resonances.resonance.analysis.settings import AnalysisSettings, settings_from_config

sys.path.insert(0, str(Path(__file__).parent))
import test_w20_undefined_phase as undefined_phase  # noqa: E402

S = settings_from_config()
GM_SUN = 4 * math.pi**2


# --- settings ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize('name', ['mmr_owner_r2_min', 'libration_return_tol', 'mmr_coorbital_qs_well_share'])
def test_a_nan_setting_is_rejected(name):
    with pytest.raises(ValueError, match='must be a number'):
        replace(AnalysisSettings(), **{name: float('nan')})


def test_extended_precision_digits_may_not_shrink():
    with pytest.raises(ValueError, match='must not exceed'):
        replace(AnalysisSettings(), mmr_strength_mp_dps=60, mmr_strength_mp_dps_max=50)
    assert replace(AnalysisSettings(), mmr_strength_mp_dps=60, mmr_strength_mp_dps_max=60).mmr_strength_mp_dps_max == 60


# --- the strength columns ------------------------------------------------------------------------------------------


def test_separatrix_half_width_uses_each_snapshots_own_mean_motion():
    """Two snapshots at a = 2 and 3 au with mu = 1e-6 and 4e-6: (4/3) a sqrt(3 mu) = 0.0046188 and 0.0138564 au; the
    record's identity coefficient (here deliberately unrelated) no longer enters."""
    theory = SimpleNamespace(values_all=[1e-6, 4e-6], a_all=[2.0, 3.0], e_all=[0.1, 0.1])
    context = SimpleNamespace(mean_motion_rad_yr=2 * math.pi / 2.0**1.5, identity_coefficient=-123.0)
    out = st._display_numbers(theory, None, float('nan'), -3, context)
    assert out['sep_halfwidth_min_au'] == pytest.approx(4 / 3 * 2.0 * math.sqrt(3e-6), rel=1e-12)
    assert out['sep_halfwidth_max_au'] == pytest.approx(4 / 3 * 3.0 * math.sqrt(12e-6), rel=1e-12)
    # T0 = 2 pi / (|k| n sqrt(3 mu)) at a = 2 au, mu = 1e-6, k = -3
    n = math.sqrt(GM_SUN / 2.0**3)
    assert out['period_theory_max_yr'] == pytest.approx(2 * math.pi / (3 * n * math.sqrt(3e-6)), rel=1e-12)


# --- costs ---------------------------------------------------------------------------------------------------------


def test_the_potential_route_budget_counts_kernels():
    """k = 8 at L = 2880 is 8 x 2880^2 = 6.6e7 kernels, at 5760 2.65e8: with a budget of 1e8 the doubling stops before
    5760 (potential_cost); the old bound |k| L <= 5e4 let 5760 through."""
    assert owner_potential.profile_kernels(-8, 2880) == 8 * 2880**2
    assert owner_potential.profile_kernels(-8, 5760) > S.mmr_owner_potential_max_kernels > owner_potential.profile_kernels(-8, 2880)
    body = averaged_potential.Ellipse(3.0, Orbit(0.15, 0.1, 0.3, 0.2))
    planet = averaged_potential.Ellipse(5.2, Orbit(0.048, 0.02, 0.25, 1.75))
    tight = replace(S, mmr_owner_potential_max_kernels=float(owner_potential.profile_kernels(-8, 720)) - 1)
    assert owner_potential.derivative_profile(body, planet, -8, 3, 0.0, tight)[-1] == owner_potential.REASON_COST


def test_the_quasi_satellite_blocks_are_bounded_in_number():
    """P_QS,min = 5.3 yr under a 55.5-yr P_c over 20 kyr: 3774 blocks of 5.3 yr fit in 4000; with a cap of 1000 the
    blocks are 20 yr long."""
    well = coorbital.QuasiSatelliteWell(np.array([0.0]), np.array([0.0]), np.array([0.5]), np.array([5.3]))
    found = quasi_satellite.QuasiSatellite(well, 1, 0.0, 0.0, 0.0, 2.65, 111.0, '', 1.3, True, 55.5, 20000.0 / 4000)
    assert found.elements_block_yr == pytest.approx(5.3)
    assert replace(found, block_min_yr=20000.0 / 1000).elements_block_yr == pytest.approx(20.0)


def test_record_tables_keep_every_table_of_one_analysis():
    """Inside the scope 600 distinct tables (more than the LRU's 512) are computed once each, however often asked."""
    coorbital.cached_table.cache_clear()
    orbits = [Orbit(0.05 + 1e-4 * i, 0.1, 0.3, 0.2) for i in range(600)]
    planet = Orbit(0.0167, 0.0, 0.0, 1.8)
    with coorbital.record_tables(S.mmr_coorbital_record_tables_mb * 1e6):
        for _ in range(2):
            for body in orbits:
                coorbital.table_for(body, planet, 16, 1e-3)
        assert coorbital.cached_table.cache_info().misses == 600
    assert coorbital._RECORD_TABLES is None
    coorbital.cached_table.cache_clear()


def test_record_tables_are_bounded_and_keep_the_first_ones():
    """A store of 100 tables' bytes (3 arrays of 16 doubles, 384 bytes each: the bound is in bytes, review C #6) over
    1200 distinct tables read twice in the same order (the passes of one record), more than the shared LRU's 512: it
    holds 100, and the second pass finds those; the LRU, read cyclically, misses every other one. Misses: 1200 on the
    first pass, 1200 - 100 = 1100 on the second, 2300 in all (an evicting store would miss 2400)."""
    coorbital.cached_table.cache_clear()
    orbits = [Orbit(0.05 + 1e-4 * i, 0.1, 0.3, 0.2) for i in range(1200)]
    planet = Orbit(0.0167, 0.0, 0.0, 1.8)
    with coorbital.record_tables(100 * 3 * 16 * 8):
        for _ in range(2):
            for body in orbits:
                coorbital.table_for(body, planet, 16, 1e-3)
        assert coorbital.record_tables_kept() == 100
        assert coorbital.cached_table.cache_info().misses == 2300
    assert coorbital.record_tables_kept() == 0
    coorbital.cached_table.cache_clear()


# --- geometry ------------------------------------------------------------------------------------------------------


def test_gap_raw_range_is_read_between_two_runs():
    """The averaged angle 0..9 rad on t = 0..9 yr: from the last sample of a run ending at 2 yr to the first of one
    starting at 6 yr the range is 6 - 2 = 4 rad; overlapping runs give 0; without the angle nan."""
    t = np.arange(10.0)
    holder = SimpleNamespace(averaged_raw=t.copy(), averaged_raw_t_yr=t)
    left, right = SimpleNamespace(t_start_yr=0.0, t_end_yr=2.0), SimpleNamespace(t_start_yr=6.0, t_end_yr=9.0)
    assert g.Geometry.gap_raw_range(holder, left, right) == pytest.approx(4.0)
    assert g.Geometry.gap_raw_range(holder, left, SimpleNamespace(t_start_yr=1.0, t_end_yr=3.0)) == 0.0
    assert math.isnan(g.Geometry.gap_raw_range(SimpleNamespace(averaged_raw=None, averaged_raw_t_yr=None), left, right))


def test_a_floor_well_needs_its_share_of_the_episode():
    """The well (centre 0, half-width 1 rad) exists on t = 0..29 of an episode spanning 0..99 yr (30 %): an episode
    centred at 0.2 rad holds it without a share, not with the default 0.5."""
    t = np.arange(100.0)
    centre = np.where(t < 30, 0.0, np.nan)
    well = g.FloorWell(t, centre, np.where(t < 30, 1.0, np.nan), 111.0, 'mmr_coorbital_period_floor_factor')
    episode = SimpleNamespace(t_start_yr=0.0, t_end_yr=99.0, centre_mod_2pi_rad=0.2)
    assert well.holds(episode)
    assert not replace(well, min_share=S.mmr_coorbital_qs_well_share).holds(episode)
    assert replace(well, min_share=0.3).holds(episode)


# --- the suspect-step mask (258640) --------------------------------------------------------------------------------


def test_fast_slow_element_steps_are_suspect_only_where_varpi_is_undefined():
    t = np.arange(0.0, 3000.0 + 0.3, 0.6)
    z = undefined_phase.E_SLOW * np.exp(2j * math.pi * t / 25000.0) + 0.0045 * np.exp(2j * math.pi * t / undefined_phase.LOOP_PERIOD_YR)
    sampling = undefined_phase._sampling(undefined_phase._series(z, t))
    assert sampling.n_slow_element_steps > 0 and sampling.varpi_undefined_fraction == 0.0
    assert sampling.phase_undefined_fraction == pytest.approx(1.0)  # the |p|-scaled proxy: every sample
    assert sampling.suspect_times_yr == ()
    noise = undefined_phase._sampling(undefined_phase._noise())  # varpi undefined everywhere: its fast steps are suspect
    assert noise.varpi_undefined_fraction == pytest.approx(1.0) and len(noise.suspect_times_yr) > 0


def test_258640_loses_the_suspect_step_flag():
    example = 'mmr2__258640__46J-19+0-27'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    result = analyse_mmr(io.load_series(example), S)
    v = result.verdict
    assert (str(v.status), str(v.angle_status), str(v.confidence)) == ('non_resonant', 'circulation', 'medium')
    assert 'unwrap_step_suspect' not in v.flags and result.sampling.suspect_times_yr == ()


def test_one_to_one_angles_are_not_judged_by_the_eq_strength():
    """1E-1 and the retrograde 1J-1r0 are 1:1 (the co-orbital law judges them); 3J-1 and the three-body 4J-2S-1 are not."""
    from resonances.resonance.analysis.mmr.models import ResonanceSpec
    from resonances.resonance.analysis.mmr.strength import direct_applicable, one_to_one

    keys = {'1E-1+0+0': True, '1J-1r0': True, '3J-1+0-2': False, '4J-2S-1+0+0-1': False}
    assert {key: one_to_one(ResonanceSpec.from_key(key)) for key in keys} == keys
    assert not direct_applicable(ResonanceSpec.from_key('1E-1+0+0')) and direct_applicable(ResonanceSpec.from_key('3J-1+0-2'))
