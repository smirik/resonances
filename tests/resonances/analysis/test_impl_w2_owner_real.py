"""Owner test B3 on benchmark records (no integration; skipped when the git-ignored data is absent).

The owner's inputs (context, preprocessing, P_cyc of the first geometry pass) are built without `analyse_mmr`: its
energy/zone steps are not the owner's and on 385266 they took 30 GB (requests-w2-owner.md §7). ~4 s in all.

Expected owners come from the methodichka (G2/G3 analyses) and the independent Codex check
(codex/B-review.md: 117306 11M-20 R^2 0.944 against 0.052 for sigma; 385266: 3N-5 turns with the period of a):
  117306 1V+5J-7+0+0+1        -> 11M-20+0+9 (forced wave on 0-28 kyr, then its own episode)
  385266 16N-27+0+11          -> 3N-5+0+2 (forced wave)
  2000 YG4 2R-8E-1 (rerun)    -> 7E-23+0+16 (captured for 6 of 20 kyr)
  1997 GK3 3M+5S-3 (rerun)    -> 10E-17+0+7 (needs Earth: the run's planets file)
  Hilda 153 3J-2+0-1          -> no owner (own libration)
  2003 UG22 8M-4S-7           -> no accepted owner; 10E-17+0+7 is the probable one (centre drifts)
"""

from dataclasses import replace

import numpy as np
import pytest

from resonances.resonance.analysis.mmr import io, owner
from resonances.resonance.analysis.mmr.analyse import _prominence
from resonances.resonance.analysis.mmr.context import build_context
from resonances.resonance.analysis.mmr.preprocess import filtered_angle, preprocess
from resonances.resonance.analysis.mmr.sampling import analyse_sampling
from resonances.resonance.analysis.geometry import measure_geometry
from resonances.resonance.analysis.mmr.models import PLANET_NAMES_ORDER
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue, OriginDetail

S = settings_from_config()

EXPECTED = {
    'mmr3__117306__1V+5J-7+0+0+1': ('11M-20+0+9', OriginDetail.OWNER_FORCED_WAVE),
    'mmr2__385266__16N-27+0+11': ('3N-5+0+2', OriginDetail.OWNER_FORCED_WAVE),
    'mmr3__2000_YG4__2R-8E-1+0+0+7__rerun': ('7E-23+0+16', OriginDetail.OWNER_LIBRATION),
    'mmr3__1997_GK3__3M+5S-3+0+0-5__rerun': ('10E-17+0+7', OriginDetail.OWNER_LIBRATION),
    'mmr2__153__3J-2+0-1': ('', OriginDetail.NONE),
}


# Records whose own run kept no planets file: a run on the same epoch and grid (planets do not depend on the body).
SAME_GRID_RUN = {'mmr3__2003_UG22__8M-4S-7+0+0+3': 'bench2-R-ias15-19.9999kyr-5.00122yr-jd2461297.5'}


def _with_run_planets(series, example):
    """Every planet the run saved (the example keeps only the angle's ones for some records)."""
    run = SAME_GRID_RUN.get(example) or io.read_meta(example).get('provenance', {}).get('run_id')
    folder = io.data_root() / 'runs' / str(run)
    if not (folder / 'planets.csv.gz').exists():
        return series
    import pandas as pd

    t = pd.read_csv(folder / 'planets.csv.gz', usecols=['t_yr'])['t_yr'].to_numpy(float)
    if len(t) != len(series.t_yr) or not np.allclose(np.sort(t), series.t_yr, atol=1e-3):
        return series
    planets = io._planets(folder, PLANET_NAMES_ORDER, np.argsort(t, kind='stable'), len(t))
    return replace(series, planets={**planets, **series.planets})


def run_owner_on(example, settings=S):
    """`owner.run_owner` on a benchmark record, P_cyc from the geometry of the filtered angle (as analyse_mmr's first pass)."""
    series = _with_run_planets(io.load_series(example), example)
    context = build_context(series, settings)
    sampling = analyse_sampling(series, context, settings)
    prep = preprocess(series, context, sampling, settings)
    geometry = measure_geometry(filtered_angle(prep, context, sampling.step_yr), _prominence(prep, settings), settings)
    return owner.run_owner(series, prep, context, geometry.period_cycle_yr, settings)


@pytest.mark.parametrize('example', sorted(EXPECTED))
def test_owner_of_real_records(example):
    if not (io.example_path(example) / 'series.csv.gz').exists():
        pytest.skip('benchmark data not available')
    result = run_owner_on(example)
    key, detail = EXPECTED[example]
    if not key:
        assert result.check == CheckValue.PASS and result.accepted is None
        return
    assert result.check == CheckValue.FAIL
    assert result.accepted.spec.key == key and result.detail == str(detail)
    assert result.accepted.r2_median >= S.mmr_owner_r2_min and result.accepted.phi0_coherence >= S.mmr_owner_phi0_coh_min


def test_probable_owner_of_2003_ug22():
    """10E-17 fits a with a reachable force but its phase drifts (its centre moves with varpi): by phase alone it is only
    probable (methodichka); its directly averaged force explains a_dot, so the potential route accepts it (W20 F3)."""
    example = 'mmr3__2003_UG22__8M-4S-7+0+0+3'
    if not (io.example_path(example) / 'series.csv.gz').exists():
        pytest.skip('benchmark data not available')
    result = run_owner_on(example, replace(S, mmr_owner_potential='off'))
    assert result.accepted is None and result.check == CheckValue.PASS
    assert owner.FLAG_PROBABLE in result.flags
    assert result.candidates[0].spec.key == '10E-17+0+7'
    result = run_owner_on(example)
    assert result.check == CheckValue.FAIL and result.accepted.spec.key == '10E-17+0+7' and result.accepted.route == 'potential'
    fit = result.accepted.potential
    assert fit.r2_median >= S.mmr_owner_r2_min and 1 / S.mmr_owner_potential_scale_max <= fit.scale_lo <= fit.scale_hi <= 1.1
