"""Sampling of clones: the draws follow the multivariate normal of the covariance in the fit's own coordinates.

Statistical checks with fixed seeds on real covariances (the condition number of 463's is ~1e13 in raw units):
- the sample mean and covariance converge to the nominal and the covariance (compared in correlation units, where
  every entry has the same scale; the Monte Carlo error of a correlation is ~1/sqrt(N));
- the squared Mahalanobis distances, recomputed here independently by a solve with the covariance, follow chi^2 with 6
  degrees of freedom (Kolmogorov-Smirnov);
- the draws are prefix-stable and differ between bodies; a draw that is no bound orbit is kept, not redrawn.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from resonances.clones import Ensemble, OrbitSolution, parse_oef, parse_sbdb
from resonances.clones.ensemble import clone_name, factor, parent_of

FIXTURES = Path(__file__).resolve().parents[2] / 'fixtures' / 'clones'
N = 20000


def solution(relative: str) -> OrbitSolution:
    path = FIXTURES / relative
    return parse_sbdb(path.read_text()) if path.suffix == '.json' else parse_oef(path.read_text())


@pytest.mark.parametrize('relative', ['astdys/463.eq1', 'neodys/2013RZ53.eq0', 'sbdb/463.json'])
def test_sample_mean_and_covariance_converge(relative):
    sol = solution(relative)
    ensemble = sol.sample(n=N, seed=11, name='test')
    draws = ensemble.clones[[f'x_{label}' for label in sol.labels]].to_numpy()
    scale = sol.sigma
    # the mean: within 4.5 standard errors per coordinate
    assert np.all(np.abs(draws.mean(axis=0) - sol.nominal) / scale < 4.5 / np.sqrt(N))
    # the covariance in correlation units: |S_ij - C_ij| / (s_i s_j) within 5 / sqrt(N)
    sample = np.cov(draws, rowvar=False)
    assert np.max(np.abs(sample - sol.covariance) / np.outer(scale, scale)) < 5 / np.sqrt(N)


@pytest.mark.parametrize(
    'relative, rtol',
    [
        ('astdys/463.eq1', 1e-6),
        ('neodys/2013RZ53.eq0', 1e-6),
        # SBDB: tp ~ 2.46e6 days stored in float64 (2e-10 d) against sigma_tp 6e-7 d along |corr(e, q)| = 0.99999: the
        # stored draw itself rounds d^2 at 1e-3; the sampler's d^2 = |z|^2 is exact
        ('sbdb/463.json', 2e-2),
    ],
)
def test_mahalanobis_distances_follow_chi2_6(relative, rtol):
    sol = solution(relative)
    ensemble = sol.sample(n=N, seed=5, name='test')
    draws = ensemble.clones[[f'x_{label}' for label in sol.labels]].to_numpy()
    # independently of the sampler's factor: d^2 = dx^T C^-1 dx by a solve in correlation units
    scale = sol.sigma
    dx = (draws - sol.nominal) / scale
    correlation = sol.covariance / np.outer(scale, scale)
    d2 = np.einsum('ij,ij->i', dx, np.linalg.solve(correlation, dx.T).T)
    np.testing.assert_allclose(ensemble.clones['d2'], d2, rtol=rtol)
    assert stats.kstest(d2, stats.chi2(df=6).cdf).pvalue > 0.01
    # P(chi^2_6 <= 9) = 1 - e^-4.5 (1 + 4.5 + 4.5^2/2) = 0.8264: the 6-D "3 sigma" ellipsoid holds 83 %
    assert np.mean(d2 <= 9) == pytest.approx(1 - np.exp(-4.5) * (1 + 4.5 + 4.5**2 / 2), abs=0.01)


def test_draws_are_prefix_stable_and_differ_between_bodies():
    sol = solution('astdys/463.eq1')
    small, large = sol.sample(n=10, seed=3, name='463'), sol.sample(n=40, seed=3, name='463')
    pd.testing.assert_frame_equal(small.clones, large.clones.iloc[:10].reset_index(drop=True))
    other = sol.sample(n=10, seed=3, name='464')
    assert not np.any(other.clones['x_lambda'].to_numpy() == small.clones['x_lambda'].to_numpy())
    assert not np.any(sol.sample(n=10, seed=4, name='463').clones['x_lambda'].to_numpy() == small.clones['x_lambda'].to_numpy())


def test_clones_of_good_orbit_stay_near_the_nominal():
    """463 (AstDyS): sigma_a = 4.9e-9 AU: the clones' a within 6 sigma of the nominal, the nominal kept apart."""
    sol = solution('astdys/463.eq1')
    ensemble = sol.sample(n=100, seed=0, name='463')
    assert len(ensemble) == 100 and ensemble.n_valid == 100
    assert ensemble.nominal['a'] == pytest.approx(2.3986620775145058)
    assert np.all(np.abs(ensemble.clones['a'] - ensemble.nominal['a']) < 6 * 4.92285e-09)


def test_unbound_draws_are_kept_as_invalid():
    """A wide cometary covariance around e = 0.9 (sigma_e = 0.1): about 16 % of draws have e >= 1; they stay, invalid."""
    covariance = np.diag([0.1, 0.01, 1.0, 0.1, 0.1, 0.1]) ** 2
    sol = OrbitSolution(
        'wide', 'test', 'cometary', np.array([0.9, 0.5, 2451545.0, 10.0, 20.0, 5.0]), covariance, epoch_mjd=51544.5, time_scale='TDB'
    )
    ensemble = sol.sample(n=2000, seed=2, name='wide')
    unbound = ensemble.clones['x_e'] >= 1
    assert len(ensemble) == 2000
    assert (~ensemble.clones['valid']).sum() == unbound.sum()
    assert unbound.mean() == pytest.approx(1 - stats.norm.cdf(1.0), abs=0.03)
    assert ensemble.clones.loc[~ensemble.clones['valid'], 'a'].isna().all()


def test_factor_of_an_ill_conditioned_covariance():
    sol = solution('astdys/463.eq1')
    lower = factor(sol.covariance)
    np.testing.assert_allclose(lower @ lower.T, sol.covariance, rtol=1e-9, atol=1e-30)


def test_rank_deficient_covariance_gives_the_pseudo_inverse_distance(tmp_path):
    """corr(a, h) = 1 exactly: rank 5, Cholesky fails, the eigen factor is used. The draws then have no component along
    the null mode, d^2 must be dX^T C^+ dX of the actual draw (here by numpy's pinv, independently of the factor), not
    |z|^2 (8.4479 instead of 8.3708 on the review's draw), and it follows chi^2 with 5 degrees of freedom. The rank is
    kept with the ensemble (6 for a full-rank covariance)."""
    sigma = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 0.1])
    correlation = np.eye(6)
    correlation[0, 1] = correlation[1, 0] = 1.0
    covariance = np.outer(sigma, sigma) * correlation
    nominal = np.array([2.4, 0.02, 0.2, 0.07, 0.09, 316.0])
    sol = OrbitSolution('rank5', 'test', 'equinoctial', nominal, covariance, epoch_mjd=61200.0)
    ensemble = sol.sample(n=N, seed=0, name='rank5')
    assert ensemble.rank == 5
    draws = ensemble.clones[[f'x_{label}' for label in sol.labels]].to_numpy()
    dx = (draws - nominal) / sigma
    np.testing.assert_allclose(dx[:, 0], dx[:, 1], atol=1e-12)  # the draws lie in the range of C
    expected = np.einsum('ij,ij->i', dx, (np.linalg.pinv(correlation) @ dx.T).T)
    np.testing.assert_allclose(ensemble.clones['d2'], expected, rtol=1e-9)
    assert stats.kstest(expected, stats.chi2(df=5).cdf).pvalue > 0.01
    assert stats.kstest(expected, stats.chi2(df=6).cdf).pvalue < 1e-6
    ensemble.save(tmp_path / 'rank5')
    assert Ensemble.load(tmp_path / 'rank5').rank == 5
    assert solution('astdys/463.eq1').sample(n=3, seed=0, name='463').rank == 6


def test_save_and_load_round_trip(tmp_path):
    ensemble = solution('neodys/2013RZ53.eq0').sample(n=7, seed=1, name='2013 RZ53')
    csv_path, json_path = ensemble.save(tmp_path / 'ens')
    loaded = Ensemble.load(csv_path)
    pd.testing.assert_frame_equal(loaded.clones, ensemble.clones)
    assert loaded.nominal == ensemble.nominal and loaded.seed == 1 and loaded.epoch_mjd == ensemble.epoch_mjd
    np.testing.assert_array_equal(loaded.solution.covariance, ensemble.solution.covariance)
    assert Ensemble.load(json_path).name == '2013 RZ53'


def test_ensemble_from_the_users_clone_list():
    rows = [
        {'a': 2.40, 'e': 0.20, 'inc': 0.2, 'Omega': 1.0, 'omega': 2.0, 'M': 3.0, 'epoch': 61200.0},
        {'a': 2.41, 'e': 0.21, 'inc': 0.2, 'Omega': 1.0, 'omega': 2.0, 'M': 3.1, 'epoch': 61200.0},
        {'a': 2.42, 'e': 1.20, 'inc': 0.2, 'Omega': 1.0, 'omega': 2.0, 'M': 3.2, 'epoch': 61200.0},
    ]
    ensemble = Ensemble.from_elements('mine', rows)
    assert ensemble.epoch_mjd == 61200.0 and ensemble.method == 'user'
    assert ensemble.clones['valid'].tolist() == [True, True, False]
    assert ensemble.clone_elements()[2] is None
    assert ensemble.clone_elements()[1]['a'] == 2.41
    with pytest.raises(ValueError, match='one epoch'):
        Ensemble.from_elements('mixed', [dict(rows[0]), {**rows[1], 'epoch': 61000.0}])


def test_clone_names():
    assert clone_name('463', 7) == '463~c007'
    assert parent_of('463~c007') == '463' and parent_of('2013 RZ53~c100') == '2013 RZ53'
    assert parent_of('463') is None
    names = solution('astdys/463.eq1').sample(n=1000, seed=0, name='463').clone_names()
    assert names[0] == '463~c0001' and names[-1] == '463~c1000'
