"""Ensembles of virtual asteroids (clones) of one object (docs/clones.md).

`sample(solution, n, seed)` draws the clones from the multivariate normal of the orbit solution's covariance in the
fit's own coordinates, X_i = X_0 + L z_i with C = L L^T and z_i ~ N(0, I_6), never from independent per-element
errors, and converts each clone exactly to Keplerian elements at the covariance's epoch. The nominal is kept apart. A
draw that is no bound orbit (e >= 1) is kept as an invalid clone, never redrawn. Each clone's squared Mahalanobis
distance d^2 = dX^T C^+ dX is stored (= |z|^2, chi^2 with 6 degrees of freedom; for a covariance float64 finds singular,
the pseudo-inverse distance over the `rank` modes kept, chi^2 with `rank` degrees of freedom), so a reader can condition
on it afterwards; there is no truncation.

The draws of a body are prefix-stable: the generator is seeded by (seed, crc32(name)), and the first 100 clones of
n = 400 are the 100 clones of n = 100.
"""

import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

from resonances.config import config as c
from resonances.logger import logger

from .elements import KEPLERIAN_KEYS
from .solution import OrbitSolution

DEFAULT_CLONES = 100
CLONE_MARK = '~c'
METHODS = ('gaussian', 'user')


def configured_count() -> int:
    """CLONES of .env (0 = off), else 0."""
    value = c.get('CLONES') if c.has('CLONES') else ''
    return int(value) if str(value).strip() else 0


def configured_seed() -> int:
    value = c.get('CLONES_SEED') if c.has('CLONES_SEED') else ''
    return int(value) if str(value).strip() else 0


def clone_name(parent: str, index: int, width: int = 3) -> str:
    """'463~c001': clone `index` (from 1) of `parent`."""
    return f'{parent}{CLONE_MARK}{index:0{width}d}'


def parent_of(name: str) -> Optional[str]:
    """The parent of a clone name ('463~c001' -> '463'), None for an ordinary body."""
    text = str(name)
    head, mark, tail = text.rpartition(CLONE_MARK)
    return head if mark and tail.isdigit() else None


# Eigen fallback: a correlation eigenvalue below -NEGATIVE_EIGEN x the largest is a real indefiniteness (raises); one
# below RANK_EIGEN x the largest (numpy's matrix_rank tolerance, 6 eps) is a null mode of the repaired matrix
NEGATIVE_EIGEN = 1e-8
RANK_EIGEN = 6 * np.finfo(float).eps


def factor_and_rank(covariance: np.ndarray) -> tuple:
    """(L, kept): L with L L^T = covariance, through the correlation matrix (the scaled matrix is far better
    conditioned: a covariance of a well-observed orbit has condition numbers of 1e13-1e19 in its raw units), and the
    columns of L that carry a mode (`kept`, all True after Cholesky).

    Cholesky; when float64 finds the matrix indefinite, the eigen factor (a warning): eigenvalues below RANK_EIGEN x the
    largest (the slightly negative ones of round-off included) are null modes, their columns zero; one below
    -NEGATIVE_EIGEN x the largest raises."""
    covariance = np.asarray(covariance, dtype=float)
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)
    correlation = 0.5 * (correlation + correlation.T)
    try:
        lower = np.linalg.cholesky(correlation)
        kept = np.ones(len(scale), dtype=bool)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(correlation)
        if values.min() < -NEGATIVE_EIGEN * values.max():
            raise ValueError(f'the covariance is not positive semi-definite (correlation eigenvalue {values.min():.3g})') from None
        kept = values > RANK_EIGEN * values.max()
        logger.warning(
            f'covariance: Cholesky failed, eigen factor used (smallest correlation eigenvalue {values.min():.3g}, rank '
            f'{int(kept.sum())} of {len(values)})'
        )
        lower = vectors * np.sqrt(np.where(kept, values, 0.0))
    return scale[:, None] * lower, kept


def factor(covariance: np.ndarray) -> np.ndarray:
    """L with L L^T = covariance (`factor_and_rank`)."""
    return factor_and_rank(covariance)[0]


def generator(seed: int, name: str) -> np.random.Generator:
    """The clones' generator of a body: SeedSequence([seed, crc32(name)]), so two bodies never share their draws."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), zlib.crc32(str(name).encode())]))


@dataclass
class Ensemble:
    """The clones of one object at `epoch_mjd` (ecliptic J2000, heliocentric).

    `clones` has one row per clone: `clone` (index from 1), the Keplerian elements a, e, inc, Omega, omega, M [rad] at
    the epoch, `valid` (a bound orbit), `d2` (squared Mahalanobis distance of the draw, pseudo-inverse over `rank`
    modes; nan for a user list) and, for a drawn ensemble, the draw in the fit's coordinates (`x_<label>`). `nominal` is
    the nominal orbit (elements dict) or None. `solution` is the orbit solution they were drawn from (None for a user
    list). `rank`: the covariance's numerical rank the draws used (6, fewer after the eigen fallback; None for a user
    list), the degrees of freedom of d2."""

    name: str
    epoch_mjd: float
    clones: pd.DataFrame
    nominal: Optional[dict] = None
    solution: Optional[OrbitSolution] = None
    seed: Optional[int] = None
    method: str = 'gaussian'
    rank: Optional[int] = None
    _at_date: dict = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.clones)

    @property
    def n_valid(self) -> int:
        return int(self.clones['valid'].sum())

    @property
    def nongrav(self) -> dict:
        return dict(self.solution.nongrav) if self.solution is not None else {}

    def clone_names(self, parent: Optional[str] = None) -> list:
        parent = self.name if parent is None else parent
        width = max(3, len(str(int(self.clones['clone'].max())))) if len(self.clones) else 3
        return [clone_name(parent, int(index), width) for index in self.clones['clone']]

    def clone_elements(self) -> list:
        """The elements dict of every clone at the ensemble's epoch ('epoch' MJD), None for an invalid draw."""
        out = []
        for row in self.clones.itertuples(index=False):
            if not row.valid:
                out.append(None)
                continue
            out.append({**{key: float(getattr(row, key)) for key in KEPLERIAN_KEYS}, 'epoch': float(self.epoch_mjd)})
        return out

    def provenance(self) -> dict:
        """What the JSON beside the CSV keeps: the solution (nominal, covariance, units, epoch, source, URL, sha256), the
        seed and method, the nominal elements."""
        return {
            'name': self.name,
            'epoch_mjd': float(self.epoch_mjd),
            'n': len(self.clones),
            'n_valid': self.n_valid,
            'seed': self.seed,
            'method': self.method,
            'covariance_rank': self.rank,
            'nominal': self.nominal,
            'solution': self.solution.to_dict() if self.solution is not None else None,
        }

    def save(self, path: Union[str, Path]) -> tuple:
        """`<path>.csv` (one row per clone) and `<path>.json` (provenance). Returns the two paths."""
        path = Path(path)
        base = path.with_suffix('') if path.suffix in ('.csv', '.json') else path
        base.parent.mkdir(parents=True, exist_ok=True)
        csv_path, json_path = base.with_name(base.name + '.csv'), base.with_name(base.name + '.json')
        self.clones.to_csv(csv_path, index=False, float_format='%.17g')
        json_path.write_text(json.dumps(self.provenance(), indent=2, default=float))
        return str(csv_path), str(json_path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Ensemble':
        """An ensemble saved by `save` (the CSV path, the JSON path or their common base)."""
        path = Path(path)
        base = path.with_suffix('') if path.suffix in ('.csv', '.json') else path
        meta = json.loads(base.with_name(base.name + '.json').read_text())
        clones = pd.read_csv(base.with_name(base.name + '.csv'))
        clones['valid'] = clones['valid'].astype(bool)
        solution = OrbitSolution.from_dict(meta['solution']) if meta.get('solution') else None
        return cls(
            name=str(meta['name']),
            epoch_mjd=float(meta['epoch_mjd']),
            clones=clones,
            nominal=meta.get('nominal'),
            solution=solution,
            seed=meta.get('seed'),
            method=meta.get('method', 'gaussian'),
            rank=meta.get('covariance_rank'),
        )

    @classmethod
    def from_elements(
        cls, name: str, clones: Union[pd.DataFrame, Iterable[dict]], epoch_mjd: Optional[float] = None, nominal: Optional[dict] = None
    ) -> 'Ensemble':
        """An ensemble from the user's own clone list: rows with a, e, inc, Omega, omega, M (radians) and, unless
        `epoch_mjd` is given, 'epoch' (MJD, one for all rows). `nominal` (an elements dict) is optional. The clones are
        taken as equally weighted members (an unweighted list: weighted samples such as a line of variations do not give
        binomial shares)."""
        frame = pd.DataFrame(list(clones) if not isinstance(clones, pd.DataFrame) else clones).copy()
        missing = [key for key in KEPLERIAN_KEYS if key not in frame.columns]
        if missing:
            raise ValueError(f'clone rows need {", ".join(KEPLERIAN_KEYS)}; missing {", ".join(missing)}')
        if epoch_mjd is None:
            if 'epoch' not in frame.columns:
                raise ValueError('give epoch_mjd or an epoch column (MJD)')
            epochs = frame['epoch'].astype(float).unique()
            if len(epochs) != 1:
                raise ValueError(f'the clones of one ensemble share one epoch, got {sorted(epochs)}')
            epoch_mjd = float(epochs[0])
        if nominal is not None and 'epoch' in nominal and abs(float(nominal['epoch']) - float(epoch_mjd)) > 1e-9:
            raise ValueError('the nominal is at another epoch than the clones')
        frame = frame[[key for key in KEPLERIAN_KEYS]].astype(float)
        valid = (frame['a'] > 0) & (frame['e'] >= 0) & (frame['e'] < 1) & np.isfinite(frame).all(axis=1)
        frame.insert(0, 'clone', np.arange(1, len(frame) + 1))
        frame['valid'] = valid.to_numpy()
        frame['d2'] = np.nan
        nominal = None if nominal is None else {**{key: float(nominal[key]) for key in KEPLERIAN_KEYS}, 'epoch': float(epoch_mjd)}
        return cls(name=str(name), epoch_mjd=float(epoch_mjd), clones=frame, nominal=nominal, method='user')


def sample(solution: OrbitSolution, n: Optional[int] = None, seed: Optional[int] = None, name: Optional[str] = None) -> Ensemble:
    """`n` clones of `solution` (CLONES when None and set, else 100), seeded by (seed, crc32(name)); seed None =
    CLONES_SEED (0 when unset). See the module's docstring."""
    n = int(n) if n is not None else (configured_count() or DEFAULT_CLONES)
    if n < 1:
        raise ValueError(f'n is at least 1, got {n}')
    seed = configured_seed() if seed is None else int(seed)
    name = str(solution.name if name is None else name)
    lower, kept = factor_and_rank(solution.covariance)
    z = generator(seed, name).standard_normal((n, 6))
    draws = solution.nominal + z @ lower.T
    kep = solution.to_keplerian(draws)
    frame = pd.DataFrame({'clone': np.arange(1, n + 1)})
    for key in KEPLERIAN_KEYS:
        frame[key] = kep[key]
    frame['valid'] = kep['valid']
    # dX = L z: dX^T C^+ dX = the sum of z_j^2 over the modes L keeps (any generalised inverse of C gives it on the
    # range of C); after Cholesky every mode, |z|^2
    frame['d2'] = np.einsum('ij,ij->i', z[:, kept], z[:, kept])
    for index, label in enumerate(solution.labels):
        frame[f'x_{label}'] = draws[:, index]
    invalid = int((~kep['valid']).sum())
    if invalid:
        logger.warning(f'{name}: {invalid} of {n} clones are no bound orbit (e >= 1); kept as invalid clones, not redrawn')
    return Ensemble(
        name=name,
        epoch_mjd=float(solution.epoch_mjd),
        clones=frame,
        nominal=solution.nominal_elements(),
        solution=solution,
        seed=seed,
        method='gaussian',
        rank=int(kept.sum()),
    )
