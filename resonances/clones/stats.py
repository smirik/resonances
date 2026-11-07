"""Status shares of an ensemble of clones with their binomial confidence intervals (clones.csv, docs/clones.md).

The share of clones with a status is a Monte Carlo estimate of a conditional probability: given the provider's orbit
solution (its Gaussian), the package's dynamical model and the integration span. Its interval is Wilson's score
interval (Wilson 1927), which, unlike the normal approximation, stays inside [0, 1] and is not empty at 0/N or N/N:

    p_hat = k / N,  centre = (p_hat + z^2 / 2N) / (1 + z^2 / N),
    half = z / (1 + z^2 / N) * sqrt(p_hat (1 - p_hat) / N + z^2 / 4N^2),  z = 1.959964 for 95 %.

Its half-width at p = 0.5 is about 1/sqrt(N): +-10 % for N = 100, +-5 % for 400, +-3 % for 1000. It says nothing about
model or classifier errors.

A draw that is no bound orbit (`Ensemble`: e < 0, e >= 1, a <= 0) is not integrated and is no observation of any
status: the shares are over the bound draws only (`cloud_row`). When more than CLONES_MAX_INVALID of the draws are
unbound, the Gaussian is no usable model of the orbit's uncertainty (author 2026-11-06: 2010 CG236, 2010 DH65, 1-day
arcs, gave 0/100 'bounded' that read as evidence against co-orbital motion): the row then has no shares at all.
"""

import math
from typing import Iterable, Optional

from resonances.resonance.analysis.verdict import Status

Z_95 = 1.959963984540054
STATUSES = tuple(str(s) for s in Status)


def wilson_interval(k: int, n: int, z: float = Z_95) -> tuple:
    """(low, high) of Wilson's score interval for k successes out of n (z = 1.96: two-sided 95 %); (nan, nan) for n = 0."""
    if n <= 0:
        return float('nan'), float('nan')
    if not 0 <= k <= n:
        raise ValueError(f'k must be within 0..n, got k={k}, n={n}')
    p = k / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half = z / denominator * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    low = 0.0 if k == 0 else max(0.0, centre - half)  # exact at 0/N and N/N (round-off leaves 1e-18 there)
    high = 1.0 if k == n else min(1.0, centre + half)
    return low, high


def status_shares(statuses: Iterable[Optional[str]], z: float = Z_95) -> dict:
    """The counts n_<status>, shares f_<status> and Wilson bounds f_<status>_lo/_hi of the six statuses, over every clone
    with a status (None = a clone without a verdict for this row, counted in n_missing and not in the shares); the modal
    status and its share (ties: the first in the order of `Status`)."""
    statuses = list(statuses)
    known = [str(s) for s in statuses if s not in (None, '')]
    unknown = [s for s in known if s not in STATUSES]
    if unknown:
        raise ValueError(f'unknown statuses {sorted(set(unknown))}')
    n = len(known)
    row = {'n_classified': n, 'n_missing': len(statuses) - n}
    counts = {status: known.count(status) for status in STATUSES}
    for status in STATUSES:
        low, high = wilson_interval(counts[status], n, z)
        row[f'n_{status}'] = counts[status]
        row[f'f_{status}'] = counts[status] / n if n else float('nan')
        row[f'f_{status}_lo'] = low
        row[f'f_{status}_hi'] = high
    modal = max(STATUSES, key=lambda status: counts[status]) if n else ''
    row['modal_status'] = modal
    row['modal_share'] = counts[modal] / n if n else float('nan')
    return row


def cloud_row(
    statuses: Iterable[Optional[str]],
    nominal_status: Optional[str],
    n_clones: int,
    n_draw_invalid: int,
    max_invalid: float,
    z: float = Z_95,
) -> dict:
    """`ensemble_row` of the bound draws (`statuses` holds no entry for an unbound draw) with `n_draw_invalid`,
    `f_draw_invalid` (its share of the n_clones draws) and `cloud_usable` (f_draw_invalid <= max_invalid). An unusable
    cloud keeps its counts but gets no shares: every f_*, the modal status and share and the nominal agreement are
    empty (NaN), so no statistic can be read from it."""
    row = ensemble_row(statuses, nominal_status, z)
    share = n_draw_invalid / n_clones if n_clones else float('nan')
    usable = bool(n_clones > 0 and share <= max_invalid)
    row.update({'n_draw_invalid': int(n_draw_invalid), 'f_draw_invalid': share, 'cloud_usable': usable})
    if not usable:
        for key in list(row):
            if (key.startswith('f_') and key != 'f_draw_invalid') or key.startswith('nominal_agreement') or key == 'modal_share':
                row[key] = float('nan')
        row['modal_status'] = ''
    return row


def upgrade_row(record: dict, max_invalid: float, z: float = Z_95) -> dict:
    """A clones.csv row (dict) in the semantics of `cloud_row`: a row written before 2026-11-06 (no f_draw_invalid, or
    NaN there) counted the unbound draws in n_invalid and in every share's denominator; its counts are taken back
    (n_invalid minus n_draw_invalid) and its shares, cloud_usable and nominal agreement recomputed. A current row is
    returned as it is."""
    share = record.get('f_draw_invalid')
    if share is not None and share == share:  # present and not NaN
        return record
    counts = {status: int(record.get(f'n_{status}', 0) or 0) for status in STATUSES}
    n_draw_invalid = int(record.get('n_draw_invalid', 0) or 0)
    counts['invalid'] = max(0, counts['invalid'] - n_draw_invalid)
    statuses = [s for s in STATUSES for _ in range(counts[s])] + [None] * int(record.get('n_missing', 0) or 0)
    nominal = record.get('nominal_status') or None
    nominal = nominal if isinstance(nominal, str) else None
    out = dict(record)
    out.update(cloud_row(statuses, nominal, int(record.get('n_clones', 0) or 0), n_draw_invalid, max_invalid, z))
    return out


def ensemble_row(statuses: Iterable[Optional[str]], nominal_status: Optional[str] = None, z: float = Z_95) -> dict:
    """status_shares plus the nominal's agreement: the share of classified clones with the nominal's status and its
    Wilson interval (a diagnostic: the nominal is one member of the cloud with no special weight)."""
    statuses = list(statuses)
    row = status_shares(statuses, z)
    known = [str(s) for s in statuses if s not in (None, '')]
    if nominal_status and known:
        agree = sum(1 for s in known if s == str(nominal_status))
        low, high = wilson_interval(agree, len(known), z)
        row.update({'nominal_agreement': agree / len(known), 'nominal_agreement_lo': low, 'nominal_agreement_hi': high})
    else:
        row.update({'nominal_agreement': float('nan'), 'nominal_agreement_lo': float('nan'), 'nominal_agreement_hi': float('nan')})
    return row
