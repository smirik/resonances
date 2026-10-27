"""Status shares of an ensemble of clones with their binomial confidence intervals (clones.csv, docs/clones.md).

The share of clones with a status is a Monte Carlo estimate of a conditional probability: given the provider's orbit
solution (its Gaussian), the package's dynamical model and the integration span. Its interval is Wilson's score
interval (Wilson 1927), which, unlike the normal approximation, stays inside [0, 1] and is not empty at 0/N or N/N:

    p_hat = k / N,  centre = (p_hat + z^2 / 2N) / (1 + z^2 / N),
    half = z / (1 + z^2 / N) * sqrt(p_hat (1 - p_hat) / N + z^2 / 4N^2),  z = 1.959964 for 95 %.

Its half-width at p = 0.5 is about 1/sqrt(N): +-10 % for N = 100, +-5 % for 400, +-3 % for 1000. It says nothing about
model or classifier errors.
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
