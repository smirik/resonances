"""Conversions of orbit-fit coordinates to the Keplerian elements the package adds to REBOUND.

Pure numpy, vectorised over rows (one row per clone). Every function returns a dict of arrays a [AU], e, inc, Omega,
omega, M [rad] (angles in [0, 2 pi)) and `valid`: False for a row that is no bound orbit (e >= 1, a <= 0, q <= 0, a
non-finite value). Such a row is a legitimate draw of the Gaussian, not an error: it is kept and counted, never redrawn.

Heliocentric, ecliptic J2000 throughout; mu = k^2 AU^3/day^2 (the Sun's GM in Gaussian units, which the package's
time unit 1/k day makes 1).
"""

import numpy as np

from resonances.data.const import K

TWO_PI = 2.0 * np.pi
MU_AU3_DAY2 = K**2
KEPLERIAN_KEYS = ('a', 'e', 'inc', 'Omega', 'omega', 'M')
# Below this eccentricity the pericentre has no direction (the Cartesian map sets omega = 0, f = the argument of latitude)
CIRCULAR_E = 1e-12


def _rows(values, width: int = 6) -> np.ndarray:
    array = np.atleast_2d(np.asarray(values, dtype=float))
    if array.shape[1] != width:
        raise ValueError(f'expected rows of {width} coordinates, got shape {array.shape}')
    return array


def _result(a, e, inc, Omega, omega, M, valid) -> dict:
    valid = valid & np.isfinite(a) & np.isfinite(e) & np.isfinite(inc) & np.isfinite(M)
    out = {
        'a': a,
        'e': e,
        'inc': inc,
        'Omega': np.mod(Omega, TWO_PI),
        'omega': np.mod(omega, TWO_PI),
        'M': np.mod(M, TWO_PI),
    }
    for key in KEPLERIAN_KEYS:
        out[key] = np.where(valid, out[key], np.nan)
    out['valid'] = valid
    return out


def equinoctial_to_keplerian(rows) -> dict:
    """OEF equinoctial (a, h = e sin varpi, k = e cos varpi, p = tan(i/2) sin Omega, q = tan(i/2) cos Omega, lambda) to
    Keplerian. lambda is in DEGREES, as in the AstDyS/NEODyS EQU records and their covariances.

    e = sqrt(h^2 + k^2), varpi = atan2(h, k), i = 2 atan(sqrt(p^2 + q^2)), Omega = atan2(p, q), omega = varpi - Omega,
    M = lambda - varpi. Valid for 0 <= i < 180 deg (tan(i/2) > 1 is a retrograde orbit)."""
    x = _rows(rows)
    a, h, k, p, q, lam = x.T
    e = np.hypot(h, k)
    varpi = np.arctan2(h, k)
    inc = 2.0 * np.arctan(np.hypot(p, q))
    Omega = np.arctan2(p, q)
    M = np.radians(lam) - varpi
    return _result(a, e, inc, Omega, varpi - Omega, M, (a > 0) & (e < 1))


def cometary_to_keplerian(rows, epoch_jd: float) -> dict:
    """Cometary (e, q [AU], tp [JD, the solution's time scale], Omega, omega, i [deg]) at `epoch_jd` to Keplerian, as in the
    JPL SBDB covariance (labels e, q, tp, node, peri, i): a = q / (1 - e), M = n (epoch - tp), n = k a^-3/2 rad/day."""
    x = _rows(rows)
    e, q, tp, node, peri, inc = x.T
    elliptic = (e >= 0) & (e < 1) & (q > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(elliptic, q / (1.0 - e), np.nan)
        n = K * np.abs(a) ** -1.5
    M = n * (float(epoch_jd) - tp)
    return _result(a, e, np.radians(inc), np.radians(node), np.radians(peri), M, elliptic)


def cartesian_to_keplerian(rows, mu: float = MU_AU3_DAY2) -> dict:
    """Heliocentric ecliptic position [AU] and velocity [AU/day] (x, y, z, vx, vy, vz) to Keplerian with `mu` [AU^3/day^2].

    The standard two-body map: h = r x v, e = ((v^2 - mu/r) r - (r.v) v) / mu, 1/a = 2/r - v^2/mu, the node from z x h,
    omega from the node to e, the argument of latitude u from the node to r, the true anomaly f = u - omega, then the
    eccentric and mean anomalies. f is never taken from e: on a circular orbit (e < CIRCULAR_E) e has no direction,
    omega is set to 0 and f = u, so the position (which depends on omega + f only) is kept."""
    x = _rows(rows)
    r_vec, v_vec = x[:, :3], x[:, 3:]
    r = np.linalg.norm(r_vec, axis=1)
    v2 = np.einsum('ij,ij->i', v_vec, v_vec)
    rv = np.einsum('ij,ij->i', r_vec, v_vec)
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec, axis=1)
    e_vec = ((v2 - mu / r)[:, None] * r_vec - rv[:, None] * v_vec) / mu
    e = np.linalg.norm(e_vec, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_a = 2.0 / r - v2 / mu
        a = 1.0 / inv_a
        inc = np.arccos(np.clip(h_vec[:, 2] / h, -1.0, 1.0))
        Omega = np.arctan2(h_vec[:, 0], -h_vec[:, 1])
        node = np.column_stack([np.cos(Omega), np.sin(Omega), np.zeros_like(Omega)])
        h_hat = h_vec / h[:, None]
        # omega: from the ascending node to the pericentre, in the orbital plane (sense of h)
        omega = np.arctan2(np.einsum('ij,ij->i', np.cross(node, e_vec), h_hat), np.einsum('ij,ij->i', node, e_vec))
        omega = np.where(e > CIRCULAR_E, omega, 0.0)
        # u: from the ascending node to r, in the sense of h (defined for any e; for i = 0 or 180 deg the node is the
        # direction Omega gives, and u is measured from it)
        u = np.arctan2(np.einsum('ij,ij->i', np.cross(node, r_vec), h_hat), np.einsum('ij,ij->i', node, r_vec))
        f = u - omega
        E = np.arctan2(np.sqrt(np.clip(1.0 - e**2, 0.0, None)) * np.sin(f), e + np.cos(f))
    M = E - e * np.sin(E)
    return _result(a, e, inc, Omega, omega, M, (inv_a > 0) & (e < 1))


def mean_longitude(elements: dict) -> np.ndarray:
    """lambda = Omega + omega + M of Keplerian elements [rad], not wrapped."""
    return np.asarray(elements['Omega']) + np.asarray(elements['omega']) + np.asarray(elements['M'])


def drift_mean_anomaly(elements: dict, days: float) -> dict:
    """The same orbits `days` later on their two-body ellipses: M += n days, n = k a^-3/2 (used only for shifts of an
    epoch of a few seconds at most, EPOCH_TOLERANCE_DAYS, where the planets' perturbation is nil)."""
    out = dict(elements)
    a = np.asarray(elements['a'], dtype=float)
    out['M'] = np.mod(np.asarray(elements['M'], dtype=float) + K * a**-1.5 * float(days), TWO_PI)
    return out
