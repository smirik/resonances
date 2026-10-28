"""Elements measured from the invariable plane (`const.INVARIABLE_PLANE`), by an exact rotation.

The package's elements are ecliptic J2000. Everything whose physics is defined against the plane of the planets'
total angular momentum (the inclination of an MMR's nodal factor, Lidov's integrals c1 and c2, the e-i coupling of a
ZLK row) reads them from this plane: the angular momentum and the perihelion direction are rotated, never the angles
subtracted. Shared by the MMR analysis and the Lidov-Kozai columns.
"""

from typing import Tuple

import numpy as np

from resonances.data import const


def rotation_to_invariable() -> np.ndarray:
    """R = Rz(Omega_I) Rx(-I) Rz(-Omega_I): ecliptic vectors into the frame of the invariable plane
    (`const.INVARIABLE_PLANE`), longitudes counted from about the same origin."""
    inc, node = np.radians(const.INVARIABLE_PLANE['inc']), np.radians(const.INVARIABLE_PLANE['Omega'])

    def rz(a):
        return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])

    def rx(a):
        return np.array([[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]])

    return rz(node) @ rx(-inc) @ rz(-node)


def to_invariable_plane(inc, Omega, omega) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(inc', Omega', varpi') [rad] of orbits given by ecliptic (inc, Omega, omega), measured from the invariable plane:
    the angular momentum and the perihelion direction are rotated; varpi' = Omega' + omega'."""
    inc, Omega, omega = (np.asarray(x, dtype=float) for x in (inc, Omega, omega))
    ci, si, cn, sn, cw, sw = np.cos(inc), np.sin(inc), np.cos(Omega), np.sin(Omega), np.cos(omega), np.sin(omega)
    h = np.stack([si * sn, -si * cn, ci])
    p = np.stack([cn * cw - sn * sw * ci, sn * cw + cn * sw * ci, sw * si])
    rotation = rotation_to_invariable()
    h, p = np.tensordot(rotation, h, axes=1), np.tensordot(rotation, p, axes=1)
    inc_new = np.arccos(np.clip(h[2], -1.0, 1.0))
    node_new = np.arctan2(h[0], -h[1])
    node_x, node_y = np.cos(node_new), np.sin(node_new)
    # h x N with N = (cos, sin, 0)
    cross = np.stack([-h[2] * node_y, h[2] * node_x, h[0] * node_y - h[1] * node_x])
    omega_new = np.arctan2(np.sum(p * cross, axis=0), p[0] * node_x + p[1] * node_y)
    return inc_new, node_new, node_new + omega_new


def inclination_to_invariable_plane(inc, Omega) -> np.ndarray:
    """The inclination [rad] from the invariable plane of orbits given by ecliptic (inc, Omega): arccos(h . n), h the
    unit angular momentum and n the plane's normal (exact; the node is not needed)."""
    inc, Omega = np.asarray(inc, dtype=float), np.asarray(Omega, dtype=float)
    inc_plane, node_plane = np.radians(const.INVARIABLE_PLANE['inc']), np.radians(const.INVARIABLE_PLANE['Omega'])
    dot = np.cos(inc) * np.cos(inc_plane) + np.sin(inc) * np.sin(inc_plane) * np.cos(Omega - node_plane)
    return np.arccos(np.clip(dot, -1.0, 1.0))
