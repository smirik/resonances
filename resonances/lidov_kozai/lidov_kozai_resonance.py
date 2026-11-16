from typing import Tuple

import numpy as np

from resonances.data.invariable_plane import FRAME_ECLIPTIC, FRAME_INVARIABLE, elements_in_invariable_plane, to_invariable_plane
from resonances.resonance.resonance import Resonance


def elements_in_frame(inc, Omega, omega, frame: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(inc, Omega, omega) [rad] of orbits given by ecliptic (inc, Omega, omega), measured from the plane `frame`
    names (ZLK_ANGLE_FRAME): 'ecliptic' returns them as they are (floats), 'invariable' rotates the orbit's angular
    momentum and perihelion vectors exactly (`data.invariable_plane.elements_in_invariable_plane`; the stored omega
    enters, never REBOUND's pomega or l, so an orbit with i > 90 deg has no jump). The Lidov-Kozai angle, its
    diagnostics, the free elements of a ZLK body and its figure read the same frame.

    From the ecliptic, the node of an orbit tilted from the invariable plane by about the plane's own 1.58 deg or less
    librates about the plane's node instead of turning, and omega = varpi - Omega with it (555527: i_inv 1.6 deg, omega
    bounded from the ecliptic, circulating from the invariable plane)."""
    if frame == FRAME_INVARIABLE:
        return elements_in_invariable_plane(inc, Omega, omega)
    if frame != FRAME_ECLIPTIC:
        raise ValueError(f"the ZLK angle frame is {FRAME_ECLIPTIC!r} or {FRAME_INVARIABLE!r}, got {frame!r}")
    return tuple(np.asarray(x, dtype=float) for x in (inc, Omega, omega))


class LidovKozaiParameters:
    """
    The integrals of the quadrupole, double-averaged circular restricted problem in Lidov's
    notation (Lidov 1961; Shevchenko 2017, eqs. 3.35, 3.43, 3.41):

        c1     = (1 - e^2) cos^2 i
        c2     = e^2 (2/5 - sin^2 i sin^2 w)
        h_quad = (2 + 3 e^2)(3 cos^2 i - 1) + 15 e^2 sin^2 i cos 2w = 6 c1 - 2 + 30 c2

    omega librates when c1 < 3/5 and c2 < 0, and circulates otherwise. The sign of h_quad
    is not a libration criterion. The criterion is exact only for a/a_pert -> 0; for
    main-belt asteroids (a/a_J ~ 0.5) it is indicative.

    The integrals are defined against the perturber's orbital plane; the package reads them
    against the invariable plane (`evaluate_invariable`, the columns c1, c2, h_quad: osculating,
    at the epoch, invariable plane). From the ecliptic they shift by up to 0.02 in c1, and the
    sign of c2 of 1373 Cincinnati flips (+0.0014 ecliptic, -0.0011 invariable).
    """

    @staticmethod
    def compute_c1(eccentricity, inclination):
        """Lidov's c1 (Kozai's Theta): (1 - e^2) cos^2 i, the squared normalized z-component of
        the angular momentum. Inclination in radians."""
        e = np.asarray(eccentricity, dtype=float)
        inc = np.asarray(inclination, dtype=float)
        return (1.0 - e**2) * np.cos(inc) ** 2

    @staticmethod
    def compute_c2(eccentricity, inclination, argument_of_pericenter):
        """Lidov's c2: e^2 (2/5 - sin^2 i sin^2 w). With c1 < 3/5, c2 < 0 means omega librates
        around 90 or 270 deg. Angles in radians."""
        e = np.asarray(eccentricity, dtype=float)
        inc = np.asarray(inclination, dtype=float)
        omega = np.asarray(argument_of_pericenter, dtype=float)
        return (e**2) * ((2.0 / 5.0) - np.sin(inc) ** 2 * np.sin(omega) ** 2)

    @staticmethod
    def compute_h_quad(eccentricity, inclination, argument_of_pericenter):
        """The quadrupole Hamiltonian in the Kinoshita-Nakai form, without its constant factor:
        (2 + 3e^2)(3 cos^2 i - 1) + 15 e^2 sin^2 i cos 2w. Angles in radians."""
        e = np.asarray(eccentricity, dtype=float)
        inc = np.asarray(inclination, dtype=float)
        omega = np.asarray(argument_of_pericenter, dtype=float)
        first_term = (2.0 + 3.0 * e**2) * (3.0 * np.cos(inc) ** 2 - 1.0)
        second_term = 15.0 * e**2 * np.sin(inc) ** 2 * np.cos(2.0 * omega)
        return first_term + second_term

    @classmethod
    def evaluate(cls, eccentricity, inclination, argument_of_pericenter):
        """(c1, c2, h_quad) of elements given in the plane they are wanted in."""
        return (
            cls.compute_c1(eccentricity, inclination),
            cls.compute_c2(eccentricity, inclination, argument_of_pericenter),
            cls.compute_h_quad(eccentricity, inclination, argument_of_pericenter),
        )

    @classmethod
    def evaluate_invariable(cls, eccentricity, inclination, node, argument_of_pericenter):
        """(c1, c2, h_quad) of ecliptic elements (radians), read in the invariable plane: i and omega
        rotated exactly (`data.invariable_plane.to_invariable_plane`), e unchanged."""
        inc, node_new, varpi = to_invariable_plane(inclination, node, argument_of_pericenter)
        return cls.evaluate(eccentricity, inc, varpi - node_new)


class LidovKozaiResonance(Resonance):
    """
    Resonance defined by the argument of pericenter (ω).

    The libration of ω around ±π/2 indicates the classic Lidov–Kozai mechanism.
    """

    def __init__(self, identifier: str = 'LK'):
        """
        Parameters
        ----------
        identifier : str, optional
            Custom identifier used in output tables. Defaults to "LK".
        """
        self.identifier = identifier or 'LK'

    @property
    def type(self) -> str:
        return 'lidov_kozai'

    def calc_angle(self, body_orbit, planets=None):
        """
        The resonant argument is the argument of pericenter ω, ecliptic as the integration stores it. The analysis
        reads ω in the plane ZLK_ANGLE_FRAME names (`elements_in_frame`; default the invariable plane) and writes that
        angle back to `body.angles` (`Simulation._analyse_long_period`).
        """
        return body_orbit.omega

    def to_s(self):
        return self.identifier

    def to_short(self):
        return self.identifier
