import numpy as np

from resonances.resonance.resonance import Resonance


class LidovKozaiParameters:
    """
    The integrals of the quadrupole, double-averaged circular restricted problem in Lidov's
    notation (Lidov 1961; Shevchenko 2017, eqs. 3.35, 3.43, 3.41):

        c1     = (1 - e^2) cos^2 i
        c2     = e^2 (2/5 - sin^2 i sin^2 w)
        h_quad = (2 + 3 e^2)(3 cos^2 i - 1) + 15 e^2 sin^2 i cos 2w = 6 c1 - 2 + 30 c2

    omega librates when c1 < 3/5 and c2 < 0, and circulates otherwise. The sign of h_quad
    is not a libration criterion. The criterion is exact only for a/a_pert -> 0; for
    main-belt asteroids (a/a_J ~ 0.5) it is indicative: 1373 Cincinnati librates at c2 > 0.
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
        """(c1, c2, h_quad)."""
        return (
            cls.compute_c1(eccentricity, inclination),
            cls.compute_c2(eccentricity, inclination, argument_of_pericenter),
            cls.compute_h_quad(eccentricity, inclination, argument_of_pericenter),
        )


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
        The resonant argument is the argument of pericenter ω.
        """
        return body_orbit.omega

    def to_s(self):
        return self.identifier

    def to_short(self):
        return self.identifier
