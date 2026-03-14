import numpy as np

from resonances.resonance.resonance import Resonance


class LidovKozaiParameters:
    """
    Helper class that provides analytical Lidov–Kozai invariants.

    The parameters follow the classic quadrupole-level definitions introduced by
    Lidov (1962) and Kozai (1962). They provide a fast analytical diagnostic to
    verify whether a body can occupy the Lidov–Kozai libration island.
    """

    @staticmethod
    def compute_c1(eccentricity, inclination):
        """
        Kozai's Θ (or Lidov's c1) – the squared normalized z-component of the angular momentum.

        Parameters
        ----------
        eccentricity : float or np.ndarray
            Orbital eccentricity.
        inclination : float or np.ndarray
            Orbital inclination in radians.

        Returns
        -------
        float or np.ndarray
            Value of c1 = (1 - e^2) * cos^2(i)
        """
        e = np.asarray(eccentricity, dtype=float)
        inc = np.asarray(inclination, dtype=float)
        return (1.0 - e**2) * np.cos(inc) ** 2

    @staticmethod
    def compute_c(eccentricity, inclination, argument_of_pericenter):
        """
        Lidov's c parameter – the additional invariant introduced by Lidov for direct libration prediction.

        Parameters
        ----------
        eccentricity : float or np.ndarray
            Orbital eccentricity.
        inclination : float or np.ndarray
            Orbital inclination in radians.
        argument_of_pericenter : float or np.ndarray
            Argument of pericenter (ω) in radians.

        Returns
        -------
        float or np.ndarray
            Value of c (c < 0 typically corresponds to libration islands)
        """
        e = np.asarray(eccentricity, dtype=float)
        inc = np.asarray(inclination, dtype=float)
        omega = np.asarray(argument_of_pericenter, dtype=float)

        sin_inc = np.sin(inc)
        sin_omega = np.sin(omega)

        return (e**2) * ((2.0 / 5.0) - (sin_inc**2) * (sin_omega**2))

    @staticmethod
    def compute_c2(eccentricity, inclination, argument_of_pericenter):
        """
        Lidov's c2 parameter – energy-like integral that predicts libration of ω.

        Parameters
        ----------
        eccentricity : float or np.ndarray
            Orbital eccentricity.
        inclination : float or np.ndarray
            Orbital inclination in radians.
        argument_of_pericenter : float or np.ndarray
            Argument of pericenter (ω) in radians.

        Returns
        -------
        float or np.ndarray
            Value of c2 based on the quadrupole Hamiltonian (c2 < 0 -> LK state)
        """
        e = np.asarray(eccentricity, dtype=float)
        inc = np.asarray(inclination, dtype=float)
        omega = np.asarray(argument_of_pericenter, dtype=float)

        cos_inc = np.cos(inc)
        sin_inc = np.sin(inc)

        first_term = (2.0 + 3.0 * e**2) * (3.0 * cos_inc**2 - 1.0)
        second_term = 15.0 * e**2 * sin_inc**2 * np.cos(2.0 * omega)
        return first_term + second_term

    @classmethod
    def evaluate(cls, eccentricity, inclination, argument_of_pericenter):
        """
        Convenience method returning both Lidov–Kozai invariants.
        """
        c1 = cls.compute_c1(eccentricity, inclination)
        c2 = cls.compute_c2(eccentricity, inclination, argument_of_pericenter)
        c = cls.compute_c(eccentricity, inclination, argument_of_pericenter)
        return c1, c2, c


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
