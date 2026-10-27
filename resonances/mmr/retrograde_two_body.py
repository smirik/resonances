import re
from math import gcd
from typing import ClassVar

from resonances.data import const
from resonances.mmr.mmr import MMR


class RetrogradeTwoBody(MMR):
    """The p/-q resonance of a body on a retrograde orbit (i > 90 deg) with one planet.

    Notation `1J-3r0`: p (the planet's coefficient), the planet's letter, q (the body's),
    `r`, and k. The commensurability is q n = p n' (n' the planet's mean motion), so the
    resonant axis is (q/p)^(2/3) a', the same as for the prograde p/q.

    The angle (Morais & Namouni 2013a, eq. 1; Li, Huang & Gong 2019, eq. 2) is

        phi = q lam* - p lam' - (p + q - 2k) varpi* + 2k Omega,
        varpi* = omega - Omega,  lam* = M + omega - Omega,  0 <= 2k <= p + q,

    i.e. q M + (2k - p) omega + p Omega - p lam'. Its order is p + q (amplitude
    ~ cos^2k(I/2) e^(p+q-2k)); k = 0 dominates near the plane, larger k at polar orbits.
    The text of Li et al. writes the condition as "pn - qn' = 0": a typo, their eqs. (1)-(2)
    and the real objects give q n - p n' = 0.

    The angle is built from the body's Omega, omega and M only. For i > 90 deg REBOUND
    returns `l` = -lam* and `pomega` = -varpi*, and those jump when i crosses 90 deg.
    Rotating the frame by delta about z shifts lam*, varpi* by -delta and lam', Omega by
    +delta: hence `rotation_weights` instead of the plain D'Alembert sum.
    """

    mmr_type: ClassVar[str] = 'two-body-retrograde'
    retrograde: ClassVar[bool] = True
    rotation_weights: ClassVar[tuple] = (-1, 1, -1, 1)  # (lam*, lam', varpi*, Omega)
    NOTATION: ClassVar[re.Pattern] = re.compile(r'^(\d+)([A-Z])-(\d+)r(\d+)$')

    def __init__(self, notation: str = None, *, p: int = None, q: int = None, k: int = None, planet: str = None):
        if notation is not None:
            p, q, k, planet = self.parse(notation)
        if None in (p, q, k, planet):
            raise ValueError('A retrograde resonance needs a notation (e.g. 1J-3r0) or all of p, q, k and planet.')
        p, q, k = int(p), int(q), int(k)
        if p < 1 or q < 1:
            raise ValueError(f'p and q of a retrograde resonance must be positive. Given p={p}, q={q}.')
        if gcd(p, q) > 1:
            raise ValueError(f'p and q of a retrograde resonance must have gcd 1. Given p={p}, q={q}.')
        if k < 0 or 2 * k > p + q:
            raise ValueError(f'k of a retrograde resonance must satisfy 0 <= 2k <= p + q. Given p={p}, q={q}, k={k}.')
        if len(planet) == 1:
            planet = self.get_planet_name_from_letter(planet)
        self.p, self.q, self.k = p, q, k
        super().__init__([q, -p, -(p + q - 2 * k), 2 * k], [planet])

    @classmethod
    def parse(cls, notation: str):
        """(p, q, k, planet letter) from `1J-3r0`; ValueError for anything else."""
        match = cls.NOTATION.match(notation.replace(' ', ''))
        if match is None:
            raise ValueError(
                f"Bad retrograde resonance notation '{notation}'. Use <p><planet letter>-<q>r<k>, "
                "e.g. 1J-3r0 for the 1/-3 resonance with Jupiter with k = 0 (k is required)."
            )
        p, letter, q, k = match.groups()
        return int(p), int(q), int(k), letter

    def calc_angle(self, body, planets):
        """body, planets[0]: REBOUND orbits. Not body.l / body.pomega (see the class docstring)."""
        lam = body.M + body.omega - body.Omega
        varpi = body.omega - body.Omega
        return self.q * lam - self.p * planets[0].l - (self.p + self.q - 2 * self.k) * varpi + 2 * self.k * body.Omega

    def order(self) -> int:
        return self.p + self.q

    def to_s(self) -> str:
        return f'{self.p}{self.get_letter_from_planet_name(self.planets_names[0])}-{self.q}r{self.k}'

    def to_short(self) -> str:
        return self.to_s()

    def calculate_resonant_axis(self):
        return const.PLANETS_AXIS[self.planets_names[0]] * (self.q / self.p) ** (2.0 / 3.0)
