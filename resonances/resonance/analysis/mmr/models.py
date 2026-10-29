"""Inputs and results of the MMR analysis (pure dataclasses; arrays are numpy float64).

Units inside the analysis: time in Gaussian years (2 pi REBOUND units, 365.2569 d), angles in
radians, a in au, rates in rad/yr, a_dot in au/yr. CSV columns carry their unit in the name
(`_deg`, `_deg_yr`, `_rad_kyr`, `_kyr`, ...); each result's `to_columns()` does the conversion.
Not measured = nan (float) or None, written as an empty cell, never 0.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from resonances.resonance.analysis.geometry import Geometry
from resonances.resonance.analysis.mmr.coorbital import CoorbitalLaw
from resonances.resonance.analysis.pendulum import EnergyAnalysis
from resonances.resonance.analysis.verdict import CheckValue, Status, Verdict, join_list

NAN = float('nan')

PLANET_LETTERS = {'R': 'Mercury', 'V': 'Venus', 'E': 'Earth', 'M': 'Mars', 'J': 'Jupiter', 'S': 'Saturn', 'U': 'Uranus', 'N': 'Neptune'}
PLANET_NAMES_ORDER = ('Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune')

RAD_TO_DEG = 180.0 / math.pi
RAD_TO_ARCSEC = RAD_TO_DEG * 3600.0

# planet times may differ from the record's by this fraction of a step (CSV rounding; the same tolerance as io.py)
SAME_GRID_TOLERANCE = 1e-3

_RETROGRADE_KEY = re.compile(r'^(\d+)([A-Z])-(\d+)r(\d+)$')
_KEY_TOKEN = re.compile(r'([+-]?\d+)([A-Z]?)')


def column_number(value: Any, scale: float = 1.0) -> Optional[float]:
    """A number for a CSV cell in CSV units: None (an empty cell) when not measured (None/NaN)."""
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value * scale


def blank_columns(columns: dict) -> dict:
    """The keys of a `to_columns()` row, empty: '' for a text cell, None (an empty number) for every other."""
    return {key: ('' if isinstance(value, str) else None) for key, value in columns.items()}


def column_list(values: Iterable[Any], scale: float = 1.0, digits: int = 6) -> str:
    """A ';'-separated list cell of numbers (rounded to `digits` significant digits)."""
    return join_list(f'{float(v) * scale:.{digits}g}' for v in values)


def planet_letter(name: str) -> str:
    """'Jupiter' -> 'J' (the package notation; Mercury is 'R')."""
    for letter, planet in PLANET_LETTERS.items():
        if planet == name:
            return letter
    raise ValueError(f'Unknown planet {name!r}')


def planet_terms(planets: Iterable[str], coefficients: Iterable[int]) -> str:
    """The planets' part of the package notation: ('Jupiter', 'Saturn'), (3, -1) -> '3J-1S' (no leading '+')."""
    return ''.join(f'{k:+d}{planet_letter(name)}' for name, k in zip(planets, coefficients)).lstrip('+')


def _planet_from_letter(letter: str, key: str) -> str:
    if letter not in PLANET_LETTERS:
        raise ValueError(f'Unknown planet letter {letter!r} in resonance {key!r}')
    return PLANET_LETTERS[letter]


class ResonanceKind(StrEnum):
    TWO_BODY = 'two_body'
    THREE_BODY = 'three_body'
    RETROGRADE = 'retrograde'


@dataclass(frozen=True)
class ResonanceSpec:
    """Coefficients of a resonant angle

        sigma = sum_j k_j lambda_j + k lambda + p varpi + sum_j p_j varpi_j + v Omega + sum_j v_j Omega_j.

    For a retrograde p'/-q' resonance (`1J-3r0`, Li et al. 2019 eq. 2) the body's terms are
    lambda* = M + omega - Omega and varpi* = omega - Omega, never REBOUND's l/pomega:
    k = q', p = -(p' + q' - 2k'), v = 2k', planet coefficient -p'.
    `key` is the package notation of the resonance (`3J-1S-1+0+0-1`, `9M-16+0+7`, `1J-3r0`).
    """

    key: str
    kind: ResonanceKind
    planets: Tuple[str, ...]
    planet_coefficients: Tuple[int, ...]  # k_j
    body_coefficient: int  # k
    body_varpi_coefficient: int  # p
    planet_varpi_coefficients: Tuple[int, ...]  # p_j
    body_node_coefficient: int = 0  # v
    planet_node_coefficients: Tuple[int, ...] = ()  # v_j (empty = zeros)
    nominal_axis_au: Optional[float] = None  # catalogue a_res, if known

    @property
    def order(self) -> int:
        """q = |sum_j k_j + k| (two- and three-body); p' + q' for a retrograde resonance."""
        if self.retrograde:
            return self.body_coefficient - self.planet_coefficients[0]
        return abs(sum(self.planet_coefficients) + self.body_coefficient)

    @property
    def retrograde(self) -> bool:
        return self.kind == ResonanceKind.RETROGRADE

    @classmethod
    def from_key(cls, key: str, nominal_axis_au: Optional[float] = None) -> 'ResonanceSpec':
        """Parse the package notation (long two-/three-body forms and `pPlanet-q r k`) without importing resonances.mmr.

        Accepted: `9M-16+0+7`, `2J-1` (short: p_j = 0, p from d'Alembert), `3J-1S-1+0+0-1`, `4J-2S-1`, `1J-3r0`.
        ValueError for anything else, including a long form whose coefficients do not sum to 0.
        """
        text = key.replace(' ', '')
        retrograde = _RETROGRADE_KEY.match(text)
        if retrograde:
            p, letter, q, k = retrograde.groups()
            return cls.retrograde_spec(int(p), int(q), int(k), _planet_from_letter(letter, key), nominal_axis_au)
        tokens = _KEY_TOKEN.findall(text)
        if not tokens or ''.join(number + letter for number, letter in tokens) != text:
            raise ValueError(f'Bad resonance notation {key!r}')
        n_planets = sum(1 for _, letter in tokens if letter)
        if n_planets not in (1, 2) or any(letter for _, letter in tokens[n_planets:]):
            raise ValueError(f'Bad resonance notation {key!r}: planet letters must lead')
        numbers = [int(number) for number, _ in tokens]
        planets = tuple(_planet_from_letter(letter, key) for _, letter in tokens[:n_planets])
        if len(numbers) == n_planets + 1:  # short form
            numbers = numbers + [0] * n_planets + [-sum(numbers)]
        if len(numbers) != 2 * n_planets + 2:
            raise ValueError(f'Bad resonance notation {key!r}: wrong number of coefficients')
        if sum(numbers) != 0:
            raise ValueError(f"Resonance {key!r} breaks the d'Alembert rule (coefficients sum to {sum(numbers)})")
        kind = ResonanceKind.TWO_BODY if n_planets == 1 else ResonanceKind.THREE_BODY
        return cls(
            key=text,
            kind=kind,
            planets=planets,
            planet_coefficients=tuple(numbers[:n_planets]),
            body_coefficient=numbers[n_planets],
            body_varpi_coefficient=numbers[-1],
            planet_varpi_coefficients=tuple(numbers[n_planets + 1 : 2 * n_planets + 1]),
            nominal_axis_au=nominal_axis_au,
        )

    @classmethod
    def retrograde_spec(cls, p: int, q: int, k: int, planet: str, nominal_axis_au: Optional[float] = None) -> 'ResonanceSpec':
        """The p/-q resonance `pP-qrk`: sigma = q lambda* - p lambda' - (p + q - 2k) varpi* + 2k Omega."""
        return cls(
            key=f'{p}{planet_letter(planet)}-{q}r{k}',
            kind=ResonanceKind.RETROGRADE,
            planets=(planet,),
            planet_coefficients=(-p,),
            body_coefficient=q,
            body_varpi_coefficient=-(p + q - 2 * k),
            planet_varpi_coefficients=(0,),
            body_node_coefficient=2 * k,
            nominal_axis_au=nominal_axis_au,
        )

    @classmethod
    def from_resonance(cls, resonance: Any) -> 'ResonanceSpec':
        """From a TwoBody/ThreeBody/RetrogradeTwoBody object, duck-typed on `.coeff`, `.planets_names`,
        `.to_s()`, `.resonant_axis` and the retrograde `.p/.q/.k` (no import of resonances.mmr)."""
        try:
            nominal = float(resonance.resonant_axis)
        except Exception:  # an object without a catalogue axis
            nominal = None
        planets = tuple(resonance.planets_names)
        if getattr(resonance, 'retrograde', False):
            return cls.retrograde_spec(int(resonance.p), int(resonance.q), int(resonance.k), planets[0], nominal)
        coefficients = [int(c) for c in resonance.coeff]
        n = len(planets)
        if len(coefficients) != 2 * n + 2:
            raise ValueError(f'{len(coefficients)} coefficients for {n} planet(s)')
        return cls(
            key=str(resonance.to_s()),
            kind=ResonanceKind.TWO_BODY if n == 1 else ResonanceKind.THREE_BODY,
            planets=planets,
            planet_coefficients=tuple(coefficients[:n]),
            body_coefficient=coefficients[n],
            body_varpi_coefficient=coefficients[-1],
            planet_varpi_coefficients=tuple(coefficients[n + 1 : 2 * n + 1]),
            nominal_axis_au=nominal,
        )

    @property
    def planet_varpi_terms(self) -> Tuple[int, ...]:
        """p_j per planet (zeros when `planet_varpi_coefficients` is empty)."""
        return self.planet_varpi_coefficients or (0,) * len(self.planets)

    def planet_node_coefficient(self, index: int) -> int:
        """v_j (0 when `planet_node_coefficients` is empty)."""
        return self.planet_node_coefficients[index] if self.planet_node_coefficients else 0


@dataclass(frozen=True)
class PlanetSeries:
    """Saved elements of one planet on the record's time grid (angles in rad, a in au)."""

    name: str
    t_yr: np.ndarray
    a: np.ndarray
    e: np.ndarray
    inc: np.ndarray
    Omega: np.ndarray
    varpi: np.ndarray
    mean_longitude: Optional[np.ndarray] = None


@dataclass(frozen=True)
class Regime:
    """How the record was produced (the analysis reads the step from the series itself)."""

    step_yr: float
    span_yr: float
    n_rows: int
    integrator: str = ''
    dt: float = NAN  # REBOUND units
    direction: str = 'forward'  # 'forward' | 'backward'
    epoch_jd: float = NAN
    decimation: int = 1
    planets_saved: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MMRSeries:
    """The input of `analyse_mmr`: raw elements of the body and the planets, and the wrapped angle.

    Times are increasing (a backward run is reversed by the caller). `angle_wrapped` is sigma mod 2pi
    as computed in the integration (exact at every sample); unwrapping is the analysis' job.

    `catalogue_longitudes` holds lambda_j at t_yr[0] [rad] of planets whose lambda is not saved: the owner search
    alone extends it as the line lambda_j0 + n_j (t - t_yr[0]) with the table n_j (owner.py, catalogue longitudes).
    """

    t_yr: np.ndarray
    a: np.ndarray
    e: np.ndarray
    inc: np.ndarray
    Omega: np.ndarray
    omega: np.ndarray
    varpi: np.ndarray
    angle_wrapped: np.ndarray
    resonance: ResonanceSpec
    regime: Regime
    planets: Dict[str, PlanetSeries] = field(default_factory=dict)
    mean_longitude: Optional[np.ndarray] = None
    mean_anomaly: Optional[np.ndarray] = None
    angle_unwrapped_numpy: Optional[np.ndarray] = None  # np.unwrap version: figure and alias_np_unwrap_differs only
    name: str = ''
    catalogue_longitudes: Dict[str, float] = field(default_factory=dict)  # planet -> lambda_j(t_yr[0]) [rad]

    def validate(self) -> None:
        """ValueError unless every array has the length of `t_yr`, times increase strictly, the
        planets of the resonance are either all present or all absent, and every catalogue longitude is a finite
        value of a known planet."""
        n = len(self.t_yr)
        if n < 2:
            raise ValueError(f'{self.name}: a record needs at least 2 rows, got {n}')
        if not np.all(np.diff(self.t_yr) > 0):
            raise ValueError(f'{self.name}: times must increase strictly (reverse a backward run)')
        arrays = {name: getattr(self, name) for name in ('a', 'e', 'inc', 'Omega', 'omega', 'varpi', 'angle_wrapped')}
        for name in ('mean_longitude', 'mean_anomaly', 'angle_unwrapped_numpy'):
            if getattr(self, name) is not None:
                arrays[name] = getattr(self, name)
        for planet, series in self.planets.items():
            for name in ('t_yr', 'a', 'e', 'inc', 'Omega', 'varpi', 'mean_longitude'):
                if getattr(series, name) is not None:
                    arrays[f'{planet}.{name}'] = getattr(series, name)
        wrong = [f'{name} ({len(values)})' for name, values in arrays.items() if len(values) != n]
        if wrong:
            raise ValueError(f'{self.name}: arrays not of the length of t_yr ({n}): {", ".join(wrong)}')
        present = [planet for planet in self.resonance.planets if planet in self.planets]
        if present and len(present) != len(self.resonance.planets):
            raise ValueError(f'{self.name}: planets of {self.resonance.key} partly saved: {present}')
        self._check_catalogue()
        self._check_planet_grids()

    def _check_catalogue(self) -> None:
        wrong = [name for name, value in self.catalogue_longitudes.items() if name not in PLANET_NAMES_ORDER or not np.isfinite(value)]
        if wrong:
            raise ValueError(f'{self.name}: catalogue longitudes of unknown planets or not finite: {wrong}')

    def _check_planet_grids(self) -> None:
        """The same length is not the same grid: the ripple and the reconstructed angle add samples index by index, so
        every planet's times must equal t_yr within SAME_GRID_TOLERANCE of a step (a half-step shift is refused)."""
        tolerance = SAME_GRID_TOLERANCE * float(np.median(np.diff(self.t_yr)))
        t = np.asarray(self.t_yr, dtype=float)
        shifted = [
            planet for planet, series in self.planets.items() if not np.all(np.abs(np.asarray(series.t_yr, dtype=float) - t) <= tolerance)
        ]
        if shifted:
            raise ValueError(f'{self.name}: planet times differ from t_yr by more than {SAME_GRID_TOLERANCE} of a step: {shifted}')

    def domain_problems(self) -> List[str]:
        """What puts the record outside the MMR model (empty = none), checked before anything is computed:

        - `k=0`: the angle has no mean longitude of the body (no a to resonate with: the identity has no c);
        - `non_finite:<names>`: NaN/inf in the body's elements, the angle or the times (the integration lost the orbit);
        - `a<=0`, `e<0`, `e>=1` anywhere: no elliptic Keplerian orbit, n(a) and the periodic angle are undefined.
        """
        problems = []
        if self.resonance.body_coefficient == 0:
            problems.append('k=0')
        arrays = ('t_yr', 'a', 'e', 'inc', 'Omega', 'omega', 'varpi', 'angle_wrapped')
        broken = [name for name in arrays if not np.all(np.isfinite(np.asarray(getattr(self, name), dtype=float)))]
        if broken:
            problems.append('non_finite:' + ','.join(broken))
        a, e = np.asarray(self.a, dtype=float), np.asarray(self.e, dtype=float)
        with np.errstate(invalid='ignore'):
            if np.any(a <= 0):
                problems.append('a<=0')
            if np.any(e < 0):
                problems.append('e<0')
            if np.any(e >= 1):
                problems.append('e>=1')
        return problems


# --- Results --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GreatInequality:
    period_yr: float  # 2 pi / |2 n_J - 5 n_S + 3 varpi_dot_S|
    dlambda_jupiter_rad: float  # half-amplitude in lambda_J (measured, or 0.34 deg)
    dlambda_saturn_rad: float  # (or 0.83 deg)
    measured: bool


@dataclass
class Context:
    """Constants of the run and the record (SPEC §4.1, columns `ctx_*`; run-level in constants.csv)."""

    planet_mean_motion_rad_yr: Dict[str, float]  # slope of the unwrapped lambda_j, else the table
    planet_mean_motion_source: Dict[str, str]  # 'measured' | 'table'
    planet_mass: Dict[str, float]  # m_j / M_sun
    planet_varpi_rate_rad_yr: Dict[str, float]  # measured varpi_dot_j, else g_j
    great_inequality: Optional[GreatInequality]
    order: int
    a_mean_au: float
    a_p01_au: float
    a_p99_au: float
    e_min: float
    e_med: float
    e_max: float
    inc_min_rad: float
    inc_max_rad: float
    q_min_au: float
    big_q_max_au: float  # aphelion Q_max
    orbital_period_yr: float
    crosses: Tuple[str, ...]  # planets of the angle whose orbits the body crosses
    mean_motion_rad_yr: float  # n(<a>) = 2 pi <a>^-3/2
    identity_coefficient: float  # c = -1.5 k n / a [rad yr^-1 au^-1]
    nominal_axis_au: float
    exact_axis_au: float  # a*: nu(a*) = 0 with the measured n_j and slow terms
    axis_from_mean_longitude_rate_au: float  # (GM / lambda_dot^2)^(1/3)
    synodic_periods_yr: Dict[str, float]  # 2 pi / |n - n_j| for the planets of the angle
    slow_rate_rad_yr: float  # p varpi_dot_ref + sum p_j varpi_dot_j
    rate_expected_rad_yr: float  # nu(<a>)
    rate_expected_lo_rad_yr: float  # nu over a 1..99 %
    rate_expected_hi_rad_yr: float
    rate_measured_rad_yr: float  # slope of the predictively unwrapped angle (set after sampling)
    mu_scale_no_eq: float  # mu_P (two-body) or mu_i mu_j (three-body), without e^q
    fair_step_fraction: float  # dt / P_orb
    # Inputs of the numbers above, kept for the CSV (defaults: a Context built by hand stays valid).
    angle_planets: Tuple[str, ...] = ()  # planets of the angle (the other dicts may hold more)
    body_coefficient: int = 0  # k
    body_varpi_coefficient: int = 0  # p
    retrograde: bool = False
    inc_med_rad: float = NAN
    body_varpi_rate_rad_yr: float = NAN  # varpi_dot (varpi* for a retrograde angle) over mmr_varpi_ref_window_yr
    planet_node_rate_rad_yr: Dict[str, float] = field(default_factory=dict)  # measured Omega_dot_j, else s_j
    planet_varpi_rate_measured_rad_yr: Dict[str, float] = field(default_factory=dict)  # nan = not saved
    planet_node_rate_measured_rad_yr: Dict[str, float] = field(default_factory=dict)
    planet_g_rad_yr: Dict[str, float] = field(default_factory=dict)  # g_j, s_j of const.PLANETARY_FREQUENCIES
    planet_s_rad_yr: Dict[str, float] = field(default_factory=dict)
    planet_axis_mean_au: Dict[str, float] = field(default_factory=dict)  # saved planets only
    planet_e_mean: Dict[str, float] = field(default_factory=dict)
    near_commensurability_periods_yr: Dict[str, float] = field(default_factory=dict)  # const.PLANET_NEAR_COMMENSURABILITIES
    # nu with the body's measured mean lambda_dot instead of n(<a>): free of the offset of the osculating <a>
    # (1999 UQ: 71.9786 against 71.9986 deg/yr; 21514: +0.04 against -0.17), not of the angle's unwrap.
    rate_from_mean_longitude_rad_yr: float = NAN

    @property
    def delta_axis_au(self) -> float:
        return self.a_mean_au - self.exact_axis_au

    def to_columns(self) -> dict:
        """CSV columns `ctx_*` (record) in CSV units."""
        planets = self.angle_planets or tuple(self.synodic_periods_yr)
        synodic = join_list(
            f'{planet_letter(name)}:{self.synodic_periods_yr[name]:.6g}' for name in planets if name in self.synodic_periods_yr
        )
        return {
            'ctx_k': self.body_coefficient,
            'ctx_p': self.body_varpi_coefficient,
            'ctx_order': self.order,
            'ctx_a_mean_au': column_number(self.a_mean_au),
            'ctx_a_p01_au': column_number(self.a_p01_au),
            'ctx_a_p99_au': column_number(self.a_p99_au),
            'ctx_e_min': column_number(self.e_min),
            'ctx_e_med': column_number(self.e_med),
            'ctx_e_max': column_number(self.e_max),
            'ctx_inc_min_deg': column_number(self.inc_min_rad, RAD_TO_DEG),
            'ctx_inc_med_deg': column_number(self.inc_med_rad, RAD_TO_DEG),
            'ctx_inc_max_deg': column_number(self.inc_max_rad, RAD_TO_DEG),
            'ctx_q_min_au': column_number(self.q_min_au),
            'ctx_Q_max_au': column_number(self.big_q_max_au),
            'ctx_P_orb_yr': column_number(self.orbital_period_yr),
            'ctx_crosses': join_list(self.crosses),
            'ctx_n_deg_yr': column_number(self.mean_motion_rad_yr, RAD_TO_DEG),
            'ctx_c_rad_yr_au': column_number(self.identity_coefficient),
            'ctx_a_res_nominal_au': column_number(self.nominal_axis_au),
            'ctx_a_star_au': column_number(self.exact_axis_au),
            'ctx_delta_a_au': column_number(self.delta_axis_au),
            'ctx_a_from_mean_longitude_rate_au': column_number(self.axis_from_mean_longitude_rate_au),
            'ctx_P_syn_yr': synodic,
            'ctx_varpi_dot_arcsec_yr': column_number(self.body_varpi_rate_rad_yr, RAD_TO_ARCSEC),
            'ctx_slow_rate_arcsec_yr': column_number(self.slow_rate_rad_yr, RAD_TO_ARCSEC),
            'ctx_rate_expected_deg_yr': column_number(self.rate_expected_rad_yr, RAD_TO_DEG),
            'ctx_rate_expected_lo_deg_yr': column_number(self.rate_expected_lo_rad_yr, RAD_TO_DEG),
            'ctx_rate_expected_hi_deg_yr': column_number(self.rate_expected_hi_rad_yr, RAD_TO_DEG),
            'ctx_rate_from_mean_longitude_deg_yr': column_number(self.rate_from_mean_longitude_rad_yr, RAD_TO_DEG),
            'ctx_rate_measured_deg_yr': column_number(self.rate_measured_rad_yr, RAD_TO_DEG),
            'ctx_mu_scale_no_eq': column_number(self.mu_scale_no_eq),
            'ctx_fair_step_fraction': column_number(self.fair_step_fraction),
        }

    def constants_rows(self) -> List[dict]:
        """constants.csv rows (one per planet + the great inequality + the other planetary near-commensurabilities).

        Planet rows: `planet, n_deg_yr, n_source, mu, a_mean_au, e_mean, g_arcsec_yr, s_arcsec_yr,
        varpi_dot_meas_arcsec_yr, Omega_dot_meas_arcsec_yr` (SPEC §11.4); the others: `planet` (the term's
        name), `period_yr` and, for the great inequality, `dlambda_jupiter_deg`, `dlambda_saturn_deg`, `measured`.
        """
        names = [name for name in PLANET_NAMES_ORDER if name in self.planet_mean_motion_rad_yr]
        rows = [self._planet_row(name) for name in names]
        if self.great_inequality is not None:
            gi = self.great_inequality
            rows.append(
                {
                    'planet': 'great_inequality',
                    'period_yr': column_number(gi.period_yr),
                    'dlambda_jupiter_deg': column_number(gi.dlambda_jupiter_rad, RAD_TO_DEG),
                    'dlambda_saturn_deg': column_number(gi.dlambda_saturn_rad, RAD_TO_DEG),
                    'measured': gi.measured,
                }
            )
        for name, period in self.near_commensurability_periods_yr.items():
            if name != 'great_inequality':
                rows.append({'planet': name, 'period_yr': column_number(period)})
        return rows

    def _planet_row(self, name: str) -> dict:
        return {
            'planet': name,
            'n_deg_yr': column_number(self.planet_mean_motion_rad_yr.get(name), RAD_TO_DEG),
            'n_source': self.planet_mean_motion_source.get(name, ''),
            'mu': column_number(self.planet_mass.get(name)),
            'a_mean_au': column_number(self.planet_axis_mean_au.get(name)),
            'e_mean': column_number(self.planet_e_mean.get(name)),
            'g_arcsec_yr': column_number(self.planet_g_rad_yr.get(name), RAD_TO_ARCSEC),
            's_arcsec_yr': column_number(self.planet_s_rad_yr.get(name), RAD_TO_ARCSEC),
            'varpi_dot_meas_arcsec_yr': column_number(self.planet_varpi_rate_measured_rad_yr.get(name), RAD_TO_ARCSEC),
            'Omega_dot_meas_arcsec_yr': column_number(self.planet_node_rate_measured_rad_yr.get(name), RAD_TO_ARCSEC),
        }


class StepVerdict(StrEnum):
    RESOLVED = 'resolved'  # |nu| dt <= mmr_step_resolved_deg over the a range
    MARGINAL = 'marginal'  # up to 180 deg, one branch
    ALIASED_UNAMBIGUOUS = 'aliased_unambiguous'  # one branch m != 0
    AMBIGUOUS = 'ambiguous'  # several branches over the a range


@dataclass
class Sampling:
    """Step, alias branch and the predictive unwrap (SPEC §3.1, §4.2; columns `smp_*`)."""

    step_yr: float
    span_yr: float
    n_rows: int
    nyquist_rad_yr: float  # pi / dt
    step_rotation_min_rad: float  # |nu| dt over the a range
    step_rotation_max_rad: float
    branches: Tuple[int, ...]  # m = round(nu dt / 2pi) over the a range
    verdict: StepVerdict
    branch_margin_turns: float
    nyquist_boundary_da_au: float
    next_branch_da_au: float
    far_ratio: float  # |<a> - a*| / max half-width
    far_from_resonance: bool
    rate_apparent_pred_rad_yr: float  # nu - 2 pi m / dt
    rate_apparent_meas_rad_yr: float  # slope of np.unwrap
    nyquist_halfwidth_au: float  # a^(5/2) / (3 |k| dt)
    ghost_axis_au: float
    ghost_distance_au: float
    angle_unwrapped: np.ndarray  # predictive unwrap [rad]
    residual_rad: np.ndarray  # per-step model error wrap(...) of the full angle, length N-1
    unwrap_resid_p99_rad: float  # of the lambda part (the alias gate, W15); the full angle without a saved planet
    unwrap_resid_max_rad: float
    unwrap_ok: bool  # the lambda part within the limits and not undefined_phase
    suspect_times_yr: Tuple[float, ...]
    revs_predictive: float
    revs_numpy: float
    alias_numpy_differs: bool
    jitter: float  # |p| de_sp / e
    jitter_low_e: bool
    undefined_phase: bool  # jitter_low_e, the full angle fails the residual limits and varpi itself is undefined
    safe_step_plain_yr: float  # 0.5 pi / max|nu|
    safe_step_pred_yr: float  # 0.5 a^(5/2) / (3 |k| da_sp)
    angle_reconstruction_rad: float  # max |sigma_saved - sum coef x saved longitudes| (mod 2pi)
    aliased_lines_yr: Tuple[float, ...]
    resample_step_yr: Optional[float]
    flags: List[str] = field(default_factory=list)
    # min(P_orb, P_syn of the angle's planets) / 2: a step at or above it folds the short-period terms of the elements
    # into slow ones (flag short_period_aliased, confidence <= medium)
    short_period_step_yr: float = NAN
    # the longest period the step folds P_orb, P_syn and their harmonics into (`sampling.folded_slowest_yr`)
    folded_slowest_yr: float = NAN
    # the largest MMR_RESAMPLE_STEPS_YR step below step_yr that folds no short period (`sampling.fold_free_step`); nan
    # when the step folds nothing or no grid step is free. A rerun there lifts the cap short_period_aliased
    fold_free_step_yr: float = NAN
    short_period_da_au: float = NAN  # std of a at periods < mmr_jitter_band_yr (da_sp of safe_step_pred)
    # smp_da_sp_over_a_excursion: short_period_da_au over the a half-excursion of the decisive episode
    # (`decide.a_scatter_ratio`), set once the episodes are measured; R-ALIAS-A and the flag a_short_period_scatter read it
    a_scatter_ratio: float = NAN
    rate_apparent_from_mean_longitude_rad_yr: float = NAN  # apparent rate of Context.rate_from_mean_longitude_rad_yr
    unwrap_method: str = 'predictive'  # mmr_unwrap
    unwrap_resid_full_p99_rad: float = NAN  # residual of the full angle (before W15 the gate)
    unwrap_resid_full_max_rad: float = NAN
    n_lambda_suspect_steps: int = 0  # |lambda-part residual| > mmr_unwrap_suspect_resid_deg
    n_slow_element_steps: int = 0  # the full step beyond that, the lambda part not: varpi turned fast (e near 0)
    phase_undefined_fraction: float = NAN  # share of samples with |p| s/|z_f| > mmr_jitter_max_rad (nan: not measured)
    phase_undefined_longest_yr: float = NAN  # longest such stretch
    varpi_undefined_fraction: float = NAN  # share of samples with varpi's own s/|z_f| > mmr_jitter_max_rad (nan: not measured)
    revs_lambda_predictive: float = NAN  # revolutions of the lambda part (alias_numpy_differs compares these two)
    revs_lambda_numpy: float = NAN

    def to_columns(self) -> dict:
        """CSV columns `smp_*`."""
        return {
            'smp_step_yr': column_number(self.step_yr),
            'smp_span_yr': column_number(self.span_yr),
            'smp_n_rows': self.n_rows,
            'smp_nyquist_deg_yr': column_number(self.nyquist_rad_yr, RAD_TO_DEG),
            'smp_step_rotation_deg_min': column_number(self.step_rotation_min_rad, RAD_TO_DEG),
            'smp_step_rotation_deg_max': column_number(self.step_rotation_max_rad, RAD_TO_DEG),
            'smp_branches': join_list(self.branches),
            'smp_verdict': str(self.verdict),
            'smp_branch_margin_turns': column_number(self.branch_margin_turns),
            'smp_nyquist_boundary_da_au': column_number(self.nyquist_boundary_da_au),
            'smp_next_branch_da_au': column_number(self.next_branch_da_au),
            'smp_far_ratio': column_number(self.far_ratio),
            'smp_far_from_resonance': self.far_from_resonance,
            'smp_rate_apparent_pred_deg_yr': column_number(self.rate_apparent_pred_rad_yr, RAD_TO_DEG),
            'smp_rate_apparent_mean_longitude_deg_yr': column_number(self.rate_apparent_from_mean_longitude_rad_yr, RAD_TO_DEG),
            'smp_rate_apparent_meas_deg_yr': column_number(self.rate_apparent_meas_rad_yr, RAD_TO_DEG),
            'smp_nyquist_halfwidth_au': column_number(self.nyquist_halfwidth_au),
            'smp_ghost_au': column_number(self.ghost_axis_au),
            'smp_ghost_distance_au': column_number(self.ghost_distance_au),
            'smp_unwrap': self.unwrap_method,
            'smp_unwrap_resid_p99_deg': column_number(self.unwrap_resid_p99_rad, RAD_TO_DEG),
            'smp_unwrap_resid_max_deg': column_number(self.unwrap_resid_max_rad, RAD_TO_DEG),
            'smp_unwrap_resid_full_p99_deg': column_number(self.unwrap_resid_full_p99_rad, RAD_TO_DEG),
            'smp_unwrap_resid_full_max_deg': column_number(self.unwrap_resid_full_max_rad, RAD_TO_DEG),
            'smp_unwrap_ok': self.unwrap_ok,
            'smp_n_suspect_steps': len(self.suspect_times_yr),
            'smp_n_lambda_suspect_steps': self.n_lambda_suspect_steps,
            'smp_n_slow_element_steps': self.n_slow_element_steps,
            'smp_phase_undefined_fraction': column_number(self.phase_undefined_fraction),
            'smp_phase_undefined_longest_yr': column_number(self.phase_undefined_longest_yr),
            'smp_varpi_undefined_fraction': column_number(self.varpi_undefined_fraction),
            'smp_suspect_times_yr': column_list(self.suspect_times_yr),
            'smp_revs_predictive': column_number(self.revs_predictive),
            'smp_revs_numpy': column_number(self.revs_numpy),
            'smp_revs_lambda_predictive': column_number(self.revs_lambda_predictive),
            'smp_revs_lambda_numpy': column_number(self.revs_lambda_numpy),
            'smp_alias_np_unwrap_differs': self.alias_numpy_differs,
            'smp_jitter': column_number(self.jitter),
            'smp_jitter_low_e': self.jitter_low_e,
            'smp_undefined_phase': self.undefined_phase,
            'smp_da_short_period_au': column_number(self.short_period_da_au),
            'smp_safe_step_plain_yr': column_number(self.safe_step_plain_yr),
            'smp_safe_step_pred_yr': column_number(self.safe_step_pred_yr),
            'smp_angle_reconstruction_rad': column_number(self.angle_reconstruction_rad),
            'smp_aliased_lines_yr': column_list(self.aliased_lines_yr),
            'smp_resample_step_yr': column_number(self.resample_step_yr),
            'smp_short_period_step_yr': column_number(self.short_period_step_yr),
            'smp_folded_slowest_yr': column_number(self.folded_slowest_yr),
            'smp_fold_free_step_yr': column_number(self.fold_free_step_yr),
        }


@dataclass(frozen=True)
class Cutoff:
    """The filter cutoff of the record (DELTA item 1: one simple rule)."""

    cutoff_yr: float  # P_c = max(synodic factor x longest P_syn, P_floor / filter_min_libration_cutoffs), or the user value
    binding: str  # 'synodic' | 'floor' | 'user' | 'centre_scale' (G2c)
    synodic_bound_yr: float
    floor_bound_yr: float  # P_floor / filter_min_libration_cutoffs
    samples_per_cutoff: float  # P_c / dt


@dataclass
class Preprocessed:
    """Ripple removed, filtered and differentiated series (SPEC §3; columns `flt_*`)."""

    cutoff: Cutoff
    t_yr: np.ndarray
    angle_unwrapped: np.ndarray  # predictive unwrap, before the ripple
    angle_clean: np.ndarray  # minus sum_j k_j dlambda_j^slow
    angle_f: np.ndarray
    rate_f: np.ndarray  # rad/yr
    angle_f_half: np.ndarray  # cutoff x0.5
    angle_f_double: np.ndarray  # cutoff x2
    a_f: np.ndarray
    adot_f: np.ndarray  # au/yr
    e_f: np.ndarray
    inc_f: np.ndarray
    window: Tuple[float, float]  # edge-trimmed
    ripple_rad: Optional[np.ndarray]  # the subtracted term (None when not removed)
    ripple_removed: bool
    ripple_half_amp_rad: float
    ripple_corr: float
    unremoved_ripple_rad: float  # predicted r_GI when not removed, else 0
    # the shortest libration period of this angle (`preprocess.libration_period_floor`): faster 'episodes' are forced
    # waves; basis = the setting it comes from: mmr_coorbital_period_floor_factor (x the planet's tadpole period) or
    # mmr_libration_period_min_yr
    period_floor_yr: float = NAN
    period_floor_basis: str = ''
    # blocks of slowly varying elements the 1:1 law tabulates its averaged potential on [yr]: P_c, but in the G1q pass at
    # the quasi-satellite floor min(the angle's own P_c, P_QS) (`quasi_satellite.QuasiSatellite.elements_block_yr`)
    elements_block_yr: float = NAN
    gain_at_period_cycle: float = NAN  # set once P_cyc is known
    filter_too_slow: bool = False
    flags: List[str] = field(default_factory=list)
    # the series the response read (smoothed again at response_cutoff_yr, `response.response_series`): drawn in the figure
    response_cutoff_yr: float = NAN
    angle_response: Optional[np.ndarray] = None  # rad: the angle the single force law was fitted on (panel (e))
    rate_response: Optional[np.ndarray] = None  # rad/yr
    adot_response: Optional[np.ndarray] = None  # au/yr

    def to_columns(self) -> dict:
        """CSV columns `flt_*` (cutoff, binding, bounds, ripple, gain, filter_too_slow)."""
        from resonances.resonance.analysis.mmr.preprocess import preprocessed_columns

        return preprocessed_columns(self)


@dataclass
class SlowComponents:
    """Slow parts with the centre window W_c (SPEC §3.5): for B5, kinematics and the slow a_dot."""

    window_yr: float
    t_yr: np.ndarray
    angle_slow: np.ndarray
    a_slow: np.ndarray
    adot_slow: np.ndarray
    varpi_slow: np.ndarray
    e_slow: np.ndarray
    window: Tuple[float, float]  # trimmed by W_c at each edge


@dataclass(frozen=True)
class ForceWindow:
    """a_dot_f = R sin(sigma_f - phi0) + C in one window."""

    t_start_yr: float
    t_end_yr: float
    r2: float
    amplitude_au_yr: float  # R = K
    phi0_rad: float
    offset_au_yr: float


@dataclass(frozen=True)
class MultipletFit:
    """a_dot_f = sum_j [A_j sin sigma_j + B_j cos sigma_j] + C, sigma_j = sigma + j (varpi - varpi_P).

    The law whose R^2 L4 reads is the one fit over the union of `windows` (`pooled`, `constant_au_yr`); panel (e) of
    the figure draws it (`response.members_model`, varpi - varpi_P from `response.multiplet_offset` with the planet's
    rate). These fields only describe the fit; nothing decides on them."""

    harmonics: Tuple[int, ...]  # j values
    coefficients: np.ndarray  # complex A_j + i B_j per window x harmonic
    r2: float
    coefficient_coherence: float  # |<(A+iB)/|A+iB|>| of the leading terms over windows
    eps: Tuple[float, ...]  # |C_j| / |C_0|
    pooled: Optional[np.ndarray] = None  # complex A_j + i B_j of the one fit over the union of the windows
    constant_au_yr: float = NAN  # its C
    windows: Tuple[Tuple[float, float], ...] = ()  # [start, end] yr of the windows
    planet: str = ''  # the planet P of varpi_P ('' when the caller fitted another offset: the owner's law)
    planet_varpi_rate_rad_yr: float = NAN  # g_P the offset was built with


@dataclass(frozen=True)
class HarmonicLaw:
    """CS12 (W23), the force law with N harmonics of the angle: a_dot = sum_{k=1..N} [A_k sin k psi + B_k cos k psi] + C.

    sigma_dot = c (a - a*) and sigma_ddot = c a_dot = -V'(sigma) for the periodic V that R0 fits as V_E with up to
    SEPARATRIX_HARMONICS_MAX harmonics, so a pendulum in a well that is not a sinusoid (a double well, a flat bottom)
    has this force law. Consulted only when the one-harmonic L4 fails or passes thin (`response.harmonic_law`); it can
    lift L4, never lower it. `taken` False keeps the old check; `reason` then says which guard refused it."""

    n_harmonics: int  # the N measured (the largest that still gained mmr_force_multi_gain, else 2)
    r2: float  # the smaller of the two half medians of the window R^2 (as the single law)
    single_r2: float  # the same statistic of the one-harmonic law
    n_windows: int
    coherence: float = NAN  # min over the leading C_k of |<C_k / |C_k|>| over the windows
    null_max: float = NAN  # the largest R^2 of the same law with a_dot shifted in time, full windows only
    oos_r2: float = NAN  # independent halves: the smaller R^2 of one half's law predicting the other
    c2_over_c1: float = NAN  # |C_2| / |C_1| of the window medians: the weight of the second harmonic
    taken: bool = False
    reason: str = ''  # '' when taken, else the first guard that refused it (`response.HARMONIC_REASONS`)

    @property
    def gain(self) -> float:
        """R^2 over the one-harmonic law's."""
        return self.r2 - self.single_r2

    @property
    def null_gain(self) -> float:
        """R^2 over the shifted null (nan without a null)."""
        return self.r2 - self.null_max


def harmonic_reason(law: Optional[HarmonicLaw]) -> Optional[str]:
    """The CSV word of the harmonic law: 'taken', the guard that refused it, or None (an empty cell: not consulted)."""
    if law is None:
        return None
    return 'taken' if law.taken else law.reason


class ResponseSource(StrEnum):
    """Where the force law, the identity or B1 the status tree reads was measured (rule R, `response.deciding_response`)."""

    RECORD = 'record'  # pooled over the accepted episodes of the window (the window itself without one)
    WITNESS = 'witness'  # on the witness episode (`EpisodeResponse.witness`) with the most turning points
    BEFORE_JUMP = 'before_jump'  # pooled over the librating episodes before the last jump of a (G1e)


@dataclass(frozen=True)
class ForcedWave:
    """An accepted episode that is a forced wave of another argument, not a libration of this angle (W20 F6,
    `decide.forced_wave_episodes`): its identity fails with the wrong sign (`wrong_sign`), or its period sits on a
    forced line of the angle's lattice and its identity fails (`on_line`). `Geometry.forced_waves` holds them; they
    are no episodes of the angle (`Geometry.episodes`)."""

    episode_id: int
    n_tp: int
    period_yr: float
    identity_slope_ratio: float
    identity_corr: float
    force_line: str
    force_line_offset: float
    wrong_sign: bool
    on_line: bool

    @property
    def route(self) -> str:
        """episodes.csv `forced_wave`: 'wrong_sign', 'lattice_line' or both joined by ';'."""
        return ';'.join(name for name, holds in (('wrong_sign', self.wrong_sign), ('lattice_line', self.on_line)) if holds)


@dataclass(frozen=True)
class EpisodeResponse:
    """The force law (L4), the identity (L3) and B1 on one run with its own period P_e (windows of
    MMR_FORCE_WINDOW_CYCLES x P_e, its own halves, the filter guard at P_e, the 1:1 law for a co-orbital), or pooled
    over the librating episodes before the last jump of a (`episode_id` None). The check values are None on a run
    that is not a librating episode: its numbers are columns, nothing judges it (episodes.csv, rule R)."""

    episode_id: Optional[int]
    before_jump: bool
    n_tp: int
    t_start_yr: float
    t_end_yr: float
    period_yr: float  # P_e (pooled: the median of the episodes' periods)
    windows: List[ForceWindow]
    force_window_yr: float
    force_r2_h1: float
    force_r2_h2: float
    multiplet: Optional[MultipletFit]
    force_law: str  # 'single' | 'multiplet' | 'harmonic' | 'coorbital' | ''
    force_amplitude_au_yr: float  # K: median over the windows
    phi0_rad: float
    phi0_coherence: float
    phi0_max_dev_rad: float
    phi0_tol_rad: float
    identity_slope_ratio: float
    identity_corr: float
    b1_required_au: float
    b1_measured_au: float
    b1_ratio: float
    coorbital: Optional[CoorbitalLaw] = None
    a_response: Optional[CheckValue] = None
    a_response_thin: bool = False
    identity: Optional[CheckValue] = None
    a_amplitude: Optional[CheckValue] = None
    filter_too_slow: bool = False  # P_e < filter_min_libration_cutoffs x P_c: the shape checks are not measured
    # a forced line of the angle's lattice within the window's resolution of 1/P_e (forced_lines.coincident_line): its
    # period, argument and offset in units of 1/(W P_e); contaminated = a measured L4 made not_measured by it
    force_line_period_yr: float = NAN
    force_line: str = ''
    force_line_offset: float = NAN
    force_contaminated: bool = False
    force_raw_check: str = ''  # the measured L4 (pass/fail) a contamination replaced, '' otherwise
    force_jump_dropped: int = 0  # force-law windows dropped next to a confirmed jump of a (`response.jump_zones`)
    # the line rule (`response.line_margin_cutoff`): the cutoff the force law read, the line of a_dot that raised it
    # (nan: not raised), the largest R^2 of the same law with a shifted (null test) and the law's R^2 above it
    force_cutoff_yr: float = NAN
    force_line_binding_yr: float = NAN
    force_r2_null_max: float = NAN
    force_null_gain: float = NAN
    force_harmonic: Optional[HarmonicLaw] = None  # CS12: the N-harmonic law when it was consulted

    @property
    def witness(self) -> bool:
        """The episode stands for the dynamics by itself (rule R): the force law passes with its full margin, B1 does
        not fail and the identity is not measured to fail (W11), or the force law passes thin and the identity and B1
        pass on the same episode. A failed identity contradicts the pendulum the force law would witness: a and sigma
        then follow another clock (37087 3J+8S-3, 1 Myr: a near-2:1 Jupiter term forces a, -8 varpi_dot reverses the
        slope). A pooled measurement is never a witness."""
        if self.episode_id is None or self.a_response != CheckValue.PASS or self.a_amplitude == CheckValue.FAIL:
            return False
        if self.a_response_thin:
            return self.identity == CheckValue.PASS and self.a_amplitude == CheckValue.PASS
        return self.identity in (CheckValue.PASS, CheckValue.NOT_MEASURED)

    def to_columns(self) -> dict:
        """episodes.csv columns of the response on this run."""
        multi = self.multiplet

        def check(value: Optional[CheckValue]) -> str:
            return str(value) if value is not None else ''

        return {
            'force_window_yr': column_number(self.force_window_yr),
            'force_n_windows': len(self.windows),
            'force_n_windows_jump': self.force_jump_dropped,
            'force_R2_h1': column_number(self.force_r2_h1),
            'force_R2_h2': column_number(self.force_r2_h2),
            'force_R2_multi': column_number(multi.r2) if multi is not None else None,
            'force_law': self.force_law,
            'force_K_au_yr': column_number(self.force_amplitude_au_yr),
            'force_phi0_deg': column_number(self.phi0_rad, RAD_TO_DEG),
            'phi0_coherence': column_number(self.phi0_coherence),
            'phi0_tol_deg': column_number(self.phi0_tol_rad, RAD_TO_DEG),
            'identity_slope_ratio': column_number(self.identity_slope_ratio),
            'identity_corr': column_number(self.identity_corr),
            'b1_ratio': column_number(self.b1_ratio),
            'check_a_response': check(self.a_response),
            'force_thin': bool(self.a_response_thin),
            'force_line_period_yr': column_number(self.force_line_period_yr),
            'force_line': self.force_line,
            'force_contaminated': bool(self.force_contaminated),
            'force_raw_check': self.force_raw_check,
            'force_cutoff_yr': column_number(self.force_cutoff_yr),
            'force_line_binding_yr': column_number(self.force_line_binding_yr),
            'force_R2_null_max': column_number(self.force_r2_null_max),
            'force_harm_N': self.force_harmonic.n_harmonics if self.force_harmonic is not None else None,
            'force_R2_harm': column_number(self.force_harmonic.r2) if self.force_harmonic is not None else None,
            'force_harm_reason': harmonic_reason(self.force_harmonic),
            'check_identity': check(self.identity),
            'check_a_amplitude': check(self.a_amplitude),
            'episode_filter_too_slow': bool(self.filter_too_slow),
            'witness': self.witness,
        }


@dataclass
class Response:
    """Response of a to the angle (SPEC §4.5; columns `rsp_*`, `check_mmr_identity/a_response/a_amplitude/slow_libration`).

    The checks are pooled over the window (`ResponseSource.RECORD`). `episode_responses` holds them on every librating
    episode (window and before the last jump of a) and `before_jump_response` pooled over those before the jump; the
    status tree reads `response.deciding_response` (rule R), which records in the `*_source` fields where each of the
    three was taken from."""

    identity_slope_ratio: float  # slope(sigma_dot_f on a_f) / c
    identity_corr: float
    identity_wave_corr: float
    identity_amp_ratio: float
    identity: CheckValue
    windows: List[ForceWindow]
    force_window_yr: float
    force_r2_h1: float
    force_r2_h2: float
    multiplet: Optional[MultipletFit]
    force_law: str  # 'single' | 'multiplet' | 'harmonic' | 'coorbital' | ''
    force_amplitude_au_yr: float  # K
    phi0_rad: float
    phi0_coherence: float
    phi0_max_dev_rad: float
    phi0_tol_rad: float
    phi0_minus_centre_coherence: float
    a_response: CheckValue
    a_response_thin: bool  # R^2 in [fail, full) or passes at one cutoff only
    omega0_force_rad_yr: float  # sqrt(|c| K)
    omega0_ratio: float  # omega0_K / omega0(period)
    b1_required_au: float  # omega dsigma / |c|
    b1_measured_au: float
    b1_ratio: float
    a_amplitude: CheckValue
    circulation_strength_ratio: float  # <|sigma_dot_f|> / (2 omega0_K)
    forced_wave_pred_au: float
    forced_wave_meas_au: float
    slow_adot_r2: float
    b5_arc_rad: float  # S
    b5_force_max_au_yr: float
    b5_omega0_max_rad_yr: float
    b5_period_min_yr: float
    centre_rate_rad_yr: float  # nu_c
    centre_rate_err_rad_yr: float
    b5_ratio: float  # (|nu_c| - dnu_c) / (2 omega0_max)
    b5_owner_halfwidth_au: float
    b5_owner_ratio: float
    b5_route: str  # 'data' | 'geometry' | 'owner' | 'none' (no theory route: mu e^q is a scale, W3-F3)
    slow_libration: CheckValue
    exact_commensurability: bool  # |nu_c| < mmr_b5_exact_sigma dnu_c
    flags: List[str] = field(default_factory=list)
    coorbital: Optional[CoorbitalLaw] = None  # the 1:1 law (`coorbital.coorbital_law`); None for other angles
    force_r2_whole: float = float('nan')  # R^2 of the one force fit over the whole window (no windows: circulation), else nan
    episode_responses: List[EpisodeResponse] = field(default_factory=list)  # every librating episode (rule R)
    before_jump_response: Optional[EpisodeResponse] = None  # pooled over the librating episodes before the last jump
    force_source: ResponseSource = ResponseSource.RECORD  # where the force law (L4) was measured
    identity_source: ResponseSource = ResponseSource.RECORD
    b1_source: ResponseSource = ResponseSource.RECORD
    witness_episode_id: Optional[int] = None  # the witness some check was taken from
    # the forced line at the period of the force law (EpisodeResponse, forced_lines): a part of the force law
    force_line_period_yr: float = NAN
    force_line: str = ''
    force_line_offset: float = NAN
    force_contaminated: bool = False  # L4 not_measured: a forced line of the lattice at its period (contamination)
    force_raw_check: str = ''  # the measured L4 (pass/fail) the contamination replaced
    force_jump_dropped: int = 0  # force-law windows dropped next to a confirmed jump of a (`response.jump_zones`)
    force_cutoff_yr: float = NAN  # the line rule (EpisodeResponse): the cutoff the force law read
    force_line_binding_yr: float = NAN
    force_r2_null_max: float = NAN
    force_null_gain: float = NAN
    force_harmonic: Optional[HarmonicLaw] = None  # CS12 (EpisodeResponse)

    def to_columns(self) -> dict:
        """CSV columns `rsp_*`."""
        from resonances.resonance.analysis.mmr.response import response_columns

        return response_columns(self)


@dataclass
class Kinematics:
    """Kinematic origin tests (SPEC §4.7; columns `kin_*`, `check_mmr_kinematic`)."""

    lambda_part_rate_rad_yr: float
    lambda_part_resid_rad: float
    p_varpi_rate_rad_yr: float
    p_varpi_rate_min_rad_yr: float
    p_varpi_rate_max_rad_yr: float
    cancel_ratio: float  # psi_lambda_dot / (p varpi_dot)
    corr_rate: float  # corr(sigma_dot_slow, p varpi_dot_slow)
    corr_detrended: float
    e_free: float  # (e_max + e_min) / 2
    e_forced: float  # (e_max - e_min) / 2
    varpi_rate_pred_min_rad_yr: float
    varpi_rate_pred_max_rad_yr: float
    modulation_period_yr: float  # 2 pi / (g - g5)
    two_omega_period_yr: float  # pi / |omega_dot|
    e_cos2omega_corr: float
    check: CheckValue
    detail: str  # OriginDetail value when the check fails ('varpi_term', 'secular_cycle', 'planet_ripple'), else ''
    # the slow centre c(t) against the uniform lambda part plus the secular part of sigma low-passed at W_c
    # (`kinematics.slow_centre_kinematics`): correlation and slope of the rates, and the intercept [rad/yr] next to the
    # lambda part's own rate (the route `decide.slow_centre_is_kinematic`, K2)
    centre_corr_rate: float = NAN
    centre_slope: float = NAN
    centre_intercept_rad_yr: float = NAN

    def to_columns(self) -> dict:
        """CSV columns `kin_*`."""
        from resonances.resonance.analysis.mmr.kinematics import kinematics_columns

        return kinematics_columns(self)


@dataclass(frozen=True)
class Strength:
    """Theoretical and required strength (SPEC §4.9; columns `str_*`, `check_mmr_strength`; `strength.py`).

    method 'direct_averaging' (two-body): the theory is mu_P a <D>''(sigma_c) of the directly averaged disturbing function
    at snapshots of the decisive episode, only those whose observed half-cycle librates in the theoretical well, `ratio`
    the end of [ratio_min, ratio_max] nearest to 1 (1 inside); when none does, check FAIL with reason
    'centre_outside_well' and the range over every evaluation as columns only; method 'direct_harmonic' (two-body,
    double precision cannot resolve the well): mu_P a 2 |C| of the resonant harmonic in extended precision, every
    evaluation counts, read as 'direct_averaging'; method 'eq_scale' (three-body, 1:1): `ratio` = required / mu e^q
    (or mu alone when e^q is not legal)."""

    mu_scale: float  # mu e^q (or mu_i mu_j e^q): a column on every path
    mu_scale_no_eq: float
    eq_legal: bool  # no crossing and mean e < mmr_strength_e_max
    mu_eff_required: float  # (omega0 / (|k| n))^2 / 3
    ratio: float  # what check_mmr_strength reads (above)
    omega0_theory_rad_yr: float  # |k| n sqrt(3 mu_theory) (direct: the geometric middle of the range; else mu_scale)
    period_theory_yr: float
    implausible: bool
    check: CheckValue
    method: str = 'eq_scale'  # 'direct_averaging' | 'direct_harmonic' | 'eq_scale'
    mu_eff_theory_min: float = NAN  # direct averaging over the snapshots (and the unknown planet phases)
    mu_eff_theory_max: float = NAN
    ratio_min: float = NAN  # required / theory over the same
    ratio_max: float = NAN
    centre_theory_deg: float = NAN  # the occupied minimum of <R> (first measured snapshot)
    centre_offset_max_deg: float = NAN  # largest distance of that minimum from the measured centre
    min_distance_hill: float = NAN  # closest approach to the planet along the average at the centre [Hill radii]
    nodes_per_turn: int = 0  # the largest quadrature used
    n_evaluations: int = 0  # snapshots x planet phases, measured
    n_consistent: int = 0  # of those, the ones whose observed swing lies in the theoretical well (the range's members)
    planet_elements: str = ''  # 'saved' | 'mean_orbit_phase_scan' (not saved: its phase scanned)
    not_measured_reason: str = ''  # why the direct theory is missing or fails ('' when measured or not attempted)
    three_body: bool = False  # a three-body angle (eq_scale): the small-divisor envelope applies
    # the small-divisor envelope (three-body, legal e^q, ratio >= mmr_strength_implausible_ratio; strength.py):
    # how |Delta| was bounded ('' = not evaluated), B = n / |Delta|, the residual ratio / B^2, the nearest candidate,
    # whether the swing cap B <= n min(P, P_cyc) / 2 pi set B (W22 R-DIV)
    divisor_kind: str = ''
    divisor_b: float = NAN
    divisor_residual: float = NAN
    divisor_candidate: str = ''
    divisor_capped: bool = False
    # display only (the figure's physics block, R1): over every evaluation, n at each one's own a
    mu_eff_theory_all_min: float = NAN
    mu_eff_theory_all_max: float = NAN
    period_theory_min_yr: float = NAN  # T0_th = 2 pi / (|k| n sqrt(3 mu_th))
    period_theory_max_yr: float = NAN
    ratio_all_min: float = NAN  # required / theory
    ratio_all_max: float = NAN
    t_obs_yr: float = NAN  # T0 of the decisive episode, 2 pi / (|k| n sqrt(3 mu_required))
    t_obs_source: str = ''  # 'episode:<id>' | 'rejected_run:<t0>-<t1>' | ''
    snap_t0_yr: float = NAN  # the window the snapshots are spread over (the decisive episode)
    snap_t1_yr: float = NAN
    e_snap_min: float = NAN  # the body's averaged e at the snapshots
    e_snap_max: float = NAN
    sep_halfwidth_min_au: float = NAN  # 2 omega0_th / |c|, c = 1.5 |k| n / a of the same snapshot
    sep_halfwidth_max_au: float = NAN

    def to_columns(self) -> dict:
        """CSV columns `str_*`."""
        from resonances.resonance.analysis.mmr.strength import strength_columns

        return strength_columns(self)


@dataclass(frozen=True)
class EpisodePhysics:
    """Pendulum and response numbers of one episode (episodes.csv, joined to geometry.Episode by id)."""

    episode_id: int
    small_period_min_yr: float  # T0 over amplitude regimes
    small_period_max_yr: float
    small_period_ratio: float
    omega0_rad_yr: float
    omega0_vmax_rad_yr: float
    omega0_vmax_ratio: float
    half_width_au: float
    a_excursion_pred_au: float
    a_excursion_meas_au: float
    mu_eff: float
    a_band_std_before_au: float
    a_band_std_in_au: float
    a_band_std_after_au: float
    h_min: float
    h_med: float
    h_max: float
    action_ratio_min: float
    action_ratio_max: float
    zone_class: str
    response: Optional[EpisodeResponse]  # force law, identity, B1 on the run (checks only on a librating episode)
    libration_origin: str = ''
    owner: str = ''
    # why the a-wave before the episode is not measured: 'record_edge' | 'jump' | 'episode' (`response.onset_clear`),
    # '' otherwise
    onset_short: str = ''
    # O1's background stretch [yr] after the clear cut (`response.onset_group`: before the episode's group, cut at an
    # earlier episode with a circulation after it) and the earlier accepted episodes it was joined with (latest first)
    onset_before_t0_yr: float = NAN
    onset_before_t1_yr: float = NAN
    onset_group_ids: Tuple[int, ...] = ()


def physics_of(episodes: Iterable[EpisodePhysics], episode_id: Optional[int]) -> Optional[EpisodePhysics]:
    """The physics of the episode with this `episode_id`; None for a None id or no such episode."""
    if episode_id is None:
        return None
    return next((physics for physics in episodes if physics.episode_id == episode_id), None)


class SubresonanceTrigger(StrEnum):
    T1 = 'T1'  # the angle turns but its centre moves (drift, centre_moves or no_closed_cycle run, TP-rich circulation)
    T2 = 'T2'  # a responds but the phi0 coherence is low
    T5 = 'T5'  # before any kinematic or drift_mechanism_unknown verdict
    T6 = 'T6'  # theory: the binomial mode j* != 0


@dataclass(frozen=True)
class SubresonanceCandidate:
    """One member of the multiplet and its numbers (a candidates.csv row, kind=subresonance)."""

    spec: ResonanceSpec  # its own key, e.g. 9M-16+2+5
    j: int  # power of the planet's eccentricity
    nodal: bool
    centre_range_rad: float
    centre_rate_rad_yr: float
    centre_max_pass_rad: float
    centre_mean_rad: float
    phi0_coherence: float
    coefficient_rel: float  # |C_j| / max |C|
    composite_rms_rad: float
    period_corr: float
    period_ratio: float
    rank: int
    accepted: bool
    reason: str


@dataclass
class SubresonanceResult:
    """The subresonance check (SPEC §6 with 'subresonance' for 'sister'; columns `sub_*`)."""

    triggers: Tuple[SubresonanceTrigger, ...]
    support: Tuple[str, ...]  # 'T3', 'T4'
    candidates: List[SubresonanceCandidate]
    accepted: Optional[SubresonanceCandidate]
    check: CheckValue  # pass = scanned, none accepted; fail = one accepted
    composite: bool  # Rouche |C_N| > sum |C_j| does not hold
    undetermined: bool  # no member can be told: beta = varpi_P - varpi librates, or several pass with test A unmeasured
    tie_by_order: bool
    theory_measured: bool
    sigma0_phi0_coherence: float
    composite_rms_rad: float
    single_rms_rad: float
    binomial_mode: int  # j*
    drift_pred_rad_yr: float
    drift_meas_rad_yr: float
    plane_offset_rad: float
    theory_cache: str = ''
    flags: List[str] = field(default_factory=list)

    def to_columns(self) -> dict:
        """CSV columns `sub_*` (sub_triggers, sub_n_scanned, sub_best, ...)."""
        ranked = sorted(self.candidates, key=lambda c: c.rank)
        best = ranked[0] if ranked else None
        return {
            'sub_triggers': join_list(str(trigger) for trigger in self.triggers),
            'sub_support': join_list(self.support),
            'sub_n_scanned': len(self.candidates),
            'sub_best': best.spec.key if best is not None else '',
            'sub_best_centre_range_deg': column_number(best.centre_range_rad, RAD_TO_DEG) if best is not None else None,
            'sub_best_phi0_coherence': column_number(best.phi0_coherence) if best is not None else None,
            'sub_accepted': self.accepted.spec.key if self.accepted is not None else '',
            'sub_accepted_j': self.accepted.j if self.accepted is not None else None,
            'sub_composite': self.composite,
            'sub_undetermined': self.undetermined,
            'sub_tie_by_order': self.tie_by_order,
            'sub_theory_measured': self.theory_measured,
            'sub_sigma0_phi0_coherence': column_number(self.sigma0_phi0_coherence),
            'sub_composite_rms_deg': column_number(self.composite_rms_rad, RAD_TO_DEG),
            'sub_single_rms_deg': column_number(self.single_rms_rad, RAD_TO_DEG),
            'sub_binomial_mode': self.binomial_mode,
            'sub_drift_pred_rad_kyr': column_number(self.drift_pred_rad_yr, KYR),
            'sub_drift_meas_rad_kyr': column_number(self.drift_meas_rad_yr, KYR),
            'sub_plane_offset_deg': column_number(self.plane_offset_rad, RAD_TO_DEG),
            'sub_theory_cache': self.theory_cache,
            'sub_flags': join_list(self.flags),
        }


@dataclass(frozen=True)
class OwnerCandidate:
    """A resonance that might drive a (a candidates.csv row, kind=owner)."""

    spec: ResonanceSpec
    rate_rad_yr: float
    r2_median: float
    r2_windows: Tuple[float, ...]
    phi0_coherence: float
    psi_share: float
    force_au_yr: float  # K_o
    force_theory_au_yr: float  # omega0_th,o^2 / |c_o|
    strength_ratio: float
    rank: int
    accepted: bool
    reason: str  # 'accepted:r2|phase|strength', else the first failed test (r2_low, phase_incoherent, ripple_only, ...)
    regime: str = ''  # 'libration' | 'forced_wave_positive' | 'forced_wave_negative' (the owner's turns per window)
    n_windows: int = 0  # force-law windows of that regime
    sigma_r2: float = NAN  # R^2 of the angle itself on the same windows
    sigma_phi0_coherence: float = NAN
    sigma_slow_turns: int = 0  # turning points of sigma below the band (prominence >= ripple factor x band ptp)
    sigma_slow_to_band: float = NAN  # peak-to-peak of the slow sigma (minus its line) / that of the band
    # the multiplet law (owner.owner_multiplet): constant-coefficient R^2, coefficient coherence, planet of varpi_P
    multiplet_r2: float = NAN
    multiplet_coherence: float = NAN
    multiplet_planet: str = ''
    sigma_multiplet_r2: float = NAN  # sigma with constant coefficients on the owner's long windows
    law: str = ''  # the law that fixes the phase: 'single' | 'multiplet' | '' (owner.phase_law)
    # the owner's period on the regime's windows, 2 pi / |mean phi_o_dot| [yr], and whether a forced wave at that period
    # lies in the a band [P/f, f P] (`owner.rate_admissible`; a libration regime is always admissible)
    owner_period_yr: float = NAN
    rate_admissible: bool = True
    # the phase lever (`owner.phase_leverage`): phi_o's coherence if sigma's own term drove a, and sigma's if the owner did
    lever_owner: float = NAN
    lever_sigma: float = NAN
    route: str = ''  # 'r2' | 'phase' | 'potential' | 'identity' (owner_identity) when accepted
    potential: Optional['PotentialFit'] = None  # the potential route (`owner_potential`), None when not evaluated


@dataclass
class OwnerResult:
    """The owner search (SPEC §4.8; columns `own_*`, `check_mmr_owner`)."""

    n_candidates: int
    candidates: List[OwnerCandidate]
    accepted: Optional[OwnerCandidate]
    check: CheckValue  # pass = none, fail = owner found
    sigma_r2: float  # R^2 of a_dot on the angle itself
    psi_rate_rad_yr: float
    psi_slow_corr: float
    psi_varpi_corr: float
    freq_r2_owner: float
    freq_r2_sigma: float
    freq_period_yr: float
    detail: str = ''  # 'owner_libration' | 'owner_forced_wave'
    flags: List[str] = field(default_factory=list)
    # where the accepted owner acts (F7): (t0, t1, regime) with regime libration | forced_wave | slip, the decisive
    # window of sigma it is compared with and the covered fraction of it (nan without an accepted owner)
    intervals: List[Tuple[float, float, str]] = field(default_factory=list)
    decisive_window: Tuple[float, float] = (NAN, NAN)
    cover: float = NAN
    step_yr: float = NAN  # the record's step: the intervals are written rounded to it
    psi_def: str = ''  # Psi = u sigma - v phi_o of the shown candidate, e.g. '17*sigma-3*phi'
    identity_text: str = ''  # sigma = (v/u) phi + Psi/u written out, e.g. '(27/5)φ + (ϖ − λ_N)/5'
    capture: Optional['SecondCapture'] = None  # the second-capture test of the accepted owner
    sigma_force_bound: float = NAN  # max |s - 1| of the potential route: the bound on sigma's own term
    sigma_period_min_yr: float = NAN  # T_obs / sqrt(|v/u| bound): the shortest own period sigma's term could have
    sigma_wave_rad: float = NAN  # band amplitude of sigma on sin/cos phi_o (measured only)
    sigma_wave_r2: float = NAN
    wave_da_meas_au: float = NAN  # band amplitude of a on sin/cos phi_o (forced-wave regime)
    owner_rate_rad_yr: float = NAN  # mean phi_o_dot on the shown regime's windows
    identity: Optional['IdentityRoute'] = None  # the route 'identity' of the accepted owner (`owner_identity`)

    def to_columns(self) -> dict:
        """CSV columns `own_*` (the accepted owner, else the best candidate)."""
        ranked = sorted(self.candidates, key=lambda c: c.rank)
        shown = self.accepted if self.accepted is not None else (ranked[0] if ranked else None)

        def of(attribute: str, scale: float = 1.0):
            return column_number(getattr(shown, attribute), scale) if shown is not None else None

        potential = shown.potential if shown is not None else None
        return {
            'own_name': self.accepted.spec.key if self.accepted is not None else '',
            'own_best': shown.spec.key if shown is not None else '',
            'own_n_candidates': self.n_candidates,
            'own_R2': of('r2_median'),
            'own_R2_windows': column_list(shown.r2_windows, digits=3) if shown is not None else '',
            'own_phi0_coherence': of('phi0_coherence'),
            'own_psi_share': of('psi_share'),
            'own_K_au_yr': of('force_au_yr'),
            'own_K_theory_au_yr': of('force_theory_au_yr'),
            'own_strength_ratio': of('strength_ratio'),
            'own_rate_rad_kyr': of('rate_rad_yr', KYR),
            'own_sigma_R2': column_number(self.sigma_r2),
            'own_psi_rate_rad_kyr': column_number(self.psi_rate_rad_yr, KYR),
            'own_psi_slow_corr': column_number(self.psi_slow_corr),
            'own_psi_varpi_corr': column_number(self.psi_varpi_corr),
            'own_freq_R2_owner': column_number(self.freq_r2_owner),
            'own_freq_R2_sigma': column_number(self.freq_r2_sigma),
            'own_freq_period_yr': column_number(self.freq_period_yr),
            'own_regime': shown.regime if shown is not None else '',
            'own_law': shown.law if shown is not None else '',
            'own_multiplet_R2': of('multiplet_r2'),
            'own_multiplet_coherence': of('multiplet_coherence'),
            'own_detail': self.detail,
            'own_flags': join_list(self.flags),
            'own_route': shown.route if shown is not None else '',
            'own_owner_period_yr': of('owner_period_yr'),
            'own_lever_owner': of('lever_owner'),
            'own_lever_sigma': of('lever_sigma'),
            'own_lever_gain': column_number(shown.phi0_coherence - shown.lever_owner) if shown is not None else None,
            'own_psi_def': self.psi_def,
            'own_identity_text': self.identity_text,
            'own_intervals': intervals_text(self.intervals, self.step_yr),
            'own_cover': column_number(self.cover),
            'own_decisive_t0_yr': column_number(self.decisive_window[0]),
            'own_decisive_t1_yr': column_number(self.decisive_window[1]),
            **(self.capture.to_columns() if self.capture is not None else SecondCapture.empty_columns()),
            **(potential.to_columns() if potential is not None else PotentialFit.empty_columns()),
            'own_wave_da_pred_au': (
                column_number(potential.k_direct / abs(self.owner_rate_rad_yr))
                if potential is not None and self.detail == 'owner_forced_wave' and abs(self.owner_rate_rad_yr) > 0
                else None
            ),
            'own_wave_da_meas_au': column_number(self.wave_da_meas_au) if self.detail == 'owner_forced_wave' else None,
            'own_nu_over_omega0': (
                column_number(abs(self.owner_rate_rad_yr) / potential.omega0_direct)
                if potential is not None and potential.omega0_direct > 0
                else None
            ),
            'own_sigma_wave_deg': column_number(self.sigma_wave_rad, RAD_TO_DEG),
            'own_sigma_wave_R2': column_number(self.sigma_wave_r2),
            'own_sigma_force_bound': column_number(self.sigma_force_bound),
            'own_sigma_period_min_yr': column_number(self.sigma_period_min_yr),
            **(self.identity.to_columns() if self.identity is not None else IdentityRoute.empty_columns()),
        }


@dataclass(frozen=True)
class IdentityRoute:
    """The numbers of an owner accepted by the route 'identity' (`owner_identity`, W22 R-OWNER-ID): on sigma's decisive
    window the fit sigma_f = c phi_f + a + b t (`coef_dev` = c / (k_sigma / k_phi) - 1), the rms of sigma_f -
    (k_sigma / k_phi) phi_f about its line over sigma's median half-amplitude (`resid`), phi_f's span there, and phi's own
    analysis: its angle status, the cover of the window by its librating episodes, its directly averaged strength
    ratio; `n_pass` candidates passed the identity numbers, `n_analysed` were analysed in full."""

    candidate: str
    coef_dev: float
    resid: float
    phi_span_rad: float
    phi_angle_status: str
    phi_cover: float
    phi_strength_ratio: float
    n_pass: int
    n_analysed: int

    def to_columns(self) -> dict:
        return {
            'own_id_coef_dev': column_number(self.coef_dev),
            'own_id_resid': column_number(self.resid),
            'own_id_phi_span_deg': column_number(self.phi_span_rad, RAD_TO_DEG),
            'own_id_phi_angle_status': self.phi_angle_status,
            'own_id_phi_cover': column_number(self.phi_cover),
            'own_id_phi_str_ratio': column_number(self.phi_strength_ratio),
            'own_id_n_pass': self.n_pass,
            'own_id_n_analysed': self.n_analysed,
        }

    @staticmethod
    def empty_columns() -> dict:
        return blank_columns(IdentityRoute('', NAN, NAN, NAN, '', NAN, NAN, 0, 0).to_columns())


def _interval_number(value: float, step_yr: float) -> str:
    """A time [yr] rounded to the record's step, with the decimals the step needs."""
    if not (np.isfinite(step_yr) and step_yr > 0):
        return f'{value:.0f}'
    decimals = max(0, -int(math.floor(math.log10(step_yr))))
    return f'{round(value / step_yr) * step_yr:.{decimals}f}'


def intervals_text(intervals: Iterable[Tuple[float, float, str]], step_yr: float = NAN) -> str:
    """'t0-t1:regime;...' [yr] (column own_intervals)."""
    return join_list(f'{_interval_number(a, step_yr)}-{_interval_number(b, step_yr)}:{kind}' for a, b, kind in intervals)


@dataclass(frozen=True)
class PotentialWindow:
    """The owner's directly averaged force in one window (`owner_potential`): the fitted scale s and R^2 of the
    band-passed prediction, the predicted K and phi0 in a_dot = K sin(phi_o - phi0), the owner's fitted ones, and the
    body's elements and closest approach there."""

    t_start_yr: float
    t_end_yr: float
    scale: float
    r2: float
    k_theory: float  # the band-passed prediction on sin/cos phi_o [au/yr] (matched to k_fit: the same filter)
    phi0_theory: float  # [rad]
    k_fit: float  # the band-passed a_dot on sin/cos phi_o [au/yr]
    phi0_fit: float
    k_profile: float  # the unfiltered prediction (2 k_o / (n a)) G m_P d<D>/dphi on sin/cos phi [au/yr]
    phi0_profile: float
    a_au: float
    e: float
    min_distance_hill: float
    nodes_per_turn: int
    mu_eff: float  # mu_P a A1, A1 the first harmonic of <D>(phi_o) [au^-1]
    omega0: float  # |k_o| n sqrt(3 mu_eff) [rad/yr]


@dataclass(frozen=True)
class PotentialFit:
    """The potential route of one candidate (owner_potential.py): reason '' when every chosen window was measured."""

    reason: str
    windows: Tuple[PotentialWindow, ...] = ()
    planet: str = ''

    @property
    def measured(self) -> bool:
        return self.reason == '' and len(self.windows) > 0

    def _values(self, name: str) -> np.ndarray:
        return np.asarray([getattr(w, name) for w in self.windows], float)

    @property
    def scales(self) -> np.ndarray:
        return self._values('scale')

    @property
    def r2_median(self) -> float:
        return float(np.median(self._values('r2'))) if self.windows else NAN

    @property
    def scale_lo(self) -> float:
        return float(np.min(self.scales)) if self.windows else NAN

    @property
    def scale_hi(self) -> float:
        return float(np.max(self.scales)) if self.windows else NAN

    @property
    def k_direct(self) -> float:
        """The owner's force from the averaged potential, unfiltered: the median K_profile [au/yr]."""
        return float(np.median(self._values('k_profile'))) if self.windows else NAN

    @property
    def omega0_direct(self) -> float:
        return float(np.median(self._values('omega0'))) if self.windows else NAN

    def to_columns(self) -> dict:
        ratio = self._values('k_fit') / self._values('k_theory') if self.windows else np.zeros(0)
        diff = np.abs(np.angle(np.exp(1j * (self._values('phi0_fit') - self._values('phi0_theory'))))) if self.windows else []
        phases = self._values('phi0_profile')
        return {
            'own_pot_reason': self.reason,
            'own_pot_n_windows': len(self.windows),
            'own_pot_R2': column_number(self.r2_median),
            'own_pot_scale': column_number(float(np.median(self.scales)) if self.windows else NAN),
            'own_pot_scale_lo': column_number(self.scale_lo),
            'own_pot_scale_hi': column_number(self.scale_hi),
            'own_pot_scale_worst': column_number(max(self.scale_hi, 1.0 / self.scale_lo) if self.scale_lo > 0 else NAN),
            'own_pot_min_distance_hill': column_number(float(np.min(self._values('min_distance_hill'))) if self.windows else NAN),
            'own_K_direct_au_yr': column_number(self.k_direct),
            'own_K_direct_min': column_number(float(np.min(self._values('k_profile'))) if self.windows else NAN),
            'own_K_direct_max': column_number(float(np.max(self._values('k_profile'))) if self.windows else NAN),
            'own_K_over_K_direct_min': column_number(float(np.min(ratio)) if len(ratio) else NAN),
            'own_K_over_K_direct_max': column_number(float(np.max(ratio)) if len(ratio) else NAN),
            'own_phi0_direct_deg': column_number(float(np.angle(np.mean(np.exp(1j * phases)))) if len(phases) else NAN, RAD_TO_DEG),
            'own_phi0_diff_max_deg': column_number(float(np.max(diff)) if len(diff) else NAN, RAD_TO_DEG),
            'own_omega0_direct_rad_yr': column_number(self.omega0_direct),
        }

    @staticmethod
    def empty_columns() -> dict:
        return blank_columns(PotentialFit('').to_columns())


@dataclass(frozen=True)
class SecondCapture:
    """Is sigma held by its own term while the owner pins phi_o (`owner.second_capture_blocks`)? x = Psi_slow / u on a
    connected block of the owner intervals covering sigma's decisive episode (the deciding one): the net turns of Psi,
    the sweep of x, the R^2 of x_dot^2 on [1, cos jx, sin jx] (the pendulum's energy integral), lambda_ad and the
    verdict ('excluded' | 'unresolved')."""

    psi_turns: float
    x_sweep_rad: float
    energy_r2: float
    adiabatic: float
    verdict: str
    t_start_yr: float = NAN
    t_end_yr: float = NAN

    def to_columns(self) -> dict:
        return {
            'own_psi_turns': column_number(self.psi_turns),
            'own_psi_sweep_deg': column_number(self.x_sweep_rad, RAD_TO_DEG),
            'own_psi_energy_R2': column_number(self.energy_r2),
            'own_adiabatic': column_number(self.adiabatic),
            'own_second_capture': self.verdict,
        }

    @staticmethod
    def empty_columns() -> dict:
        return blank_columns(SecondCapture(NAN, NAN, NAN, NAN, '').to_columns())


@dataclass(frozen=True)
class Baseline:
    """Is the record long enough (SPEC §4.10; `base_*`, `check_mmr_baseline`)."""

    required_yr: float  # (libration_min_turning_points / 2) P + 2 filter_edge_trim_cutoffs P_c
    # the physical estimate N = round_up(max(required, f_min T)), column base_extend_to_years; the verdict of an
    # insufficient_time angle asks for the next check instead (`decide.next_check`, N8)
    extend_to_years: Optional[float]
    check: CheckValue
    period_yr: float = NAN  # the period P the requirement was computed from
    # what set P (`decide.baseline_period`): 'tp_spacing' | 'arc_slow' | 'two_omega' | 'modulation' | 'stall_rate' |
    # 'stall_force' | 'cycle' | 'b5' | ''
    period_source: str = ''
    # an insufficient_time angle whose P fits the record: what is missing instead of time (the reject reason of the run
    # P was read from, or 'period_fits_record'); '' when time is what is missing (or the angle has enough of it)
    cause: str = ''

    def to_columns(self) -> dict:
        """CSV columns `base_*`."""
        return {
            'base_required_kyr': column_number(self.required_yr, 1 / KYR),
            'base_period_yr': column_number(self.period_yr),
            'base_period_source': self.period_source,
            'base_extend_to_years': column_number(self.extend_to_years),
            'base_cause': self.cause,
        }


@dataclass
class MMRAnalysis:
    """Everything `analyse_mmr` measured and decided for one angle of one body."""

    series_name: str
    resonance: ResonanceSpec
    context: Optional[Context]  # None only for a record refused before the analysis (MMRSeries.domain_problems)
    sampling: Optional[Sampling]
    prep: Optional[Preprocessed]
    slow: Optional[SlowComponents]
    geometry: Optional[Geometry]
    energy: Optional[EnergyAnalysis]
    response: Optional[Response]
    kinematics: Optional[Kinematics]
    subresonances: Optional[SubresonanceResult]
    owner: Optional[OwnerResult]
    strength: Optional[Strength]
    baseline: Optional[Baseline]
    episode_physics: List[EpisodePhysics]
    verdict: Verdict
    settings_hash: str
    confidence_cells_sha: str
    derived: List['MMRAnalysis'] = field(default_factory=list)  # accepted subresonance rows (depth 1)
    parent: str = ''  # for a derived row: the key of the catalogue angle
    multiplet_status: Optional[Status] = None
    decisive_episode_id: Optional[int] = None  # the episode the verdict rests on (ep_* columns)
    extra_columns: Dict[str, Any] = field(default_factory=dict)  # orchestration numbers (L1 counts, filter x2 test)
    candidate_top_n: Tuple[int, int] = (0, 0)  # (subresonances, owners) written to candidates.csv in top mode
    derived_series: Optional[MMRSeries] = None  # a derived row: the series of its own angle (for its figure)
    filter_order: Optional[int] = None  # FILTER_ORDER of the analysis (the figure smooths the member tracks with it)
    # the CSV rows taken just before `drop_arrays` (the rows read prep and the centre track, which it releases)
    frozen_numbers: Optional[Dict[str, Any]] = None
    frozen_episode_rows: Optional[List[dict]] = None

    def _results(self) -> list:
        return [
            self.context,
            self.sampling,
            self.prep,
            self.geometry,
            self.energy,
            self.response,
            self.kinematics,
            self.subresonances,
            self.owner,
            self.strength,
            self.baseline,
        ]

    def _identity(self) -> dict:
        return {
            'name': self.series_name,
            'resonance': self.resonance.key,
            'resonance_type': 'mmr',
            'added_by': ADDED_BY_SUBRESONANCE if self.parent else '',
            'parent': self.parent,
        }

    def summary_row(self) -> dict:
        """The summary.csv row: identity, verdict, checks, margins, the ledger's columns and the short
        fixed set (SPEC §11.1), `settings_hash`, `confidence_cells_sha`."""
        numbers = self.numbers_row()
        row = self._identity()
        row.update(self.verdict.to_flat_dict())
        if self.multiplet_status is not None:
            row['multiplet_status'] = str(self.multiplet_status)
        ledger_columns = [entry.column for entry in self.verdict.decision_path if entry.column]
        for column in list(dict.fromkeys(ledger_columns + list(SUMMARY_FIXED_COLUMNS))):
            row.setdefault(column, numbers.get(column))
        row['settings_hash'] = self.settings_hash
        row['confidence_cells_sha'] = self.confidence_cells_sha
        return row

    def numbers_row(self) -> dict:
        """The mmr_numbers.csv row: every `*_columns()` of the results, keyed by (name, resonance)."""
        if self.frozen_numbers is not None:
            return dict(self.frozen_numbers)
        row = {'name': self.series_name, 'resonance': self.resonance.key}
        for result in self._results():
            if result is not None:
                row.update(result.to_columns())
        row.update(self._decisive_episode_columns())
        row.update(self.extra_columns)
        return row

    def _decisive_episode_columns(self) -> dict:
        """`ep_*`: the episode the verdict rests on (the covering one, else the longest accepted)."""
        if self.decisive_episode_id is None:
            return {}
        rows = [r for r in self.episode_rows() if r['episode_id'] == self.decisive_episode_id]
        if not rows:
            return {}
        skip = ('name', 'resonance', 'tp_times_yr')
        return {f'ep_{key}': value for key, value in rows[0].items() if key not in skip}

    def episode_rows(self) -> List[dict]:
        """episodes.csv rows: geometry.Episode joined with EpisodePhysics by episode_id."""
        if self.frozen_episode_rows is not None:
            return [dict(row) for row in self.frozen_episode_rows]
        if self.geometry is None:
            return []
        physics = {p.episode_id: p for p in self.episode_physics}
        cutoff = self.prep.cutoff.cutoff_yr if self.prep is not None else None
        window = self.geometry.window
        return [_episode_row(self, run, physics.get(run.episode_id), cutoff, window) for run in self.geometry.runs]

    def candidate_rows(self, top_only: bool = True) -> List[dict]:
        """candidates.csv rows (subresonances and owners; top-N and the accepted one when `top_only`)."""
        rows = []
        top_sub, top_owner = self.candidate_top_n
        if self.subresonances is not None:
            for c in self.subresonances.candidates:
                if not top_only or c.accepted or c.rank <= top_sub:
                    rows.append(_subresonance_candidate_row(self, c))
        if self.owner is not None:
            for c in self.owner.candidates:
                if not top_only or c.accepted or c.rank <= top_owner:
                    rows.append(_owner_candidate_row(self, c))
        return rows

    def drop_arrays(self) -> None:
        """Release the arrays (memory: 40 angles x 2e5 rows would hold ~1 GB). The CSV rows are frozen first, so
        `numbers_row`, `summary_row` and `episode_rows` give the same values after it; a figure re-runs the analysis."""
        if self.frozen_numbers is None:
            self.frozen_numbers = self.numbers_row()
            self.frozen_episode_rows = self.episode_rows()
        empty = np.empty(0)
        self.prep = None
        self.slow = None
        if self.sampling is not None:  # None for a record refused before sampling (domain_problems)
            self.sampling.angle_unwrapped = empty
            self.sampling.residual_rad = empty
        self.derived_series = None
        if self.geometry is not None:
            self.geometry.centre.t_yr = empty
            self.geometry.centre.centre_rad = empty
            self.geometry.record_centres = []
            self.geometry.floor_well = None  # G1q: the quasi-satellite well per sample
            self.geometry.averaged_raw, self.geometry.averaged_raw_t_yr = None, None
            self.geometry.forget_derived()  # they read the centre and the averaged angle dropped here
        if self.energy is not None:
            self.energy.model_frame = None
            self.energy.h_avgpot = ()  # sep_h_avgpot_* and sep_J_avgpot_* are frozen in the numbers above
            self.energy.j_avgpot = ()
        for derived in self.derived:
            derived.drop_arrays()


ADDED_BY_SUBRESONANCE = 'subresonance_check'
KYR = 1e3
# The short fixed set of summary.csv (SPEC §11.1 item 5) besides the ledger's columns
SUMMARY_FIXED_COLUMNS = (
    'ctx_a_mean_au',
    'ctx_delta_a_au',
    'smp_step_yr',
    'smp_verdict',
    'flt_cutoff_yr',
    'geo_n_tp',
    'geo_period_cycle_yr',
    'geo_centre_max_pass_rev',
    'rsp_force_R2_h1',
    'rsp_force_R2_h2',
    'rsp_b1_ratio',
    'sep_h_med',
    'own_name',
    'sub_best',
    'base_extend_to_years',  # the physical estimate N next to extend_to_years (the next check, N8)
)


def _episode_row(analysis: MMRAnalysis, run, physics: Optional[EpisodePhysics], cutoff_yr, window) -> dict:
    witness_id = getattr(analysis.response, 'witness_episode_id', None)  # the witness the status tree read (rule R)
    row = {
        'name': analysis.series_name,
        'resonance': analysis.resonance.key,
        'episode_id': run.episode_id,
        'accepted': bool(run.accepted),
        'subtype': str(run.shape),
        'reject_reason': str(run.reject_reason),
        't_start_yr': run.t_start_yr,
        't_end_yr': run.t_end_yr,
        'n_tp': run.n_tp,
        'tp_times_yr': column_list(run.tp_times_yr),
        'period_med_yr': column_number(run.period_med_yr),
        'period_min_yr': column_number(run.period_min_yr),
        'period_max_yr': column_number(run.period_max_yr),
        'half_amp_min_deg': column_number(run.half_amp_min_rad, RAD_TO_DEG),
        'half_amp_med_deg': column_number(run.half_amp_med_rad, RAD_TO_DEG),
        'half_amp_max_deg': column_number(run.half_amp_max_rad, RAD_TO_DEG),
        'centre_mod_2pi_deg': column_number(run.centre_mod_2pi_rad, RAD_TO_DEG),
        'centre_shift_deg': column_number(run.centre_shift_rad, RAD_TO_DEG),
        'centre_shift_over_amp': column_number(run.centre_shift_over_amp),
        'bands_overlap': bool(run.bands_overlap),
        'centre_estimates': run.centre_estimates,
        'centre_stationarity': str(run.centre_stationarity),
        'closed_cycles': run.closed_cycles,
        'closure_ratio_min': column_number(run.closure_ratio_min),
        'centre_returns': run.centre_returns,
        'coverage_start_gap_P': column_number(run.coverage_start_gap_p),
        'coverage_end_gap_P': column_number(run.coverage_end_gap_p),
        'window_start_gap_yr': column_number(run.window_start_gap_yr),
        'window_end_gap_yr': column_number(run.window_end_gap_yr),
        'tp_step_max_yr': column_number(run.tp_step_max_yr),
        'window_start_hidden_P': column_number(run.window_start_hidden_p),
        'window_end_hidden_P': column_number(run.window_end_hidden_p),
        'before_last_jump': bool(run.before_last_jump),
        'tp_prominence_deg': column_number(run.tp_prominence_rad, RAD_TO_DEG),
        **moving_centre_columns(run),
        'exit_observed': bool(run.exit_observed),
        'entry_observed': bool(run.entry_observed),
        'closed_by_return': bool(run.closed_by_return),
        'return_t_a_yr': column_number(run.return_t_a_yr),
        'return_t_b_yr': column_number(run.return_t_b_yr),
        'return_level_fraction': column_number(run.return_level_fraction),
        # the route that made the accepted run a forced wave, no episode of the angle (`decide.forced_wave_episodes`)
        'forced_wave': next((w.route for w in analysis.geometry.forced_waves if w.episode_id == run.episode_id), ''),
        'cutoff_yr': column_number(cutoff_yr),
        'window_start_yr': column_number(window[0]),
        'window_end_yr': column_number(window[1]),
    }
    if physics is not None:
        row.update(
            {
                'T0_min_kyr': column_number(physics.small_period_min_yr, 1 / KYR),
                'T0_max_kyr': column_number(physics.small_period_max_yr, 1 / KYR),
                'T0_ratio': column_number(physics.small_period_ratio),
                'omega0_rad_yr': column_number(physics.omega0_rad_yr),
                'omega0_vmax_rad_yr': column_number(physics.omega0_vmax_rad_yr),
                'omega0_vmax_ratio': column_number(physics.omega0_vmax_ratio),
                'half_width_au': column_number(physics.half_width_au),
                'a_excursion_pred_au': column_number(physics.a_excursion_pred_au),
                'a_excursion_meas_au': column_number(physics.a_excursion_meas_au),
                'mu_eff': column_number(physics.mu_eff),
                'a_band_std_before_au': column_number(physics.a_band_std_before_au),
                'a_band_std_in_au': column_number(physics.a_band_std_in_au),
                'a_band_std_after_au': column_number(physics.a_band_std_after_au),
                'onset_not_measured': physics.onset_short,
                'onset_before_t0_yr': column_number(physics.onset_before_t0_yr),
                'onset_before_t1_yr': column_number(physics.onset_before_t1_yr),
                'onset_group': ';'.join(str(i) for i in physics.onset_group_ids),
                'h_min': column_number(physics.h_min),
                'h_med': column_number(physics.h_med),
                'h_max': column_number(physics.h_max),
                'J_over_Js_min': column_number(physics.action_ratio_min),
                'J_over_Js_max': column_number(physics.action_ratio_max),
                'zone_class': physics.zone_class,
                **(physics.response.to_columns() if physics.response is not None else {}),
                'deciding_witness': physics.episode_id == witness_id,
                'libration_origin': physics.libration_origin,
                'owner': physics.owner,
            }
        )
    return row


def moving_centre_columns(run) -> dict:
    """The frame that accepted a run and the numbers of its moving centre (`geometry._classify_run`; in mmr_numbers.csv
    with ep_ for the decisive episode): band margins in sigma and psi, the half-swing margin (a column only), the well
    track's status ('' = not evaluated: the sigma test decided), its drift over the run and the RMS of the full-cycle
    centres about it, about their mean and about the reversed track (`geometry.well_track_valid`)."""
    return {
        'frame': str(run.frame),
        'margin_sigma_deg': column_number(run.margin_sigma_rad, RAD_TO_DEG),
        'margin_psi_deg': column_number(run.margin_psi_rad, RAD_TO_DEG),
        'half_swing_margin_deg': column_number(run.half_swing_margin_rad, RAD_TO_DEG),
        'centre_track': str(run.well_status),
        'centre_model_drift_deg': column_number(run.well_drift_rad, RAD_TO_DEG),
        'centre_track_rms_deg': column_number(run.well_rms_rad, RAD_TO_DEG),
        'centre_fixed_rms_deg': column_number(run.well_rms_fixed_rad, RAD_TO_DEG),
        'centre_reversed_rms_deg': column_number(run.well_rms_reversed_rad, RAD_TO_DEG),
    }


def _candidate_base(analysis: MMRAnalysis, kind: str, spec: ResonanceSpec) -> dict:
    coefficients = (
        list(spec.planet_coefficients) + [spec.body_coefficient, spec.body_varpi_coefficient] + list(spec.planet_varpi_coefficients)
    )
    return {
        'name': analysis.series_name,
        'resonance': analysis.resonance.key,
        'kind': kind,
        'candidate': spec.key,
        'coefficients': join_list(coefficients),
        'order': spec.order,
    }


def _subresonance_candidate_row(analysis: MMRAnalysis, c: SubresonanceCandidate) -> dict:
    row = _candidate_base(analysis, 'subresonance', c.spec)
    row.update(
        {
            'j': c.j,
            'nodal': c.nodal,
            'centre_range_deg': column_number(c.centre_range_rad, RAD_TO_DEG),
            'centre_rate_rad_kyr': column_number(c.centre_rate_rad_yr, KYR),
            'centre_max_pass_deg': column_number(c.centre_max_pass_rad, RAD_TO_DEG),
            'centre_mean_deg': column_number(c.centre_mean_rad, RAD_TO_DEG),
            'phi0_coherence': column_number(c.phi0_coherence),
            'coef_rel': column_number(c.coefficient_rel),
            'composite_rms_deg': column_number(c.composite_rms_rad, RAD_TO_DEG),
            'period_corr': column_number(c.period_corr),
            'period_ratio': column_number(c.period_ratio),
            'rank': c.rank,
            'accepted': c.accepted,
            'reason': c.reason,
        }
    )
    if hasattr(c, 'extra_columns'):
        row.update(c.extra_columns())
    return row


def _owner_candidate_row(analysis: MMRAnalysis, c: OwnerCandidate) -> dict:
    row = _candidate_base(analysis, 'owner', c.spec)
    row.update(
        {
            'R2': column_number(c.r2_median),
            'phi0_coherence': column_number(c.phi0_coherence),
            'psi_share': column_number(c.psi_share),
            'K_au_yr': column_number(c.force_au_yr),
            'K_theory_au_yr': column_number(c.force_theory_au_yr),
            'strength_ratio': column_number(c.strength_ratio),
            'rank': c.rank,
            'accepted': c.accepted,
            'reason': c.reason,
            'regime': c.regime,
            'n_windows': c.n_windows,
            'sigma_R2': column_number(c.sigma_r2),
            'sigma_phi0_coherence': column_number(c.sigma_phi0_coherence),
            'sigma_slow_turns': c.sigma_slow_turns,
            'sigma_slow_to_band': column_number(c.sigma_slow_to_band),
            'law': c.law,
            'multiplet_R2': column_number(c.multiplet_r2),
            'multiplet_coherence': column_number(c.multiplet_coherence),
            'multiplet_planet': c.multiplet_planet,
            'sigma_multiplet_R2': column_number(c.sigma_multiplet_r2),
            'owner_period_yr': column_number(c.owner_period_yr),
            'rate_admissible': c.rate_admissible,
            'lever_owner': column_number(c.lever_owner),
            'lever_sigma': column_number(c.lever_sigma),
            'route': c.route,
            'pot_reason': c.potential.reason if c.potential is not None else '',
            'pot_R2': column_number(c.potential.r2_median) if c.potential is not None else None,
            'pot_scale_lo': column_number(c.potential.scale_lo) if c.potential is not None else None,
            'pot_scale_hi': column_number(c.potential.scale_hi) if c.potential is not None else None,
        }
    )
    return row
