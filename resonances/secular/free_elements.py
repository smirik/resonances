"""Free (proper) eccentricity and inclination vectors, and the free argument of pericentre.

An osculating element vector is the sum of a *forced* part, imposed by the planets and
turning at *their* frequencies, and a *free* part carrying the object's own secular
motion:

    z_e(t) = e * exp(i * varpi) = z_forced + z_free      (driven by g5..g8)
    z_i(t) = sin(i/2) * exp(i * Omega)                   (driven by s5..s8)

Vectors add and angles do not, so the whole computation lives in the non-singular
variables built by `non_singular_elements`. What it is for: the Lidov-Kozai angle is
omega = varpi - Omega, so

    omega_free(t) = arg(z_e_free) - arg(z_i_free)

separates a genuine ZLK librator from an object whose osculating omega only *looks*
confined because the forced eccentricity offsets the (k, h) cloud.

The forced part is removed by a complex least-squares fit at the *known* planetary
frequencies from `PLANETARY_FREQUENCIES`. Fitting known frequencies rather than searching
for them matters: a frequency analysis (NAFF/FMFT) of a real integration splits the free
mode into five or six neighbouring terms — on 4257 the spread is +-0.5 arcsec/yr against a
Rayleigh resolution of 0.13 — and any of those splinters can be mistaken for a planetary
line and subtracted by accident.

Which modes can be fitted depends on the baseline, and the rule is pairwise rather than
per-mode. Two modes whose rates differ by less than the Rayleigh resolution never drift a
full turn apart over the window, so no fit can tell them apart and least squares splits
their amplitude arbitrarily. `cluster` therefore groups the candidates by 2/T and only one
member of each group is fitted — see `choose_basis` for how that member is picked, and why
the choice is a modelling assumption rather than a free convention.

This module is pure signal processing: it knows nothing about Body, Resonance or status.
The decision built on top of it lives in `resonance/omega_free_gate.py`.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import decimate

from resonances.data.const import PLANETARY_FREQUENCIES
from resonances.logger import logger
from .proper_angle import non_singular_elements

ARCSEC_PER_TURN = 360.0 * 3600.0  # one full turn, in arcsec; also arcsec/yr per cycle/yr
RAD_PER_ARCSEC = 2.0 * np.pi / ARCSEC_PER_TURN

# Order in which a cluster's representative is picked. This is NOT arbitrary and it is not
# free: measured across the reference set, fitting a different member of the same cluster
# moves 3752 from 2 to -9 and 591986 from -2 to -9. The order encodes what actually
# dominates the forcing — Jupiter's g5 for the eccentricity vector of a main-belt orbit,
# and the constant offset between the ecliptic and the invariable plane (the s5 mode, zero
# by construction) for the slow inclination modes. On baselines from 300 kyr to 1 Myr it
# reproduces the {g5, g6} / {const, s6} basis that was validated independently against
# 10 Myr integrations. Errors from a different choice always go towards withholding a
# verdict, never towards a false confirmation.
PRIORITY_E = ('g5', 'g6', 'g7', 'g8', 'const')
PRIORITY_I = ('const', 's6', 's7', 's8')

# Points where |z_free| falls below this fraction of its median are dropped before the
# phase is unwrapped: near the origin the argument is undefined and unwrapping picks up
# spurious half-turns that look like circulation.
DEFAULT_MASK_QUANTILE = 0.15

# An angle whose span over the baseline is less than a full turn is confined. The threshold
# is topological rather than tuned, and it must not be tightened: inside a mean-motion
# resonance the Lidov-Kozai cycle is asymmetric and its centre shifts away from 90/270
# degrees — to 0/180 for Hildas and by +-60 degrees for Trojans (Vinogradova 2024).
LIBRATION_RANGE = 2.0 * np.pi


def candidate_frequencies(kind: str, frequencies: Optional[dict] = None) -> Dict[str, float]:
    """Candidate modes for one vector, arcsec/yr, in representative-priority order.

    'const' is a mode at exactly zero frequency. For the inclination vector it is real
    physics under another name — s5 is identically zero because the planetary angular
    momentum is conserved, and the vector it defines is the invariable plane, so an
    inclination measured from the ecliptic carries a constant offset of about 1.6 degrees.
    For the eccentricity vector there is no conserved direction and every g is non-zero;
    'const' there is a technical absorber for the slow arcs of unresolved modes.
    """
    source = PLANETARY_FREQUENCIES if frequencies is None else frequencies
    if kind == 'e':
        return {label: (0.0 if label == 'const' else source[label]) for label in PRIORITY_E}
    if kind == 'i':
        return {label: (source.get('s5', 0.0) if label == 'const' else source[label]) for label in PRIORITY_I}
    raise ValueError(f"kind must be 'e' or 'i', got {kind!r}")


def cluster(candidates: Dict[str, float], baseline: float) -> List[List[str]]:
    """Group modes the baseline cannot tell apart, keeping each group in priority order.

    Two rates separated by less than 2/T accumulate less than two turns of relative phase
    over the window, so a fit cannot attribute amplitude between them. Grouping is by
    chaining: A joins B and B joins C puts all three together even when A and C are far
    apart, because each link is individually unresolvable.
    """
    threshold = 2.0 * ARCSEC_PER_TURN / baseline if baseline > 0 else float('inf')
    ordered = sorted(candidates, key=lambda label: candidates[label])
    groups: List[List[str]] = []
    current = [ordered[0]]
    for previous, label in zip(ordered, ordered[1:]):
        if candidates[label] - candidates[previous] < threshold:
            current.append(label)
        else:
            groups.append(current)
            current = [label]
    groups.append(current)
    return [[label for label in candidates if label in members] for members in groups]


def choose_basis(
    candidates: Dict[str, float], baseline: float, representatives: Optional[Sequence[str]] = None
) -> Tuple[Dict[str, float], List[str], List[List[str]]]:
    """Reduce the candidates to one representative per cluster.

    Returns the basis (label -> arcsec/yr), the dropped labels, and the clusters. By default
    the representative is the highest-priority member, `candidates` being in priority order
    — see PRIORITY_E for why that ordering carries a physical assumption. `representatives`
    forces the choice instead, which is what makes that assumption testable: exactly one
    label per cluster, in any order.
    """
    groups = cluster(candidates, baseline)
    if representatives is None:
        chosen = [members[0] for members in groups]
    else:
        chosen = [next(label for label in members if label in representatives) for members in groups]
    basis = {label: candidates[label] for label in chosen}
    dropped = [label for members in groups for label in members if label not in basis]
    return basis, dropped, groups


def remove_forced(z: np.ndarray, times: np.ndarray, frequencies: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, Dict[str, complex]]:
    """Fit and subtract terms at `frequencies` (arcsec/yr) from a complex series.

    The model is linear in the complex amplitudes because the rates are fixed, so one
    lstsq does it: |c_j| is the amplitude of mode j and arg(c_j) its phase at t = 0.
    Least squares rather than a notch, because the fundamentals are independent but not
    orthogonal on a finite baseline — lstsq handles that exactly, while a notch also
    removes whatever else shares the bin.
    """
    z = np.asarray(z, dtype=complex)
    if not frequencies:
        return z, np.zeros_like(z), {}
    design = np.exp(1j * np.outer(times, [value * RAD_PER_ARCSEC for value in frequencies.values()]))
    coefficients, *_ = np.linalg.lstsq(design, z, rcond=None)
    forced = design @ coefficients
    return z - forced, forced, dict(zip(frequencies.keys(), coefficients))


def mask_origin(z_free: np.ndarray, quantile: float = DEFAULT_MASK_QUANTILE) -> np.ndarray:
    """Which samples have a well-defined argument: those far enough from the origin."""
    amplitude = np.abs(z_free)
    median = float(np.median(amplitude))
    if median <= 0:
        return np.ones(amplitude.shape, dtype=bool)
    return amplitude >= quantile * median


def rate(times: np.ndarray, angle: np.ndarray) -> float:
    """Least-squares slope of an unwrapped angle, in arcsec/yr."""
    if len(times) < 3:
        return float('nan')
    return float(np.polyfit(times, angle, 1)[0] / RAD_PER_ARCSEC)


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Median over a sliding window, used to smooth the beating of |z_forced|."""
    window = max(1, min(int(window), len(values)))
    if window <= 1:
        return values
    view = np.lib.stride_tricks.sliding_window_view(values, window)
    return np.median(view, axis=1)


def _beat_window(basis: Dict[str, float], dt: float) -> int:
    """Samples covering a tenth of the shortest beat period between fitted modes.

    |z_forced| is a sum of arrows, so its length oscillates at the difference frequencies
    of every pair. A window well inside the shortest beat smooths sampling noise without
    smoothing the beat itself. A single-line basis has no beats, hence no smoothing.
    """
    rates = list(basis.values())
    gaps = [abs(a - b) for index, a in enumerate(rates) for b in rates[index + 1 :] if abs(a - b) > 0]
    if not gaps or dt <= 0:
        return 1
    return max(1, int(round(ARCSEC_PER_TURN / max(gaps) / 10.0 / dt)))


@dataclass
class FreeElements:
    """Forced/free split of one body's eccentricity and inclination vectors."""

    times: np.ndarray  # years, possibly resampled
    z_e: np.ndarray  # osculating e * exp(i varpi)
    z_i: np.ndarray  # osculating sin(i/2) * exp(i Omega)
    z_e_free: np.ndarray
    z_i_free: np.ndarray
    z_e_forced: np.ndarray
    z_i_forced: np.ndarray
    basis_e: Dict[str, float]  # modes actually fitted, arcsec/yr
    basis_i: Dict[str, float]
    dropped_e: List[str]  # cluster members the baseline could not separate
    dropped_i: List[str]
    amplitudes_e: Dict[str, complex]  # fitted complex amplitude per mode
    amplitudes_i: Dict[str, complex]
    keep_e: np.ndarray  # boolean mask of samples with a defined argument
    keep_i: np.ndarray
    varpi_free: np.ndarray  # unwrapped, on the keep_e grid
    Omega_free: np.ndarray  # unwrapped, on the keep_i grid
    omega_free: np.ndarray  # unwrapped, on the intersection of the two masks
    b_varpi: float  # arcsec/yr
    b_Omega: float
    b_omega: float
    clusters_e: List[List[str]] = field(default_factory=list)
    clusters_i: List[List[str]] = field(default_factory=list)
    # The fundamentals this split was built from, carried along so that anything judging
    # the residual compares it against the same numbers that were subtracted.
    frequencies: Dict[str, float] = field(default_factory=lambda: dict(PLANETARY_FREQUENCIES))

    @property
    def baseline(self) -> float:
        return float(self.times[-1] - self.times[0])

    @property
    def resolution(self) -> float:
        """One Rayleigh width over this baseline, arcsec/yr — the frequency tolerance."""
        return ARCSEC_PER_TURN / self.baseline

    @property
    def cluster_threshold(self) -> float:
        """The 2/T grouping threshold, arcsec/yr."""
        return 2.0 * self.resolution

    @property
    def omega_times(self) -> np.ndarray:
        """Times where both arguments are defined — the grid omega_free lives on."""
        return self.times[self.keep_e & self.keep_i]

    @property
    def omega_range(self) -> float:
        """Span of the free omega, radians, trimmed by half a percent at each end.

        Quantiles rather than a strict peak-to-peak: a span is an extreme statistic and one
        surviving glitch ruins the maximum. The known cost is that a sharp excursion right
        at the end of the window gets trimmed too.
        """
        if len(self.omega_free) == 0:
            return float('nan')
        return float(np.quantile(self.omega_free, 0.995) - np.quantile(self.omega_free, 0.005))

    @property
    def librates(self) -> bool:
        """Whether the free omega stays inside a full turn over the baseline."""
        return bool(self.omega_range < LIBRATION_RANGE)

    @property
    def revolutions(self) -> float:
        """Turns accumulated by the free omega over the baseline."""
        if len(self.omega_free) == 0:
            return float('nan')
        return float((self.omega_free[-1] - self.omega_free[0]) / (2.0 * np.pi))

    @property
    def e_free(self) -> float:
        """Median length of the free eccentricity vector.

        Median rather than mean: |z_free| of a real Lidov-Kozai object oscillates — that is
        the resonance — with dips towards zero and occasional spikes. For a circle the
        median is the radius, so nothing is biased away.
        """
        return float(np.median(np.abs(self.z_e_free)))

    @property
    def e_forced(self) -> float:
        return float(np.median(np.abs(self.z_e_forced)))

    @property
    def rho(self) -> float:
        """e_free / e_forced: does the free circle enclose the origin?

        Above one it does and varpi can turn freely; below one the osculating pericentre is
        pinned by the planets and any libration of it is geometry. Annotation only, never a
        verdict — for a large-amplitude librator e_forced is inflated by the same free
        signal the fit could not represent, so read it in the small-amplitude regime.
        """
        forced = self.e_forced
        return self.e_free / forced if forced > 0 else float('nan')

    @property
    def e_free_series(self) -> np.ndarray:
        """Free eccentricity over time, on the full grid."""
        return np.abs(self.z_e_free)

    @property
    def omega_free_wrapped(self) -> np.ndarray:
        """Free omega folded into [0, 2*pi), on the `omega_times` grid."""
        return np.mod(self.omega_free, 2.0 * np.pi)

    def free_eccentricity_vector(self) -> Tuple[np.ndarray, np.ndarray]:
        """(e cos omega, e sin omega) built from free quantities, on the `omega_times` grid.

        The free counterpart of `Body.eccentricity_vector()`. Drawn against the osculating
        one it shows directly whether the cloud that keeps omega bounded is the object's
        own free circle or the forced offset: a genuine librator keeps a comparable ring,
        a carousel collapses to a small ring displaced from the origin.
        """
        amplitude = self.e_free_series[self.keep_e & self.keep_i]
        return amplitude * np.cos(self.omega_free), amplitude * np.sin(self.omega_free)

    @property
    def mask_fraction_e(self) -> float:
        return float(1.0 - self.keep_e.mean())

    @property
    def mask_fraction_i(self) -> float:
        return float(1.0 - self.keep_i.mean())

    @property
    def slope_consistency(self) -> float:
        """|b_omega - (b_varpi - b_Omega)|, arcsec/yr.

        Not an identity: the three slopes are fitted on three different grids, because the
        two masks are independent and omega_free lives on their intersection. A large value
        means the masks disagree badly enough that the rates are not comparable.
        """
        return float(abs(self.b_omega - (self.b_varpi - self.b_Omega)))

    def rho_envelope(self) -> Tuple[float, float]:
        """(rho_min, rho_max) over the beating of |z_e_forced|."""
        dt = float(np.median(np.diff(self.times))) if len(self.times) > 1 else 0.0
        smoothed = _rolling_median(np.abs(self.z_e_forced), _beat_window(self.basis_e, dt))
        low, high = float(np.min(smoothed)), float(np.max(smoothed))
        e_free = self.e_free
        return (e_free / high if high > 0 else float('nan'), e_free / low if low > 0 else float('nan'))

    def drift_flag(self) -> int:
        """Whether the free omega shows a reproducible monotonic drift inside its span.

        Informational: the span may be under a full turn while a steady ramp is plainly
        visible, which is what makes an object worth integrating for longer. Reproducible
        means the two halves of the window agree on the slope.
        """
        times, angle = self.omega_times, self.omega_free
        if len(times) < 8 or not np.isfinite(self.b_omega):
            return 0
        half = len(times) // 2
        spread = abs(rate(times[:half], angle[:half]) - rate(times[half:], angle[half:])) / 2.0
        drift = abs(self.b_omega) * self.baseline * RAD_PER_ARCSEC
        return int(drift > np.pi and abs(self.b_omega) > 2.0 * spread)


def decimation_steps(factor: int) -> List[int]:
    """Split a decimation factor into steps of at most 10.

    scipy's FIR design stays accurate up to about a factor of 10 per pass; a single large
    factor produces a filter whose transition band swallows the signal. Steps that do not
    divide the remaining factor are shrunk rather than rounded, so the product is exact and
    the thinned time grid matches the decimated series.
    """
    steps = []
    while factor > 1:
        step = min(factor, 10)
        while factor % step and step > 1:
            step -= 1
        if step == 1:
            break
        steps.append(step)
        factor //= step
    return steps


def _resample(z: np.ndarray, steps: Sequence[int]) -> np.ndarray:
    """Decimate a complex series with an anti-aliasing filter on each component."""
    for step in steps:
        z = decimate(z.real, step, ftype='fir', zero_phase=True) + 1j * decimate(z.imag, step, ftype='fir', zero_phase=True)
    return z


def free_elements(
    times: np.ndarray,
    ecc: np.ndarray,
    inc: np.ndarray,
    Omega: np.ndarray,
    omega: np.ndarray,
    sampling_years: Optional[float] = 500.0,
    frequencies: Optional[dict] = None,
    mask_quantile: float = DEFAULT_MASK_QUANTILE,
    representatives: Optional[Sequence[str]] = None,
) -> Optional[FreeElements]:
    """Split the eccentricity and inclination vectors into forced and free parts.

    Parameters
    ----------
    times : np.ndarray
        Sampling times in years, uniform.
    ecc, inc, Omega, omega : np.ndarray
        Osculating elements; angles in radians. `omega` is the argument of pericentre, the
        longitude is rebuilt internally as varpi = Omega + omega.
    sampling_years : float, optional
        Resample to about this step first, with an anti-aliasing filter. Secular work needs
        nothing finer, and short-period terms folded into the secular band would otherwise
        show up as free motion. None keeps the integration grid.
    frequencies : dict, optional
        Planetary fundamentals in arcsec/yr, defaults to `PLANETARY_FREQUENCIES`.
    mask_quantile : float
        Samples with |z_free| below this fraction of the median are dropped before the
        argument is unwrapped.
    representatives : sequence of str, optional
        Force which member of each cluster is fitted, overriding the priority order. For
        research and for testing that the priority is a safe assumption; leave unset.

    Returns
    -------
    FreeElements, or None when the series are too short to say anything.
    """
    times = np.asarray(times, dtype=float)
    if len(times) < 16:
        logger.warning("Free elements: fewer than 16 samples, skipping")
        return None

    varpi = np.asarray(Omega) + np.asarray(omega)
    k, h, q, p = non_singular_elements(ecc, inc, Omega, varpi)
    z_e, z_i = k + 1j * h, q + 1j * p

    dt = float(np.median(np.diff(times)))
    steps = decimation_steps(1 if not sampling_years else max(1, int(round(sampling_years / dt))))
    if steps:
        z_e, z_i = _resample(z_e, steps), _resample(z_i, steps)
        times = times[:: int(np.prod(steps))][: len(z_e)]
    if len(times) < 16:
        logger.warning("Free elements: fewer than 16 samples after resampling, skipping")
        return None

    baseline = float(times[-1] - times[0])
    basis_e, dropped_e, clusters_e = choose_basis(candidate_frequencies('e', frequencies), baseline, representatives)
    basis_i, dropped_i, clusters_i = choose_basis(candidate_frequencies('i', frequencies), baseline, representatives)
    if dropped_e or dropped_i:
        # info, not warning: below 10 Myr some clustering is the normal case, it is recorded
        # in the summary columns, and a batch of thousands of bodies would emit one each.
        logger.info(
            f"Free elements: a {baseline / 1e3:.0f} kyr baseline merges "
            f"{', '.join(dropped_e + dropped_i)} into their neighbours; "
            f"fitting {'+'.join(basis_e)} / {'+'.join(basis_i)}"
        )

    z_e_free, z_e_forced, amplitudes_e = remove_forced(z_e, times, basis_e)
    z_i_free, z_i_forced, amplitudes_i = remove_forced(z_i, times, basis_i)

    keep_e, keep_i = mask_origin(z_e_free, mask_quantile), mask_origin(z_i_free, mask_quantile)
    both = keep_e & keep_i
    varpi_free = np.unwrap(np.angle(z_e_free[keep_e]))
    Omega_free = np.unwrap(np.angle(z_i_free[keep_i]))
    # Unwrapped separately and only then subtracted: each vector has its own passages near
    # the origin, so a difference taken before unwrapping would inherit both sets of jumps.
    omega_free = np.unwrap(np.angle(z_e_free[both])) - np.unwrap(np.angle(z_i_free[both]))

    return FreeElements(
        times=times,
        z_e=z_e,
        z_i=z_i,
        z_e_free=z_e_free,
        z_i_free=z_i_free,
        z_e_forced=z_e_forced,
        z_i_forced=z_i_forced,
        basis_e=basis_e,
        basis_i=basis_i,
        dropped_e=dropped_e,
        dropped_i=dropped_i,
        amplitudes_e=amplitudes_e,
        amplitudes_i=amplitudes_i,
        keep_e=keep_e,
        keep_i=keep_i,
        varpi_free=varpi_free,
        Omega_free=Omega_free,
        omega_free=omega_free,
        b_varpi=rate(times[keep_e], varpi_free),
        b_Omega=rate(times[keep_i], Omega_free),
        b_omega=rate(times[both], omega_free),
        clusters_e=clusters_e,
        clusters_i=clusters_i,
        frequencies=dict(PLANETARY_FREQUENCIES if frequencies is None else frequencies),
    )


SUMMARY_COLUMNS = (
    'free_basis_e',
    'free_basis_i',
    'free_e',
    'free_e_forced',
    'free_rho',
    'free_rho_min',
    'free_rho_max',
    'free_omega_range_deg',
    'free_omega_revolutions',
    'free_b_varpi',
    'free_b_Omega',
    'free_b_omega',
    'free_drift_flag',
    'free_mask_e',
    'free_mask_i',
    'free_resolution',
)


def summary_fields(elements: Optional[FreeElements]) -> dict:
    """The decomposition columns of one summary.csv row.

    Measurements only — the verdict built on them belongs to `omega_free_gate`. Note that
    `free_e` / `free_e_forced` come from this fit, while the older `e_free` / `e_forced`
    columns come from the (k, h) scatter in `cross_spectrum` and exist for every resonance
    type; the two are different estimators and are expected to differ.
    """
    if elements is None:
        return {column: None for column in SUMMARY_COLUMNS}
    rho_min, rho_max = elements.rho_envelope()
    return {
        'free_basis_e': '+'.join(elements.basis_e),
        'free_basis_i': '+'.join(elements.basis_i),
        'free_e': elements.e_free,
        'free_e_forced': elements.e_forced,
        'free_rho': elements.rho,
        'free_rho_min': rho_min,
        'free_rho_max': rho_max,
        'free_omega_range_deg': np.degrees(elements.omega_range),
        'free_omega_revolutions': elements.revolutions,
        'free_b_varpi': elements.b_varpi,
        'free_b_Omega': elements.b_Omega,
        'free_b_omega': elements.b_omega,
        'free_drift_flag': elements.drift_flag(),
        'free_mask_e': elements.mask_fraction_e,
        'free_mask_i': elements.mask_fraction_i,
        'free_resolution': elements.resolution,
    }


def series_rows(elements: Optional[FreeElements]) -> Optional[dict]:
    """Columns of the per-body `omega_free.csv`, as equal-length arrays on the full grid."""
    if elements is None:
        return None
    both = elements.keep_e & elements.keep_i
    columns: Dict[str, np.ndarray] = {
        'times': elements.times,
        'abs_ze_free': np.abs(elements.z_e_free),
        'abs_ze_forced': np.abs(elements.z_e_forced),
        'abs_zi_free': np.abs(elements.z_i_free),
        'abs_zi_forced': np.abs(elements.z_i_forced),
        'masked_e': (~elements.keep_e).astype(int),
        'masked_i': (~elements.keep_i).astype(int),
    }
    for name, mask, values in (
        ('varpi_free_unwrapped', elements.keep_e, elements.varpi_free),
        ('Omega_free_unwrapped', elements.keep_i, elements.Omega_free),
        ('omega_free_unwrapped', both, elements.omega_free),
    ):
        full = np.full(len(elements.times), np.nan)
        full[mask] = values
        columns[name] = full
    return columns
