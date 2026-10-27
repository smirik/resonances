"""The owner: another resonance that drives a, the angle following by the identity (SPEC §4.8, test B3; W20 R5 §6).

Two angles of the body, sigma = k lambda + s (the row, A) and phi_o = k_o lambda + s_o (a resonance B), satisfy the exact
identity sigma = (v/u) phi_o + Psi/u, u = k_o/d, v = k/d, d = gcd(|k|, |k_o|), Psi = u sigma - v phi_o free of the
body's lambda (`psi_combination`; its rate is a combination of the planets' mean motions and the slow frequencies). It
holds whoever drives a, so a shared oscillation (psi without power at the period) is necessary, not proof. B owns
sigma's oscillation on an interval I when B's force explains a_dot there, by one of three routes (`accept`,
`potential_pass`): the R^2 gain over sigma; the phase with the Psi lever (`better_than_sigma`); the potential, B's
directly averaged force (`owner_potential.py`, two-body B). The row then reads non_resonant / induced_libration: on I no
trapping in A independent of B is established, which does not mean that A's term is absent. The owner takes sigma's
episode only when its intervals cover at least mmr_owner_cover_min of it (F7: `owner_intervals`, `covered_fraction`;
decide.owner_in_episode), and a second capture of sigma by its own term is tested on the owner's intervals that meet the
episode (`second_capture`; not excluded: cap second_capture_unresolved).

Steps of `run_owner`:
  1. The reference period P: the strongest a_dot oscillation within a factor mmr_owner_band_factor of P_cyc
     (without P_cyc over [filter_min_libration_cutoffs P_c, the longest period giving mmr_force_min_windows
     windows]). A circulating owner forces a at its own rate, not at the period of sigma's episode (117306:
     1895 yr against P_cyc 3359 yr), and a circulating sigma has no P_cyc at all (2000 YG4).
  2. Candidates (`enumerate_owners`): two-body phi_o for P in mmr_owner_planets (k1 <= mmr_owner_coef_max,
     order <= mmr_owner_order_max, gcd 1) and three-body angles with the angle's two planets
     (|coefficients| <= mmr_owner_coef_max_3b, gcd 1), kept when the rate from the mean motions comes within
     mmr_owner_rate_factor x 2 pi / min(P, P_cyc) of 0 somewhere in the a range of the window (an owner may
     hold the body for part of the record: 2000 YG4 sits in 7E-23 for 6 of 20 kyr). Angles with sigma's
     lambda part (sigma, its subresonances, harmonics) are not owners: SPEC §6 handles them.
  3. Each candidate (`AngleBasis`, `fit_on_frame`; `owner_angle` + `measure_candidate` for one given angle): phi_o
     from the saved longitudes, unwrapped by its model rate, minus the planets' slow ripple sum k_j dlambda_j (as
     sigma), filtered at P_c. The
     force law is fitted to the band-passed a_dot ([P / f, f P]) in windows of mmr_force_window_cycles x P,
     separately where the owner librates (< 1 turn per window) and where it circulates one way or the other
     (a forced wave): the law is the same, but windows across the owner's resonance mix the regimes and blur
     phi0 (117306 near 30 kyr). The windows keep clear of the band-pass edges (filter_edge_trim_cutoffs f P).
     Per regime also the owner's mean rate and the Psi lever (`phase_leverage`).
  4. Acceptance in a regime (`accept`): a forced wave only at an owner period inside the a band (`rate_admissible`,
     R6: 2001 PG14's 27E-47 turns at 1.47 yr), R^2 >= mmr_owner_r2_min, a fixed phase (the phi0 coherence >=
     mmr_owner_phi0_coh_min, or the multiplet law below), psi share <= mmr_owner_psi_share_max, not a ripple (in a
     forced-wave regime sigma does not turn back below the owner's band by more than the owner can move it), and the
     owner explains a_dot better than sigma on the same windows by one of two routes (`better_than_sigma`):
       r2        R^2 >= R^2(sigma) + mmr_owner_r2_gain (SPEC: sigma's law fails where the owner's holds);
       phase     C_B >= L_owner + mmr_owner_phi0_coh_gain and C_sigma < L_sigma + gain: at small amplitude the identity
                 fits a_dot to ANY slowly moving angle with lambda (R^2 ~ 1 for both); if sigma's own term drove a, the
                 owner's phase would move with -Psi_w / v from window to window, so its coherence could not exceed
                 L_owner = |<exp(i Psi_w / v)>|; if the owner drives, sigma's is L_sigma = |<exp(i Psi_w / u)>| (with
                 the owner's slips, `slip_corrected`). L_owner > 1 - gain: phase_unresolved (Psi too steady to tell).
     The third route, 'potential' (`potential_pass`), is tried on the best mmr_owner_potential_top_n candidates that pass
     R^2, psi share and the ripple test when none was accepted: B's averaged force (2 k_o / (n a)) G m_P d<D>/dphi_o
     along the recorded phi_o, band-passed as a, explains a_dot (median R^2 >= mmr_owner_r2_min, every window scale in
     [1 / mmr_owner_potential_scale_max, mmr_owner_potential_scale_max]). It needs no fixed phase: an owner whose
     equilibrium moves with e, i and varpi (Gallardo 2006a, Icarus 184, pp. 30-31; F3: 2003 UG22's 10E-17, 101703's
     3:1 multiplet) is followed window by window.
     Frequency selection makes R^2 alone meaningless (any angle turning at the wave's rate fits it); the
     phase, the comparison with sigma and the force's size carry the decision (385266: the rank-2 13N-22 has R^2 0.83
     on its potential, but its own term is 1/170 of the force). The force K_o over the scale K_th = 2 |k| n a prod(mu)
     (`strength_ratio`, a force scale without the coefficient) is an order of magnitude, not an upper bound (Gallardo
     2006a p. 30, Nesvorny & Morbidelli 1998 p. 3031), so it neither rejects nor accepts an owner
     (W3-F3); it only orders the candidates and qualifies `owner_probable` (`within_scale`).
  5. The owner's description (`describe_owner`): its intervals (every passing regime's windows, the slips between two
     librations), their cover of sigma's decisive episode, the second capture on the connected block of intervals that
     meets the episode (x = Psi_slow / u with Psi from sigma and phi_o with the planets' slow ripple removed: the
     ripple of lambda_P is an outside forcing that would break any energy integral), the bound on sigma's own term
     max |s - 1| (with T_obs / sqrt(|v/u| bound) its shortest own period), the identity written out.

The multiplet law (W15, `owner_multiplet`; mmr_owner_law = 'multiplet'). An owner's force is not one term but the
members phi_j = phi_o + j (varpi - varpi_P), j = 0, -sgn p, ..., -p (`response.multiplet_harmonics`:
e^(|p|-|j|) e_P^|j|, the same order), and when two members are comparable the phase of their sum turns with varpi - varpi_P: the single phi0 then
drifts by a full turn per modulation period although the owner drives a (101703 41J-14: 3J-1+0-2, R^2 0.995, phi0
coherence 0.41; e e_J outweighs e^2, Murray & Dermott 1999 ch. 8). Where the single phase fails, the candidate is
fitted with every member at once (`response.multiplet_fit`, the law the angle's own L4 uses in `_best_multiplet`), in
windows of max(modulation 2 pi / |varpi_dot - g_P|, the force window) inside the regime: the phase is fixed when the
constant-coefficient fit over the windows has R^2 >= mmr_owner_r2_min and the leading coefficients keep their phase
from window to window (coherence >= mmr_owner_phi0_coh_min). Fewer than mmr_force_min_windows such windows: not
measured (the members are not separable over less than a modulation period). A candidate fixed only by the multiplet
law must beat sigma on the same long windows: its R^2 >= sigma's constant-coefficient R^2 (`sigma_long_r2`) +
mmr_owner_r2_gain; the phase route compares single phases and is closed to it. varpi: the body's, low-passed at P_c;
varpi_P: g_P t (context.planet_varpi_rate_rad_yr; its phase only turns the coefficients, a drift of it is multiplied
by j). A three-body candidate is fitted over one planet's varpi at a time (no mixed members). Two windows give a weak
coherence (two random phases pass 0.8 with probability 0.41): the long-window R^2 carries the decision (Codex W15).

Resolution (W15, `resolved_windows`). The fit grid keeps two samples per P_c, but the low-pass keeps an angle's linear
carrier: a window where a column of the fit (the candidate, each multiplet member, sigma and sigma's members in the
comparison) moves by pi or more between grid samples is not fitted (its sine would alias into the band); a candidate
with such a window is measured on the full grid (`dense_frame`, the same band-passed series), and sigma's
unresolved window has R^2 0 (its sine has a period below P_c, outside the band).

Slow turns of a libration (W15). A slow turning point counts for the ripple test only when sigma's slow part swings
by less than a full turn on at least one side of it (to the neighbouring turning point, or to the region's edge, a
lower bound there): a libration of sigma turns back within 2 pi; a reversal of circulation (sigma_dot changing sign as
a crosses a*, 101703 at 5.5 and 11.6 kyr: 613 and 640 deg on either side) is not one, and a circulating owner's forced
wave may sit on it. 129772 4N-7 keeps its turn (half a slow libration of 130-184 deg over 100 kyr).

Cost (W3-M2). A candidate angle is an integer combination of a few unwrapped series (lambda and varpi of the body,
lambda_j minus its slow ripple and varpi_j of each planet), and every step of the measurement is linear in the angle
up to the sine fit: the low-pass, the slope, the turns per window. `AngleBasis` filters each series once; a candidate
is then a combination on the fit grid (two samples per P_c), its turns a combination of per-window spans, its rate a
combination of slopes, and the unwrap check a bound from the largest per-step error of each series (exact only when
the bound is not decisive). The full-resolution angle is formed only for the few candidates whose psi is needed and
for the top one. Equal to unwrapping each candidate by its own model rate whenever its per-step error stays below pi,
which every candidate passing `unwrap_ok` (mmr_unwrap_max_resid_deg < 180) does.

The ripple test guards a lattice coincidence. At sigma's exact commensurability k1 n_P = k n every angle
j1 lambda_P - j2 lambda with j1 k - j2 k1 = +-1 turns at n_P / k: the planet's shortest forced term, a wave
of a at P = k P_P. When sigma's cycles are counted on that ripple (TNOs in high-order Neptune resonances:
307261 11N-18, P_cyc 3080 yr = 18 P_N, while sigma librates over ~30 kyr) the term fits a_dot with a fixed
phase and passes every other test, yet it moves sigma by 1-10 deg of a 30-150 deg libration: it owns the
ripple, not the libration. Ripple = in a forced-wave regime sigma below the band (low-passed at f P) turns
back in the regime's windows with a prominence of at least mmr_owner_ripple_factor x the band's
peak-to-peak. A circulating owner forces a periodic wave with no slow part, so it cannot have made that
turn. One turn is enough (129772 4N-7: 100 kyr hold half a cycle of 130-184 deg, the owner 7N-12 moves
sigma by 1 deg). Ripples (all forced waves): slow/band 6-260 with 1-12 turns; the forced-wave owners do not
turn (117306: its slow sigma circulates, 3.4; 385266). In the libration regime the test is off: the owner's
own slow motion (a modulated libration) reaches sigma through the identity (1997 GK3: 10E-17 librates,
sigma's slow part turns twice at 5x the band). A per-window share of variance does not separate ripples
from owners (windows of 3 P see a 30-kyr libration as a line: 0.25-0.94 for the ripples).

check_mmr_owner: pass = searched, no owner (or, set by the decision, an owner acting outside sigma's episode, F7);
fail = owner found; not_measured = no saved longitudes, no period, too few windows, no candidate could be unwrapped, no
owner accepted while a candidate on catalogue longitudes was too close to call (below), or the best candidate is
phase_unresolved and its potential route not applicable (flag owner_unresolved). Retrograde angles: not_applicable.

Catalogue longitudes. A planet whose lambda was not saved but whose lambda_j0 is known (`MMRSeries.catalogue_longitudes`)
enters the search as the line lambda_j0 + n_j (t - t0), n_j the table rate (`catalogue_line`); here only, not in the
ripple, the sampling, the subresonances or anything else (a varpi_j or Omega_j from the secular modes errs by 13-170 deg
over 20 kyr, so the subresonance scan stays not_measured). A constant error of lambda_j0 only turns phi0 in every window
(R^2 and the coherence are unchanged); an error alpha of the rate turns phi0 by alpha t from window to window, and over
window centres spread uniformly across T_eff the coherence falls to |sinc(alpha T_eff / 2)|. The line has no slow
ripple (the low-pass at P_c left its effect within G(1 - G) <= 1/4 of it anyway). Its phase error over T_eff is bounded
by D = sum_j |k_j| min(SPREAD_j T_eff, 2 SLOW_MAX_j + OFFSET_j T_eff) (`catalogue_phase_error`, const tables measured on
the package model), and the coherence by Delta C = 0.218 D (`coherence_tolerance`: the steepest slope of sinc(D/2)).
A candidate whose coherence lies within Delta C of mmr_owner_phi0_coh_min, or of sigma's coherence +
mmr_owner_phi0_coh_gain when that route decides, is `catalogue_ambiguous` and neither accepted nor rejected
(`catalogue_ambiguous`). Without an accepted owner such a candidate makes the check not_measured
(owner_catalogue_ambiguous); an accepted owner stays accepted, flagged when an ambiguous candidate would outrank it.
Every search on catalogue longitudes carries the flag catalogue_longitudes, which caps the confidence at medium. On
2001 HW7 5J+6S-2 (20 kyr) the catalogue Earth finds 5E-16+0+11 with the R^2 and coherence of the saved planets (0.944,
0.994); on 2003 UG22 8M-4S-7 the catalogue Saturn (9 lambda_S: D = 1.1 rad) would accept the false 5M-9S-4+0+0+8
(coherence 0.954 against 0.883 with the saved planets): ambiguous.

Co-orbital (1:1) angles: not_applicable (`FLAG_COORBITAL`). With sigma = lambda_P - lambda (body coefficient 1) every
candidate k1 lambda_P' - k2 lambda + p varpi is k2 sigma + (k1 lambda_P' - k2 lambda_P + p varpi): the angle under test
times k2 plus a combination of the planets alone, which turns slowly only at a near-commensurability of the planets
(524522 1V-1: 21E-13+0-8 = 13 sigma + the Earth-Venus 13:8). Such a candidate moves with sigma, so its fit measures
the 1:1 itself, and as a term of the disturbing function it is of order |p| in e against the 1:1's zeroth order with
the same planet's mass (the d'Alembert characteristic: the lowest power of e is |p|, Murray & Dermott 1999
ch. 6): it cannot own the 1:1. Whether a moves under the planet's own force is the co-orbital law's question
(`coorbital.py`, check_mmr_a_response).
`owner_probable` (own_flags) marks a best candidate that fits a (R^2, small psi share, no ripple, a force within the
mu scale) but fails the phase or the comparison with sigma (2003 UG22: 10E-17 with a drifting centre).
"""

from __future__ import annotations

import functools
import itertools
import math
from fractions import Fraction
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import signal

from resonances.data import const
from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.filtering import band_pass
from resonances.resonance.analysis.mmr.context import linear_slope, mean_motion, table_mean_motion, trend_line
from resonances.resonance.analysis.mmr import owner_potential as potential_module
from resonances.resonance.analysis.mmr.models import (
    PLANET_NAMES_ORDER,
    Context,
    ForceWindow,
    MMRSeries,
    OwnerCandidate,
    OwnerResult,
    PotentialFit,
    Preprocessed,
    ResonanceSpec,
    SecondCapture,
    planet_letter,
    planet_terms,
)
from resonances.resonance.analysis.mmr.preprocess import coorbital_planet
from resonances.resonance.analysis.mmr.response import (
    LAW_MULTIPLET,
    LAW_SINGLE,
    correlation,
    multiplet_fit,
    multiplet_harmonics,
    samples_between,
    finite_median,
    sine_coefficients,
    sine_fits_in_windows,
    sliding_windows,
    window_bounds,
)
from resonances.resonance.analysis.mmr.sampling import predictive_unwrap
from resonances.resonance.analysis.geometry import find_turning_points, wrap_pi
from resonances.resonance.analysis.ledger import Ledger, Op
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, CheckValue, Flag, OriginDetail

NAN = float('nan')
TWO_PI = 2 * math.pi

REGIME_LIBRATION = 'libration'  # the owner turns less than once per window
# the owner circulates and forces a at its rate; the two directions are apart: between them it crossed its resonance
REGIME_FORCED_POSITIVE = 'forced_wave_positive'
REGIME_FORCED_NEGATIVE = 'forced_wave_negative'

GAIN_R2, GAIN_PHASE, GAIN_POTENTIAL = 'r2', 'phase', 'potential'
GAIN_IDENTITY = 'identity'  # the route of `owner_identity` (W22 R-OWNER-ID): phi librates and its strength passes
PHASE_UNRESOLVED = 'phase_unresolved'  # `better_than_sigma`: the phases cannot separate the owner from sigma

# interval kinds of an accepted owner (`owner_intervals`)
INTERVAL_LIBRATION, INTERVAL_FORCED, INTERVAL_SLIP = 'libration', 'forced_wave', 'slip'
CAPTURE_EXCLUDED, CAPTURE_UNRESOLVED = 'excluded', 'unresolved'

# Reasons of a candidate (candidates.csv `reason`): the first test that fails, or 'accepted:<route>'.
REASON_ACCEPTED = 'accepted'
REASON_UNWRAP = 'unwrap_failed'
REASON_WINDOWS = 'too_few_windows'
REASON_R2 = 'r2_low'
REASON_COHERENCE = 'phase_incoherent'
REASON_PSI = 'psi_share_high'
REASON_RIPPLE = 'ripple_only'
REASON_NOT_BETTER = 'not_better_than_sigma'
REASON_CATALOGUE_AMBIGUOUS = 'catalogue_ambiguous'  # catalogue longitudes: a coherence within Delta C of a threshold
REASON_RATE = 'rate_outside_band'  # a forced wave at the owner's period would lie outside the a band [P/f, f P]
REASON_PHASE_UNRESOLVED = 'phase_unresolved'  # no route passed and the phase route could not separate A from B
REASON_POTENTIAL_SCALE = 'potential_scale_out'  # the potential's R^2 passes but a window scale lies outside the band
REASON_POTENTIAL_UNCONFIRMED = 'potential_unconfirmed'  # mmr_owner_potential_confirm: the averaged force disagrees

# own_flags
FLAG_NO_LONGITUDES = 'owner_no_longitudes'
FLAG_NO_PERIOD = 'owner_no_period'
FLAG_TOO_FEW_WINDOWS = 'owner_too_few_windows'
FLAG_UNWRAP_FAILED = 'owner_unwrap_failed'
FLAG_SEARCH_OFF = 'owner_search_off'
FLAG_RETROGRADE = 'owner_retrograde'
FLAG_COORBITAL = 'owner_coorbital'  # a 1:1 angle: every candidate is k sigma + a planetary combination (module doc)
FLAG_PROBABLE = 'owner_probable'
FLAG_CATALOGUE = str(Flag.CATALOGUE_LONGITUDES)
FLAG_CATALOGUE_AMBIGUOUS = str(Flag.OWNER_CATALOGUE_AMBIGUOUS)
FLAG_UNRESOLVED = str(Flag.OWNER_UNRESOLVED)

# max over D of |d sinc(D / 2) / dD| (at D = 4.16 rad): |C(alpha) - C(alpha')| <= this x |alpha - alpha'| T_eff
SINC_HALF_SLOPE_MAX = 0.2181


def _finite_or(value, default: float) -> float:
    return float(value) if value is not None and np.isfinite(value) else default


def slope(t_yr: np.ndarray, values: np.ndarray) -> float:
    """Least-squares slope [unit/yr] in closed form (np.polyfit is 10x slower and runs per candidate); nan below 2 points."""
    t = np.asarray(t_yr, float)
    if len(t) < 2:
        return NAN
    dt = t - t.mean()
    denominator = float(np.dot(dt, dt))
    return float(np.dot(dt, np.asarray(values, float)) / denominator) if denominator > 0 else NAN


# --------------------------------------------------------------------------------------------- when


def owner_required(a_response: str, amplitude_max_rad: float, oscillates: bool, settings: AnalysisSettings) -> bool:
    """Search when the angle oscillates (episode, drift, turning points); mandatory when the force law is
    not_measured (A < mmr_a_response_min_amp_deg: the force law cannot tell an induced libration from an own
    one there); never with mmr_owner_search='off'. `amplitude_max_rad` is informative only."""
    if settings.mmr_owner_search == 'off':
        return False
    return bool(oscillates) or str(a_response) == str(CheckValue.NOT_MEASURED)


OWNER_R2 = '2 owner R2'  # the ledger name of the search's decisive comparison (`record_decisive`)


def record_decisive(owner: OwnerResult, settings: AnalysisSettings, ledger: Ledger) -> None:
    """Step 2 of the status tree on `ledger` (`decide._foreign`): the search's decisive comparison, the accepted
    owner's R^2 (None without one) against mmr_owner_r2_min. A route that records its own comparisons when it
    accepts adds none: the route 'identity' measures nothing on a (`OwnerResult.identity`, `owner_identity.record`)."""
    if owner.identity is not None:
        return
    value = owner.accepted.r2_median if owner.accepted is not None else None
    ledger.compare_setting(OWNER_R2, value, Op.GE, settings, 'mmr_owner_r2_min', '', 'own_R2')


def owner_search_matters(angle_status, angle_subtype) -> bool:
    """False when the status tree never reads the owner, so the search is skipped (W3-M2): an alias (C5 decides from
    the sampling alone: non_resonant/alias or invalid/alias), an undefined angle (invalid) or too little time
    (uncertain/baseline_needed; neither status carries a margin or a confidence). Pass the angle the tree reads,
    `decide.status_basis(angle_decision)` (a G1c circulation is judged as episodic)."""
    status, subtype = str(angle_status), str(angle_subtype)
    if subtype == str(AngleSubtype.ALIAS):
        return False
    return status not in (str(AngleStatus.UNDEFINED), str(AngleStatus.INSUFFICIENT_TIME))


# --------------------------------------------------------------------------------------------- candidates


def owner_key(planets: Sequence[str], planet_coefficients: Sequence[int], body_coefficient: int, body_varpi_coefficient: int) -> str:
    """Package notation with every eccentricity on the body: ('Mars',), (11,), -20, 9 -> '11M-20+0+9';
    two planets: ('Venus', 'Jupiter'), (1, 5), -7, 1 -> '1V+5J-7+0+0+1'."""
    head = planet_terms(planets, planet_coefficients)
    return f'{head}{body_coefficient:+d}{"+0" * len(planets)}{body_varpi_coefficient:+d}'


def owner_spec(planets: Sequence[str], planet_coefficients: Sequence[int], body_coefficient: int) -> ResonanceSpec:
    """The candidate angle with p = -(sum k_j + k) on the body's varpi (d'Alembert)."""
    p = -(sum(planet_coefficients) + body_coefficient)
    return ResonanceSpec.from_key(owner_key(planets, planet_coefficients, body_coefficient, p))


def _lambda_vector(spec: ResonanceSpec, names: Sequence[str]) -> np.ndarray:
    coefficients = dict(zip(spec.planets, spec.planet_coefficients))
    return np.array([coefficients.get(name, 0) for name in names] + [spec.body_coefficient], dtype=float)


def same_lambda_part(first: ResonanceSpec, second: ResonanceSpec) -> bool:
    """True when the lambda parts are proportional: the same resonance, a subresonance or a harmonic of it."""
    names = sorted(set(first.planets) | set(second.planets))
    u, v = _lambda_vector(first, names), _lambda_vector(second, names)
    return bool(np.allclose(u * np.dot(v, v), v * np.dot(u, v)))


def candidate_rate(spec: ResonanceSpec, planet_n: Mapping[str, float], body_n: float, body_varpi_rate: float) -> float:
    """phi_o_dot = sum k_j n_j + k n + p varpi_dot [rad/yr] (the candidates have p_j = 0)."""
    lam = sum(k * planet_n[name] for name, k in zip(spec.planets, spec.planet_coefficients))
    return float(lam + spec.body_coefficient * body_n + spec.body_varpi_coefficient * body_varpi_rate)


def _reaches(rate_lo: np.ndarray, rate_hi: np.ndarray, bound: float) -> np.ndarray:
    """The rate interval between the two ends comes within `bound` of zero."""
    low, high = np.minimum(rate_lo, rate_hi), np.maximum(rate_lo, rate_hi)
    return (high >= -bound) & (low <= bound)


def _two_body(planet: str, n_planet: float, body_n: Tuple[float, float], varpi_rate: float, bound: float, settings) -> List[Tuple]:
    """(planets, planet coefficients, k) of k1 lambda_P - k2 lambda + (k2 - k1) varpi reaching the bound."""
    k1 = np.arange(1, settings.mmr_owner_coef_max + 1)[:, None]
    k2 = k1 + np.arange(-settings.mmr_owner_order_max, settings.mmr_owner_order_max + 1)[None, :]
    k1 = np.broadcast_to(k1, k2.shape)
    rates = [k1 * n_planet - k2 * n + (k2 - k1) * varpi_rate for n in body_n]
    keep = (k2 >= 1) & (np.gcd(k1, k2) == 1) & _reaches(rates[0], rates[1], bound)
    return [((planet,), (int(a),), -int(b)) for a, b in zip(k1[keep], k2[keep])]


def _three_body(
    planets: Tuple[str, str], planet_n: Mapping[str, float], body_n: Tuple[float, float], varpi_rate: float, bound: float, limit: int
):
    """(planets, (m1, m2), m): m1 > 0, m2, m != 0, |m_i|, |p| <= limit, gcd 1, reaching the bound."""
    values = np.arange(-limit, limit + 1)
    m1, m2, m = np.meshgrid(np.arange(1, limit + 1), values, values, indexing='ij')
    p = -(m1 + m2 + m)
    planet_part = m1 * planet_n[planets[0]] + m2 * planet_n[planets[1]] + p * varpi_rate
    keep = (m2 != 0) & (m != 0) & (np.abs(p) <= limit) & _reaches(planet_part + m * body_n[0], planet_part + m * body_n[1], bound)
    keep &= np.gcd(np.gcd(m1, np.abs(m2)), np.abs(m)) == 1
    return [(planets, (int(a), int(b)), int(c)) for a, b, c in zip(m1[keep], m2[keep], m[keep])]


def rate_bound(period_yr: float, period_cycle_yr: float, settings: AnalysisSettings) -> float:
    """mmr_owner_rate_factor x 2 pi / min(P, P_cyc) [rad/yr]: an owner librates (rate ~ 0) or turns with the wave."""
    periods = [p for p in (period_yr, period_cycle_yr) if p is not None and np.isfinite(p) and p > 0]
    if not periods:
        return NAN
    return settings.mmr_owner_rate_factor * TWO_PI / min(periods)


def planet_mean_motion(context: Context, name: str) -> float:
    """n_j [rad/yr]: the context's (measured or table), else the table."""
    value = context.planet_mean_motion_rad_yr.get(name, NAN)
    return value if np.isfinite(value) else table_mean_motion(name)


def enumerate_owners(
    spec: ResonanceSpec,
    context: Context,
    period_cycle_yr: float,
    available_planets: Tuple[str, ...],
    settings: AnalysisSettings,
    *,
    period_yr: Optional[float] = None,
    body_n_range: Optional[Tuple[float, float]] = None,
    body_varpi_rate: Optional[float] = None,
) -> List[ResonanceSpec]:
    """Two- and three-body candidates whose rate (from mean motions) comes within the rate factor of 2 pi / P.

    P = min(`period_yr`, P_cyc). The body's mean motion spans `body_n_range` (default: n at the context's a 1 and
    99 %); planets: `context.planet_mean_motion_rad_yr`, else the table; varpi_dot: `body_varpi_rate` or the
    context's. Only planets in `available_planets` (a saved mean longitude) qualify. Sorted by the distance of
    the rate at the middle of the range from zero."""
    bound = rate_bound(period_yr if period_yr is not None else NAN, period_cycle_yr, settings)
    if not np.isfinite(bound):
        return []
    if body_n_range is None:
        body_n_range = (float(mean_motion(context.a_p01_au)), float(mean_motion(context.a_p99_au)))
    varpi_rate = _finite_or(context.body_varpi_rate_rad_yr if body_varpi_rate is None else body_varpi_rate, 0.0)
    planet_n = {name: planet_mean_motion(context, name) for name in available_planets}
    raw = []
    for planet in settings.mmr_owner_planets:
        if planet in planet_n:
            raw += _two_body(planet, planet_n[planet], body_n_range, varpi_rate, bound, settings)
    if len(spec.planets) == 2 and all(name in planet_n for name in spec.planets):
        raw += _three_body(tuple(spec.planets), planet_n, body_n_range, varpi_rate, bound, settings.mmr_owner_coef_max_3b)
    candidates = [owner_spec(*item) for item in raw]
    candidates = [c for c in candidates if not same_lambda_part(c, spec)]
    middle = float(np.mean(body_n_range))
    return sorted(candidates, key=lambda c: abs(candidate_rate(c, planet_n, middle, varpi_rate)))


# --------------------------------------------------------------------------------------------- the angle


def _saved_longitude(series: MMRSeries, name: str) -> bool:
    planet = series.planets.get(name)
    return planet is not None and planet.mean_longitude is not None


def catalogue_planets(series: MMRSeries) -> Tuple[str, ...]:
    """Planets whose lambda the search takes from the catalogue line: a catalogue longitude and no saved lambda."""
    return tuple(name for name in PLANET_NAMES_ORDER if name in series.catalogue_longitudes and not _saved_longitude(series, name))


def catalogue_line(series: MMRSeries, name: str) -> np.ndarray:
    """lambda_j0 + n_j (t - t0) [rad], unwrapped, with lambda_j0 = `series.catalogue_longitudes[name]` and the table n_j."""
    t = np.asarray(series.t_yr, float)
    return series.catalogue_longitudes[name] + table_mean_motion(name) * (t - t[0])


def planet_longitude(series: MMRSeries, name: str) -> np.ndarray:
    """The planet's saved lambda, else its catalogue line mod 2 pi [rad]."""
    if _saved_longitude(series, name):
        return np.asarray(series.planets[name].mean_longitude, float)
    return np.mod(catalogue_line(series, name), TWO_PI)


def catalogue_phase_error(spec: ResonanceSpec, catalogued: Sequence[str], span_yr: float) -> float:
    """D = sum_j |k_j| min(SPREAD_j T, 2 SLOW_MAX_j + OFFSET_j T) [rad] over the planets of `spec` in `catalogued`: the
    largest change of the candidate's phase error across a span T of window centres (module docstring; tables
    const.PLANET_N_SPREAD_DEG_PER_GAUSS_YR, PLANET_N_OFFSET_DEG_PER_GAUSS_YR, PLANET_LAMBDA_SLOW_MAX_DEG)."""
    total = 0.0
    for name, k in zip(spec.planets, spec.planet_coefficients):
        if name not in catalogued or k == 0:
            continue
        drift = math.radians(const.PLANET_N_SPREAD_DEG_PER_GAUSS_YR[name]) * span_yr
        bounded = (
            2 * math.radians(const.PLANET_LAMBDA_SLOW_MAX_DEG[name]) + math.radians(const.PLANET_N_OFFSET_DEG_PER_GAUSS_YR[name]) * span_yr
        )
        total += abs(k) * min(drift, bounded)
    return total


def wrapped_angle(candidate: ResonanceSpec, series: MMRSeries) -> np.ndarray:
    """phi_o mod 2 pi from the body's and the planets' longitudes (a planet's catalogue line when its lambda is not saved;
    a planet's varpi only where the candidate has it, which needs the planet saved)."""
    phi = candidate.body_coefficient * np.asarray(series.mean_longitude, float)
    phi = phi + candidate.body_varpi_coefficient * np.asarray(series.varpi, float)
    for name, k, p in zip(candidate.planets, candidate.planet_coefficients, candidate.planet_varpi_coefficients):
        phi = phi + k * planet_longitude(series, name)
        if p:
            phi = phi + p * np.asarray(series.planets[name].varpi, float)
    return np.mod(phi, TWO_PI)


def owner_angle_with_residual(
    candidate: ResonanceSpec,
    series: MMRSeries,
    planet_n: Mapping[str, float],
    body_n_offset: float = 0.0,
    varpi_rate: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """(phi_o unwrapped [rad], per-step model error [rad], length N-1): phi_o mod 2 pi from the saved longitudes,
    unwrapped by sum k_j n_j + k (n(a) + offset) + p varpi_dot (varpi_dot: slope over the record)."""
    t = np.asarray(series.t_yr, float)
    a = np.asarray(series.a, float)
    if varpi_rate is None:
        varpi_rate = slope(t, np.unwrap(np.asarray(series.varpi, float)))
    body = mean_motion(0.5 * (a[1:] + a[:-1])) + body_n_offset
    lam = sum(k * planet_n[name] for name, k in zip(candidate.planets, candidate.planet_coefficients))
    rate = lam + candidate.body_coefficient * body + candidate.body_varpi_coefficient * varpi_rate
    return predictive_unwrap(t, wrapped_angle(candidate, series), rate)


def owner_angle(candidate: ResonanceSpec, series: MMRSeries, planet_n: Mapping[str, float]) -> np.ndarray:
    """phi_o from the saved lambda, lambda_P (or its catalogue line), varpi with the predictive unwrap [rad]."""
    return owner_angle_with_residual(candidate, series, planet_n)[0]


def unwrap_ok(residual_rad: np.ndarray, settings: AnalysisSettings) -> bool:
    """The model stitches the angle: residual p99 and max within sigma's unwrap limits (mmr_unwrap_max_resid_*).

    The 99th percentile (linear interpolation, as np.percentile) is compared through a count of the samples above the
    limit; np.percentile runs only when the limit falls between the two order statistics it interpolates."""
    residual = np.degrees(np.abs(np.asarray(residual_rad, float)))
    if not len(residual):
        return True
    if not np.all(np.isfinite(residual)) or np.max(residual) > settings.mmr_unwrap_max_resid_deg:
        return False
    limit = settings.mmr_unwrap_max_resid_p99_deg
    position = (len(residual) - 1) * 0.99
    below = int(np.floor(position))  # the percentile interpolates the sorted values [below] and [below + 1]
    above_limit = int(np.count_nonzero(residual > limit))
    if above_limit <= len(residual) - below - 2 or (above_limit == len(residual) - below - 1 and position == below):
        return True
    if above_limit >= len(residual) - below:
        return False
    return bool(np.percentile(residual, 99) <= limit)


def planet_slow_deviations(
    t_yr: np.ndarray, planets: Mapping[str, np.ndarray], planet_n: Mapping[str, float], cutoff_yr: float, filter_order: int
) -> Dict[str, np.ndarray]:
    """dlambda_j^slow = lowpass_{P_c}(lambda_j - its line) [rad] per planet: the ripple of a term is sum k_j dlambda_j
    (as `preprocess.planet_ripple`, computed once for every candidate)."""
    t = np.asarray(t_yr, float)
    result = {}
    for name, longitude in planets.items():
        unwrapped, _ = predictive_unwrap(t, np.asarray(longitude, float), planet_n[name])
        line = trend_line(t, unwrapped)
        result[name] = filtering.lowpass(t, unwrapped - line, cutoff_yr, filter_order, detrend=True)
    return result


def body_mean_motion(t_yr: np.ndarray, mean_longitude: np.ndarray, a_au: np.ndarray, mask: np.ndarray) -> float:
    """Slope of lambda unwrapped with n(a) over the mask [rad/yr] (exact at steps of several revolutions)."""
    a_au = np.asarray(a_au, float)
    unwrapped, _ = predictive_unwrap(t_yr, mean_longitude, mean_motion(0.5 * (a_au[1:] + a_au[:-1])))
    return linear_slope(np.asarray(t_yr)[mask], unwrapped[mask])


# --------------------------------------------------------------------------------------------- the frame


def dominant_period(t_yr: np.ndarray, values: np.ndarray, period_lo_yr: float, period_hi_yr: float, padding: int = 8) -> float:
    """Period [yr] of the highest peak of the Hann-windowed, detrended power spectrum within [lo, hi]; nan when no
    frequency of the grid falls there. `padding` (zero padding) only refines the grid."""
    t = np.asarray(t_yr, float)
    x = np.asarray(values, float)
    if len(t) < 4 or not (np.isfinite(period_lo_yr) and np.isfinite(period_hi_yr) and period_hi_yr > period_lo_yr):
        return NAN
    x = (x - trend_line(t, x)) * np.hanning(len(x))
    size = len(x) * padding
    step = float(np.median(np.diff(t)))
    spacing = 1.0 / (size * step)  # the grid of np.fft.rfftfreq(size, step)
    first = max(int(np.floor(1.0 / (period_hi_yr * spacing))) - 1, 1)
    last = min(int(np.ceil(1.0 / (period_lo_yr * spacing))) + 1, size // 2)
    if last < first:
        return NAN
    index = np.arange(first, last + 1)
    periods = 1.0 / (index * spacing)
    inside = (periods >= period_lo_yr) & (periods <= period_hi_yr)
    if not np.any(inside):
        return NAN
    # the padded rfft at only the grid frequencies of the band: a zoom FFT of the unpadded series (same values; the
    # padded length 8 (10^k + 1) has a large prime factor, and its Bluestein rfft held 1.2 GB at 10^6 samples)
    zoom = signal.ZoomFFT(len(x), [first * spacing, last * spacing], m=len(index), fs=1.0 / step, endpoint=True)
    power = np.abs(zoom(x)) ** 2
    return float(periods[np.argmax(np.where(inside, power, -np.inf))])


def windows_needed_span(period_yr: float, settings: AnalysisSettings) -> float:
    """Span [yr] that holds mmr_force_min_windows sliding windows of mmr_force_window_cycles x P."""
    length = settings.mmr_force_window_cycles * period_yr
    return length * (1 + (settings.mmr_force_min_windows - 1) * settings.mmr_force_window_step)


def reference_period(prep: Preprocessed, period_cycle_yr: float, mask: np.ndarray, settings: AnalysisSettings) -> float:
    """P of the owner test [yr]: the a_dot peak within a factor mmr_owner_band_factor of P_cyc; without P_cyc the peak
    over [filter_min_libration_cutoffs P_c, the longest period that still gives mmr_force_min_windows windows]."""
    t = np.asarray(prep.t_yr, float)[mask]
    if len(t) < 4:
        return NAN
    low = settings.filter_min_libration_cutoffs * prep.cutoff.cutoff_yr
    high = (t[-1] - t[0]) / windows_needed_span(1.0, settings)
    if np.isfinite(period_cycle_yr) and period_cycle_yr > 0:
        factor = settings.mmr_owner_band_factor
        low, high = max(low, period_cycle_yr / factor), min(high, period_cycle_yr * factor)
        if not high > low:
            return float(period_cycle_yr)
    return dominant_period(t, np.asarray(prep.adot_f, float)[mask], low, high)


def theory_force(spec: ResonanceSpec, body_n: float, a_au: float) -> float:
    """K_th = omega0_th^2 / |c| = 2 |k| n a mu [au/yr]: omega0_th = |k| n sqrt(3 mu), c = 1.5 k n / a, mu = the product
    of the angle's planet masses (no e^q). A scale, not a bound: the disturbing-function coefficient is left out, and it
    can exceed 1 by orders of magnitude (W3-F3). `strength_ratio` = K_o / K_th is a diagnostic column."""
    mu = float(np.prod([const.PLANETS_MASS[name] for name in spec.planets]))
    return float(2 * abs(spec.body_coefficient) * body_n * a_au * mu)


@dataclass
class OwnerFrame:
    """What every candidate is compared with: band-passed a and a_dot, sigma and its windows, on the analysis window.

    The force-law windows are the same for every angle (sliding windows on the fit grid): `windows` holds them,
    `span_first`/`span_last` their first and last sample on the full grid (turns per window), and the sigma_* arrays
    sigma's fit per window (nan where it has none)."""

    t_yr: np.ndarray
    mask: np.ndarray
    fit_mask: np.ndarray  # mask thinned to two samples per P_c: the filtered series hold nothing faster (cost)
    period_yr: float
    window_yr: float
    a_band: np.ndarray
    adot_band: np.ndarray
    sigma_f: np.ndarray  # sigma at P_c (ripple removed)
    sigma_clean: np.ndarray  # sigma unfiltered (ripple removed)
    sigma_band: np.ndarray
    sigma_windows: Dict[float, ForceWindow]  # keyed by the rounded start time
    sigma_slow: np.ndarray  # sigma below the band: low-passed at f P (ripple removed)
    ripple_factor: float  # mmr_owner_ripple_factor
    sigma_theory_au_yr: float
    body_n: float
    body_n_offset: float  # measured lambda_dot minus <n(a)> on the window
    body_varpi_rate: float
    a_mean_au: float
    varpi_unwrapped: Optional[np.ndarray] = None
    slow_cache: Dict[Tuple[float, ...], Tuple[int, float]] = field(default_factory=dict)  # by regime window starts
    windows: List[Tuple[float, float]] = field(default_factory=list)
    span_first: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    span_last: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    sigma_r2_windows: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sigma_force_windows: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sigma_phi0_windows: np.ndarray = field(default_factory=lambda: np.zeros(0))
    varpi_fit: Optional[np.ndarray] = None  # the body's varpi unwrapped, low-passed at P_c, on the fit grid (multiplet)
    planet_varpi_rates: Mapping[str, float] = field(default_factory=dict)  # varpi_dot_P (context; multiplet law)
    sigma_spec: Optional[ResonanceSpec] = None  # the angle under test (its own multiplet in `sigma_long_r2`)
    band_factor: float = NAN  # mmr_owner_band_factor: the a band [P/f, f P] (rate admissibility of a forced wave)

    @property
    def sigma_r2(self) -> float:
        return finite_median([w.r2 for w in self.sigma_windows.values()])

    @functools.cached_property
    def on_fit_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(t, sigma_slow, sigma_band, mask) on the fit grid, taken once (`slow_oscillation` runs per regime)."""
        fit = self.fit_mask
        return self.t_yr[fit], self.sigma_slow[fit], self.sigma_band[fit], self.mask[fit]

    def spans(self, values: np.ndarray) -> np.ndarray:
        """values(last sample of each window) - values(first sample) on the full grid; nan below two samples."""
        values = np.asarray(values, float)
        result = values[self.span_last] - values[self.span_first]
        return np.where(self.span_last > self.span_first, result, NAN)


def band_window(t_yr: np.ndarray, period_yr: float, settings: AnalysisSettings) -> np.ndarray:
    """Samples clear of the band-pass edges: filter_edge_trim_cutoffs x its longest cutoff (f P) from either end of
    the record (the record's P_c trim is far too short for a filter at f P)."""
    t = np.asarray(t_yr, float)
    trim = settings.filter_edge_trim_cutoffs * settings.mmr_owner_band_factor * period_yr
    return filtering.window_mask(t, filtering.edge_window(t, trim))


def fit_stride(t_yr: np.ndarray, cutoff_yr: float) -> int:
    """Every k-th sample keeps two per P_c: series low-passed at P_c lose nothing, the fits run k times faster."""
    step = float(np.median(np.diff(np.asarray(t_yr, float)))) if len(t_yr) > 1 else NAN
    if not (np.isfinite(step) and step > 0 and np.isfinite(cutoff_yr)):
        return 1
    return max(1, int(cutoff_yr / (2 * step)))


def _start_key(start: float) -> float:
    return round(float(start), 6)


def slow_oscillation(
    t_yr: np.ndarray, sigma_slow: np.ndarray, sigma_band: np.ndarray, region: np.ndarray, factor: float
) -> Tuple[int, float]:
    """(turning points of sigma below the band with prominence >= factor x the band's peak-to-peak that turn back
    within a full turn (`libration_turns`), slow-to-band ratio = peak-to-peak of the slow sigma minus its line / that
    of the band) on `region`; (0, nan) without a band.

    Prominences are measured over the region's span, the turning points counted inside the region."""
    if np.count_nonzero(region) < 3:
        return 0, NAN
    t, slow = t_yr[region], sigma_slow[region]
    band_ptp = float(np.ptp(sigma_band[region]))
    slow_ptp = float(np.ptp(slow - trend_line(t, slow)))
    if not band_ptp > 0:
        return 0, NAN
    span = (region.argmax(), len(region) - region[::-1].argmax())
    tp = find_turning_points(t_yr[span[0] : span[1]], sigma_slow[span[0] : span[1]], factor * band_ptp, (t[0], t[-1]))
    if not len(tp):
        return 0, slow_ptp / band_ptp
    inside = region[span[0] : span[1]][tp.indices]
    swing_ok = libration_turns(slow[0], np.asarray(sigma_slow[span[0] : span[1]], float)[tp.indices], slow[-1])
    return int(np.count_nonzero(inside & swing_ok)), slow_ptp / band_ptp


def libration_turns(first: float, values: np.ndarray, last: float) -> np.ndarray:
    """Per turning point (`values` in time order): the swing to the previous or the next extreme (a neighbouring turning
    point, or the region's first/last value at the ends: a lower bound of the swing there) is below 2 pi on at least
    one side. A libration turns back within a full turn; a reversal of circulation has more than 2 pi on both sides."""
    extremes = np.concatenate([[first], np.asarray(values, float), [last]])
    swings = np.abs(np.diff(extremes))
    return np.minimum(swings[:-1], swings[1:]) < TWO_PI


def _sigma_windows(windows, a_sin, b_cos, offset, r2) -> Dict[float, ForceWindow]:
    return {
        _start_key(start): ForceWindow(
            start, end, float(r2[i]), float(np.hypot(a_sin[i], b_cos[i])), float(np.arctan2(-b_cos[i], a_sin[i])), float(offset[i])
        )
        for i, (start, end) in enumerate(windows)
        if np.isfinite(a_sin[i])
    }


def build_frame(
    prep: Preprocessed,
    series: MMRSeries,
    period_yr: float,
    settings: AnalysisSettings,
    window=None,
    *,
    context: Optional[Context] = None,
    stride: Optional[int] = None,
) -> OwnerFrame:
    """The band [P/f, fP] of a, a_dot and sigma on `window` (default prep.window); sigma's force-law windows; the
    body's mean motion (measured lambda_dot when lambda is saved, else n(<a>)) and varpi rate on the window.
    `context` gives the planets' varpi rates of the multiplet law (none: the law is not measured); `stride` overrides
    the fit grid's thinning (`fit_stride`; 1 = every sample, the dense verification)."""
    t = np.asarray(prep.t_yr, float)
    factor, order = settings.mmr_owner_band_factor, settings.filter_order
    mask = filtering.window_mask(t, prep.window if window is None else window) & band_window(t, period_yr, settings)
    a_band = band_pass(t, np.asarray(prep.a_f, float), period_yr, factor, order)
    adot_band = filtering.time_derivative(t, a_band)
    sigma_f = np.asarray(prep.angle_f, float)
    sigma_clean = np.asarray(prep.angle_clean, float)
    length = settings.mmr_force_window_cycles * period_yr
    fit_mask = mask & (np.arange(len(t)) % (stride or fit_stride(t, prep.cutoff.cutoff_yr)) == 0)
    sigma_band = band_pass(t, sigma_clean, period_yr, factor, order)
    a = np.asarray(series.a, float)
    n_mean = float(np.mean(mean_motion(a[mask]))) if np.any(mask) else NAN
    body_n = n_mean
    if series.mean_longitude is not None and np.count_nonzero(mask) > 1:
        body_n = body_mean_motion(t, series.mean_longitude, a, mask)
    a_mean = float(np.mean(np.asarray(prep.a_f, float)[mask])) if np.any(mask) else NAN
    varpi_unwrapped = np.unwrap(np.asarray(series.varpi, float))
    varpi_rate = linear_slope(t[mask], varpi_unwrapped[mask]) if np.count_nonzero(mask) > 1 else NAN
    varpi_low = None
    if context is not None and settings.mmr_owner_law == LAW_MULTIPLET:
        varpi_low = filtering.lowpass(t, varpi_unwrapped, prep.cutoff.cutoff_yr, order, grid_rtol=settings.filter_grid_rtol)
    grid = grid_fields(t, fit_mask, sigma_f, adot_band, length, varpi_low, settings)
    return OwnerFrame(
        t_yr=t,
        mask=mask,
        period_yr=float(period_yr),
        window_yr=float(length),
        a_band=a_band,
        adot_band=adot_band,
        sigma_f=sigma_f,
        sigma_clean=sigma_clean,
        sigma_band=sigma_band,
        sigma_slow=filtering.lowpass(t, sigma_clean, factor * period_yr, order, grid_rtol=settings.filter_grid_rtol),
        ripple_factor=float(settings.mmr_owner_ripple_factor),
        sigma_theory_au_yr=theory_force(series.resonance, body_n, a_mean),
        body_n=float(body_n),
        body_n_offset=float(body_n - n_mean),
        body_varpi_rate=float(varpi_rate),
        a_mean_au=a_mean,
        varpi_unwrapped=varpi_unwrapped,
        planet_varpi_rates=dict(context.planet_varpi_rate_rad_yr) if context is not None else {},
        sigma_spec=series.resonance,
        band_factor=float(factor),
        **grid,
    )


def grid_fields(t, fit_mask, sigma_f, adot_band, length, varpi_low, settings) -> dict:
    """The OwnerFrame fields of a fit grid: the force-law windows, their sample spans, sigma's fits per window (sigma
    turning faster than the grid resolves: R^2 0, no phase; its sine lies outside the band) and varpi on the grid."""
    t_fit = t[fit_mask]
    windows = sliding_windows(t_fit, length, settings.mmr_force_window_step)
    a_sin, b_cos, offset, r2 = sine_fits_in_windows(t_fit, sigma_f[fit_mask], adot_band[fit_mask], windows)
    resolved = resolved_windows(t_fit, [sigma_f[fit_mask]], windows)
    a_sin, b_cos, r2 = (np.where(resolved | ~np.isfinite(x), x, 0.0) for x in (a_sin, b_cos, r2))
    first, stop = window_bounds(t, windows)
    fitted = np.isfinite(a_sin)
    return dict(
        fit_mask=fit_mask,
        sigma_windows=_sigma_windows(windows, a_sin, b_cos, offset, r2),
        windows=list(windows),
        span_first=first,
        span_last=stop - 1,
        sigma_r2_windows=np.where(fitted, r2, NAN),
        sigma_force_windows=np.where(fitted, np.hypot(a_sin, b_cos), NAN),
        sigma_phi0_windows=np.where(fitted & resolved, np.arctan2(-b_cos, a_sin), NAN),
        varpi_fit=varpi_low[fit_mask] if varpi_low is not None else None,
    )


def dense_frame(frame: OwnerFrame, cutoff_yr: float, settings: AnalysisSettings) -> OwnerFrame:
    """The frame on the full grid (every sample of its mask), sharing its band-passed series: for a candidate the
    thinned grid does not resolve (`run_owner`)."""
    varpi_low = None
    if frame.varpi_fit is not None:
        order, rtol = settings.filter_order, settings.filter_grid_rtol
        varpi_low = filtering.lowpass(frame.t_yr, frame.varpi_unwrapped, cutoff_yr, order, grid_rtol=rtol)
    grid = grid_fields(frame.t_yr, frame.mask, frame.sigma_f, frame.adot_band, frame.window_yr, varpi_low, settings)
    return replace(frame, slow_cache={}, **grid)


# --------------------------------------------------------------------------------------------- measuring


@dataclass(frozen=True)
class RegimeFit:
    """The force law of the owner and of sigma on the windows of one regime of the owner."""

    regime: str
    starts: Tuple[float, ...]
    r2: float
    r2_windows: Tuple[float, ...]
    coherence: float
    force_au_yr: float
    sigma_r2: float
    sigma_coherence: float
    sigma_force_au_yr: float
    sigma_slow_turns: int = 0  # turning points of sigma below the band (`slow_oscillation`)
    sigma_slow_to_band: float = NAN
    # the multiplet law (`owner_multiplet`): R^2 of the constant-coefficient fit, coherence of the leading coefficients,
    # the planet of varpi_P and the number of its windows; nan / '' / 0 when not measured
    multiplet_r2: float = NAN
    multiplet_coherence: float = NAN
    multiplet_planet: str = ''
    multiplet_windows: int = 0
    sigma_multiplet_r2: float = NAN  # sigma with constant coefficients on the same long windows (`sigma_long_r2`)
    owner_rate_rad_yr: float = NAN  # mean phi_o_dot over the windows (`rate_admissible`)
    rate_admissible: bool = True
    lever_owner: float = NAN  # `phase_leverage` on the regime's windows
    lever_sigma: float = NAN


@dataclass
class OwnerFit:
    """Everything measured for one candidate (an OwnerCandidate holds one regime of it)."""

    spec: ResonanceSpec
    rate_rad_yr: float
    theory_au_yr: float
    regimes: List[RegimeFit]
    unwrap_ok: bool
    phi_f: Optional[np.ndarray]  # dropped by run_owner after judging (memory: hundreds of candidates)
    psi: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)  # by regime
    catalogue_planets: Tuple[str, ...] = ()  # its planets whose lambda is the catalogue line
    unresolved: int = 0  # windows of the frame its sine is not resolved in (`resolved_windows`)


def psi_combination(sigma_spec: ResonanceSpec, candidate: ResonanceSpec) -> Tuple[int, int]:
    """(u, v) of Psi = u sigma - v phi_o, the combination without the body's lambda: d = gcd(|k|, |k_o|), u = k_o / d,
    v = k / d (k, k_o the body's coefficients of sigma and phi_o), signs flipped together so that u > 0. Identity:
    sigma = (v / u) phi_o + Psi / u. (0, 0) when either angle holds no lambda of the body."""
    k, k_o = int(sigma_spec.body_coefficient), int(candidate.body_coefficient)
    if k == 0 or k_o == 0:
        return 0, 0
    d = math.gcd(abs(k), abs(k_o))
    u, v = k_o // d, k // d
    return (u, v) if u > 0 else (-u, -v)


def phase_leverage(psi_window_means: np.ndarray, u: int, v: int, psi_sigma_means: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """(L_owner, L_sigma) = (|<exp(i Psi_w / v)>|, |<exp(i Psi'_w / u)>|) over the window means Psi_w (R5 §1.2);
    Psi' = `psi_sigma_means`, default Psi (`slip_corrected` for a libration regime with slips).

    If sigma's own term drove a with a fixed phase, the owner's phase in window w would be (u/v) phi0_sigma - Psi_w / v:
    its coherence cannot exceed L_owner. If the owner drives, sigma's coherence is L_sigma. A uniform drift of Psi by Delta
    across the windows gives |sinc(Delta / 2)| (Delta = pi: 2 / pi). nan without windows or without (u, v)."""
    psi = np.asarray(psi_window_means, float)
    other = psi if psi_sigma_means is None else np.asarray(psi_sigma_means, float)
    keep = np.isfinite(psi) & np.isfinite(other)
    if not np.any(keep) or u == 0 or v == 0:
        return NAN, NAN
    return float(abs(np.mean(np.exp(1j * psi[keep] / v)))), float(abs(np.mean(np.exp(1j * other[keep] / u))))


def slip_corrected(psi_means: np.ndarray, phi_means: np.ndarray, index: np.ndarray, v: int) -> np.ndarray:
    """Psi_w + 2 pi v N_w on the windows `index` (time order) of a libration regime: N_w counts the owner's slips, the
    2 pi steps of its window mean phi_w across each gap between non-adjacent windows (round((phi_w - phi_prev) / 2 pi)).

    With the owner librating, sigma's phase in window w is (v/u) phi0_B + (Psi_w + 2 pi v N_w) / u: a slip of phi_o by
    2 pi moves sigma by 2 pi v / u and leaves Psi unchanged (R5 §1.1), so sigma's coherence under the owner is read from
    the corrected Psi. A slip happens in a window the owner turns in, so it falls in a gap of the libration windows."""
    psi, phi = np.asarray(psi_means, float)[index], np.asarray(phi_means, float)[index]
    steps = np.zeros(len(index))
    for j in range(1, len(index)):
        if index[j] != index[j - 1] + 1 and np.isfinite(phi[j]) and np.isfinite(phi[j - 1]):
            steps[j] = np.round((phi[j] - phi[j - 1]) / TWO_PI)
    return psi + TWO_PI * v * np.cumsum(steps)


def window_means(t_fit: np.ndarray, values: np.ndarray, windows: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Mean of `values` (on the increasing grid `t_fit`) in each window [start, end]; nan for an empty window."""
    t_fit = np.asarray(t_fit, float)
    if not len(windows):
        return np.zeros(0)
    first = np.searchsorted(t_fit, [a for a, _ in windows], side='left')
    stop = np.searchsorted(t_fit, [b for _, b in windows], side='right')
    total = np.concatenate([[0.0], np.cumsum(np.asarray(values, float))])
    count = stop - first
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(count > 0, (total[stop] - total[first]) / np.maximum(count, 1), NAN)


def rate_admissible(owner_period_yr: float, period_yr: float, factor: float) -> bool:
    """A forced wave drives a at the owner's own period: it can make the a-wave of period P only when that period lies
    in the band [P / f, f P] (R6 #1: a candidate turning at 1.47 yr cannot force a 181-yr oscillation). An unmeasured
    period is not a reason to reject."""
    if not (np.isfinite(owner_period_yr) and np.isfinite(period_yr) and np.isfinite(factor) and factor > 0):
        return True
    return bool(period_yr / factor <= owner_period_yr <= factor * period_yr)


def regimes_from_windows(
    frame: OwnerFrame,
    r2: np.ndarray,
    force: np.ndarray,
    phi0: np.ndarray,
    turns: np.ndarray,
    psi_means: Optional[np.ndarray] = None,
    psi_uv: Tuple[int, int] = (0, 0),
    phi_means: Optional[np.ndarray] = None,
) -> List[RegimeFit]:
    """Split the windows by the owner's turns: |turns| < 1 libration, else a forced wave of either direction; libration
    first. Per window of `frame.windows`: the owner's R^2, force R, phase phi0 (nan where the window has no fit) and
    turns (`frame.spans` of the filtered angle). sigma's numbers come from the same windows. A regime also gets the
    owner's mean rate (turns over the window's span), whether a forced wave at that rate is admissible
    (`rate_admissible`; a libration always is) and the phase lever (`phase_leverage` of `psi_means`, the per-window
    means of Psi = u sigma - v phi_o with `psi_uv` = (u, v); in a libration regime sigma's lever is corrected for the
    owner's slips with the window means `phi_means` of phi_o, `slip_corrected`)."""
    fitted = np.isfinite(force) & np.isfinite(turns)
    duration = frame.t_yr[frame.span_last] - frame.t_yr[frame.span_first] if len(frame.windows) else np.zeros(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        rates = np.where(duration > 0, np.asarray(turns, float) / np.where(duration > 0, duration, 1.0), NAN)
    libration = fitted & (np.abs(turns) < TWO_PI)
    groups = (
        (REGIME_LIBRATION, libration),
        (REGIME_FORCED_POSITIVE, fitted & ~libration & (turns > 0)),
        (REGIME_FORCED_NEGATIVE, fitted & ~libration & (turns <= 0)),
    )
    fits = []
    for regime, member in groups:
        index = np.flatnonzero(member)
        if not len(index):
            continue
        group = [frame.windows[i] for i in index]
        rate = float(np.nanmean(rates[index])) if np.any(np.isfinite(rates[index])) else NAN
        period = TWO_PI / abs(rate) if np.isfinite(rate) and rate != 0 else NAN
        levers = (NAN, NAN)
        if psi_means is not None:
            corrected = None
            if regime == REGIME_LIBRATION and phi_means is not None:
                corrected = slip_corrected(psi_means, phi_means, index, psi_uv[1])
            levers = phase_leverage(psi_means[index], *psi_uv, corrected)
        fits.append(
            RegimeFit(
                regime=regime,
                starts=tuple(start for start, _ in group),
                r2=finite_median(r2[index]),
                r2_windows=tuple(float(x) for x in r2[index]),
                coherence=filtering.circular_resultant(phi0[index]),
                force_au_yr=finite_median(force[index]),
                sigma_r2=finite_median(frame.sigma_r2_windows[index]),
                sigma_coherence=filtering.circular_resultant(frame.sigma_phi0_windows[index]),
                sigma_force_au_yr=finite_median(frame.sigma_force_windows[index]),
                **_slow_numbers(frame, [start for start, _ in group], key=member.tobytes()),
                owner_rate_rad_yr=rate,
                rate_admissible=regime == REGIME_LIBRATION or rate_admissible(period, frame.period_yr, frame.band_factor),
                lever_owner=levers[0],
                lever_sigma=levers[1],
            )
        )
    return fits


def ripple_turns(regime: RegimeFit) -> int:
    """Slow turns of sigma that the owner cannot have made: those of a forced-wave regime (0 for libration)."""
    return 0 if regime.regime == REGIME_LIBRATION else int(regime.sigma_slow_turns)


def _slow_numbers(frame: OwnerFrame, starts: Sequence[float], key=None) -> dict:
    """`slow_oscillation` on the windows of one regime (cached by `key`, default the rounded starts: it depends on sigma
    and the windows only)."""
    key = tuple(_start_key(start) for start in starts) if key is None else key
    if key not in frame.slow_cache:
        t_fit, slow_fit, band_fit, mask_fit = frame.on_fit_grid
        region = _windows_mask(t_fit, starts, frame.window_yr) & mask_fit
        frame.slow_cache[key] = slow_oscillation(t_fit, slow_fit, band_fit, region, frame.ripple_factor)
    turns, ratio = frame.slow_cache[key]
    return {'sigma_slow_turns': turns, 'sigma_slow_to_band': ratio}


def _windows_mask(t: np.ndarray, starts: Sequence[float], length: float) -> np.ndarray:
    """Samples of an increasing `t` inside any window [start, start + length]: O(len(t) + windows)."""
    t = np.asarray(t, float)
    begins = np.asarray(starts, dtype=float)
    first = np.searchsorted(t, begins, side='left')
    stop = np.searchsorted(t, begins + length, side='right')
    marks = np.zeros(len(t) + 1, dtype=np.int64)
    np.add.at(marks, first, 1)
    np.add.at(marks, stop, -1)
    return np.cumsum(marks[:-1]) > 0


def psi_numbers(
    phi_clean: np.ndarray, frame: OwnerFrame, mask: np.ndarray, sigma_k: int, owner_k: int, varpi: np.ndarray, settings: AnalysisSettings
) -> Tuple[float, float, float, float]:
    """psi = sigma - (k/k_o) phi_o on `mask`: (share = std of psi / std of sigma in the band [P/f, fP], psi rate [rad/yr],
    corr of the slow sigma and psi (low-passed at f P), corr of psi and varpi)."""
    t = frame.t_yr
    if np.count_nonzero(mask) < 3 or owner_k == 0:
        return NAN, NAN, NAN, NAN
    psi = frame.sigma_clean - (sigma_k / owner_k) * phi_clean
    factor, order = settings.mmr_owner_band_factor, settings.filter_order
    psi_band = band_pass(t, psi, frame.period_yr, factor, order)
    sigma_std = float(np.std(frame.sigma_band[mask]))
    share = float(np.std(psi_band[mask]) / sigma_std) if sigma_std > 0 else NAN
    slow_sigma = frame.sigma_slow  # lowpass(sigma_clean, f P), the same series
    slow_psi = filtering.lowpass(t, psi, factor * frame.period_yr, order, grid_rtol=settings.filter_grid_rtol)
    return share, slope(t[mask], psi[mask]), correlation(slow_sigma[mask], slow_psi[mask]), correlation(psi[mask], varpi[mask])


def _psi_needed(regime: RegimeFit, settings: AnalysisSettings, tolerance: float = 0.0) -> bool:
    """psi is read by `accept` only after R^2 and the phase passed: computed for those regimes (cost), for a
    coherence (single, or of the multiplet law with its R^2 passing) short of the threshold by at most `tolerance`
    (catalogue longitudes: `catalogue_ambiguous` reads it)."""
    floor = settings.mmr_owner_phi0_coh_min - tolerance
    multiplet = multiplet_fixed(replace(regime, multiplet_coherence=_finite_or(regime.multiplet_coherence, -np.inf) + tolerance), settings)
    return _finite_or(regime.r2, -np.inf) >= settings.mmr_owner_r2_min and (_finite_or(regime.coherence, -np.inf) >= floor or multiplet)


def add_psi(fit: OwnerFit, regime: RegimeFit, phi_clean: np.ndarray, frame: OwnerFrame, series: MMRSeries, settings: AnalysisSettings):
    """psi numbers of one regime of the candidate (into `fit.psi`)."""
    region = _windows_mask(frame.t_yr, regime.starts, frame.window_yr) & frame.mask
    varpi = frame.varpi_unwrapped if frame.varpi_unwrapped is not None else np.unwrap(np.asarray(series.varpi, float))
    k, k_o = series.resonance.body_coefficient, fit.spec.body_coefficient
    fit.psi[regime.regime] = psi_numbers(phi_clean, frame, region, k, k_o, varpi, settings)


def regime_pieces(starts: Sequence[float], length: float) -> List[Tuple[float, float]]:
    """The union of the windows [start, start + length] as disjoint pieces in time order."""
    pieces: List[Tuple[float, float]] = []
    for start in sorted(float(s) for s in starts):
        if pieces and start <= pieces[-1][1]:
            pieces[-1] = (pieces[-1][0], max(pieces[-1][1], start + length))
        else:
            pieces.append((start, start + length))
    return pieces


def owner_multiplet(
    spec: ResonanceSpec, phi_fit: np.ndarray, frame: OwnerFrame, regime: RegimeFit, settings: AnalysisSettings
) -> RegimeFit:
    """The multiplet law of the candidate on one regime (module docstring): `regime` with multiplet_* of the planet whose
    fit has the highest R^2. Unchanged when the candidate has one member (p = 0), the frame has no varpi (no context),
    or no planet gives mmr_force_min_windows windows of max(modulation period, force window) inside the regime.

    `phi_fit`: the candidate filtered at P_c on the fit grid; the members are phi + j (varpi_f - g_P t)."""
    harmonics = multiplet_harmonics(spec.body_varpi_coefficient, 0)
    if frame.varpi_fit is None or len(harmonics) < 2:
        return regime
    t_fit = frame.t_yr[frame.fit_mask]
    inside = _windows_mask(t_fit, regime.starts, frame.window_yr)
    if np.count_nonzero(inside) < 2:
        return regime
    pieces = regime_pieces(regime.starts, frame.window_yr)
    fits = [_planet_multiplet(harmonics, phi_fit, frame, planet, inside, pieces, settings) for planet in spec.planets]
    fits = [f for f in fits if f is not None and np.isfinite(f[0].r2)]
    best = max(fits, key=lambda f: f[0].r2, default=None)
    if best is None:
        return regime
    fit, planet, count, windows, offset = best
    return replace(
        regime,
        multiplet_r2=float(fit.r2),
        multiplet_coherence=float(fit.coefficient_coherence),
        multiplet_planet=planet,
        multiplet_windows=count,
        sigma_multiplet_r2=sigma_long_r2(frame, planet, windows, offset, settings),
    )


def _planet_multiplet(harmonics, phi_fit, frame: OwnerFrame, planet: str, inside: np.ndarray, pieces, settings):
    """(fit, planet, valid windows, windows, offset) of the members over one planet's varpi, None when its rate is
    unknown or fewer than mmr_force_min_windows windows of max(modulation period, force window) have every member
    fitted (the members are not separable over less than a modulation period)."""
    t_fit = frame.t_yr[frame.fit_mask]
    rate = frame.planet_varpi_rates.get(planet, NAN)
    if not np.isfinite(rate):
        return None
    offset = frame.varpi_fit - rate * t_fit
    nu = abs(slope(t_fit[inside], offset[inside]))
    if not (np.isfinite(nu) and nu > 0):
        return None
    length = max(TWO_PI / nu, frame.window_yr)
    windows: List[Tuple[float, float]] = []
    for start, end in pieces:
        windows += sliding_windows(t_fit[samples_between(t_fit, start, end)], length, settings.mmr_force_window_step)
    # the member columns are formed one at a time as `resolved_windows` reads them (none is kept: N floats each)
    columns = itertools.chain((phi_fit + j * offset for j in harmonics), _sigma_columns(frame, planet, offset))
    resolved = resolved_windows(t_fit, columns, windows)
    windows = [w for w, ok in zip(windows, resolved) if ok]  # owner members and sigma's comparison laws resolved
    if len(windows) < settings.mmr_force_min_windows:
        return None
    fit = multiplet_fit(t_fit, phi_fit, frame.adot_band[frame.fit_mask], offset, harmonics, windows, settings.mmr_force_power_share)
    valid = int(np.count_nonzero(np.all(np.isfinite(fit.coefficients), axis=1)))
    return (fit, planet, valid, windows, offset) if valid >= settings.mmr_force_min_windows else None


def _sigma_harmonics(frame: OwnerFrame, planet: str) -> Tuple[int, ...]:
    """sigma's own members over `planet` (`multiplet_harmonics`), (0,) when the planet is not sigma's."""
    spec = frame.sigma_spec
    if spec is None or spec.retrograde or planet not in spec.planets:
        return (0,)
    planet_p = spec.planet_varpi_terms[list(spec.planets).index(planet)]
    return multiplet_harmonics(spec.body_varpi_coefficient, planet_p)


def _sigma_columns(frame: OwnerFrame, planet: str, offset: np.ndarray) -> Iterator[np.ndarray]:
    """sigma's members over `planet` on the fit grid (the columns of `sigma_long_r2`), formed one at a time."""
    sigma_fit = frame.sigma_f[frame.fit_mask]
    return (sigma_fit + j * offset for j in _sigma_harmonics(frame, planet))


def sigma_long_r2(frame: OwnerFrame, planet: str, windows: Sequence[Tuple[float, float]], offset: np.ndarray, settings) -> float:
    """R^2 of sigma on the owner's long windows with constant coefficients: the better of the single law and sigma's
    own multiplet over `planet` (when it is a planet of sigma): the comparison a multiplet owner must win (Codex W15)."""
    t_fit = frame.t_yr[frame.fit_mask]
    sigma_fit, target = frame.sigma_f[frame.fit_mask], frame.adot_band[frame.fit_mask]
    share = settings.mmr_force_power_share
    values = [multiplet_fit(t_fit, sigma_fit, target, np.zeros_like(t_fit), (0,), windows, share).r2]
    harmonics = _sigma_harmonics(frame, planet)
    if len(harmonics) > 1:
        values.append(multiplet_fit(t_fit, sigma_fit, target, offset, harmonics, windows, share).r2)
    values = [v for v in values if np.isfinite(v)]
    return float(max(values)) if values else NAN


def needs_multiplet(regime: RegimeFit, settings: AnalysisSettings) -> bool:
    """mmr_owner_law = 'multiplet', the regime has its windows and its R^2 passes, and the single phase does not."""
    return (
        settings.mmr_owner_law == LAW_MULTIPLET
        and len(regime.r2_windows) >= settings.mmr_force_min_windows
        and _finite_or(regime.r2, -np.inf) >= settings.mmr_owner_r2_min
        and not _finite_or(regime.coherence, -np.inf) >= settings.mmr_owner_phi0_coh_min
    )


def add_multiplets(fit: OwnerFit, phi_fit: np.ndarray, frame: OwnerFrame, settings: AnalysisSettings) -> None:
    """The multiplet law on every regime of the candidate that needs it (`needs_multiplet`), into `fit.regimes`."""
    fit.regimes = [owner_multiplet(fit.spec, phi_fit, frame, r, settings) if needs_multiplet(r, settings) else r for r in fit.regimes]


def multiplet_fixed(numbers, settings: AnalysisSettings) -> bool:
    """The multiplet law fixes the phase: R^2 >= mmr_owner_r2_min and coefficient coherence >= mmr_owner_phi0_coh_min
    (`numbers`: a RegimeFit or an OwnerCandidate), with mmr_owner_law = 'multiplet'."""
    r2, coherence = numbers.multiplet_r2, numbers.multiplet_coherence
    return bool(
        settings.mmr_owner_law == LAW_MULTIPLET
        and np.isfinite(r2)
        and r2 >= settings.mmr_owner_r2_min
        and np.isfinite(coherence)
        and coherence >= settings.mmr_owner_phi0_coh_min
    )


def phase_law(candidate: OwnerCandidate, settings: AnalysisSettings) -> str:
    """The law that fixes the candidate's phase: LAW_SINGLE (phi0 coherence >= mmr_owner_phi0_coh_min), else
    LAW_MULTIPLET (`multiplet_fixed`), else ''."""
    coherence = candidate.phi0_coherence
    if np.isfinite(coherence) and coherence >= settings.mmr_owner_phi0_coh_min:
        return LAW_SINGLE
    return LAW_MULTIPLET if multiplet_fixed(candidate, settings) else ''


def resolved_windows(t_fit: np.ndarray, angles: Iterable[np.ndarray], windows: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Per window: every angle of `angles` (the columns of a fit: an angle, or each member phi + j theta, on the fit
    grid) moves by less than pi between consecutive samples inside it (Nyquist). The low-pass at P_c keeps an angle's
    linear carrier, so sin(phi) of a fast circulating angle aliases into the band on the thinned grid (2001 PG14
    33S-2: 4E-7+0+3 turns 28 times in 355 yr; R^2 0.93 at the 50-yr fit step, 6e-5 on the data's 5-yr step; a member
    j = 2 of a multiplet can alias while its base angle does not: Codex W15). A candidate's unresolved short window and
    an unresolved long multiplet window are not fitted; sigma's unresolved short window has R^2 0 and no phase (its sine
    has a period below 2 grid steps = P_c, outside the band). `run_owner` measures a candidate with an unresolved window
    on the full grid."""
    t_fit = np.asarray(t_fit, float)
    if len(t_fit) < 2 or not len(windows):
        return np.zeros(len(windows), dtype=bool)
    first = np.searchsorted(t_fit, [a for a, _ in windows], side='left')
    stop = np.searchsorted(t_fit, [b for _, b in windows], side='right')
    last_step = np.maximum(stop - 1, first)  # steps first .. stop - 2 lie inside the window
    bad = np.zeros(len(windows), dtype=np.int64)
    for angle in angles:
        step = np.abs(np.diff(np.asarray(angle, float)))
        fast = ~(np.isfinite(step) & (step < np.pi))  # step i joins samples i and i + 1
        count = np.concatenate([[0], np.cumsum(fast)])
        bad += count[last_step] - count[first]
    return ((stop - first) >= 2) & (bad == 0)


def fit_on_frame(
    candidate: ResonanceSpec,
    phi_fit: np.ndarray,
    turns: np.ndarray,
    rate_rad_yr: float,
    unwrapped: bool,
    frame: OwnerFrame,
    catalogue_planets: Tuple[str, ...] = (),
    settings: Optional[AnalysisSettings] = None,
) -> OwnerFit:
    """The force law of the filtered angle on the frame's windows (`phi_fit` on the fit grid, `turns` per window), split
    by regime, without psi. `catalogue_planets`: the candidate's planets on catalogue lines (`coherence_tolerance`).
    With `settings`, the multiplet law on the regimes that need it (`add_multiplets`)."""
    t_fit = frame.t_yr[frame.fit_mask]
    a_sin, b_cos, _, r2 = sine_fits_in_windows(t_fit, phi_fit, frame.adot_band[frame.fit_mask], frame.windows)
    # a window the grid does not resolve is not fitted here; `run_owner` measures such a candidate on the full grid
    resolved = resolved_windows(t_fit, [phi_fit], frame.windows)
    force = np.where(resolved, np.hypot(a_sin, b_cos), NAN)
    phi0 = np.where(resolved, np.arctan2(-b_cos, a_sin), NAN)
    r2 = np.where(resolved, r2, NAN)
    psi_uv = psi_combination(frame.sigma_spec, candidate) if frame.sigma_spec is not None else (0, 0)
    psi_means = phi_means = None
    if psi_uv[0]:
        psi_fit = psi_uv[0] * frame.sigma_f[frame.fit_mask] - psi_uv[1] * np.asarray(phi_fit, float)
        psi_means = window_means(t_fit, psi_fit, frame.windows)
        phi_means = window_means(t_fit, phi_fit, frame.windows)
    fit = OwnerFit(
        spec=candidate,
        unresolved=int(np.count_nonzero(~resolved & np.isfinite(a_sin))),
        rate_rad_yr=float(rate_rad_yr),
        theory_au_yr=theory_force(candidate, frame.body_n, frame.a_mean_au),
        regimes=regimes_from_windows(frame, r2, force, phi0, turns, psi_means, psi_uv, phi_means),
        unwrap_ok=bool(unwrapped),
        phi_f=None,
        catalogue_planets=tuple(catalogue_planets),
    )
    if settings is not None:
        add_multiplets(fit, phi_fit, frame, settings)
    return fit


def measure_candidate(
    candidate: ResonanceSpec,
    phi: np.ndarray,
    frame: OwnerFrame,
    series: MMRSeries,
    cutoff_yr: float,
    settings: AnalysisSettings,
    ripple: Optional[np.ndarray] = None,
    residual: Optional[np.ndarray] = None,
    *,
    psi_all: bool = False,
) -> OwnerFit:
    """The force law of phi_o (unwrapped, minus `ripple`, filtered at P_c) by regime, sigma on the same windows, and psi
    for the regimes that pass R^2 and the coherence (every regime with `psi_all`). The angle is given at full
    resolution here (`fit_owner`); `run_owner` measures its candidates through `AngleBasis` instead."""
    t, mask = frame.t_yr, frame.mask
    phi_clean = np.asarray(phi, float) - (0.0 if ripple is None else ripple)
    phi_f = filtering.lowpass_with(t, phi_clean, cutoff_yr, settings)
    ok = True if residual is None else unwrap_ok(residual, settings)
    rate = slope(t[mask], phi_clean[mask])
    result = fit_on_frame(candidate, phi_f[frame.fit_mask], frame.spans(phi_f), rate, ok, frame, settings=settings)
    result.phi_f = phi_f
    for regime in result.regimes:
        if psi_all or _psi_needed(regime, settings):
            add_psi(result, regime, phi_clean, frame, series, settings)
    return result


# --------------------------------------------------------------------------------------------- the candidates as combinations


BODY_LAMBDA, BODY_VARPI = 'lambda', 'varpi'


@dataclass
class BasisSeries:
    """One unwrapped series of the basis: the series minus its least-squares line (`detrended`, full grid), the line
    (`intercept` + `rate` (t - t0)), the low-pass at P_c of `detrended` on the fit grid and its turns per window
    (the line is added back once per candidate, from the combined scalars: less cancellation), the slope of the series
    on the window and the largest per-step model error of its unwrap [deg] (`residual` kept for the exact check)."""

    detrended: np.ndarray
    intercept: float
    rate: float
    filtered_fit: np.ndarray
    spans: np.ndarray
    window_slope: float
    residual: np.ndarray
    residual_max_deg: float


class AngleBasis:
    """The candidate angles as integer combinations of the unwrapped series they are made of (module docstring, Cost).

    Components: the body's lambda (model rate n(a) + the measured offset), the body's varpi (record rate), each planet's
    lambda minus its slow deviation (its ripple, as for sigma) or its catalogue line (`catalogue_line`: no ripple, no
    unwrap error) and, when a candidate needs it, a planet's varpi.
    Memory: two full-grid arrays per component (the detrended series and its unwrap error), never one per candidate."""

    def __init__(
        self,
        series: MMRSeries,
        frame: OwnerFrame,
        planet_n: Mapping[str, float],
        deviations: Mapping[str, np.ndarray],
        cutoff_yr: float,
        varpi_rate: float,
        settings: AnalysisSettings,
    ):
        self.series, self.frame, self.planet_n, self.deviations = series, frame, planet_n, deviations
        self.cutoff_yr, self.varpi_rate, self.settings = cutoff_yr, varpi_rate, settings
        self.t = np.asarray(frame.t_yr, float)
        self.parts: Dict[str, BasisSeries] = {}

    def _build(self, name: str) -> BasisSeries:
        t, series = self.t, self.series
        if name == BODY_LAMBDA:
            a = np.asarray(series.a, float)
            rate = mean_motion(0.5 * (a[1:] + a[:-1])) + self.frame.body_n_offset
            unwrapped, residual = predictive_unwrap(t, np.asarray(series.mean_longitude, float), rate)
        elif name == BODY_VARPI:
            unwrapped, residual = predictive_unwrap(t, np.asarray(series.varpi, float), self.varpi_rate)
        else:
            kind, planet = name.split(':', 1)
            saved = series.planets.get(planet)
            if kind == 'lambda' and not _saved_longitude(series, planet):
                unwrapped, residual = catalogue_line(series, planet), np.zeros(max(len(t) - 1, 0))
            elif kind == 'lambda':
                unwrapped, residual = predictive_unwrap(t, np.asarray(saved.mean_longitude, float), self.planet_n[planet])
                unwrapped = unwrapped - self.deviations[planet]
            else:
                unwrapped, residual = predictive_unwrap(t, np.asarray(saved.varpi, float), 0.0)
        elapsed = t - t[0]
        rate, intercept = np.polyfit(elapsed, unwrapped, 1)
        detrended = unwrapped - (intercept + rate * elapsed)
        del unwrapped
        filtered = filtering.lowpass_with(t, detrended, self.cutoff_yr, self.settings)
        frame = self.frame
        fit = frame.fit_mask
        part = BasisSeries(
            detrended=detrended,
            intercept=float(intercept),
            rate=float(rate),
            filtered_fit=filtered[fit],
            spans=frame.spans(filtered),
            window_slope=slope(t[frame.mask], detrended[frame.mask]) + float(rate),
            residual=residual,
            residual_max_deg=float(np.degrees(np.max(np.abs(residual)))) if len(residual) else 0.0,
        )
        return part

    def part(self, name: str) -> BasisSeries:
        if name not in self.parts:
            self.parts[name] = self._build(name)
        return self.parts[name]

    @staticmethod
    def coefficients(spec: ResonanceSpec) -> Dict[str, int]:
        """{component: integer coefficient} of the angle (zero coefficients left out)."""
        result = {BODY_LAMBDA: spec.body_coefficient, BODY_VARPI: spec.body_varpi_coefficient}
        for planet, k, p in zip(spec.planets, spec.planet_coefficients, spec.planet_varpi_terms):
            result[f'lambda:{planet}'] = result.get(f'lambda:{planet}', 0) + k
            result[f'varpi:{planet}'] = result.get(f'varpi:{planet}', 0) + p
        return {name: int(c) for name, c in result.items() if c != 0}

    def _combine(self, coefficients: Mapping[str, int], attribute: str):
        return sum(c * getattr(self.part(name), attribute) for name, c in coefficients.items())

    def _line(self, coefficients: Mapping[str, int]) -> Tuple[float, float]:
        """(intercept, rate) of the candidate's line, combined as scalars."""
        return float(self._combine(coefficients, 'intercept')), float(self._combine(coefficients, 'rate'))

    def filtered_fit(self, coefficients: Mapping[str, int]) -> np.ndarray:
        """The candidate filtered at P_c on the fit grid [rad] (up to a multiple of 2 pi)."""
        intercept, rate = self._line(coefficients)
        t_fit = self.t[self.frame.fit_mask]
        return self._combine(coefficients, 'filtered_fit') + (intercept + rate * (t_fit - self.t[0]))

    def turns(self, coefficients: Mapping[str, int]) -> np.ndarray:
        """Turns of the filtered candidate in each window of the frame [rad]."""
        _, rate = self._line(coefficients)
        frame = self.frame
        return self._combine(coefficients, 'spans') + rate * (self.t[frame.span_last] - self.t[frame.span_first])

    def rate(self, coefficients: Mapping[str, int]) -> float:
        """Slope of the candidate on the window [rad/yr]."""
        return float(self._combine(coefficients, 'window_slope'))

    def clean(self, coefficients: Mapping[str, int]) -> np.ndarray:
        """The candidate unwrapped, ripple removed, on the full grid [rad] (up to a multiple of 2 pi)."""
        intercept, rate = self._line(coefficients)
        return self._combine(coefficients, 'detrended') + (intercept + rate * (self.t - self.t[0]))

    def unwrap_ok(self, coefficients: Mapping[str, int]) -> bool:
        """`unwrap_ok` of the candidate's own per-step error wrap(sum c_i r_i): decided by the bound sum |c_i| max|r_i|
        when that is within both limits, else computed."""
        settings = self.settings
        bound = sum(abs(c) * self.part(name).residual_max_deg for name, c in coefficients.items())
        if bound <= min(settings.mmr_unwrap_max_resid_p99_deg, settings.mmr_unwrap_max_resid_deg):
            return True
        return unwrap_ok(wrap_pi(self._combine(coefficients, 'residual')), settings)


def _candidate(fit: OwnerFit, regime: Optional[RegimeFit]) -> OwnerCandidate:
    """The OwnerCandidate of one regime of the candidate (blank numbers without a regime)."""
    force = regime.force_au_yr if regime is not None else NAN
    theory = fit.theory_au_yr
    common = dict(
        spec=fit.spec,
        rate_rad_yr=fit.rate_rad_yr,
        force_au_yr=force,
        force_theory_au_yr=theory,
        strength_ratio=float(force / theory) if theory > 0 and np.isfinite(force) else NAN,
        rank=0,
        accepted=False,
        reason='',
    )
    if regime is None:
        return OwnerCandidate(r2_median=NAN, r2_windows=(), phi0_coherence=NAN, psi_share=NAN, **common)
    return OwnerCandidate(
        r2_median=regime.r2,
        r2_windows=regime.r2_windows,
        phi0_coherence=regime.coherence,
        psi_share=fit.psi.get(regime.regime, (NAN,))[0],
        regime=regime.regime,
        n_windows=len(regime.r2_windows),
        sigma_r2=regime.sigma_r2,
        sigma_phi0_coherence=regime.sigma_coherence,
        sigma_slow_turns=regime.sigma_slow_turns,
        sigma_slow_to_band=regime.sigma_slow_to_band,
        multiplet_r2=regime.multiplet_r2,
        multiplet_coherence=regime.multiplet_coherence,
        multiplet_planet=regime.multiplet_planet,
        sigma_multiplet_r2=regime.sigma_multiplet_r2,
        owner_period_yr=TWO_PI / abs(regime.owner_rate_rad_yr) if _finite_or(regime.owner_rate_rad_yr, 0.0) else NAN,
        rate_admissible=regime.rate_admissible,
        lever_owner=regime.lever_owner,
        lever_sigma=regime.lever_sigma,
        **common,
    )


def fit_owner(
    candidate: ResonanceSpec,
    phi: np.ndarray,
    prep: Preprocessed,
    context: Context,
    period_cycle_yr: float,
    settings: AnalysisSettings,
    *,
    series: Optional[MMRSeries] = None,
    frame: Optional[OwnerFrame] = None,
    ripple: Optional[np.ndarray] = None,
) -> OwnerCandidate:
    """R^2 of a_dot_f (band [P/f, f P], P = `period_cycle_yr`) on sin/cos phi_o per window, phi0 coherence, psi share of
    sigma = (k/k_o) phi_o + psi, force K_o and theory K_th,o = omega0_th,o^2 / |c_o|, for the owner's regime with the
    higher R^2 (`run_owner` judges every regime). `phi` is the unwrapped angle (`owner_angle`), `ripple` the planets'
    slow term (default 0). Needs `series` (the body's lambda and varpi); `frame` is built from it when not given."""
    if series is None:
        raise ValueError('fit_owner needs the series (the body lambda and varpi)')
    frame = build_frame(prep, series, period_cycle_yr, settings, context=context) if frame is None else frame
    fit = measure_candidate(candidate, phi, frame, series, prep.cutoff.cutoff_yr, settings, ripple, psi_all=True)
    best = max(fit.regimes, key=lambda r: _finite_or(r.r2, -np.inf), default=None)
    return _candidate(fit, best)


# --------------------------------------------------------------------------------------------- acceptance


def better_than_sigma(
    candidate: OwnerCandidate, sigma_r2: float, sigma_coherence: float, sigma_strength_ratio: float, settings: AnalysisSettings
) -> str:
    """The route by which the owner explains a_dot better than sigma on the same windows ('' = none):
    'r2' (R^2 gain >= mmr_owner_r2_gain) or 'phase' with the Psi lever (R5 §1.3, W20): the owner's coherence C_B >=
    L_owner + mmr_owner_phi0_coh_gain (more than it could have if sigma's own term drove a) and sigma's C_sigma <
    L_sigma + gain (no more than the identity gives it if the owner drives); PHASE_UNRESOLVED when L_owner > 1 - gain
    (Psi nearly constant: the phases cannot separate A from B). Without a measured lever (a candidate built by hand)
    the old comparison C_B >= C_sigma + gain. An unmeasured R^2 or coherence of sigma is 0. `sigma_strength_ratio`
    (sigma's force over its mu scale) is not a route: the scale is no bound (W3-F3); the argument stays for the callers."""
    gain = settings.mmr_owner_phi0_coh_gain
    if candidate.r2_median >= _finite_or(sigma_r2, 0.0) + settings.mmr_owner_r2_gain:
        return GAIN_R2
    lever_owner, lever_sigma = candidate.lever_owner, candidate.lever_sigma
    if not (np.isfinite(lever_owner) and np.isfinite(lever_sigma)):
        return GAIN_PHASE if candidate.phi0_coherence >= _finite_or(sigma_coherence, 0.0) + gain else ''
    if lever_owner > 1 - gain:
        return PHASE_UNRESOLVED
    if candidate.phi0_coherence >= lever_owner + gain and _finite_or(sigma_coherence, 0.0) < lever_sigma + gain:
        return GAIN_PHASE
    return ''


def accept(
    candidate: OwnerCandidate,
    sigma_r2: float,
    settings: AnalysisSettings,
    *,
    sigma_coherence: float = NAN,
    sigma_strength_ratio: float = NAN,
    sigma_slow_turns: int = 0,
) -> Tuple[bool, str]:
    """Accepted when median R^2 >= mmr_owner_r2_min, the phase is fixed (`phase_law`: coherence >=
    mmr_owner_phi0_coh_min, or the multiplet law), psi share <= mmr_owner_psi_share_max, no slow turn of sigma
    (`slow_oscillation`; else a ripple) and the owner beats sigma (`better_than_sigma`: R^2 >= sigma_r2 +
    mmr_owner_r2_gain, or the phase; a phase fixed by the multiplet law only by R^2); (accepted, reason). The force
    over its mu scale (`strength_ratio`) is not tested (W3-F3).

    The reason is the first test that fails, in that order, or 'accepted:<route>'. An unmeasured number fails its test.
    Before all: a forced-wave regime whose owner turns at a period outside the a band is not an owner there
    (`rate_admissible`, reason rate_outside_band). The potential route is tried after this, on the best candidates left
    (`run_owner`)."""
    if not candidate.rate_admissible:
        return False, REASON_RATE
    if not (np.isfinite(candidate.r2_median) and candidate.r2_median >= settings.mmr_owner_r2_min):
        return False, REASON_R2
    law = phase_law(candidate, settings)
    if not law:
        return False, REASON_COHERENCE
    if not (np.isfinite(candidate.psi_share) and candidate.psi_share <= settings.mmr_owner_psi_share_max):
        return False, REASON_PSI
    if sigma_slow_turns > 0:
        return False, REASON_RIPPLE
    route = better_than_sigma(candidate, sigma_r2, sigma_coherence, sigma_strength_ratio, settings)
    if law == LAW_MULTIPLET:  # R^2 of the long-window laws, owner against sigma (`sigma_long_r2`); no phase route
        gain = candidate.multiplet_r2 - _finite_or(candidate.sigma_multiplet_r2, 0.0)
        route = GAIN_R2 if gain >= settings.mmr_owner_r2_gain else ''
    if route == PHASE_UNRESOLVED:
        return False, REASON_PHASE_UNRESOLVED
    if not route:
        return False, REASON_NOT_BETTER
    return True, f'{REASON_ACCEPTED}:{route}'


def coherence_tolerance(fit: OwnerFit, regime: Optional[RegimeFit]) -> float:
    """Delta C = SINC_HALF_SLOPE_MAX x D [coherence] of a candidate on catalogue longitudes in one regime, T_eff = the span
    of its window centres (`catalogue_phase_error`); 0 when no planet of the candidate is on a catalogue line."""
    if not fit.catalogue_planets or regime is None or len(regime.starts) < 2:
        return 0.0
    span = float(max(regime.starts) - min(regime.starts))
    return SINC_HALF_SLOPE_MAX * catalogue_phase_error(fit.spec, fit.catalogue_planets, span)


def catalogue_ambiguous(candidate: OwnerCandidate, regime: RegimeFit, tolerance: float, settings: AnalysisSettings) -> bool:
    """True when the catalogue lines leave the decision open: R^2, psi and the ripple test pass, and the coherence lies
    within `tolerance` (`coherence_tolerance`) of mmr_owner_phi0_coh_min, or, when the R^2 route does not carry the
    candidate, of sigma's coherence + mmr_owner_phi0_coh_gain (`better_than_sigma`); for a phase fixed by the multiplet
    law, its coefficient coherence within `tolerance` of mmr_owner_phi0_coh_min. R^2 and psi get no tolerance: the
    rate error moves phi0 from window to window and hardly changes the fit within one short window."""
    if tolerance <= 0 or regime is None:
        return False
    coherence = candidate.phi0_coherence
    if not (np.isfinite(candidate.r2_median) and candidate.r2_median >= settings.mmr_owner_r2_min):
        return False
    if not (np.isfinite(candidate.psi_share) and candidate.psi_share <= settings.mmr_owner_psi_share_max) or ripple_turns(regime) > 0:
        return False
    single_fails = not (np.isfinite(coherence) and coherence >= settings.mmr_owner_phi0_coh_min + tolerance)
    near_multiplet = multiplet_fixed(
        replace(candidate, multiplet_coherence=_finite_or(candidate.multiplet_coherence, -np.inf) + tolerance), settings
    )
    if single_fails and near_multiplet and abs(candidate.multiplet_coherence - settings.mmr_owner_phi0_coh_min) <= tolerance:
        return True  # the multiplet law decides within the catalogue's tolerance, on either side of its threshold
    if not np.isfinite(coherence):
        return False
    if abs(coherence - settings.mmr_owner_phi0_coh_min) <= tolerance:
        return True
    if coherence < settings.mmr_owner_phi0_coh_min or candidate.r2_median >= _finite_or(regime.sigma_r2, 0.0) + settings.mmr_owner_r2_gain:
        return False
    return abs(coherence - _finite_or(regime.sigma_coherence, 0.0) - settings.mmr_owner_phi0_coh_gain) <= tolerance


@dataclass
class Judged:
    """A candidate's verdict: the regime it rests on, the OwnerCandidate of that regime, accepted and the reason;
    `passing` every regime of the candidate that passes (its intervals, F7) and `frame` the frame it was measured on."""

    fit: OwnerFit
    regime: Optional[RegimeFit]
    candidate: OwnerCandidate
    accepted: bool
    reason: str
    passing: List[RegimeFit] = field(default_factory=list)
    frame: Optional[OwnerFrame] = None
    # when no regime passes: the verdict on every usable regime (the potential route tries each, `potential_pass`)
    alternatives: List['Judged'] = field(default_factory=list)


def judge(fit: OwnerFit, frame: OwnerFrame, settings: AnalysisSettings) -> Judged:
    """The candidate on each regime with at least mmr_force_min_windows windows: the passing regime with the most
    windows (then R^2), else a regime left open by the catalogue lines (`catalogue_ambiguous`, reason
    catalogue_ambiguous; the highest R^2 of them), else the regime with the highest R^2; without a passing regime the
    verdicts on all of them go to `alternatives`."""
    if not fit.unwrap_ok:
        return Judged(fit, None, _candidate(fit, None), False, REASON_UNWRAP)
    usable = [r for r in fit.regimes if len(r.r2_windows) >= settings.mmr_force_min_windows]
    if not usable:
        best = max(fit.regimes, key=lambda r: len(r.r2_windows), default=None)
        return Judged(fit, best, _candidate(fit, best), False, REASON_WINDOWS, frame=frame)
    verdicts = []
    for regime in usable:
        candidate = _candidate(fit, regime)
        sigma_ratio = regime.sigma_force_au_yr / frame.sigma_theory_au_yr if frame.sigma_theory_au_yr > 0 else NAN
        accepted, reason = accept(
            candidate,
            regime.sigma_r2,
            settings,
            sigma_coherence=regime.sigma_coherence,
            sigma_strength_ratio=sigma_ratio,
            sigma_slow_turns=ripple_turns(regime),
        )
        if catalogue_ambiguous(candidate, regime, coherence_tolerance(fit, regime), settings):
            accepted, reason = False, REASON_CATALOGUE_AMBIGUOUS
        candidate = replace(candidate, law=phase_law(candidate, settings))
        verdicts.append(Judged(fit, regime, candidate, accepted, reason))
    passing = [j for j in verdicts if j.accepted]
    if passing:
        best = max(passing, key=lambda j: (len(j.candidate.r2_windows), _finite_or(j.candidate.r2_median, -np.inf)))
        return replace(best, passing=[j.regime for j in passing], frame=frame)
    ambiguous = [j for j in verdicts if j.reason == REASON_CATALOGUE_AMBIGUOUS]
    shown = max(ambiguous or verdicts, key=lambda j: _finite_or(j.candidate.r2_median, -np.inf))
    return replace(shown, frame=frame, alternatives=[replace(j, frame=frame) for j in verdicts])


def owner_unresolved(judged: Judged, settings: AnalysisSettings) -> bool:
    """The best candidate is not accepted only because its phases cannot separate it from sigma (phase_unresolved) and
    the potential route cannot measure it (not applicable: three-body, catalogue line, switched off): the search did
    not decide, check_mmr_owner not_measured (flag owner_unresolved)."""
    if judged.accepted or judged.reason != REASON_PHASE_UNRESOLVED:
        return False
    fit = judged.candidate.potential
    return fit is None or fit.reason in potential_module.NOT_APPLICABLE


def within_scale(candidate: OwnerCandidate, settings: AnalysisSettings) -> bool:
    """K_o <= mmr_owner_strength_margin x K_th,o: the force is within the order of magnitude of the mu scale. Orders
    the candidates and qualifies `owner_probable` (neither changes a status); never accepts or rejects an owner, the
    scale being no bound (W3-F3)."""
    return bool(np.isfinite(candidate.strength_ratio) and candidate.strength_ratio <= settings.mmr_owner_strength_margin)


def probable(judged: Judged, settings: AnalysisSettings) -> bool:
    """Fits a (R^2 >= mmr_owner_r2_min) with a force within the mu scale (`within_scale`), a small psi share and
    sigma's motion in its band, but the phase or the comparison with sigma fails: a likely owner that is not proven
    (`owner_probable`; the status does not change). The owner of a ripple (66652 4N-7: 3N-5 moves sigma by 0.7 of
    166 deg) is not probable."""
    c = judged.candidate
    turns = ripple_turns(judged.regime) if judged.regime is not None else 0
    return (
        not judged.accepted
        and judged.reason in (REASON_COHERENCE, REASON_NOT_BETTER, REASON_PHASE_UNRESOLVED, REASON_POTENTIAL_SCALE)
        and np.isfinite(c.psi_share)
        and c.psi_share <= settings.mmr_owner_psi_share_max
        and turns == 0
        and within_scale(c, settings)
    )


# --------------------------------------------------------------------------------------------- the search


def _empty(check: CheckValue, flags: List[str], sigma_r2: float = NAN, period_yr: float = NAN, n_candidates: int = 0) -> OwnerResult:
    return OwnerResult(
        n_candidates=n_candidates,
        candidates=[],
        accepted=None,
        check=check,
        sigma_r2=sigma_r2,
        psi_rate_rad_yr=NAN,
        psi_slow_corr=NAN,
        psi_varpi_corr=NAN,
        freq_r2_owner=NAN,
        freq_r2_sigma=NAN,
        freq_period_yr=period_yr,
        flags=flags,
    )


def available_planets(series: MMRSeries, settings: AnalysisSettings) -> Tuple[str, ...]:
    """Planets with a saved mean longitude or a catalogue line (`catalogue_planets`) among mmr_owner_planets and the
    angle's planets, in solar-system order."""
    wanted = set(settings.mmr_owner_planets) | set(series.resonance.planets)
    have = {name for name in series.planets if _saved_longitude(series, name)} | set(catalogue_planets(series))
    return tuple(name for name in PLANET_NAMES_ORDER if name in wanted and name in have)


def frequency_test(frame: OwnerFrame, phi_f: np.ndarray, settings: AnalysisSettings) -> Tuple[float, float]:
    """(R^2 of a on sin/cos phi_o, R^2 of a on sin/cos sigma), one fit each over the windows where both turn at least
    once (SPEC §4.8 own_freq_*: a waves at the owner's rate, not at sigma's); nan when there is no such window."""
    t, mask = frame.t_yr, frame.mask
    if not np.any(mask):
        return NAN, NAN
    starts = []
    start, end = t[mask][0], t[mask][-1]
    while start + frame.window_yr <= end:
        starts.append(start)
        start += settings.mmr_force_window_step * frame.window_yr
    if not starts:
        return NAN, NAN
    begins = np.asarray(starts, dtype=float)
    first = np.searchsorted(t, begins, side='left')
    last = np.searchsorted(t, begins + frame.window_yr, side='right') - 1
    enough = last > first
    sigma_turns = np.where(enough, frame.sigma_f[last] - frame.sigma_f[first], 0.0)
    owner_turns = np.where(enough, phi_f[last] - phi_f[first], 0.0)
    both = enough & (np.abs(sigma_turns) >= TWO_PI) & (np.abs(owner_turns) >= TWO_PI)
    union = _windows_mask(t, begins[both], frame.window_yr) & mask
    if np.count_nonzero(union) < 4:
        return NAN, NAN
    return sine_coefficients(phi_f[union], frame.a_band[union])[3], sine_coefficients(frame.sigma_f[union], frame.a_band[union])[3]


def _not_searched(series: MMRSeries, settings: AnalysisSettings) -> Optional[OwnerResult]:
    """The result when there is nothing to search: a retrograde or 1:1 angle (not_applicable, module doc), the search
    switched off or no saved longitudes (not_measured); None when the search runs."""
    if series.resonance.retrograde:
        return _empty(CheckValue.NOT_APPLICABLE, [FLAG_RETROGRADE])
    if coorbital_planet(series.resonance) is not None:
        return _empty(CheckValue.NOT_APPLICABLE, [FLAG_COORBITAL])
    if settings.mmr_owner_search == 'off':
        return _empty(CheckValue.NOT_MEASURED, [FLAG_SEARCH_OFF])
    if series.mean_longitude is None or not available_planets(series, settings):
        return _empty(CheckValue.NOT_MEASURED, [FLAG_NO_LONGITUDES])
    return None


def run_owner(
    series: MMRSeries,
    prep: Preprocessed,
    context: Context,
    period_cycle_yr: float,
    settings: AnalysisSettings,
    *,
    window: Optional[Tuple[float, float]] = None,
    decisive_window: Optional[Tuple[float, float]] = None,
    decisive_period_yr: float = NAN,
) -> OwnerResult:
    """The search; check pass when none is accepted, fail when one is, not_measured without planet longitudes,
    with mmr_owner_search='off' or when the planets' lambda cannot be unwrapped.

    `window` (default prep.window, the edge-trimmed record) is where the force law is fitted. The candidates are
    measured one at a time through `AngleBasis` and only their numbers are kept (W3-M2). After the R^2 and phase routes
    the potential route (`potential_pass`). `decisive_window` (default the fitted window) is sigma's decisive episode:
    the accepted owner's intervals are compared with it (`OwnerResult.cover`, F7) and the second capture is tested on
    the intervals that meet it; `decisive_period_yr` is that episode's period (lambda_ad, the bound on sigma's term)."""
    skipped = _not_searched(series, settings)
    if skipped is not None:
        return skipped
    planets = available_planets(series, settings)
    t = np.asarray(prep.t_yr, float)
    mask = filtering.window_mask(t, prep.window if window is None else window)
    period = reference_period(prep, period_cycle_yr, mask, settings)
    if not (np.isfinite(period) and period > 0):
        return _empty(CheckValue.NOT_MEASURED, [FLAG_NO_PERIOD])
    frame = build_frame(prep, series, period, settings, window, context=context)
    if len(frame.sigma_windows) < settings.mmr_force_min_windows:
        return _empty(CheckValue.NOT_MEASURED, [FLAG_TOO_FEW_WINDOWS], frame.sigma_r2, period)
    planet_n = {name: planet_mean_motion(context, name) for name in planets}
    a = np.asarray(series.a, float)[frame.mask]
    n_range = (float(mean_motion(np.max(a))) + frame.body_n_offset, float(mean_motion(np.min(a))) + frame.body_n_offset)
    specs = enumerate_owners(
        series.resonance,
        context,
        period_cycle_yr,
        planets,
        settings,
        period_yr=period,
        body_n_range=n_range,
        body_varpi_rate=frame.body_varpi_rate,
    )
    catalogued = set(catalogue_planets(series)) & set(planets)
    used = sorted({p for s in specs for p in s.planets})
    longitudes = {name: series.planets[name].mean_longitude for name in used if name not in catalogued}
    deviations = planet_slow_deviations(t, longitudes, planet_n, prep.cutoff.cutoff_yr, settings.filter_order)
    basis = AngleBasis(series, frame, planet_n, deviations, prep.cutoff.cutoff_yr, slope(t, frame.varpi_unwrapped), settings)

    judged, dense = [], None
    for spec in specs:
        coefficients = basis.coefficients(spec)
        fit = fit_on_frame(
            spec,
            basis.filtered_fit(coefficients),
            basis.turns(coefficients),
            basis.rate(coefficients),
            basis.unwrap_ok(coefficients),
            frame,
            tuple(name for name in spec.planets if name in catalogued),
            settings,
        )
        if fit.unresolved:  # the thinned grid aliases this candidate somewhere: measured on the full grid (Codex W15)
            dense = dense if dense is not None else dense_frame(frame, prep.cutoff.cutoff_yr, settings)
            full = measure_candidate(spec, basis.clean(coefficients), dense, series, prep.cutoff.cutoff_yr, settings)
            full.unwrap_ok, full.catalogue_planets, full.phi_f = fit.unwrap_ok, fit.catalogue_planets, None  # memory
            judged.append(judge(full, dense, settings))
            continue
        needs_psi = [regime for regime in fit.regimes if _psi_needed(regime, settings, coherence_tolerance(fit, regime))]
        if needs_psi:
            phi_clean = basis.clean(coefficients)
            for regime in needs_psi:
                add_psi(fit, regime, phi_clean, frame, series, settings)
            del phi_clean
        judged.append(judge(fit, frame, settings))
    potential_pass(judged, basis, series, context, settings)
    result = result_from(judged, frame, period, settings, len(specs), catalogue=bool(catalogued))
    if judged and result.candidates:
        top = min(judged, key=lambda j: _rank_key(j, settings))
        phi_clean = basis.clean(basis.coefficients(result.candidates[0].spec))
        _describe_top(result, top, phi_clean, frame, series, prep, settings)
        decisive = tuple(decisive_window) if decisive_window is not None else (float(t[frame.mask][0]), float(t[frame.mask][-1]))
        describe_owner(result, top, phi_clean, basis, series, decisive, decisive_period_yr, settings)
    return result


# --------------------------------------------------------------------------------------------- the potential route


def _phi_fit(judged: Judged, basis: AngleBasis) -> np.ndarray:
    """The candidate low-passed at P_c on the fit grid of the frame it was judged on."""
    coefficients = basis.coefficients(judged.fit.spec)
    if judged.frame is None or judged.frame is basis.frame:
        return basis.filtered_fit(coefficients)
    settings = basis.settings
    phi_f = filtering.lowpass_with(basis.t, basis.clean(coefficients), basis.cutoff_yr, settings)
    return phi_f[judged.frame.fit_mask]


def potential_of(judged: Judged, basis: AngleBasis, series: MMRSeries, context: Context, settings: AnalysisSettings) -> PotentialFit:
    """`owner_potential.owner_potential` of the judged candidate on its regime's windows."""
    spec = judged.fit.spec
    reason = potential_module.not_applicable(spec, series, context, settings)
    if reason or judged.regime is None:
        return PotentialFit(reason or potential_module.REASON_NO_WINDOWS)
    frame = judged.frame if judged.frame is not None else basis.frame
    phi_wrapped = wrapped_angle(spec, series)
    return potential_module.owner_potential(
        spec, judged.regime.starts, frame, phi_wrapped, _phi_fit(judged, basis), series, context, settings
    )


def potential_accepts(fit: PotentialFit, settings: AnalysisSettings) -> Tuple[bool, str]:
    """(accepted, reason): route 'potential' when measured, median R^2 >= mmr_owner_r2_min and every window scale lies in
    [1 / mmr_owner_potential_scale_max, mmr_owner_potential_scale_max] (a negative scale: the body sits on the owner's
    maximum, the wrong sign); reason potential_scale_out when only the scale fails, '' otherwise."""
    if not fit.measured or not _finite_or(fit.r2_median, -np.inf) >= settings.mmr_owner_r2_min:
        return False, ''
    limit = settings.mmr_owner_potential_scale_max
    scales = fit.scales
    if not (np.all(np.isfinite(scales)) and np.all(scales >= 1.0 / limit) and np.all(scales <= limit)):
        return False, REASON_POTENTIAL_SCALE
    return True, ''


def potential_confirms(fit: PotentialFit, settings: AnalysisSettings) -> bool:
    """mmr_owner_potential_confirm: every window's fitted K over the directly averaged one within [1/x, x]
    (x = mmr_owner_potential_scale_max) and |phi0 - phi0_direct| <= mmr_owner_potential_phase_max_deg."""
    limit = settings.mmr_owner_potential_scale_max
    for w in fit.windows:
        ratio = w.k_fit / w.k_theory if w.k_theory > 0 else NAN
        phase = abs(float(np.angle(np.exp(1j * (w.phi0_fit - w.phi0_theory)))))
        if not (np.isfinite(ratio) and 1.0 / limit <= ratio <= limit and phase <= math.radians(settings.mmr_owner_potential_phase_max_deg)):
            return False
    return True


def _potential_gates(judged: Judged, basis: AngleBasis, series: MMRSeries, context: Context, settings: AnalysisSettings) -> bool:
    """A candidate the potential route may still accept: not accepted for want of a phase or a gain over sigma, while its
    R^2, psi share (computed here when missing), ripple test and rate pass, and the route applies to it."""
    regime = judged.regime
    if judged.accepted or regime is None or not judged.fit.unwrap_ok:
        return False
    if judged.reason not in (REASON_COHERENCE, REASON_NOT_BETTER, REASON_PHASE_UNRESOLVED):
        return False
    if not regime.rate_admissible or ripple_turns(regime) > 0 or not _finite_or(regime.r2, -np.inf) >= settings.mmr_owner_r2_min:
        return False
    if potential_module.not_applicable(judged.fit.spec, series, context, settings):
        return False
    if regime.regime not in judged.fit.psi:
        frame = judged.frame if judged.frame is not None else basis.frame
        add_psi(judged.fit, regime, basis.clean(basis.coefficients(judged.fit.spec)), frame, series, settings)
        judged.candidate = replace(judged.candidate, psi_share=judged.fit.psi[regime.regime][0])
    return _finite_or(judged.candidate.psi_share, np.inf) <= settings.mmr_owner_psi_share_max


def potential_pass(judged: List[Judged], basis: AngleBasis, series: MMRSeries, context: Context, settings: AnalysisSettings) -> None:
    """The potential route (R5 §1.4-1.5), in place. The best accepted owner (by R^2 or phase) gets its potential for
    the columns, and with mmr_owner_potential_confirm a measured potential that disagrees (`potential_confirms`)
    rejects it (the next accepted one is tried). When no owner is accepted after that, the potential may accept one
    (`_potential_fallback`). The route of every accepted candidate goes to `OwnerCandidate.route`."""
    if settings.mmr_owner_potential != 'off':
        for item in sorted((j for j in judged if j.accepted), key=lambda j: _rank_key(j, settings)):
            fit = potential_of(item, basis, series, context, settings)
            item.candidate = replace(item.candidate, potential=fit)
            if settings.mmr_owner_potential_confirm and fit.measured and not potential_confirms(fit, settings):
                item.accepted, item.reason, item.passing = False, REASON_POTENTIAL_UNCONFIRMED, []
                continue
            break
        if not any(j.accepted for j in judged):
            _potential_fallback(judged, basis, series, context, settings)
    for item in judged:
        if item.accepted:
            item.candidate = replace(item.candidate, route=item.reason.split(':', 1)[1])


def _potential_fallback(judged: List[Judged], basis: AngleBasis, series: MMRSeries, context: Context, settings: AnalysisSettings) -> None:
    """The candidates without an accepted owner: each usable regime of a candidate that passes the gates
    (`_potential_gates`; Codex W20 A #8: not only the regime with the best R^2, which may fail on its rate while another
    qualifies) is measured by `potential_of`, the best mmr_owner_potential_top_n candidates by their best gated R^2.
    A candidate is accepted by 'potential' when `potential_accepts` holds on one of its regimes: shown on the one with
    the most windows (then R^2), every regime it accepts counted in `passing` (the owner's intervals, F7). A candidate
    the route does not accept keeps its shown regime, with that regime's potential and reason when it was measured."""
    options = []
    for item in judged:
        entries = [entry for entry in (item.alternatives or [item]) if _potential_gates(entry, basis, series, context, settings)]
        if entries:
            options.append((item, sorted(entries, key=lambda entry: -_finite_or(entry.regime.r2, -np.inf))))
    options.sort(key=lambda option: -_finite_or(option[1][0].regime.r2, -np.inf))
    for item, entries in options[: settings.mmr_owner_potential_top_n]:
        passed = []
        for entry in entries:
            fit = potential_of(entry, basis, series, context, settings)
            entry.candidate = replace(entry.candidate, potential=fit)
            ok, reason = potential_accepts(fit, settings)
            if ok:
                passed.append(entry)
            elif reason:
                entry.reason = reason
        if passed:
            best = max(passed, key=lambda entry: (len(entry.candidate.r2_windows), _finite_or(entry.candidate.r2_median, -np.inf)))
            item.regime, item.candidate = best.regime, best.candidate
            item.accepted, item.reason, item.passing = True, f'{REASON_ACCEPTED}:{GAIN_POTENTIAL}', [entry.regime for entry in passed]
            continue
        own = next((entry for entry in entries if entry.regime is item.regime), None)
        if own is not None and own is not item:
            item.candidate, item.reason = own.candidate, own.reason


# --------------------------------------------------------------------------------------------- the owner's description


def owner_intervals(regimes: Sequence[RegimeFit], window_yr: float) -> List[Tuple[float, float, str]]:
    """Where the owner acts (F7): the union of each passing regime's windows (libration or forced_wave), and a slip
    between two libration intervals where only forced or unfitted windows lie between them (its 2 pi slip of phi_o);
    sorted by time."""
    pieces: List[Tuple[float, float, str]] = []
    for regime in regimes:
        kind = INTERVAL_LIBRATION if regime.regime == REGIME_LIBRATION else INTERVAL_FORCED
        pieces += [(a, b, kind) for a, b in regime_pieces(regime.starts, window_yr)]
    pieces.sort()
    librations = [(a, b) for a, b, kind in pieces if kind == INTERVAL_LIBRATION]
    forced = [(a, b) for a, b, kind in pieces if kind == INTERVAL_FORCED]
    for (_, end), (start, _) in zip(librations[:-1], librations[1:]):
        pieces += [(a, b, INTERVAL_SLIP) for a, b in _subtract((end, start), forced)]
    return sorted(pieces)


def _subtract(span: Tuple[float, float], covered: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """The parts of `span` outside every interval of `covered`."""
    parts = [span] if span[1] > span[0] else []
    for a, b in covered:
        parts = [piece for lo, hi in parts for piece in ((lo, min(hi, a)), (max(lo, b), hi)) if piece[1] > piece[0]]
    return parts


def connected_blocks(intervals: Sequence[Tuple[float, float, str]]) -> List[Tuple[float, float]]:
    """The union of the intervals as disjoint blocks: librations joined by their slips are one block (the owner held
    throughout, R5 §1.7: the second capture is tested on the block that covers sigma's episode)."""
    blocks: List[Tuple[float, float]] = []
    for a, b, _ in sorted(intervals):
        if blocks and a <= blocks[-1][1]:
            blocks[-1] = (blocks[-1][0], max(blocks[-1][1], b))
        else:
            blocks.append((a, b))
    return blocks


def covered_fraction(intervals: Sequence[Tuple[float, float, str]], window: Tuple[float, float]) -> float:
    """|union of the intervals within the window| / |window| (nan for an empty window)."""
    lo, hi = float(window[0]), float(window[1])
    if not hi > lo:
        return NAN
    total, reach = 0.0, lo
    for a, b in sorted((max(a, lo), min(b, hi)) for a, b, _ in intervals):
        if b <= max(a, reach):
            continue
        total += b - max(a, reach)
        reach = b
    return total / (hi - lo)


def capture_terms(settings: AnalysisSettings) -> int:
    """Terms of the energy fit: the constant and a cosine and a sine per harmonic (mmr_owner_capture_harmonics)."""
    return 1 + 2 * settings.mmr_owner_capture_harmonics


def capture_energy_r2(x: np.ndarray, x_dot: np.ndarray, settings: AnalysisSettings) -> float:
    """R^2 of x_dot^2 on [1, cos jx, sin jx], j = 1..mmr_owner_capture_harmonics: a pendulum-like x conserves
    x_dot^2 = 2 E - 2 V(x), V periodic in x (R5 §1.7); nan below mmr_owner_capture_samples_per_term samples per term."""
    x, y = np.asarray(x, float), np.asarray(x_dot, float) ** 2
    harmonics = settings.mmr_owner_capture_harmonics
    columns = [np.ones_like(x)] + [f(j * x) for j in range(1, harmonics + 1) for f in (np.cos, np.sin)]
    if len(x) < settings.mmr_owner_capture_samples_per_term * len(columns) or not np.var(y) > 0:
        return NAN
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return 1.0 - float(np.var(y - design @ coefficients)) / float(np.var(y))


def second_capture(
    t_yr: np.ndarray,
    sigma_clean: np.ndarray,
    phi_clean: np.ndarray,
    uv: Tuple[int, int],
    span: Tuple[float, float],
    period_yr: float,
    omega0_rad_yr: float,
    settings: AnalysisSettings,
) -> SecondCapture:
    """Is sigma also held by its own term while the owner pins phi_o (R5 §1.7)? Both resonances hold at once only if
    Psi = u sigma - v phi_o stays bounded, and a Psi held by A's term obeys the energy integral in x = Psi / u. On `span`
    (an owner interval block that meets sigma's decisive episode), with Psi low-passed at mmr_owner_capture_lowpass_periods x P
    and its filter edges trimmed (one connected block of them: `second_capture_blocks`): the net turns of Psi, the sweep
    of x, `capture_energy_r2` and lambda_ad =
    |<Psi_dot / v>| / omega0 (Shevchenko 2020, p. 9). Excluded when x sweeps at least mmr_owner_capture_sweep_deg and the
    energy R^2 < mmr_owner_r2_min; otherwise unresolved."""
    u, v = uv
    t = np.asarray(t_yr, float)
    cut = settings.mmr_owner_capture_lowpass_periods * period_yr
    psi = u * np.asarray(sigma_clean, float) - v * np.asarray(phi_clean, float)
    inside = (t >= span[0] + cut) & (t <= span[1] - cut)
    unresolved = SecondCapture(NAN, NAN, NAN, NAN, CAPTURE_UNRESOLVED, float(span[0]), float(span[1]))
    if not u or not v or np.count_nonzero(inside) < settings.mmr_owner_capture_samples_per_term * capture_terms(settings):
        return unresolved
    slow = filtering.lowpass_or_none(t, psi, cut, settings)
    if slow is None:
        return unresolved
    rate = filtering.time_derivative(t, slow)[inside]
    x, x_dot = slow[inside] / u, rate / u
    sweep = float(np.ptp(x))
    energy = capture_energy_r2(x, x_dot, settings)
    adiabatic = abs(float(np.mean(rate)) / v) / omega0_rad_yr if omega0_rad_yr > 0 else NAN
    excluded = sweep >= math.radians(settings.mmr_owner_capture_sweep_deg) and _finite_or(energy, np.inf) < settings.mmr_owner_r2_min
    turns = float(slow[inside][-1] - slow[inside][0]) / TWO_PI
    return SecondCapture(
        turns, sweep, energy, adiabatic, CAPTURE_EXCLUDED if excluded else CAPTURE_UNRESOLVED, float(span[0]), float(span[1])
    )


def second_capture_blocks(
    t_yr: np.ndarray,
    sigma_clean: np.ndarray,
    phi_clean: np.ndarray,
    uv: Tuple[int, int],
    blocks: Sequence[Tuple[float, float]],
    period_yr: float,
    omega0_rad_yr: float,
    settings: AnalysisSettings,
) -> SecondCapture:
    """`second_capture` on each connected block of the owner's intervals that meets sigma's decisive episode, separately
    (Codex W20 A #6): a stretch between two blocks, where the owner does not hold phi_o, is no part of the test.
    Excluded only when every block is measured (a finite energy R^2) and excludes (W20 round C, review B #3: a block
    whose capture was never tested leaves it open); no block is unresolved. The numbers shown are those of the block
    that decides: the block with the highest energy R^2 when excluded (the weakest exclusion), else the first measured
    block that does not exclude, else the first unmeasured one."""
    tests = [second_capture(t_yr, sigma_clean, phi_clean, uv, block, period_yr, omega0_rad_yr, settings) for block in blocks]
    if not tests:
        return SecondCapture(NAN, NAN, NAN, NAN, CAPTURE_UNRESOLVED)
    measured = [test for test in tests if np.isfinite(test.energy_r2)]
    if len(measured) == len(tests) and all(test.verdict == CAPTURE_EXCLUDED for test in tests):
        return max(tests, key=lambda test: test.energy_r2)
    open_ones = [test for test in measured if test.verdict != CAPTURE_EXCLUDED] or [
        test for test in tests if not np.isfinite(test.energy_r2)
    ]
    return replace(open_ones[0], verdict=CAPTURE_UNRESOLVED)


ANGLE_SYMBOLS = {'lambda': 'λ', 'varpi': 'ϖ', 'Omega': 'Ω'}


def angle_terms(spec: ResonanceSpec) -> Dict[str, int]:
    """{component: integer coefficient} of an angle: 'lambda', 'varpi', 'Omega' of the body and '<kind>:<planet>' of the
    planets (zero coefficients left out)."""
    terms = AngleBasis.coefficients(spec)
    if spec.body_node_coefficient:
        terms['Omega'] = int(spec.body_node_coefficient)
    for index, planet in enumerate(spec.planets):
        node = spec.planet_node_coefficient(index)
        if node:
            terms[f'Omega:{planet}'] = int(node)
    return {name: c for name, c in terms.items() if c}


def psi_terms(sigma_spec: ResonanceSpec, candidate: ResonanceSpec) -> Dict[str, int]:
    """{component: coefficient} of Psi = u sigma - v phi_o (`psi_combination`): no body lambda."""
    u, v = psi_combination(sigma_spec, candidate)
    sigma, phi = angle_terms(sigma_spec), angle_terms(candidate)
    names = list(dict.fromkeys(list(sigma) + list(phi)))
    terms = {name: u * sigma.get(name, 0) - v * phi.get(name, 0) for name in names}
    return {name: c for name, c in terms.items() if c}


def _symbol(name: str) -> str:
    kind, _, planet = name.partition(':')
    return ANGLE_SYMBOLS[kind] + (f'_{planet_letter(planet)}' if planet else '')


def _term_order(name: str) -> Tuple[int, int, int]:
    kind, _, planet = name.partition(':')
    position = PLANET_NAMES_ORDER.index(planet) + 1 if planet else 0
    return position, ('lambda', 'varpi', 'Omega').index(kind), 0


def _linear_text(terms: Mapping[str, int]) -> str:
    """'ϖ − λ_N' of {component: coefficient} (body first, then the planets in order)."""
    text = ''
    for name in sorted(terms, key=_term_order):
        c = terms[name]
        size = '' if abs(c) == 1 else str(abs(c))
        sign = ('−' if c < 0 else '') if not text else (' − ' if c < 0 else ' + ')
        text += f'{sign}{size}{_symbol(name)}'
    return text


def identity_text(sigma_spec: ResonanceSpec, candidate: ResonanceSpec) -> str:
    """The exact identity sigma = (v/u) phi + Psi/u written out (R1 §2): 385266 16N-27+0+11 with owner 3N-5+0+2 gives
    '(27/5)φ + (ϖ − λ_N)/5'; '' without a combination."""
    u, v = psi_combination(sigma_spec, candidate)
    if not u:
        return ''
    ratio = Fraction(v, u)
    size = '' if abs(ratio) == 1 else (str(abs(ratio.numerator)) if ratio.denominator == 1 else f'({abs(ratio)})')
    text = f'{"−" if ratio < 0 else ""}{size}φ'
    psi = _linear_text(psi_terms(sigma_spec, candidate))
    if psi:
        single = len(psi_terms(sigma_spec, candidate)) == 1 and not psi.startswith('−')
        body = psi if (u == 1 or single) else f'({psi})'
        text += f' + {body}' + (f'/{u}' if u != 1 else '')
    return text


def psi_definition(sigma_spec: ResonanceSpec, candidate: ResonanceSpec) -> str:
    """'u*sigma-v*phi' (column own_psi_def), '' without a combination."""
    u, v = psi_combination(sigma_spec, candidate)
    return f'{u}*sigma{-v:+d}*phi' if u else ''


def _sine_fit(angle: np.ndarray, values: np.ndarray) -> Tuple[float, float]:
    """(K, R^2) of values = K sin(angle - phi0) + C (`sine_coefficients`; nan below three samples)."""
    if len(angle) < 3:
        return NAN, NAN
    a_sin, b_cos, _, r2 = sine_coefficients(angle, values)
    return float(np.hypot(a_sin, b_cos)), r2


def band_gain(period_yr: float, frame: OwnerFrame, settings: AnalysisSettings) -> float:
    """Gain of the owner band [P/f, f P] at `period_yr`: the difference of the two low-passes' gains (nan outside it)."""
    step = float(np.median(np.diff(frame.t_yr)))
    order = settings.filter_order
    gain = filtering.butterworth_gain(period_yr, frame.period_yr / frame.band_factor, order, step)
    gain -= filtering.butterworth_gain(period_yr, frame.period_yr * frame.band_factor, order, step)
    return float(gain) if np.isfinite(gain) and gain > 0 else NAN


def describe_owner(
    result: OwnerResult,
    top: Judged,
    phi_clean: np.ndarray,
    basis: AngleBasis,
    series: MMRSeries,
    decisive_window: Tuple[float, float],
    decisive_period_yr: float,
    settings: AnalysisSettings,
) -> None:
    """The numbers of the shown candidate (rank 1) that the verdict and the figure read, in place: Psi and the identity;
    for an accepted owner its intervals and their cover of `decisive_window` (F7), the second capture on the intervals
    that meet it, the bound on sigma's own term from the potential route, sigma's and a's band wave on phi_o (a's corrected
    for the band's gain at the owner's period: comparable with K_direct / |nu_o|)."""
    sigma_spec, spec = series.resonance, top.fit.spec
    uv = psi_combination(sigma_spec, spec)
    t = basis.t
    result.psi_def, result.identity_text = psi_definition(sigma_spec, spec), identity_text(sigma_spec, spec)
    result.decisive_window = (float(decisive_window[0]), float(decisive_window[1]))
    result.step_yr = float(np.median(np.diff(t))) if len(t) > 1 else NAN
    regime, frame = top.regime, top.frame if top.frame is not None else basis.frame
    if regime is not None:
        result.owner_rate_rad_yr = regime.owner_rate_rad_yr
        t_fit = frame.t_yr[frame.fit_mask]
        region = _windows_mask(t_fit, regime.starts, frame.window_yr) & frame.mask[frame.fit_mask]
        angle = _phi_fit(top, basis)[region]
        forced = regime.regime != REGIME_LIBRATION and _finite_or(regime.owner_rate_rad_yr, 0.0) != 0
        # a forced wave is at the owner's own period, a libration's a-wave at the band's period P
        gain = band_gain(TWO_PI / abs(regime.owner_rate_rad_yr) if forced else frame.period_yr, frame, settings)
        wave, result.sigma_wave_r2 = _sine_fit(angle, frame.sigma_band[frame.fit_mask][region])
        result.sigma_wave_rad = wave / gain
        result.wave_da_meas_au = _sine_fit(angle, frame.a_band[frame.fit_mask][region])[0] / gain
    if result.accepted is None:
        return
    result.intervals = owner_intervals(top.passing or ([regime] if regime is not None else []), frame.window_yr)
    result.cover = covered_fraction(result.intervals, decisive_window)
    meeting = [(a, b) for a, b in connected_blocks(result.intervals) if b > decisive_window[0] and a < decisive_window[1]]
    if meeting and uv[0]:
        period = decisive_period_yr if _finite_or(decisive_period_yr, 0.0) > 0 else frame.period_yr
        result.capture = second_capture_blocks(t, frame.sigma_clean, phi_clean, uv, meeting, frame.period_yr, TWO_PI / period, settings)
    fit = result.accepted.potential
    if fit is not None and fit.measured and uv[0]:
        bound = max(abs(fit.scale_lo - 1.0), abs(fit.scale_hi - 1.0))
        result.sigma_force_bound = bound
        if _finite_or(decisive_period_yr, 0.0) > 0 and bound > 0:
            result.sigma_period_min_yr = decisive_period_yr / math.sqrt(abs(uv[1] / uv[0]) * bound)


def _rank_key(judged: Judged, settings: AnalysisSettings):
    """Accepted first, then a force within the mu scale (`within_scale`: a display order, no acceptance), then R^2."""
    return (not judged.accepted, not within_scale(judged.candidate, settings), -_finite_or(judged.candidate.r2_median, -np.inf))


def result_from(
    judged: List[Judged], frame: OwnerFrame, period: float, settings: AnalysisSettings, n_candidates: int, catalogue: bool = False
) -> OwnerResult:
    """OwnerResult of the judged candidates: ranked (accepted, within the mu scale, R^2), the owner is rank 1 when accepted;
    psi and the frequency test of rank 1; `owner_probable` when rank 1 is a probable, not accepted, owner.

    `catalogue`: the search had planets on catalogue lines (flag catalogue_longitudes). A candidate left open by them
    (reason catalogue_ambiguous) makes the check not_measured when none is accepted, and flags an accepted owner that
    it would outrank if it were accepted (owner_catalogue_ambiguous: the owner may be another one)."""
    if judged and not any(j.fit.unwrap_ok for j in judged):
        return _empty(CheckValue.NOT_MEASURED, [FLAG_UNWRAP_FAILED], frame.sigma_r2, period, n_candidates)
    judged = sorted(judged, key=lambda j: _rank_key(j, settings))
    candidates = [
        replace(j.candidate, rank=rank, accepted=j.accepted and rank == 1, reason=j.reason) for rank, j in enumerate(judged, start=1)
    ]
    accepted = candidates[0] if candidates and candidates[0].accepted else None
    sigma_r2, psi, freq, detail, flags = frame.sigma_r2, (NAN, NAN, NAN, NAN), (NAN, NAN), '', []
    ambiguous = [j for j in judged if j.reason == REASON_CATALOGUE_AMBIGUOUS]
    if catalogue:
        flags.append(FLAG_CATALOGUE)
    if ambiguous and (accepted is None or _rank_key(replace(ambiguous[0], accepted=True), settings) < _rank_key(judged[0], settings)):
        flags.append(FLAG_CATALOGUE_AMBIGUOUS)
    check = CheckValue.FAIL if accepted is not None else (CheckValue.NOT_MEASURED if ambiguous else CheckValue.PASS)
    if judged:
        top = judged[0]
        if top.regime is not None:
            sigma_r2 = top.regime.sigma_r2
            psi = top.fit.psi.get(top.regime.regime, psi)
        if top.fit.phi_f is not None:
            freq = frequency_test(frame, top.fit.phi_f, settings)
        if accepted is not None:
            libration = top.regime.regime == REGIME_LIBRATION
            detail = str(OriginDetail.OWNER_LIBRATION if libration else OriginDetail.OWNER_FORCED_WAVE)
        elif probable(top, settings):
            flags.append(FLAG_PROBABLE)
        if accepted is None and owner_unresolved(top, settings):
            flags.append(FLAG_UNRESOLVED)
            check = CheckValue.NOT_MEASURED
    return OwnerResult(
        n_candidates=n_candidates,
        candidates=candidates,
        accepted=accepted,
        check=check,
        sigma_r2=sigma_r2,
        psi_rate_rad_yr=psi[1],
        psi_slow_corr=psi[2],
        psi_varpi_corr=psi[3],
        freq_r2_owner=freq[0],
        freq_r2_sigma=freq[1],
        freq_period_yr=period,
        detail=detail,
        flags=flags,
    )


def _describe_top(
    result: OwnerResult,
    top: Judged,
    phi_clean: np.ndarray,
    frame: OwnerFrame,
    series: MMRSeries,
    prep: Preprocessed,
    settings: AnalysisSettings,
) -> None:
    """psi and the frequency test of rank 1 (its arrays were dropped after judging): recomputed from its angle."""
    phi_f = filtering.lowpass_with(frame.t_yr, phi_clean, prep.cutoff.cutoff_yr, settings)
    result.freq_r2_owner, result.freq_r2_sigma = frequency_test(frame, phi_f, settings)
    if top.regime is None:
        return
    if top.regime.regime not in top.fit.psi:
        add_psi(top.fit, top.regime, phi_clean, frame, series, settings)
    psi = top.fit.psi[top.regime.regime]
    result.psi_rate_rad_yr, result.psi_slow_corr, result.psi_varpi_corr = psi[1], psi[2], psi[3]
    if result.candidates and not np.isfinite(result.candidates[0].psi_share):
        result.candidates[0] = replace(result.candidates[0], psi_share=psi[0])
    shown = replace(top, candidate=replace(top.candidate, psi_share=psi[0]))
    if result.accepted is None and FLAG_PROBABLE not in result.flags and probable(shown, settings):
        result.flags.append(FLAG_PROBABLE)
