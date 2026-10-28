"""Columns of a Lidov-Kozai row that change no status: the e-i coupling at the decisive period, the attribution of a
kinematic libration to a forced mode, and the guard that keeps a forced-vector beat from deciding the scale.

e-i coupling (`measure_ei_coupling`, columns zlk_ei_*). At the decisive period P (P_cyc of the decisive geometry; for
a circulating omega the quadrupole harmonic pi/|d omega/dt|) e and the inclination from the invariable plane (exact
rotation; the tilt min(i, 180 - i) of a retrograde orbit, an orbit that crosses 90 deg is not measured) are fitted with
one sinusoid each on the record less one period at each edge, and band-passed around P for their correlation r. Two
laws predict delta i / delta e:

    zlk           sqrt(1 - e^2) cos i conserved (the quadrupole integral):  di/de = -e cot i / (1 - e^2), antiphase
    forced_beat   sqrt(1 - e^2)(1 - cos i) conserved (the momentum conjugate to Omega when the forcing depends on
                  varpi only, as a forced-eccentricity beat does):          di/de = +e tan(i/2) / (1 - e^2), in phase
                  (for the tilt of a retrograde orbit: -e cot(tilt/2) / (1 - e^2))

A law is matched when the phase of i against e is within ZLK_EI_PHASE_TOL_DEG of its sign, r has its sign with
|r| >= ZLK_EI_MIN_BAND_R, and |delta i|/|delta e| is within a factor ZLK_EI_AMPLITUDE_FACTOR of it; none when every
number is finite and neither law matches. A band the sampling cannot hold (above Nyquist: undersampled), one the filter
cannot pass (band_unmeasured) or a law that is not finite (law_undefined) is not_measured, never none.
Coupling is not libration: 15527 circulates and shows zlk coupling at its 2.05-kyr harmonic, 1866 (kinematic) and
88899 too; the librators 3040, 162474, 4257, 1373, 1981 show it at their period (phase 178-182 deg, r ~ -0.995);
591986's 6-kyr cycle is a forced beat (phase ~ 330 deg, delta i 0.0041 deg against 0.0037 predicted).

Attribution (`attribute_forcing`, columns zlk_forcing_*): when the gate says kinematic, the fitted eccentricity mode
with the largest |c_j| (named with the cluster the baseline merged into it) is what pins omega, provided that
    the free circle stays off the origin: rho_max < 1, or the strict pin max_t |z_free| + sum_{k != j} |c_k| < |c_j|
    (the largest free radius over the record, not its median: the bound must hold at every sample);
    the predicted beat 2 pi/|b_varpi - g_j| is P_cyc within a Rayleigh width (what the osculating omega librates with);
    the free node follows the mode: |b_Omega - g_j| T < 1 turn (else the centre varpi_j - Omega would turn);
    the per-mode amplitudes exist (a run restored from a file written before they were persisted has none).
Otherwise 'not_attributed' with the first condition that failed.

Centre-scale guard (`forced_beat_centre`): a libration at the centre scale whose period is within a Rayleigh width of a
beat 2 pi/|g_k - g_j| of two fitted forced modes with nonzero amplitudes, whose reconstructed forced phase wave explains
the centre (shape and size, `slow.forced_beat_model`), while rho_max < 1 at the fine period, is the forced vector
turning its direction, not a libration of omega: the fine scale stays decisive (a clean synthetic 591986 with
e_g6 = 0.017 otherwise decides at 54 kyr instead of 5.95 kyr). No amplitudes (an old file), a zero term or a wave of
another size: no rejection. The caller applies it only where the gate acts (`analyse_zlk`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt

from resonances.data.invariable_plane import inclination_to_invariable_plane
from resonances.resonance.analysis.ledger import Op
from resonances.resonance.analysis.secular.models import ARCSEC_PER_TURN
from resonances.resonance.analysis.verdict import AngleStatus, CheckValue

NAN = float('nan')
BAND_ORDER = 3  # Butterworth order of the e-i band (forward-backward)


class Coupling:
    ZLK = 'zlk'
    FORCED_BEAT = 'forced_beat'
    NONE = 'none'
    NOT_MEASURED = 'not_measured'


@dataclass
class EICoupling:
    """The e-i coupling of one ZLK row at one period (module docstring)."""

    coupling: str = Coupling.NOT_MEASURED
    # not_measured: no_period, no_node, undersampled, orbit_flip, short_record, flat_series, band_unmeasured,
    # law_undefined
    reason: str = ''
    period_yr: float = NAN
    period_source: str = ''  # libration (P_cyc) | circulation_harmonic (pi/|omega_dot|)
    phase_deg: float = NAN  # phase of i against e at P, [0, 360)
    band_r: float = NAN
    amp_e: float = NAN
    amp_i_deg: float = NAN
    ratio_h: float = NAN  # |delta i| / |delta e| over the constant-H (ZLK) law
    ratio_beat: float = NAN  # ... over the forced-beat law
    retrograde: bool = False

    @property
    def confirmed_ei(self) -> Optional[bool]:
        """The compatibility flag: ZLK coupling at the decisive period (None when not measured)."""
        return None if self.coupling == Coupling.NOT_MEASURED else self.coupling == Coupling.ZLK

    @property
    def check(self) -> CheckValue:
        """check_zlk_ei: pass for zlk coupling, fail for another or none, not_measured."""
        if self.coupling == Coupling.NOT_MEASURED:
            return CheckValue.NOT_MEASURED
        return CheckValue.PASS if self.coupling == Coupling.ZLK else CheckValue.FAIL

    def to_columns(self) -> Dict[str, Any]:
        return {
            'confirmed_ei': self.confirmed_ei,
            'zlk_ei_coupling': self.coupling,
            'zlk_ei_reason': self.reason,
            'zlk_ei_period_yr': self.period_yr,
            'zlk_ei_period_source': self.period_source,
            'zlk_ei_phase_deg': self.phase_deg,
            'zlk_ei_band_r': self.band_r,
            'zlk_ei_amp_e': self.amp_e,
            'zlk_ei_amp_i_deg': self.amp_i_deg,
            'zlk_ei_ratio_h': self.ratio_h,
            'zlk_ei_ratio_beat': self.ratio_beat,
        }


@dataclass
class Attribution:
    """Which forced mode holds a kinematic libration (module docstring); numbers only when attributed."""

    attributed: bool = False
    reason: str = ''  # not attributed: why
    mode: str = ''  # 'g5 (g5+g7+g8+const)'
    forcing_e: float = NAN  # |c_j|
    free_over_forcing: float = NAN  # e_free / |c_j|
    strict_pin: Optional[bool] = None
    beat_period_yr: float = NAN  # 2 pi / |b_varpi - g_j|
    node_minus_mode_arcsec_yr: float = NAN  # b_Omega - g_j
    node_minus_mode_turns: float = NAN  # over the free elements' baseline

    def to_columns(self) -> Dict[str, Any]:
        return {
            'zlk_attribution': 'attributed' if self.attributed else 'not_attributed',
            'zlk_attribution_reason': self.reason,
            'zlk_forcing_mode': self.mode,
            'zlk_forcing_e': self.forcing_e,
            'zlk_free_over_forcing': self.free_over_forcing,
            'zlk_forcing_strict_pin': self.strict_pin,
            'zlk_beat_period_yr': self.beat_period_yr,
            'zlk_node_minus_mode_arcsec_yr': self.node_minus_mode_arcsec_yr,
            'zlk_node_minus_mode_turns': self.node_minus_mode_turns,
        }


@dataclass
class ZlkDiagnostics:
    """The columns of `measure_ei_coupling` and `attribute_forcing` (`LongPeriodAnalysis.zlk`)."""

    ei: EICoupling = field(default_factory=EICoupling)
    attribution: Attribution = field(default_factory=Attribution)

    def to_columns(self) -> Dict[str, Any]:
        return {**self.ei.to_columns(), **self.attribution.to_columns()}


# ------------------------------------------------------------------------------------------------ e-i coupling


def decisive_period(angle_status: str, period_cycle_yr: float, t_yr: np.ndarray, omega: np.ndarray) -> tuple:
    """(P [yr], source): P_cyc of a bounded omega; pi/|mean d omega/dt| of a circulating one (the cos 2 omega harmonic
    of the quadrupole Hamiltonian, at which e and i trade); (nan, '') otherwise."""
    if str(angle_status) == str(AngleStatus.CIRCULATION):
        t, x = np.asarray(t_yr, dtype=float), np.unwrap(np.asarray(omega, dtype=float))
        if len(t) > 2 and t[-1] > t[0]:
            rate = float(np.polyfit(t, x, 1)[0])
            if rate != 0 and np.isfinite(rate):
                return np.pi / abs(rate), 'circulation_harmonic'
        return NAN, ''
    if period_cycle_yr is not None and np.isfinite(period_cycle_yr) and period_cycle_yr > 0:
        return float(period_cycle_yr), 'libration'
    return NAN, ''


def _sinusoid(t: np.ndarray, x: np.ndarray, period: float) -> complex:
    """Complex amplitude A of x = Re(A exp(2 pi i t / P)) + a quadratic trend, by least squares."""
    tau = (t - t.mean()) / (t[-1] - t[0])
    phase = 2 * np.pi * t / period
    design = np.column_stack([np.cos(phase), np.sin(phase), np.ones_like(t), tau, tau**2])
    c, *_ = np.linalg.lstsq(design, x, rcond=None)
    return complex(c[0], -c[1])


def _band_resolved(t: np.ndarray, period: float, halfwidth: float) -> bool:
    """Whether the band [(1 - h)/P, (1 + h)/P] lies below the Nyquist frequency of the sampling (else e and i are
    aliased there and the coupling cannot be measured, whatever the fits say)."""
    step = float(np.median(np.diff(t))) if len(t) > 1 else NAN
    low, high = (1.0 - halfwidth) / period, (1.0 + halfwidth) / period
    return bool(np.isfinite(step) and step > 0 and 0 < low < high < 0.5 / step)


def _band_r(t: np.ndarray, e: np.ndarray, i: np.ndarray, period: float, halfwidth: float, inside: np.ndarray) -> float:
    """Correlation of e and i band-passed to [(1 - h)/P, (1 + h)/P] (Butterworth, forward-backward) on `inside`; nan
    when the band cannot be filtered (`_band_resolved` false, a record too short for the filter) or is flat."""
    if not _band_resolved(t, period, halfwidth):
        return NAN
    step = float(np.median(np.diff(t)))
    low, high = (1.0 - halfwidth) / period, (1.0 + halfwidth) / period
    try:
        sos = butter(BAND_ORDER, [low, high], btype='band', fs=1.0 / step, output='sos')
        eb, ib = sosfiltfilt(sos, e - e.mean()), sosfiltfilt(sos, i - i.mean())
    except ValueError:
        return NAN
    eb, ib = eb[inside], ib[inside]
    if not (np.all(np.isfinite(eb)) and np.all(np.isfinite(ib))) or np.std(eb) == 0 or np.std(ib) == 0:
        return NAN
    return float(np.corrcoef(eb, ib)[0, 1])


def _matches(law: float, ratio: float, phase_deg: float, r: float, settings) -> bool:
    """A law of sign sign(law): phase within tolerance of 180 (negative) or 0 (positive), r of that sign and strong
    enough, |delta i|/|delta e| within the amplitude factor of |law| (`ratio` = measured / |law|)."""
    if not (np.isfinite(law) and law != 0 and np.isfinite(ratio) and np.isfinite(r)):
        return False
    target = 180.0 if law < 0 else 0.0
    off = abs((phase_deg - target + 180.0) % 360.0 - 180.0)
    factor = settings.zlk_ei_amplitude_factor
    return off <= settings.zlk_ei_phase_tol_deg and np.sign(law) * r >= settings.zlk_ei_min_band_r and 1 / factor <= ratio <= factor


def measure_ei_coupling(t_yr, e, inc, Omega, period_yr: float, period_source: str, settings) -> EICoupling:
    """The e-i coupling at `period_yr` (module docstring); `inc`, `Omega` ecliptic [rad]."""
    result = EICoupling(period_yr=float(period_yr), period_source=period_source)
    if not (np.isfinite(period_yr) and period_yr > 0):
        result.reason = 'no_period'
        return result
    if Omega is None:
        result.reason = 'no_node'
        return result
    t, e = np.asarray(t_yr, dtype=float), np.asarray(e, dtype=float)
    if not _band_resolved(t, period_yr, settings.zlk_ei_band_halfwidth):
        result.reason = 'undersampled'  # the band reaches the Nyquist frequency: aliased, no measurement
        return result
    inclination = inclination_to_invariable_plane(inc, Omega)
    if np.nanmin(inclination) < np.pi / 2 < np.nanmax(inclination):
        result.reason = 'orbit_flip'
        return result
    result.retrograde = bool(np.nanmedian(inclination) > np.pi / 2)
    tilt = np.pi - inclination if result.retrograde else inclination
    inside = (t >= t[0] + period_yr) & (t <= t[-1] - period_yr)
    if inside.sum() < 8 or (t[-1] - t[0] - 2 * period_yr) < settings.zlk_ei_min_cycles * period_yr:
        result.reason = 'short_record'
        return result
    amp_e, amp_i = _sinusoid(t[inside], e[inside], period_yr), _sinusoid(t[inside], tilt[inside], period_yr)
    if abs(amp_e) == 0 or abs(amp_i) == 0:
        result.reason = 'flat_series'
        return result
    e_mean, i_mean = float(np.mean(e[inside])), float(np.mean(tilt[inside]))
    law_h = -e_mean / np.tan(i_mean) / (1 - e_mean**2)
    law_beat = (-e_mean / np.tan(i_mean / 2) if result.retrograde else e_mean * np.tan(i_mean / 2)) / (1 - e_mean**2)
    measured = abs(amp_i) / abs(amp_e)
    result.phase_deg = float(np.degrees(np.angle(amp_i) - np.angle(amp_e)) % 360.0)
    result.band_r = _band_r(t, e, tilt, period_yr, settings.zlk_ei_band_halfwidth, inside)
    result.amp_e, result.amp_i_deg = abs(amp_e), float(np.degrees(abs(amp_i)))
    result.ratio_h = measured / abs(law_h) if law_h else NAN
    result.ratio_beat = measured / abs(law_beat) if law_beat else NAN
    laws = ((Coupling.ZLK, law_h, result.ratio_h), (Coupling.FORCED_BEAT, law_beat, result.ratio_beat))
    found = [(abs(np.log(ratio)), name) for name, law, ratio in laws if _matches(law, ratio, result.phase_deg, result.band_r, settings)]
    if found:
        result.coupling = min(found)[1]
        return result
    # 'none' is a finite measurement that rejects both laws; a missing number rejects nothing
    if not np.isfinite(result.band_r):
        result.reason = 'band_unmeasured'  # the band could not be filtered (too short for the filter) or is flat
    elif not np.isfinite(result.phase_deg) or not all(np.isfinite(law) and law != 0 and np.isfinite(ratio) for _, law, ratio in laws):
        result.reason = 'law_undefined'  # an amplitude or a law's slope is not finite (e.g. i at 0 or 90 deg)
    else:
        result.coupling = Coupling.NONE
    return result


# ------------------------------------------------------------------------------------------------- attribution


def _within_rayleigh(period_a: float, period_b: float, resolution: float) -> bool:
    if not (np.isfinite(period_a) and np.isfinite(period_b) and period_a > 0 and period_b > 0):
        return False
    return abs(ARCSEC_PER_TURN / period_a - ARCSEC_PER_TURN / period_b) < resolution


def attribute_forcing(elements, gate_outcome: str, period_cycle_yr: float, rho_max: float) -> Attribution:
    """The forced mode that holds a kinematic libration, or why none is named (module docstring)."""
    if gate_outcome != 'kinematic':
        return Attribution(reason='gate_not_kinematic')
    if elements is None or not elements.amplitudes_e:
        return Attribution(reason='no_mode_amplitudes')
    amplitudes = {label: abs(value) for label, value in elements.amplitudes_e.items()}
    star = max(amplitudes, key=amplitudes.get)
    if star == 'const' or star not in elements.basis_e:
        return Attribution(reason='constant_term_dominates')
    forcing = amplitudes[star]
    e_free = elements.e_free
    # the strict pin bounds the forced radius from below by |c_j| - sum_k |c_k| over the whole record, so the free
    # radius must be bounded from above over the same record: its largest value, never the median (a free circle of
    # 0.01 for 70 % of the record and 0.10 for the rest against |c_j| = 0.05 reaches the origin)
    free_radius = np.abs(np.asarray(elements.z_e_free))
    free_radius = free_radius[np.isfinite(free_radius)]
    free_max = float(np.max(free_radius)) if len(free_radius) else NAN
    others = sum(value for label, value in amplitudes.items() if label != star)
    strict = bool(np.isfinite(free_max) and free_max + others < forcing)
    if not (rho_max < 1.0 or strict):
        return Attribution(reason='free_circle_reaches_origin')
    rate = elements.basis_e[star]
    gap = abs(elements.b_varpi - rate)
    beat = ARCSEC_PER_TURN / gap if np.isfinite(gap) and gap > 0 else NAN
    if not _within_rayleigh(beat, period_cycle_yr, elements.resolution):
        return Attribution(reason='beat_period_mismatch')
    node = float(elements.b_Omega - rate)
    turns = abs(node) * elements.baseline / ARCSEC_PER_TURN
    if not turns < 1.0:
        return Attribution(reason='node_does_not_follow_mode')
    group = next((members for members in elements.clusters_e if star in members), [star])
    return Attribution(
        attributed=True,
        mode=star if len(group) < 2 else f'{star} ({"+".join(group)})',
        forcing_e=forcing,
        free_over_forcing=e_free / forcing,
        strict_pin=strict,
        beat_period_yr=beat,
        node_minus_mode_arcsec_yr=node,
        node_minus_mode_turns=turns,
    )


# ----------------------------------------------------------------------------------------- centre-scale guard


def forced_beat_centre(elements, slow_period_yr: float, rho_max_fine: float, track=None, t_series=None, settings=None, ledger=None) -> str:
    """'forced_beat:<gk>-<gj>' when a centre-scale libration of period `slow_period_yr` is a measured beat of the
    fitted forced modes that explains the centre while the free circle stays off the origin; '' otherwise.

    All of: a beat of two modes with nonzero per-mode amplitudes within a Rayleigh width of the period (none without
    the amplitudes: an old file); the phase wave of the reconstructed forced vector explains `track` (the fine
    geometry's centre, the slow wave the centre scale would decide on) in shape and size (`slow.forced_beat_model`:
    r >= ZLK_CENTRE_MODEL_R_MIN, gain within ZLK_CENTRE_MODEL_AMP_FACTOR); rho_max at the fine period < 1. With a
    ledger the comparisons are recorded (centre_beat_distance, centre_beat_model_r, centre_beat_model_gain,
    centre_beat_rho_max)."""
    from resonances.resonance.analysis.secular import slow

    if elements is None or not np.isfinite(slow_period_yr) or track is None or settings is None:
        return ''
    beats = elements.forced_beats()
    if not beats:
        return ''
    label, beat = min(beats.items(), key=lambda item: abs(ARCSEC_PER_TURN / slow_period_yr - ARCSEC_PER_TURN / item[1]))
    distance = abs(ARCSEC_PER_TURN / slow_period_yr - ARCSEC_PER_TURN / beat)
    model = slow.forced_beat_model(track, t_series, elements, slow_period_yr, settings)
    r = model.r if model is not None else NAN
    gain = model.gain if model is not None else NAN
    factor = settings.zlk_centre_model_amp_factor
    if ledger is not None:
        near = ledger.compare('centre_beat_distance', distance, Op.LT, elements.resolution, 'arcsec/yr', 'sec_centre_rejected')
        explained = near and ledger.compare_setting(
            'centre_beat_model_r', r, Op.GE, settings, 'zlk_centre_model_r_min', column='zlk_centre_model_r'
        )
        explained = explained and ledger.compare(
            'centre_beat_model_gain',
            abs(np.log(gain)) if np.isfinite(gain) and gain > 0 else NAN,
            Op.LE,
            float(np.log(factor)),
            '|ln gain|',
            'zlk_centre_model_gain',
        )
        pinned = explained and ledger.compare('centre_beat_rho_max', rho_max_fine, Op.LT, 1.0, '', 'zlk_gate_rho_max')
    else:
        near = distance < elements.resolution
        explained = near and model is not None and model.explains
        pinned = explained and np.isfinite(rho_max_fine) and rho_max_fine < 1.0
    if not (near and explained and pinned):
        return ''
    return f'forced_beat:{model.label}'
