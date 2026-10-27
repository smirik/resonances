"""Inputs and results of the secular and Lidov-Kozai analysis (`analyse_secular`, `analyse_zlk`).

Both angles are slow (periods of 10^4–10^7 yr) and are judged by the same shared geometry as an MMR angle:
filter, turning points, episodes, the centre and its passes (`analysis.geometry`), the ledger and the
confidence rule. What differs is the cutoff of the filter (`secular.preprocess`), the physics column of a
secular angle (the response of e and i, `secular.response`) and the free-omega gate of the Lidov-Kozai angle
(`analysis.zlk`). Pure: arrays and dataclasses in and out, no Body or Simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from resonances.resonance.analysis.geometry import FilteredAngle, Geometry
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.verdict import ResonanceType, Verdict, finite_or_none, join_list

RAD_TO_DEG = 180.0 / np.pi
# Columns of the shared Geometry that only an MMR fills (slips, stalls, pattern B, jumps of a, the arc's force bounds):
# a secular or ZLK angle has no slow variable answering it by an identity, and no jumps are searched
MMR_ONLY_PREFIXES = ('stall_', 'jump_', 'geo_slip', 'geo_n_slips', 'geo_stall', 'geo_pattern_b', 'geo_h_rev', 'geo_rate_min')
MMR_ONLY_COLUMNS = ('regime_start_yr', 'geo_n_revolutions', 'geo_rate_a_corr')
ARCSEC_PER_TURN = 1296000.0  # a rate of f arcsec/yr turns once in 1296000 / f yr


@dataclass(frozen=True)
class AngleTerm:
    """One term of a secular angle: `coefficient` x (the body's varpi or Omega when `index` is None, else the
    longitude of planetary mode g_index or s_index). `mode` is 'g' (perihelia) or 's' (nodes)."""

    mode: str
    index: Optional[int]
    coefficient: int


@dataclass
class LongPeriodSeries:
    """One secular or Lidov-Kozai angle of one body on a uniform, increasing grid of years.

    `angle` is the unwrapped angle the package classifies (the reduced proper angle of a secular resonance,
    omega for ZLK) [rad]; `varpi` and `Omega` are the body's osculating longitudes [rad] (the free rates g, s of a
    secular angle come from them); `terms` the angle's terms (secular only) and `planet_frequencies` the modes'
    frequencies [arcsec/yr] the angle was built with.
    """

    resonance_type: ResonanceType
    key: str
    name: str
    t_yr: np.ndarray
    angle: np.ndarray
    a: np.ndarray
    e: np.ndarray
    inc: np.ndarray
    varpi: Optional[np.ndarray] = None
    Omega: Optional[np.ndarray] = None
    terms: Tuple[AngleTerm, ...] = ()
    planet_frequencies: Mapping[str, float] = field(default_factory=dict)

    @property
    def step_yr(self) -> float:
        return float(np.median(np.diff(self.t_yr))) if len(self.t_yr) > 1 else float('nan')

    @property
    def span_yr(self) -> float:
        return float(self.t_yr[-1] - self.t_yr[0]) if len(self.t_yr) > 1 else 0.0


@dataclass
class Cutoff:
    """The cutoff P_c of the angle's filter and the prominence of its turning points (`secular.preprocess`).

    P_c = max(F_GI x P_GI, SECULAR_FILTER_FORCED_FACTOR x P_w, P_min / FILTER_MIN_LIBRATION_CUTOFFS)
    with P_min the type's shortest libration and P_w the longest forced
    wave 2 pi/|f - f_j| of the angle's own planetary modes (secular angles with two or more modes). The prominence
    is max(floor, K r), r the ripple: the 95th percentile of what sigma_f holds faster than the type's shortest
    libration (SECULAR_/ZLK_LIBRATION_PERIOD_MIN_YR), K the type's ripple factor. `fast_amplitude_rad` is the 95th
    percentile of |sigma - sigma_f| (what the filter removed)."""

    cutoff_yr: float
    binding: str  # great_inequality | forced_wave | shortest_libration
    gi_bound_yr: float
    forced_bound_yr: float
    libration_bound_yr: float = float('nan')
    forced_waves_yr: Dict[str, float] = field(default_factory=dict)  # mode label -> period of its forced wave
    body_g_arcsec_yr: float = float('nan')
    body_s_arcsec_yr: float = float('nan')
    fast_amplitude_rad: float = float('nan')
    ripple_rad: float = float("nan")  # r: what sigma_f holds faster than the shortest libration
    prominence_rad: float = float('nan')
    prominence_binding: str = ''  # floor | ripple

    @property
    def forced_wave_yr(self) -> float:
        return max(self.forced_waves_yr.values()) if self.forced_waves_yr else float('nan')

    def to_columns(self) -> Dict[str, Any]:
        waves = join_list(f'{label}={period:.6g}' for label, period in self.forced_waves_yr.items())
        return {
            'flt_cutoff_yr': self.cutoff_yr,
            'flt_cutoff_binding': self.binding,
            'flt_cutoff_gi_bound_yr': self.gi_bound_yr,
            'flt_cutoff_forced_bound_yr': self.forced_bound_yr,
            'flt_cutoff_libration_bound_yr': self.libration_bound_yr,
            'flt_forced_wave_yr': self.forced_wave_yr,
            'flt_forced_waves_yr': waves,
            'flt_fast_amp_deg': self.fast_amplitude_rad * RAD_TO_DEG,
            'flt_ripple_deg': self.ripple_rad * RAD_TO_DEG,
            'flt_tp_prominence_binding': self.prominence_binding,
            'sec_g_arcsec_yr': self.body_g_arcsec_yr,
            'sec_s_arcsec_yr': self.body_s_arcsec_yr,
        }


@dataclass
class Response:
    """How e (and sin(i/2)) answer the angle: least squares x_dot = A sin(sigma_f) + B cos(sigma_f) + C on the window.

    Lagrange's planetary equations (Murray & Dermott 1999, §6.8, p. 251) with a resonant secular term
    R = K e^|p| sin^|q|(i/2) cos(sigma), sigma = p varpi + q Omega + ..., give e_dot ~ -(p / (n a^2 e)) dR/dsigma
    ~ p sin(sigma) and the same for sin(i/2) with q: a resonant angle drives e (g-terms) and i (s-terms) at its own
    phase; a bounded angle that is only geometry does not. A column only (SECULAR_RESPONSE_R2_MIN): it changes no
    status and no confidence (author, Q11).
    """

    variables: Tuple[str, ...]  # 'e' and/or 'i': the body's elements the angle contains
    r2_e: float = float('nan')
    amplitude_e_per_kyr: float = float('nan')
    phase_e_deg: float = float('nan')
    r2_i: float = float('nan')
    amplitude_i_per_kyr: float = float('nan')
    phase_i_deg: float = float('nan')
    check: str = ''

    def to_columns(self) -> Dict[str, Any]:
        return {
            'rsp_variables': join_list(self.variables),
            'rsp_e_R2': self.r2_e,
            'rsp_e_amp_per_kyr': self.amplitude_e_per_kyr,
            'rsp_e_phase_deg': self.phase_e_deg,
            'rsp_i_R2': self.r2_i,
            'rsp_i_amp_per_kyr': self.amplitude_i_per_kyr,
            'rsp_i_phase_deg': self.phase_i_deg,
        }


@dataclass
class GateResult:
    """The free-omega gate of a Lidov-Kozai angle as the analysis read it (`omega_free_gate`)."""

    outcome: str  # GateOutcome value, or '' when the gate did not run
    n_cycles: Optional[float] = None
    matches: Tuple[str, ...] = ()
    min_cycles: float = float('nan')
    libration_period_yr: float = float('nan')
    rho_max: float = float('nan')  # e_free / e_forced at its largest over the beating of the forced vector

    def to_columns(self) -> Dict[str, Any]:
        return {
            'zlk_gate': self.outcome,
            'zlk_gate_rho_max': self.rho_max,
            'zlk_gate_n_cycles': self.n_cycles,
            'zlk_gate_min_cycles': self.min_cycles,
            'zlk_gate_matches': join_list(self.matches),
            'zlk_gate_period_yr': self.libration_period_yr,
        }


@dataclass
class LongPeriodAnalysis:
    """Everything one secular or Lidov-Kozai analysis measured and decided."""

    series: LongPeriodSeries
    verdict: Verdict
    ledger: Ledger
    cutoff: Optional[Cutoff] = None
    angle: Optional[FilteredAngle] = None
    geometry: Optional[Geometry] = None
    response: Optional[Response] = None
    gate: Optional[GateResult] = None
    decisive_episode_id: Optional[int] = None
    n_librating_episodes: Optional[int] = None  # accepted episodes not faster than the shortest libration
    scale: str = ''  # fine | centre: the geometry the verdict rests on (`analyse.SCALE_*`)
    fine_geometry: Optional[Geometry] = None  # the geometry at P_c when the verdict rests on the centre scale
    unphysical: bool = False
    unphysical_reason: str = ''
    filter_order: int = 2  # FILTER_ORDER of the settings (the figure filters e and i the same way)
    settings_hash: str = ''
    confidence_cells_sha: str = ''
    frozen_numbers: Optional[Dict[str, Any]] = None
    frozen_episode_rows: Optional[List[Dict[str, Any]]] = None

    @property
    def libration_period_yr(self) -> float:
        """P_cyc: the median full period of the accepted episodes (nan without turning points)."""
        return float(self.geometry.period_cycle_yr) if self.geometry is not None else float('nan')

    def numbers_row(self) -> Dict[str, Any]:
        """Every measured number of the analysis in CSV units (also `body.classifier_numbers[key]`)."""
        if self.frozen_numbers is not None:
            return dict(self.frozen_numbers)
        s = self.series
        row: Dict[str, Any] = {
            'smp_span_yr': s.span_yr,
            'smp_step_yr': s.step_yr,
            'smp_n_rows': len(s.t_yr),
            'ctx_e_min': _stat(s.e, np.min),
            'ctx_e_max': _stat(s.e, np.max),
            'ctx_inc_min_deg': _stat(s.inc, np.min, RAD_TO_DEG),
            'ctx_inc_max_deg': _stat(s.inc, np.max, RAD_TO_DEG),
            'ctx_unphysical': self.unphysical_reason,
            'sec_n_librating_episodes': self.n_librating_episodes,
            'sec_scale': self.scale,
            'sec_scale_cutoff_yr': self.angle.cutoff_yr if self.angle is not None else None,
            'sec_fine_n_tp': len(self.fine_geometry.turning_points) if self.fine_geometry is not None else None,
            'sec_fine_period_cycle_yr': self.fine_geometry.period_cycle_yr if self.fine_geometry is not None else None,
        }
        for result in (self.cutoff, self.geometry, self.response, self.gate):
            if result is not None:
                row.update(result.to_columns())
        for column in [column for column in row if column.startswith(MMR_ONLY_PREFIXES) or column in MMR_ONLY_COLUMNS]:
            del row[column]
        row.update(self._decisive_episode_columns())
        return row

    def summary_row(self) -> Dict[str, Any]:
        """The summary.csv columns of the analysis: the verdict, then every measured number."""
        row = {'name': self.series.name, 'resonance': self.series.key}
        row.update(self.verdict.to_flat_dict())
        row.update(self.numbers_row())
        row['settings_hash'] = self.settings_hash
        row['confidence_cells_sha'] = self.confidence_cells_sha
        return row

    def episode_rows(self) -> List[Dict[str, Any]]:
        """episodes.csv rows: every run of turning points, accepted or rejected with its reason."""
        if self.frozen_episode_rows is not None:
            return [dict(row) for row in self.frozen_episode_rows]
        if self.geometry is None:
            return []
        return [_episode_row(self, run) for run in self.geometry.runs]

    def _decisive_episode_columns(self) -> Dict[str, Any]:
        if self.decisive_episode_id is None:
            return {}
        rows = [row for row in self.episode_rows() if row['episode_id'] == self.decisive_episode_id]
        skip = ('name', 'resonance', 'tp_times_yr')
        return {f'ep_{key}': value for key, value in rows[0].items() if key not in skip} if rows else {}

    def drop_arrays(self) -> None:
        """Freeze the CSV rows and release the arrays (a batch keeps only the arrays of the rows it draws)."""
        if self.frozen_numbers is None:
            self.frozen_numbers = self.numbers_row()
            self.frozen_episode_rows = self.episode_rows()
        self.angle = None
        if self.geometry is not None:
            self.geometry.centre.t_yr = np.empty(0)
            self.geometry.centre.centre_rad = np.empty(0)


def _stat(values, function, scale: float = 1.0) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(function(finite) * scale) if len(finite) else float('nan')


def _episode_row(analysis: LongPeriodAnalysis, run) -> Dict[str, Any]:
    return {
        'name': analysis.series.name,
        'resonance': analysis.series.key,
        'episode_id': run.episode_id,
        'accepted': bool(run.accepted),
        'subtype': str(run.shape),
        'reject_reason': str(run.reject_reason),
        't_start_yr': run.t_start_yr,
        't_end_yr': run.t_end_yr,
        'n_tp': run.n_tp,
        'tp_times_yr': join_list(f'{t:.6g}' for t in run.tp_times_yr),
        'period_med_yr': finite_or_none(run.period_med_yr),
        'period_min_yr': finite_or_none(run.period_min_yr),
        'period_max_yr': finite_or_none(run.period_max_yr),
        'half_amp_min_deg': finite_or_none(run.half_amp_min_rad, RAD_TO_DEG),
        'half_amp_med_deg': finite_or_none(run.half_amp_med_rad, RAD_TO_DEG),
        'half_amp_max_deg': finite_or_none(run.half_amp_max_rad, RAD_TO_DEG),
        'centre_mod_2pi_deg': finite_or_none(run.centre_mod_2pi_rad, RAD_TO_DEG),
        'centre_shift_deg': finite_or_none(run.centre_shift_rad, RAD_TO_DEG),
        'centre_shift_over_amp': finite_or_none(run.centre_shift_over_amp),
        'bands_overlap': bool(run.bands_overlap),
        'centre_estimates': run.centre_estimates,
        'centre_stationarity': str(run.centre_stationarity),
        'closed_cycles': run.closed_cycles,
        'closure_ratio_min': finite_or_none(run.closure_ratio_min),
        'centre_returns': run.centre_returns,
        'coverage_start_gap_P': finite_or_none(run.coverage_start_gap_p),
        'coverage_end_gap_P': finite_or_none(run.coverage_end_gap_p),
    }
