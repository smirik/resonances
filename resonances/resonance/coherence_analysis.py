"""Body-level orchestration of the cross-spectral analysis.

`cross_spectrum.py` holds the pure signal processing; this module knows about
`Body`, `Resonance` and `SimulationConfig`, and resolves which pairs to analyse
from the per-resonance-type presets.

Nothing here decides a resonance status. The Lidov-Kozai status decision moved to
`resonance/omega_free_gate.py`, which works on the forced/free split rather than on
cross spectra; what the e-i coherence contributes is `confirmed_ei`, independent
corroboration that eccentricity and inclination really trade at constant H.

The move was forced by baselines. The coherence band tops out at `T / (5 * n_seg)`,
which is 25 kyr on a 500 kyr run at four segments, while real Lidov-Kozai lines in
the reference set sit at 27-58 kyr — so on the short integrations this package now
targets, the check would be judging on a line outside its own validity band.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

from resonances.logger import logger
from resonances.resonance.cross_spectrum import (
    CoherenceLine,
    CrossSpectrum,
    DEFAULT_PAIRS,
    DEFAULT_PERIOD_MIN,
    DEFAULT_PHASE_TARGETS,
    cross_spectrum,
    decisive_level,
    levels_agreeing,
    parse_pair,
    phase_matches,
)


class CoherenceCheck(StrEnum):
    """Outcome of a phase-criterion check, reported alongside the boolean verdict.

    PASSED / FAILED are measurements. The remaining values say why no measurement
    was made, so that a missing verdict is never mistaken for a negative one.
    """

    PASSED = 'passed'
    FAILED = 'failed'
    WEAK_EXCHANGE = 'weak_exchange'  # phase fits, but e and i do not actually trade in the time domain
    SHORT_BASELINE = 'short_baseline'  # no segmentation level had a usable period band
    NOT_APPLICABLE = 'not_applicable'  # the angle does not librate, so there is nothing to confirm
    DISABLED = 'disabled'
    NO_TARGET = 'no_target'  # no phase target configured for this pair


@dataclass
class PairAnalysis:
    """Everything computed for one pair of series of one body-resonance."""

    pair: str
    levels: List[CrossSpectrum]
    decisive: Optional[CrossSpectrum] = None
    top_line: Optional[CoherenceLine] = None
    phase_target: Optional[Tuple[float, float]] = None
    check: CoherenceCheck = CoherenceCheck.NO_TARGET
    n_levels_valid: int = 0
    n_levels_passed: int = 0
    r_broadband: float = float('nan')  # Pearson r of the two raw series, as a coarse sanity check

    @property
    def r_band(self) -> float:
        """Band-passed correlation at the decisive line, or NaN if there is none."""
        return self.top_line.r_band if self.top_line is not None else float('nan')

    @property
    def coherent(self) -> Optional[bool]:
        """True/False when the criterion was evaluated, None when it could not be."""
        if self.check == CoherenceCheck.PASSED:
            return True
        if self.check == CoherenceCheck.FAILED:
            return False
        return None


def resolve_series(body, resonance, key: str) -> Optional[np.ndarray]:
    """Map a series key onto the array on `body`.

    'sigma' is the RAW unwrapped resonant angle — `prepare_angles` has already
    unwrapped it. The filtered variant must not be used: the low-pass filter
    reshapes exactly the frequency content the cross spectrum is measuring.
    """
    if key == 'sigma':
        return body.angles_unwrapped.get(resonance.to_s())
    return {'a': body.axis, 'e': body.ecc, 'i': body.inc}.get(key)


def _config_lookup(config, attr: str, resonance_type: str, fallback: dict):
    """Read a per-resonance-type setting, falling back to the built-in preset.

    A user override for one resonance type replaces the preset for that type only;
    types the user did not mention keep their defaults.
    """
    override = getattr(config, attr, None)
    if isinstance(override, dict) and resonance_type in override:
        return override[resonance_type]
    return fallback.get(resonance_type)


def pairs_for(config, resonance_type: str) -> List[str]:
    """Which pairs to analyse for this resonance type."""
    pairs = _config_lookup(config, 'coherence_pairs', resonance_type, DEFAULT_PAIRS)
    return list(pairs) if pairs else []


def phase_target_for(config, resonance_type: str, pair: str) -> Optional[Tuple[float, float]]:
    """The (target_deg, tolerance_deg) a pair's phase is expected to match, if any."""
    targets = _config_lookup(config, 'coherence_phase_targets', resonance_type, DEFAULT_PHASE_TARGETS) or {}
    target = targets.get(pair)
    if target is None:
        return None
    # Restored simulations deliver JSON lists rather than tuples.
    return float(target[0]), float(target[1])


def period_band_for(config, resonance_type: str) -> Tuple[Optional[float], Optional[float]]:
    """Search band for this resonance type; an explicit config value wins over the preset."""
    period_min = config.coherence_period_min
    if period_min is None:
        period_min = DEFAULT_PERIOD_MIN.get(resonance_type)
    return period_min, config.coherence_period_max


def analyse_pair(
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    pair: str,
    config,
    phase_target,
    period_min: Optional[float] = None,
    period_max: Optional[float] = None,
    psd_cache=None,
) -> PairAnalysis:
    """Run the multi-level cross spectrum for one pair and apply its phase criterion."""
    levels = [
        cross_spectrum(
            times,
            x,
            y,
            n_seg=n_seg,
            pair=pair,
            alpha=config.coherence_alpha,
            period_min=period_min,
            period_max=period_max,
            window=config.coherence_window,
            max_lines=config.coherence_max_lines,
            psd_cache=psd_cache,
        )
        for n_seg in sorted(config.coherence_n_segments)
    ]

    analysis = PairAnalysis(pair=pair, levels=levels, phase_target=phase_target)
    analysis.n_levels_valid = sum(1 for level in levels if level.valid)
    analysis.decisive = decisive_level(levels)
    if x.std() > 0 and y.std() > 0:
        analysis.r_broadband = float(np.corrcoef(x, y)[0, 1])

    if analysis.decisive is None:
        analysis.check = CoherenceCheck.SHORT_BASELINE
        return analysis

    analysis.top_line = analysis.decisive.top_line
    if phase_target is None:
        analysis.check = CoherenceCheck.NO_TARGET
        return analysis

    target_deg, tolerance_deg = phase_target
    if analysis.top_line is None:
        analysis.check = CoherenceCheck.FAILED
        return analysis

    passed = phase_matches(analysis.top_line.phase_deg, target_deg, tolerance_deg)
    analysis.check = CoherenceCheck.PASSED if passed else CoherenceCheck.FAILED
    analysis.n_levels_passed = sum(
        1
        for level in levels
        if level.valid and level.top_line is not None and phase_matches(level.top_line.phase_deg, target_deg, tolerance_deg)
    )
    return analysis


def analyse_resonance(body, resonance, config, psd_cache=None) -> Dict[str, PairAnalysis]:
    """Analyse every configured pair for one body-resonance combination."""
    results: Dict[str, PairAnalysis] = {}
    times = body.times / (2 * np.pi)
    period_min, period_max = period_band_for(config, resonance.type)

    for pair in pairs_for(config, resonance.type):
        try:
            x_key, y_key = parse_pair(pair)
        except ValueError as e:
            logger.warning(f"{body.name}/{resonance.to_s()}: {e}")
            continue

        x, y = resolve_series(body, resonance, x_key), resolve_series(body, resonance, y_key)
        if x is None or y is None:
            logger.warning(f"{body.name}/{resonance.to_s()}: no data for coherence pair '{pair}'")
            continue

        results[pair] = analyse_pair(
            times,
            x,
            y,
            pair,
            config,
            phase_target_for(config, resonance.type, pair),
            period_min=period_min,
            period_max=period_max,
            psd_cache=psd_cache,
        )

    return results


@dataclass
class ZLKGate:
    """Outcome of the Lidov-Kozai e-i exchange check for one body-resonance.

    Diagnostic since the free-omega gate took over the status decision — see
    `resonance/omega_free_gate.py`. What survives here is the independent piece of
    evidence: a genuine Lidov-Kozai resonance trades eccentricity for inclination at
    constant H, which shows up as a coherent antiphase e-i line. Absence of that line no
    longer demotes anything on its own, because at the baselines this package now targets
    the coherence band tops out below the real Lidov-Kozai periods.
    """

    check: CoherenceCheck
    coherent: Optional[bool] = None
    line_period: Optional[float] = None
    line_phase: Optional[float] = None
    line_gamma2: Optional[float] = None
    line_r_band: Optional[float] = None

    @classmethod
    def not_applicable(cls, check: CoherenceCheck) -> 'ZLKGate':
        return cls(check=check)

    @property
    def confirmed_ei(self) -> Optional[bool]:
        """Whether e and i were seen exchanging; None when the check could not run."""
        return self.coherent


ZLK_PAIR = 'e-i'


def evaluate_ei_exchange(analyses: Dict[str, PairAnalysis], config) -> ZLKGate:
    """Measure whether e and i actually trade at the strongest coherent line.

    Two conditions must both hold, because neither is sufficient on its own:

      1. the strongest coherent e-i line sits within the antiphase tolerance;
      2. e and i band-passed around that line actually anticorrelate, at least
         `zlk_min_anticorrelation`.

    Condition 2 exists because a cross-phase read off a single frequency bin can be
    misleading when close lines blend into it. On 591986 the phase alone says antiphase
    on a 5 Myr window and +124 degrees on 7.5 Myr and longer, while the band-passed
    correlation stays near -0.2 throughout — there is no exchange in any window.

    The result never changes a status. PASSED / FAILED are measurements; the remaining
    values say why no measurement was made, so a missing verdict is never read as a
    negative one.
    """
    if not getattr(config, 'zlk_coherence_check', True):
        return ZLKGate.not_applicable(CoherenceCheck.DISABLED)

    analysis = analyses.get(ZLK_PAIR)
    if analysis is None or analysis.phase_target is None:
        return ZLKGate.not_applicable(CoherenceCheck.NO_TARGET)
    if analysis.check == CoherenceCheck.SHORT_BASELINE:
        return ZLKGate.not_applicable(CoherenceCheck.SHORT_BASELINE)

    line = analysis.top_line
    check = analysis.check
    if check == CoherenceCheck.PASSED and not _exchanges(analysis.r_band, config):
        check = CoherenceCheck.WEAK_EXCHANGE

    return ZLKGate(
        check=check,
        coherent=check == CoherenceCheck.PASSED,
        line_period=line.period if line else None,
        line_phase=line.phase_deg if line else None,
        line_gamma2=line.gamma2 if line else None,
        line_r_band=line.r_band if line else None,
    )


def _exchanges(r_band: float, config) -> bool:
    """Whether e and i anticorrelate strongly enough to call it an exchange.

    The -0.5 default is PROVISIONAL: it sits between the two objects measured so far
    (591986 reaches -0.38 at worst and is not in resonance; 162474 reaches -0.79 and is),
    but two objects do not calibrate a threshold. Revisit once a larger sample exists.
    An unmeasurable correlation (NaN) is treated as no evidence of exchange.
    """
    threshold = getattr(config, 'zlk_min_anticorrelation', 0.5)
    return bool(np.isfinite(r_band) and r_band <= -abs(threshold))


SUMMARY_COLUMNS = (
    'confirmed_ei',
    'zlk_coherent',
    'zlk_check',
    'zlk_line_period',
    'zlk_line_phase',
    'zlk_line_gamma2',
    'zlk_line_r_band',
)


def summary_fields(analyses: Dict[str, PairAnalysis], gate: Optional[ZLKGate]) -> dict:
    """The coherence columns of one `summary.csv` row.

    `confirmed_ei` is the spec's flag: e and i were seen trading at the strongest coherent
    line. It is corroboration next to the status, not an input to it.
    """
    if gate is None:
        return {column: None for column in SUMMARY_COLUMNS}
    return {
        'confirmed_ei': gate.confirmed_ei,
        'zlk_coherent': gate.coherent,
        'zlk_check': str(gate.check),
        'zlk_line_period': gate.line_period,
        'zlk_line_phase': gate.line_phase,
        'zlk_line_gamma2': gate.line_gamma2,
        'zlk_line_r_band': gate.line_r_band,
    }


def coherence_rows(body_name: str, resonance_key: str, analyses: Dict[str, PairAnalysis]) -> List[dict]:
    """Flatten every detected line into `coherence.csv` rows, one row per line."""
    rows = []
    for pair, analysis in analyses.items():
        target = analysis.phase_target
        for level in analysis.levels:
            if not level.valid:
                continue
            for rank, line in enumerate(level.lines):
                rows.append(
                    {
                        'body': body_name,
                        'resonance': resonance_key,
                        'pair': pair,
                        'n_seg': line.n_seg,
                        'rank': rank,
                        'decisive': level is analysis.decisive and rank == 0,
                        'period': line.period,
                        'gamma2': line.gamma2,
                        'gamma2_crit': level.gamma2_crit,
                        'phase_deg': line.phase_deg,
                        'phase_std': line.phase_std,
                        'amp_ratio': line.amp_ratio,
                        'r_band': line.r_band,
                        'r_broadband': analysis.r_broadband,
                        'cross_amplitude': line.cross_amplitude,
                        'n_bins': line.n_bins,
                        'period_min': level.period_min,
                        'period_max': level.period_max,
                        'phase_target': target[0] if target else None,
                        'phase_tolerance': target[1] if target else None,
                        'n_levels_valid': analysis.n_levels_valid,
                        'n_levels_passed': analysis.n_levels_passed,
                        'n_levels_agreeing': levels_agreeing(analysis.levels, line) if rank == 0 else None,
                    }
                )
    return rows
