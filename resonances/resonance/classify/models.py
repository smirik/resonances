from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from typing import Optional, List, Dict, Any
import numpy as np


class ResonanceStatus(IntEnum):
    LIBRATION = 2
    TRANSIENT = 1
    NON_RESONANT = 0
    TRANSIENT_UNCERTAIN = -1
    LIBRATION_UNCERTAIN = -2
    NEAR_SEPARATRIX = -3
    PROBABLY_SLOW_CIRCULATION = -4
    PROBABLY_NEAR_SEPARATRIX = -5
    UNCERTAIN = -9
    CHAOTIC = -99


@dataclass
class SegmentMetrics:
    """Metrics for a segment of resonant angle time series. Defaults allow use as placeholder."""

    revolutions_true: float = np.nan  # Number of true revolutions
    trend_to_oscillation: float = np.nan  # Ratio of trend to oscillation

    sign_dominance: float = np.nan  # Dominance of the sign of σ̇
    sigma_dot_ratio: float = np.nan  # |⟨σ̇⟩| / std(σ̇)

    phi_rad: float = np.nan  # Mean angle in radians
    phi_deg: float = np.nan  # Mean angle in degrees
    R: float = np.nan  # Circular variance
    revolutions: float = np.nan  # Number of revolutions
    amplitude: float = np.nan  # Amplitude of σ
    n_zero_crossings: Optional[int] = None  # Number of zero crossings
    cv_intervals: float = np.nan  # CV of the intervals between zero crossings

    ls_snr: Optional[float] = None  # SNR of the periodogram
    ls_fap: Optional[float] = None  # FAP of the periodogram
    ls_period: Optional[float] = None  # Period from the periodogram

    mean_sigma_dot: Optional[float] = None  # Mean of the derivative of σ
    std_sigma_dot: Optional[float] = None  # Standard deviation of the derivative of σ


@dataclass
class SegmentCounts:
    """Counts of good and reasonable segments per window step and totals."""

    n_good_total: int = 0
    n_reasonable_total: int = 0
    n_good_0_1: int = 0
    n_reasonable_0_1: int = 0
    n_good_0_2: int = 0
    n_reasonable_0_2: int = 0
    n_good_0_3: int = 0
    n_reasonable_0_3: int = 0


@dataclass
class ClassifyParams:
    """All thresholds for the classification algorithm.

    Controls the decision logic in classify_from_metrics. Each field corresponds
    to one threshold used in the classification tree. Defaults match the
    hardcoded values that were previously embedded in classify_resonance.

    Attributes:
        window_steps: Window length percentages for segment analysis.
        window_step_percentage: Step size as fraction of total length.
        rev_libration: Max revolutions_true for global libration branch.
        tto_pure_libration: Max TTO for "pure libration" subtype.
        tto_partial_libration: Max TTO for "partial libration" subtype.
        tto_non_resonant: TTO above which classified as non-resonant circulation.
        tto_transient_global: Max global TTO to still qualify as transient.
        tto_near_separatrix: Max global TTO for near-separatrix with reasonable segments.
        good_seg_max_rev: Max revolutions_true for a segment to count as "good".
        good_seg_max_tto: Max TTO for a segment to count as "good".
        reasonable_seg_max_rev_1: Max rev (condition 1) for "reasonable" segment.
        reasonable_seg_max_tto_1: Max TTO (condition 1) for "reasonable" segment.
        reasonable_seg_max_rev_2: Max rev (condition 2) for "reasonable" segment.
        reasonable_seg_max_tto_2: Max TTO (condition 2) for "reasonable" segment.
    """

    window_steps: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    window_step_percentage: float = 0.05

    # Global thresholds
    rev_libration: float = 1.0
    tto_pure_libration: float = 0.5
    tto_partial_libration: float = 2.5
    tto_non_resonant: float = 6.0
    tto_transient_global: float = 3.0
    tto_near_separatrix: float = 6.0

    # Segment quality thresholds
    good_seg_max_rev: float = 1.0
    good_seg_max_tto: float = 0.5
    reasonable_seg_max_rev_1: float = 2.0
    reasonable_seg_max_tto_1: float = 1.0
    reasonable_seg_max_rev_2: float = 1.0
    reasonable_seg_max_tto_2: float = 1.5


def flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
    """Recursively flatten nested dicts and enums."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        elif isinstance(v, Enum):
            items.append((new_key, v.value))
        else:
            items.append((new_key, v))
    return dict(items)


@dataclass
class ResonanceClassifyResult:
    status: ResonanceStatus = ResonanceStatus.UNCERTAIN
    type: str = ''
    subtype: str = ''
    confidence: Optional[str] = None
    comments: Optional[str] = None
    metrics: SegmentMetrics = field(default_factory=SegmentMetrics)
    segment_counts: SegmentCounts = field(default_factory=SegmentCounts)

    def to_flat_dict(self) -> Dict[str, Any]:
        return flatten_dict(asdict(self))

    @classmethod
    def columns(cls) -> List[str]:
        return list(cls().to_flat_dict().keys())
