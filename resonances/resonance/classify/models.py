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

    def to_flat_dict(self) -> Dict[str, Any]:
        return flatten_dict(asdict(self))

    @classmethod
    def columns(cls) -> List[str]:
        return list(cls().to_flat_dict().keys())
