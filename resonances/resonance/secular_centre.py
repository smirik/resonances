"""The centre check: is a bounded secular angle a libration, or a drift the baseline is too short to see turn?

The classifier calls an angle LIBRATION as soon as it stays within a turn; an angle with a small
divisor can drift 110 deg in 10 Myr and never turn (2110 in g-g6-2g7+2g8). The *centre* — the angle
smoothed over `secular_centre_window` of the baseline — tells them apart:

    the centre turns back (a turning point of at least `secular_turning_prominence` deg) -> LIBRATION
    the centre stays within `secular_centre_drift` deg (a libration faster than the window) -> LIBRATION
    the centre moves monotonically by more than that, within a turn     -> BASELINE_TOO_SHORT (-6)

Runs only on LIBRATION and only for secular resonances. -6 is a status of its own, not the classifier's
PROBABLY_SLOW_CIRCULATION (-4, trend dominating the oscillation): bounded over the baseline, and only
a longer integration can tell libration from circulation. The record goes to summary.csv like the
free-omega gate's. The choice of the window is discussed in docs/plots.md, "Secular diagnostics".
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from resonances.resonance import oscillations as osc
from resonances.resonance.classify.models import ResonanceStatus

SUMMARY_COLUMNS = ('centre_check', 'centre_drift_deg', 'centre_turning_points')


@dataclass
class CentreCheck:
    """The centre of one secular angle and what the check made of it."""

    line: np.ndarray  # the centre, degrees, NaN within half a window of either end
    turning: np.ndarray  # indices of its turning points
    drift_deg: float  # its range
    lowered: bool

    @property
    def comment(self) -> str:
        if not self.lowered:
            return ''
        return f'centre: drifts {self.drift_deg:.0f} deg without turning back; a longer integration would tell'


def analyse(times: np.ndarray, sigma_unwrapped: np.ndarray, config) -> CentreCheck:
    """The centre of an unwrapped angle (times in years) and whether it only drifts."""
    line = np.degrees(osc.smooth(times, sigma_unwrapped, config.secular_centre_window * abs(times[-1] - times[0])))
    drift = float(np.nanmax(line) - np.nanmin(line)) if np.isfinite(line).any() else 0.0
    turning = osc.turning_points(line, config.secular_turning_prominence)
    lowered = len(turning) == 0 and config.secular_centre_drift < drift < 360.0
    return CentreCheck(line=line, turning=turning, drift_deg=drift, lowered=lowered)


def check_centre(status: ResonanceStatus, times: np.ndarray, sigma_unwrapped: np.ndarray, config):
    """(status, CentreCheck or None): LIBRATION becomes BASELINE_TOO_SHORT when its centre only drifts."""
    if status != ResonanceStatus.LIBRATION:
        return status, None
    check = analyse(times, sigma_unwrapped, config)
    return (ResonanceStatus.BASELINE_TOO_SHORT if check.lowered else status), check


def summary_fields(check: Optional[CentreCheck]) -> dict:
    """The centre-check columns of one summary.csv row (empty unless the check ran)."""
    if check is None:
        return {column: None for column in SUMMARY_COLUMNS}
    return {
        'centre_check': 'lowered' if check.lowered else 'kept',
        'centre_drift_deg': round(check.drift_deg, 1),
        'centre_turning_points': len(check.turning),
    }
