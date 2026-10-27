"""Rule G1q: the 1:1 quasi-satellite floor from the averaged potential (W20, F1).

The floor of a co-orbital angle is half its planet's linear tadpole period (`preprocess.libration_period_floor`: 111 yr
for the Earth) and its cutoff half of that (55.5 yr). A quasi-satellite librates about theta = lambda - lambda_P = 0
much faster: 8 yr for 2010 CG236 (e = 0.05), 51-55 yr for 2010 NT134 and 2023 FW13, 67-84 yr for 164207. The cutoff
erases it and the record looks like a stall. Its period comes from the same averaged direct + indirect potential the 1:1
force law reads: the well of R~(theta) about 0 (`coorbital.quasi_satellite_well`), omega_QS^2 = 3 mu n^2 R~''(theta_c).

When the angle (averaged over one orbit of the body, which drops the epicycle) stays inside that well for at least
(libration_min_turning_points / 2) P_QS, P_QS the longest small-amplitude period along that stretch, and the first pass
found no libration over the record, the record is analysed again with the floor mmr_coorbital_period_floor_factor x
P_QS,min (the cutoff follows, `preprocess.choose_cutoff`). Episodes centred outside the well keep the tadpole floor
(`geometry.FloorWell`), so the forced waves a short cutoff lets through in horseshoe parts stay rejected. The second
pass is kept when it holds a librating episode centred in the well (`analyse._quasi_satellite_pass`).

Pure: arrays in, no Body/Simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from resonances.resonance.analysis import geometry as geometry_module
from resonances.resonance.analysis.ledger import Ledger, Op
from resonances.resonance.analysis.mmr import coorbital as coorbital_module, preprocess as preprocess_module
from resonances.resonance.analysis.mmr.models import Context, MMRSeries, Preprocessed
from resonances.resonance.analysis.settings import AnalysisSettings

NAN = float('nan')
FLOOR_SETTING = 'mmr_coorbital_period_floor_factor'
# G1q's ledger names the explained verdict column reads (plotting/verdict_text.py)
QS_VISIT = 'G1q the orbit-averaged angle stays in the quasi-satellite well'
QS_FLOOR = 'G1q quasi-satellite floor from the averaged potential'
# geo_qs_pass: what the rule did
PASS_KEPT, PASS_REJECTED, PASS_RESAMPLE = 'kept', 'rejected', 'resample'


@dataclass(frozen=True)
class QuasiSatellite:
    """The quasi-satellite well of a 1:1 record and the numbers of rule G1q."""

    well: coorbital_module.QuasiSatelliteWell
    sign: int  # theta = sign x sigma
    visit_yr: float  # the longest stretch of the orbit-averaged theta inside the well
    visit_start_yr: float
    visit_needed_yr: float  # (libration_min_turning_points / 2) x the longest P_QS along that stretch
    floor_yr: float  # mmr_coorbital_period_floor_factor x P_QS,min
    tadpole_floor_yr: float  # the angle's own floor (first pass)
    tadpole_setting: str
    cutoff_yr: float  # the cutoff the quasi-satellite floor gives
    realizable: bool  # that cutoff can be filtered at the record's step (`preprocess.filter_realizable`)
    own_cutoff_yr: float  # the angle's own cutoff (first pass)
    block_min_yr: float = 0.0  # span / mmr_coorbital_qs_max_blocks: the shortest block of the pass's force law
    well_share: float = 0.0  # mmr_coorbital_qs_well_share (`geometry.FloorWell.min_share`)

    @property
    def period_min_yr(self) -> float:
        return self.well.period_min_yr

    @property
    def period_floor(self) -> Tuple[float, str]:
        return self.floor_yr, FLOOR_SETTING

    @property
    def elements_block_yr(self) -> float:
        """Blocks of the 1:1 law in this pass: one P_QS,min, not longer than the angle's own P_c. The elements of a
        quasi-satellite change over its libration (2010 CG236, P_QS = 6 yr: blocks of 55.5 yr give the force law R^2
        0.53 and K/K_theory 0.67 and fail the energy integral; blocks of 6 yr 0.71, 1.18 and pass); blocks of the
        1.5-yr cutoff follow the orbital terms of e and cost 646 tables for no better fit. Never shorter than
        `block_min_yr` (mmr_coorbital_qs_max_blocks blocks over the record: each may need a table)."""
        return float(max(min(self.own_cutoff_yr, self.period_min_yr), self.block_min_yr))

    def floor_well(self) -> geometry_module.FloorWell:
        """The well in the angle's own units (sigma = sign x theta): inside it the quasi-satellite floor holds, outside
        it (or where the well exists on less than mmr_coorbital_qs_well_share of an episode) the tadpole floor."""
        return geometry_module.FloorWell(
            self.well.t_yr,
            self.sign * self.well.centre_rad,
            self.well.halfwidth_rad,
            float(self.tadpole_floor_yr),
            self.tadpole_setting,
            self.well_share,
        )

    def visits(self, settings: AnalysisSettings, ledger: Ledger):
        """G1q's first comparison: the visit is long enough (a truthy ledger entry)."""
        return ledger.compare(
            QS_VISIT,
            self.visit_yr,
            Op.GE,
            self.visit_needed_yr,
            'yr',
            'geo_qs_visit_yr',
            setting='libration_min_turning_points',
        )

    def keeps_outside(self, second: int, first: int, ledger: Ledger):
        """G1q's third comparison: the pass at the quasi-satellite floor keeps as many librating episodes outside the
        well (`second`) as the first pass had (`first`) (a truthy ledger entry)."""
        return ledger.compare(
            'G1q the pass keeps the librating episodes outside the well', second, Op.GE, first, '', 'geo_qs_n_outside_episodes'
        )

    def lowers_floor(self, ledger: Ledger):
        """G1q's second comparison: the quasi-satellite floor is below the tadpole floor (a truthy ledger entry)."""
        return ledger.compare(
            QS_FLOOR,
            self.floor_yr,
            Op.LT,
            self.tadpole_floor_yr,
            'yr',
            'flt_qs_floor_yr',
            setting=FLOOR_SETTING,
        )


def pass_inputs(found: Optional[QuasiSatellite]) -> tuple:
    """(period floor, elements block [yr], floor well) of the quasi-satellite pass; (None, None, None) for any other."""
    if found is None:
        return None, None, None
    return found.period_floor, found.elements_block_yr, found.floor_well()


def _longest_run(inside: np.ndarray) -> Tuple[int, int]:
    """(first, last) index of the longest run of True; (-1, -1) without one."""
    padded = np.concatenate([[False], np.asarray(inside, dtype=bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    if not len(edges):
        return -1, -1
    starts, ends = edges[::2], edges[1::2] - 1
    longest = int(np.argmax(ends - starts))
    return int(starts[longest]), int(ends[longest])


def measure(
    series: MMRSeries, context: Context, prep: Preprocessed, step_yr: float, settings: AnalysisSettings
) -> Optional[QuasiSatellite]:
    """The well along the record and the visit of the angle (module docstring); None for an angle that is not a
    prograde co-orbital, with the rule switched off (mmr_coorbital_qs_floor), without the planet's mass or mean motion,
    or when no block of the record has a quasi-satellite well. The tables are those of the force law at the first
    pass (blocks of P_c), so they come from the cache."""
    spec = series.resonance
    planet = preprocess_module.coorbital_planet(spec)
    if planet is None or not settings.mmr_coorbital_qs_floor:
        return None
    inputs = preprocess_module.coorbital_inputs(series, context, planet)
    if inputs is None:
        return None
    mu, n_planet, elements = inputs
    t = np.asarray(series.t_yr, dtype=float)
    well = coorbital_module.well_along(
        t, series.e, series.inc, series.omega, series.Omega, planet, elements, mu, n_planet, prep.cutoff.cutoff_yr, settings
    )
    if well is None:
        return None
    sign = 1 if spec.body_coefficient > 0 else -1
    averaged = geometry_module.orbit_averaged(t, prep.angle_unwrapped, float(getattr(context, 'orbital_period_yr', NAN)))
    first, last = _longest_run(well.inside(sign * averaged))
    visit, start, needed = 0.0, NAN, NAN
    if first >= 0:
        visit, start = float(t[last] - t[first]), float(t[first])
        needed = settings.libration_min_turning_points / 2 * float(np.nanmax(well.period_yr[first : last + 1]))
    floor = settings.mmr_coorbital_period_floor_factor * well.period_min_yr
    cutoff, _, _ = preprocess_module.angle_cutoff(spec, context, step_yr, settings, (floor, FLOOR_SETTING))
    return QuasiSatellite(
        well=well,
        sign=sign,
        visit_yr=visit,
        visit_start_yr=start,
        visit_needed_yr=needed,
        floor_yr=float(floor),
        tadpole_floor_yr=float(prep.period_floor_yr),
        tadpole_setting=prep.period_floor_basis,
        cutoff_yr=float(cutoff.cutoff_yr),
        realizable=preprocess_module.filter_realizable(cutoff.cutoff_yr, step_yr),
        own_cutoff_yr=float(prep.cutoff.cutoff_yr),
        block_min_yr=float(t[-1] - t[0]) / settings.mmr_coorbital_qs_max_blocks,
        well_share=settings.mmr_coorbital_qs_well_share,
    )


def columns(found: Optional[QuasiSatellite], outcome: str = '') -> dict:
    """`flt_qs_period_yr` (P_QS,min), `flt_qs_floor_yr`, `flt_qs_cutoff_yr`, `geo_qs_visit_yr`, `geo_qs_visit_needed_yr`,
    `geo_qs_n_outside_episodes` (filled by the kept pass), `geo_qs_pass` (kept / rejected / resample, empty when the
    rule did not run); nan and empty without a well, and for an angle that librates over the record at the first pass
    or an aliased step (the well is not measured there)."""
    if found is None:
        return {
            'flt_qs_period_yr': NAN,
            'flt_qs_floor_yr': NAN,
            'flt_qs_cutoff_yr': NAN,
            'geo_qs_visit_yr': NAN,
            'geo_qs_visit_needed_yr': NAN,
            'geo_qs_n_outside_episodes': NAN,
            'geo_qs_pass': outcome,
        }
    return {
        'flt_qs_period_yr': found.period_min_yr,
        'flt_qs_floor_yr': found.floor_yr,
        'flt_qs_cutoff_yr': found.cutoff_yr,
        'geo_qs_visit_yr': found.visit_yr,
        'geo_qs_visit_needed_yr': found.visit_needed_yr,
        'geo_qs_n_outside_episodes': NAN,
        'geo_qs_pass': outcome,
    }
