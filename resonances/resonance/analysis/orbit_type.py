"""The orbit's type in its librating episodes: a short text for the figure's header and the `orbit_type` column.

Descriptive only: it changes no status and no confidence; '' when no episode was accepted. A verdict whose angle word
neither librates nor is episodic (circulation, drift, ...) names no libration: only the side.

  co-orbital (prograde 1:1): the regime of theta = lambda - lambda_P over the librating episodes, from the force law's
      pieces (`coorbital.CoorbitalLaw.regime_sequence`), else from the episodes' turning points (`coorbital.piece_regimes`):
      'tadpole L4', 'tadpole L5', 'horseshoe', 'quasi-satellite';
      'compound: horseshoe → quasi-satellite' (the episodes' regimes in time order);
      'compound (horseshoe ↔ quasi-satellite)' (one episode passes both 0° and 180°)
  retrograde 1/-1: 'retrograde co-orbital, libration about 0°'
  other MMR: '[retrograde, ][interior | exterior, ]symmetric libration about 0° | 180°' or 'asymmetric libration
      about X°' (and its mirror 360° - X when another accepted episode librates there), ', moving centre' when the
      episode's centre shift exceeds MMR_L6_SHIFT_FULL x its mean amplitude (the L6 margin's own threshold);
      'libration about a moving centre (mean X°)' for an episode accepted about its moving well (psi frame);
      interior / exterior: two-body angles only, n/n_P = |k_P|/|k| above or below 1; 'forced wave about X°' when
      only forced waves of another argument were accepted (`Geometry.forced_waves`)
  any MMR whose verdict denies or doubts this resonance and names a cause (non_resonant or uncertain, induced,
      kinematic or implausible_strength; `not_this_resonance`): 'episodes about X° (not this resonance)'
  secular: 'σ libration about 180°'; ZLK: 'ω libration about 90°' (the measured centre)

The episode described is the decisive one, else the longest accepted. Symmetric: its centre is within
MMR_SUBRESONANCE_CENTRE_TOL_DEG of 0° or 180°, the tolerance of the subresonance test B for the same question (no
setting of its own); the same tolerance finds the mirror centre. Angles in [0°, 360°).

Pure: no Body/Simulation.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from resonances.resonance.analysis.geometry import FRAME_PSI, Episode, Geometry
from resonances.resonance.analysis.mmr import coorbital, preprocess
from resonances.resonance.analysis.mmr.models import ResonanceKind, ResonanceSpec
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.confidence import Cap
from resonances.resonance.analysis.verdict import LibrationOrigin, Reason, ResonanceType, Status

REGIME_NAMES = {
    coorbital.REGIME_TADPOLE_L4: 'tadpole L4',
    coorbital.REGIME_TADPOLE_L5: 'tadpole L5',
    coorbital.REGIME_HORSESHOE: 'horseshoe',
    coorbital.REGIME_QUASI_SATELLITE: 'quasi-satellite',
    coorbital.REGIME_COMPOUND: 'compound (horseshoe ↔ quasi-satellite)',
}
ANGLE_SYMBOL = {ResonanceType.SECULAR: 'σ', ResonanceType.LIDOV_KOZAI: 'ω'}


def accepted_runs(geometry: Optional[Geometry]) -> List[Episode]:
    """Every accepted run, in time order (the episodes before the last jump of a included)."""
    if geometry is None:
        return []
    return [run for run in geometry.runs if run.accepted]


def _degrees(angle_rad: float) -> float:
    return float(np.mod(round(float(np.degrees(angle_rad))), 360))


def _distance_deg(a_deg: float, b_deg: float) -> float:
    return abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)


def described_episode(runs: Sequence[Episode], decisive_id: Optional[int]) -> Optional[Episode]:
    """The decisive episode when it is accepted, else the longest accepted one; None without one."""
    finite = [run for run in runs if np.isfinite(run.centre_mod_2pi_rad)]
    decisive = [run for run in finite if run.episode_id == decisive_id]
    if decisive:
        return decisive[0]
    return max(finite, key=lambda run: run.t_end_yr - run.t_start_yr, default=None)


def centre_moves(shift_over_amp: float, shift_max: float) -> bool:
    """Whether an episode's centre moves: its shift between the first and last full cycle exceeds `shift_max` x its
    mean amplitude (MMR_L6_SHIFT_FULL, the L6 margin's own threshold). The one source of the header's ', moving centre'
    and of the verdict sentence (`verdict_text`); the subtype `stationary` is another test, the cycles' bands overlap
    (`geometry.centre_bands_overlap`), which a damped libration about a shifting centre can pass (2003 TX9 3E+2J-10:
    shift 1.45 x the mean amplitude, every band shares a point with the first wide cycle's)."""
    return bool(np.isfinite(shift_over_amp) and shift_over_amp > shift_max)


def libration_text(episode: Episode, runs: Sequence[Episode], tolerance_deg: float, shift_max: float) -> str:
    """'symmetric libration about 180°', 'asymmetric libration about 72° and 288°' (the mirror centre of another
    accepted episode), with ', moving centre' when the episode's centre shift / mean amplitude exceeds `shift_max`;
    'libration about a moving centre (mean 21°)' for an episode accepted in the frame of its moving well (psi; W20
    round D: 2013 AS27 31E-36 read 'symmetric about 0°' above a text 'about 21°')."""
    centre = _degrees(episode.centre_mod_2pi_rad)
    if episode.frame == FRAME_PSI:  # accepted about its moving well: the mean centre, not a nominal one
        return f'libration about a moving centre (mean {centre:.0f}°)'
    nominal = min((0.0, 180.0), key=lambda value: _distance_deg(centre, value))
    if _distance_deg(centre, nominal) <= tolerance_deg:
        text = f'symmetric libration about {nominal:.0f}°'
    else:
        mirrors = [_degrees(run.centre_mod_2pi_rad) for run in runs if np.isfinite(run.centre_mod_2pi_rad)]
        mirror = [value for value in mirrors if _distance_deg(value, 360.0 - centre) <= tolerance_deg]
        about = f'{centre:.0f}°' + (f' and {mirror[0]:.0f}°' if mirror else '')
        text = f'asymmetric libration about {about}'
    if centre_moves(episode.centre_shift_over_amp, shift_max):
        text += ', moving centre'
    return text


def coorbital_text(sequence: Sequence[str]) -> str:
    """'horseshoe', 'compound: horseshoe → quasi-satellite', ... from the episodes' regimes in time order."""
    names = [REGIME_NAMES.get(name, name.replace('_', ' ')) for name in sequence if name]
    if not names:
        return ''
    if len(names) == 1:
        return names[0]
    return 'compound: ' + ' → '.join(names)


def _coorbital_sequence(spec: ResonanceSpec, runs: Sequence[Episode], law: Optional[coorbital.CoorbitalLaw]) -> Sequence[str]:
    """The regimes of the librating episodes: the force law's pieces, else the episodes' turning points."""
    if law is not None and law.regime_sequence:
        return law.regime_sequence
    sign = 1 if spec.body_coefficient > 0 else -1  # theta = sign x sigma (`coorbital.coorbital_law`)
    theta = np.concatenate([sign * np.asarray(run.tp_values_rad, dtype=float) for run in runs])
    labels = np.concatenate([np.full(run.n_tp, i) for i, run in enumerate(runs)])
    return coorbital.piece_regimes(theta, labels)


def _side(spec: ResonanceSpec) -> str:
    """'interior' / 'exterior' of a two-body angle (n/n_P = |k_P|/|k|); '' otherwise."""
    if spec.kind == ResonanceKind.THREE_BODY:
        return ''
    k_planet, k_body = abs(spec.planet_coefficients[0]), abs(spec.body_coefficient)
    if k_planet == k_body or not k_body:
        return ''
    return 'interior' if k_planet > k_body else 'exterior'


LIBRATING_WORDS = ('libration', 'episodic')
NOT_THIS_WORD = 'episodes about'
# what makes the episodes of a non_resonant or uncertain verdict "not this resonance" in the header (W20 Q8b)
NOT_THIS_ORIGINS = (LibrationOrigin.INDUCED, LibrationOrigin.KINEMATIC)
NOT_THIS_STATUSES = (Status.NON_RESONANT, Status.UNCERTAIN)


def not_this_resonance(status: str, origin: str, reason: str, caps: Sequence[str]) -> bool:
    """Whether the header names the episodes as not this resonance's (W20 Q8b, author 23.10): a non_resonant or
    uncertain verdict whose episodes another resonance drives (induced), geometry bounds (kinematic), or whose period
    needs an implausible force (implausible_strength, reason or cap)."""
    if str(status) not in NOT_THIS_STATUSES:
        return False
    implausible = str(reason) == Reason.IMPLAUSIBLE_STRENGTH or str(Cap.IMPLAUSIBLE_STRENGTH) in [str(c) for c in caps]
    return str(origin) in NOT_THIS_ORIGINS or implausible


def mmr_orbit_type(
    spec: ResonanceSpec,
    geometry: Optional[Geometry],
    decisive_id: Optional[int],
    law: Optional[coorbital.CoorbitalLaw],
    settings: AnalysisSettings,
    angle_status: str = '',
    not_this: bool = False,
) -> str:
    """The orbit type of an MMR angle (module docstring); '' without an accepted episode. With `not_this`
    (`not_this_resonance`: the verdict denies or doubts this resonance and names a cause) the episodes are no
    libration of this angle: 'episodes about X° (not this resonance)' (author, W20 Q8b, 23.10). With `angle_status` (the
    verdict's word) other than libration or episodic, no libration is named (W20 round C: 'libration about 214°' in the
    header of a circulation, 2897 and 21028 drifts): the side alone. A forced wave of another argument
    (`Geometry.forced_waves`) is described only without a real episode, and as what sigma does there: 'forced wave
    about 214°' (W20 round D: 385266 16N-27, episodic through its forced waves only)."""
    runs = accepted_runs(geometry)
    waves = geometry.forced_wave_ids if geometry is not None else frozenset()
    episode = described_episode([run for run in runs if run.episode_id not in waves] or runs, decisive_id)
    if episode is None:
        return ''
    if angle_status and str(angle_status) not in LIBRATING_WORDS:
        return ', '.join(part for part in ('retrograde' if spec.retrograde else '', _side(spec)) if part)
    if not_this and episode.episode_id not in waves:  # a forced wave keeps its own, more specific name
        parts = [
            'retrograde' if spec.retrograde else '',
            _side(spec),
            f'{NOT_THIS_WORD} {_degrees(episode.centre_mod_2pi_rad):.0f}° (not this resonance)',
        ]
        return ', '.join(part for part in parts if part)
    coorbital_angle = preprocess.law_planet(spec) is not None  # the one 1:1 predicate (W20 round E, review D #6)
    if coorbital_angle and spec.retrograde:
        return f'retrograde co-orbital, libration about {_degrees(episode.centre_mod_2pi_rad):.0f}°'
    if coorbital_angle:
        return coorbital_text(_coorbital_sequence(spec, runs, law))
    libration = libration_text(episode, runs, settings.mmr_subresonance_centre_tol_deg, settings.mmr_l6_shift_full)
    if episode.episode_id in waves:
        libration = libration.replace('libration', 'forced wave', 1)
    parts = ['retrograde' if spec.retrograde else '', _side(spec), libration]
    return ', '.join(part for part in parts if part)


def long_period_orbit_type(resonance_type: ResonanceType, geometry: Optional[Geometry], decisive_id: Optional[int]) -> str:
    """'σ libration about 180°' (secular), 'ω libration about 90°' (ZLK): the described episode's measured centre;
    '' without an accepted episode."""
    episode = described_episode(accepted_runs(geometry), decisive_id)
    if episode is None:
        return ''
    return f'{ANGLE_SYMBOL.get(resonance_type, "σ")} libration about {_degrees(episode.centre_mod_2pi_rad):.0f}°'
