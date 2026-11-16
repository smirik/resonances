"""The roles of the other resonance in an MMR verdict (W28 A3a; cards g04-6, g06-1, g06-2, g06-3, g05-3).

The decision tree names one other key (`StatusDecision.owner`): an owner, the accepted member of the multiplet, or an
owner that F7/P2 refused. The verdict splits it by its physical role, so that the `owner` column holds only a
resonance that makes the deciding episode:

- `owner`: an accepted owner the tree kept (origin induced: F7 cover >= mmr_owner_cover_min of every deciding episode
  and P2), in its libration regime (role owner, origin_detail owner_libration) or circulating with a forced wave at
  the period of the cycles (role driver, owner_forced_wave);
- `neighbours`: an accepted owner F7 or P2 refused (flags owner_outside_episode, owner_psi_too_fast): the body is in
  that resonance before or after the row's own episodes (1999 GL4: 1V-5 at 0.6-4.4 kyr, then 4J-1), not in both at
  once. Each block of its measured regimes as 'KEY t0–t1 before|during|after' [yr], where the block's midpoint lies
  against the hull of the row's own episodes (no tolerance: the times are printed with it);
- `resonant_argument`: rule C, the members of one commensurability are one resonance (origin subresonance): the
  member that librates, whose verdict the row takes (31935 9M-16+0+7: 9M-16+2+5); 'composite' when the multiplet
  selection (A6b) reads the decisive episode as a composite regime and the row's own libration stands (origin_detail
  composite_multiplet without a member, `decide.composite_regime_detail`);
- `cause_role` / `cause_term`: the role and the term behind the verdict: owner, driver or neighbour with the full key,
  or kinematic with the secular part of sigma that makes the cycles (285637 3M+7J-6+0+0-4: '-4varpi'; a ripple of the
  great inequality: '2J-5S');
- `commensurability`: the key without its varpi and node coefficients ('9M-16', '2M-6J-1', '1J-3r'): a statistic
  counts a body once per commensurability.

Nothing here changes a status, an angle word or a level."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from resonances.resonance.analysis.mmr.models import (
    OwnerResult,
    ResonanceSpec,
    interval_number,
    planet_letter,
    planet_terms,
)
from resonances.resonance.analysis.mmr.owner import connected_blocks
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CauseRole, Flag, LibrationOrigin, OriginDetail, join_list

COMPOSITE = 'composite'  # resonant_argument when the multiplet's regime is composite and no member is chosen (A6b)
GREAT_INEQUALITY = '2J-5S'  # the planetary term of a planet_ripple (2 n_J - 5 n_S, `context.great_inequality`)
BEFORE, DURING, AFTER = 'before', 'during', 'after'
REFUSED_FLAGS = (str(Flag.OWNER_OUTSIDE_EPISODE), str(Flag.OWNER_PSI_TOO_FAST))


@dataclass(frozen=True)
class Roles:
    owner: str = ''
    neighbours: str = ''
    resonant_argument: str = ''
    cause_role: CauseRole = CauseRole.NONE
    cause_term: str = ''
    commensurability: str = ''


def commensurability(spec: ResonanceSpec) -> str:
    """The key of the commensurability: the planets' and the body's mean-longitude coefficients only ('9M-16+0+7' ->
    '9M-16', '2M-6J-1+0+0+5' -> '2M-6J-1'); a retrograde key without its k ('1J-3r0' -> '1J-3r')."""
    if spec.retrograde:
        return spec.key[: spec.key.rindex('r') + 1]
    return f'{planet_terms(spec.planets, spec.planet_coefficients)}{spec.body_coefficient:+d}'


def secular_term(spec: ResonanceSpec) -> str:
    """The secular part of sigma, sigma minus its lambda part (`kinematics.lambda_part`), as a term: the body's
    p varpi, the planets' p_j varpi_j and the node terms ('-4varpi', '+2varpi+1varpi_M'); varpi* and Omega for a
    retrograde angle (its own body elements). '' for an angle without one."""
    body_varpi = 'varpi*' if spec.retrograde else 'varpi'
    terms = [(spec.body_varpi_coefficient, body_varpi), (spec.body_node_coefficient, 'Omega')]
    for index, name in enumerate(spec.planets):
        letter = planet_letter(name)
        terms += [(spec.planet_varpi_terms[index], f'varpi_{letter}'), (spec.planet_node_coefficient(index), f'Omega_{letter}')]
    return ''.join(f'{coefficient:+d}{name}' for coefficient, name in terms if coefficient)


def _own_windows(owner: OwnerResult, settings: AnalysisSettings) -> List[Tuple[float, float]]:
    """The windows of the row's own episodes (those the owner does not induce, `EpisodeCover.induced`), else the
    decisive window."""
    own = [
        (c.t0_yr, c.t1_yr)
        for c in owner.episode_covers
        if not c.induced(settings.mmr_owner_cover_min, settings.mmr_owner_psi_turns_per_cycle_max)
        and math.isfinite(c.cover)  # nan: the owner was not measured there (record edge), neither own nor induced
        and math.isfinite(c.t0_yr)
        and math.isfinite(c.t1_yr)
    ]
    if own:
        return own
    window = tuple(float(t) for t in owner.decisive_window)
    return [window] if all(math.isfinite(t) for t in window) else []


def position(block: Tuple[float, float], windows: Sequence[Tuple[float, float]]) -> str:
    """Where a block of the neighbour's regimes lies against the hull of `windows` (by its midpoint): before, during
    or after; '' without a window."""
    if not windows:
        return ''
    middle = 0.5 * (block[0] + block[1])
    if middle < min(w[0] for w in windows):
        return BEFORE
    if middle > max(w[1] for w in windows):
        return AFTER
    return DURING


def neighbours_text(owner: OwnerResult, settings: AnalysisSettings) -> str:
    """'KEY t0–t1 before|during|after;...' for a refused owner: one item per connected block of its measured regimes,
    the times rounded to the record's step (`interval_number`, as own_intervals); the key alone without intervals."""
    key = owner.accepted.spec.key
    blocks = connected_blocks(owner.intervals)
    if not blocks:
        return key
    windows = _own_windows(owner, settings)
    items = []
    for block in blocks:
        where = position(block, windows)
        span = f'{interval_number(block[0], owner.step_yr)}–{interval_number(block[1], owner.step_yr)}'
        items.append(' '.join(part for part in (key, span, where) if part))
    return join_list(items)


def _refused(flags: Sequence[str], owner: Optional[OwnerResult]) -> bool:
    return owner is not None and owner.accepted is not None and any(flag in REFUSED_FLAGS for flag in flags)


def verdict_roles(
    spec: ResonanceSpec,
    origin: LibrationOrigin | str,
    origin_detail: OriginDetail | str,
    key: str,
    flags: Sequence[str],
    owner: Optional[OwnerResult],
    settings: AnalysisSettings,
) -> Roles:
    """The roles of the decision's other key (`StatusDecision.owner`) by its origin (module docstring): induced -> owner
    or driver; joint -> owner with the role joint (A2b); subresonance -> resonant_argument; a refused owner (its flags)
    -> neighbours; kinematic -> the secular term. A row may carry an argument and a neighbour together; `cause_role` then names the neighbour.
    """
    origin, detail = str(origin), str(origin_detail)
    common = commensurability(spec)
    if origin == LibrationOrigin.INDUCED and key:
        role = CauseRole.DRIVER if detail == OriginDetail.OWNER_FORCED_WAVE else CauseRole.OWNER
        return Roles(owner=key, cause_role=role, cause_term=key, commensurability=common)
    if origin == LibrationOrigin.JOINT and key:  # A2b: the other resonance of a joint capture
        return Roles(owner=key, cause_role=CauseRole.JOINT, cause_term=key, commensurability=common)
    argument = key if origin == LibrationOrigin.SUBRESONANCE else (COMPOSITE if detail == OriginDetail.COMPOSITE_MULTIPLET else '')
    if _refused(flags, owner):
        neighbour = owner.accepted.spec.key
        return Roles(
            neighbours=neighbours_text(owner, settings),
            resonant_argument=argument,
            cause_role=CauseRole.NEIGHBOUR,
            cause_term=neighbour,
            commensurability=common,
        )
    if origin == LibrationOrigin.KINEMATIC:
        term = GREAT_INEQUALITY if detail == OriginDetail.PLANET_RIPPLE else secular_term(spec)
        return Roles(cause_role=CauseRole.KINEMATIC, cause_term=term, commensurability=common)
    return Roles(resonant_argument=argument, commensurability=common)
