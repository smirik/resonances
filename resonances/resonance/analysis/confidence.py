"""Confidence of a verdict from its margins (shared rule; SPEC §7.2 with the author's 2026-10-15 answers).

Each margin is full | thin | fail | not_measured | not_applicable and belongs to a class
(`MarginClass`): decisive margins never fail in a final verdict (the verdict would be another),
support margins may fail and each failure lowers the level, cap margins limit the level.

  confidence_raw:
    certain — every applicable margin full, no decisive not_measured, no cap
    high    — no fail; any number of thin; a not_measured decisive physics margin counts as thin,
              but when every physics margin is not_measured the level is at most medium
    medium  — exactly one failed support margin
    low     — two or more failed support margins
  confidence = min(raw, caps): the level shown (certain included). uncertain and invalid: no level.

The calibration cell (status, angle_status, reason) of a verdict is only reported: its author-label counts in the
cells file (n_examples, n_errors and the Wilson 95 % interval of the accuracy, `CellRecord`) go to summary.csv next
to the level and never change it (author, 2026-10-16: the current labels are enough, show the real levels now).

Every applicable not_measured margin counts as thin (a check that could not be made is not a pass).

Which margins are "physics" is the caller's list (MMR: decide.PHYSICS_MARGINS = L4, R0, Z, W, F): the measurements
that can tell the angle's own force from an oscillation that another resonance induces in a - the force law, the
potential with a, the zone, the owner search (no owner found) and the strength. The identity (L3), B1 and one pendulum
(L5) are not in it: they show that a carries the oscillation, which an induced libration shows as well (at small
amplitude the identity alone makes a_dot ~ -(sigma - sigma_s) and T0 constant; PHYSICS N5). So a clean small-amplitude
Hilda (L4, R0 not measured below MMR_A_RESPONSE_MIN_AMP_DEG) is capped at medium only until the owner search or the
strength check has measured something; once the owner search found no owner it is not capped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from resonances.resonance.analysis.ledger import Margin
from resonances.resonance.analysis.verdict import (
    CONFIDENCE_ORDER,
    Confidence,
    MarginClass,
    MarginValue,
    Status,
    angle_status_from_text,
    finite_or_none,
    lower_confidence,
)

PACKAGE_CELLS = Path(__file__).resolve().parents[2] / 'data' / 'confidence_cells.json'
NAN = float('nan')


class Cap(StrEnum):
    """Limits of the level; the value is also the `caps` column entry."""

    THREE_TP_EPISODE = 'three_tp_episode'
    # an exit (entry) with less than libration_coverage_periods of record after (before) it, on a verdict that rests on
    # that edge: a libration, or an episodic angle without a revolution > 2 pi between its librating episodes
    EXIT_SUSPECTED = 'exit_suspected'
    ENTRY_SUSPECTED = 'entry_suspected'
    # an episodic angle whose crossings no mechanism of the separatrix model explains (SPEC R3); not for a 1:1 angle
    # (MMR_COORBITAL_LIFTS_UNEXPLAINED): the pendulum separatrix model does not apply there (coorbital.py), its
    # crossings are close encounters (2026 FW12: a capture at an Earth approach of ~0.65 r_H)
    TRANSIENT_UNEXPLAINED = 'transient_unexplained'
    # the verdict rests on one cycle closed only by its phase return, two turning points (`geometry.phase_return`;
    # LIBRATION_RETURN_CYCLE_CAPS_MEDIUM): less than the three of a closed full cycle (2016 CA138, 3753 at 1 kyr)
    ONE_RETURN_CYCLE = 'one_return_cycle'
    FORCE_LAW_DISAGREES = 'force_law_disagrees'
    SLOW_LIBRATION_NOT_EXCLUDED = 'slow_libration_not_excluded'
    PLANETS_MISSING = 'planets_missing'
    OWNER_SEARCH_SKIPPED = 'owner_search_skipped'
    # the owner search ran on catalogue longitudes of some planet (not saved): approximate, whatever it found
    CATALOGUE_LONGITUDES = 'catalogue_longitudes'
    NODAL_SUBRESONANCE_CANDIDATE = 'nodal_subresonance_candidate'
    SUBRESONANCE_TIE = 'subresonance_tie'
    SUBRESONANCE_T6_ONLY = 'subresonance_t6_only'  # the scan ran on the theory trigger alone
    SUBRESONANCE_THEORY_UNCONFIRMED = 'subresonance_theory_unconfirmed'  # test D not measured or failed
    # no member can be told: beta = varpi_P - varpi librates, or several members pass with test A unmeasured
    SUBRESONANCE_UNDETERMINED = 'subresonance_undetermined'
    JITTER_LOW_E = 'jitter_low_e'
    ALL_PHYSICS_NOT_MEASURED = 'all_physics_not_measured'
    ZONE_NOT_MEASURED = 'zone_not_measured'
    L2_WITHOUT_ZONE = 'l2_without_zone'
    ONSET_FAILED = 'onset_failed'
    # a transient whose crossings all fall among cycles inside the model's separatrix zone ('separatrix_chaos' before
    # 2026-10-17)
    EPISODES_IN_SEPARATRIX_ZONE = 'episodes_in_separatrix_zone'
    # near_separatrix of the angle's geometry (G3, pattern A or B, no episode) while the zone is not measured
    # (check_mmr_separatrix_zone not pass): the pattern alone does not place the angle at the separatrix (564160 3N-4)
    NEAR_SEPARATRIX_WITHOUT_ZONE = 'near_separatrix_without_zone'
    # the force law and the potential with a are not measured, and neither is the strength by direct averaging: the
    # status rests on the geometry alone (rule 5u/6); the owner search cannot lift it (it excludes a foreign force,
    # it does not show this angle's own)
    GEOMETRY_ONLY = 'geometry_only'
    # rule 5u/6 as geometry_only, but the strength on the order-of-magnitude scale (required / mu_i mu_j e^q) lies
    # within [1/x, x] (mmr_strength_check_ratio): a scale that agrees, not a computed force (W22 R-F; 2348, 24820)
    STRENGTH_SCALE_ONLY = 'strength_scale_only'
    # non_resonant because the episode needs an implausibly strong force (rule 5s): a computation against an estimate
    # or a first-order average, not a measurement of the angle's dynamics
    IMPLAUSIBLE_STRENGTH = 'implausible_strength'
    # the status rests on librating episodes before the last jump of the slow variable: their response is measured
    # on them (rule R), but their energy (potential with a, zone) is not (the last regime's is not theirs)
    EPISODE_BEFORE_JUMP = 'episode_before_jump'
    # rule R: a witness episode's force law holds while another librating episode's is measured and fails
    EPISODE_FORCE_LAW_MIXED = 'episode_force_law_mixed'
    # a verdict without libration while a run of turning points is short of an episode by at most
    # libration_near_episode_tp_deficit TPs: one more turn back would make it transient
    NEAR_EPISODE = 'near_episode'
    # the output step folds short-period terms of the elements into slow ones (dt >= min(P_orb, P_syn) / 2): the
    # record cannot measure how much of them reached sigma_f and a_f (only a finely sampled pilot can)
    SHORT_PERIOD_ALIASED = 'short_period_aliased'
    # an induced libration whose owner does not exclude a second capture of sigma by its own term (x = Psi / u sweeps
    # less than mmr_owner_capture_sweep_deg, or x_dot^2 follows a potential in x): both resonances may hold at once
    SECOND_CAPTURE_UNRESOLVED = 'second_capture_unresolved'
    # a 1:1 verdict on one cycle (three_tp_episode) whose co-orbital energy integral, read by rule R
    # (rsp_coorb_energy_R2), fails (< mmr_force_r2_fail): one cycle does not show the energy conserved (W22 R-WJ, 2024 WJ)
    ONE_CYCLE_ENERGY_NOT_CONSERVED = 'one_cycle_energy_not_conserved'
    # G2a cut a run at a step of its libration action with fewer than LIBRATION_JUMP_CYCLES energy cycles on a side, or
    # one disjoint cycle of the larger swings (test 7 in sample) (`geometry.ActionStep.provisional`): one cycle shows
    # the other level, two confirm it (W24, Codex)
    ACTION_STEP_PROVISIONAL = 'action_step_provisional'


CAP_CEILING = {
    Cap.THREE_TP_EPISODE: Confidence.HIGH,
    Cap.ZONE_NOT_MEASURED: Confidence.HIGH,
    Cap.EXIT_SUSPECTED: Confidence.MEDIUM,
    Cap.ENTRY_SUSPECTED: Confidence.MEDIUM,
    Cap.TRANSIENT_UNEXPLAINED: Confidence.MEDIUM,
    Cap.ONE_RETURN_CYCLE: Confidence.MEDIUM,
    Cap.FORCE_LAW_DISAGREES: Confidence.MEDIUM,
    Cap.SLOW_LIBRATION_NOT_EXCLUDED: Confidence.MEDIUM,
    Cap.PLANETS_MISSING: Confidence.MEDIUM,
    Cap.OWNER_SEARCH_SKIPPED: Confidence.MEDIUM,
    Cap.CATALOGUE_LONGITUDES: Confidence.MEDIUM,
    Cap.NODAL_SUBRESONANCE_CANDIDATE: Confidence.MEDIUM,
    Cap.SUBRESONANCE_TIE: Confidence.MEDIUM,
    Cap.SUBRESONANCE_T6_ONLY: Confidence.MEDIUM,
    Cap.SUBRESONANCE_THEORY_UNCONFIRMED: Confidence.MEDIUM,
    Cap.SUBRESONANCE_UNDETERMINED: Confidence.MEDIUM,
    Cap.JITTER_LOW_E: Confidence.MEDIUM,
    Cap.ALL_PHYSICS_NOT_MEASURED: Confidence.MEDIUM,
    Cap.L2_WITHOUT_ZONE: Confidence.MEDIUM,
    # a transient whose crossings all fall inside the model's separatrix zone: its boundaries may be the layer's own
    # crossings, not a capture or an escape, so where it starts and ends (and whether it is 'transient' rather than
    # near_separatrix) is not sharp (52356; the author's decision of 2026-10-16)
    Cap.EPISODES_IN_SEPARATRIX_ZONE: Confidence.MEDIUM,
    Cap.NEAR_SEPARATRIX_WITHOUT_ZONE: Confidence.MEDIUM,
    Cap.EPISODE_BEFORE_JUMP: Confidence.MEDIUM,
    Cap.EPISODE_FORCE_LAW_MIXED: Confidence.MEDIUM,
    Cap.NEAR_EPISODE: Confidence.MEDIUM,
    Cap.SHORT_PERIOD_ALIASED: Confidence.MEDIUM,
    Cap.SECOND_CAPTURE_UNRESOLVED: Confidence.MEDIUM,
    Cap.STRENGTH_SCALE_ONLY: Confidence.MEDIUM,
    Cap.ONE_CYCLE_ENERGY_NOT_CONSERVED: Confidence.MEDIUM,
    Cap.ACTION_STEP_PROVISIONAL: Confidence.MEDIUM,
    Cap.ONSET_FAILED: Confidence.LOW,
    Cap.GEOMETRY_ONLY: Confidence.LOW,
    Cap.IMPLAUSIBLE_STRENGTH: Confidence.LOW,
}


@dataclass(frozen=True)
class CellRecord:
    """The author-label record of one calibration cell, reported next to the level (it changes no level):
    labelled verdicts of the cell at confidence high or certain, their errors, the Wilson 95 % interval of the
    accuracy (nan without examples)."""

    n_examples: int
    n_errors: int
    accuracy_low_95: float = NAN
    accuracy_high_95: float = NAN

    def verdict_fields(self) -> dict:
        """The `cell_*` fields of a Verdict."""
        return dict(
            cell_n_examples=self.n_examples,
            cell_n_errors=self.n_errors,
            cell_accuracy_low_95=finite_or_none(self.accuracy_low_95),
            cell_accuracy_high_95=finite_or_none(self.accuracy_high_95),
        )


@dataclass(frozen=True)
class ConfidenceCells:
    """The calibration cells (status, angle_status, reason) -> CellRecord of the cells file, and its sha256."""

    records: Mapping[Tuple[str, str, str], CellRecord] = field(default_factory=dict)
    sha: str = ''


@dataclass
class ConfidenceAssessment:
    """The confidence columns of one verdict."""

    confidence: Confidence
    confidence_raw: Confidence
    thin_margins: List[str] = field(default_factory=list)  # margin ids and U:<name>
    failed_support: List[str] = field(default_factory=list)
    caps: List[str] = field(default_factory=list)
    cell: Optional[CellRecord] = None  # the verdict's calibration cell, reported only

    def cell_fields(self) -> dict:
        """The `cell_*` fields of the Verdict (empty without a cell)."""
        return self.cell.verdict_fields() if self.cell is not None else {}


def file_sha(path: Path) -> str:
    """sha256 of a file (hex); '' for a missing file."""
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ''


def load_cells(path: Optional[str | Path] = None) -> ConfidenceCells:
    """Read the cells file ({"version": 1, "cells": [{"status", "angle_status", "reason", "n_examples", "n_errors",
    "accuracy_95": [low, high]}]}, written by benchmark/tools/calibrate.py). '' or None = the package file; a missing
    file gives no cells. An older file's angle_status 'transient' is read as 'episodic'."""
    path = Path(path) if path else PACKAGE_CELLS
    entries = json.loads(path.read_text()).get('cells', []) if path.exists() else []
    records = {}
    for entry in entries:
        low, high = entry.get('accuracy_95') or (NAN, NAN)
        records[cell_of(entry['status'], angle_status_from_text(entry['angle_status']), entry['reason'])] = CellRecord(
            int(entry.get('n_examples', 0)), int(entry.get('n_errors', 0)), float(low), float(high)
        )
    return ConfidenceCells(records, file_sha(path))


def cells_from_settings(settings) -> ConfidenceCells:
    """The cells file named by `confidence_cells` (empty = the package file)."""
    return load_cells(settings.confidence_cells)


def cell_of(status: Status | str, angle_status: str, reason: str) -> Tuple[str, str, str]:
    """The calibration cell of a verdict."""
    return (str(status), str(angle_status), str(reason))


def raw_level(margins: Mapping[str, MarginValue], physics_margins: Sequence[str]) -> Tuple[Confidence, List[str], List[str], bool]:
    """confidence_raw by the rule above. Returns (level, thin ids, failed support ids,
    all physics not_measured). `physics_margins`: ids of the decisive physics margins (MMR: L4, Z, R0).

    A failed cap margin is not counted (its cap does the limiting); a failed decisive margin should not
    reach a final verdict and is counted like a support failure if it does.
    """
    thin, failed = [], []
    for margin_id, value in margins.items():
        if value.margin == Margin.THIN or value.margin == Margin.NOT_MEASURED:
            thin.append(margin_id)
        elif value.margin == Margin.FAIL and value.margin_class != MarginClass.CAP:
            failed.append(margin_id)
    physics = [margins[m].margin for m in physics_margins if m in margins and margins[m].margin != Margin.NOT_APPLICABLE]
    all_physics_missing = bool(physics) and all(margin == Margin.NOT_MEASURED for margin in physics)
    if len(failed) >= 2:
        level = Confidence.LOW
    elif len(failed) == 1:
        level = Confidence.MEDIUM
    else:
        level = Confidence.HIGH if thin else Confidence.CERTAIN
    return level, thin, failed, all_physics_missing


def _rank(level: Confidence) -> int:
    return CONFIDENCE_ORDER.index(level)


def apply_caps(level: Confidence, caps: Iterable[Cap | str]) -> Confidence:
    """min(level, CAP_CEILING[cap] for each cap); NONE stays NONE."""
    level = Confidence(level)
    if level == Confidence.NONE:
        return level
    return lower_confidence(level, *(CAP_CEILING[Cap(cap)] for cap in caps))


def cap_binds(confidence: Confidence | str, confidence_raw: Confidence | str, caps: Iterable[Cap | str], cap: Cap | str) -> bool:
    """Whether `cap` holds the shown level down: the level of `assess` (min(raw, caps)) with this cap alone dropped is
    above `confidence`. False when the cap is absent, another cap sits at the same ceiling or lower, or the record has
    no level."""
    caps = [str(c) for c in caps]
    if str(cap) not in caps or Confidence(confidence) == Confidence.NONE:
        return False
    without = apply_caps(Confidence(confidence_raw), [c for c in caps if c != str(cap)])
    return _rank(without) < _rank(Confidence(confidence))


def assess(
    status: Status,
    angle_status: str,
    reason: str,
    margins: Mapping[str, MarginValue],
    caps: Iterable[Cap | str],
    ledger_thin: Sequence[str],
    physics_margins: Sequence[str],
    cells: ConfidenceCells,
) -> ConfidenceAssessment:
    """The full confidence of a verdict: raw level, caps, the U margins from the ledger (`ledger_thin`
    are 'U:<name>' items, added to thin margins); the shown level is min(raw, caps). The cell's record in `cells`
    is attached for the output only. uncertain and invalid get Confidence.NONE for both levels and no cell."""
    level, thin, failed, all_physics_missing = raw_level(margins, physics_margins)
    thin = thin + [item for item in ledger_thin if item not in thin]
    if level == Confidence.CERTAIN and thin:
        level = Confidence.HIGH
    caps = list(dict.fromkeys(str(Cap(cap)) for cap in caps))
    if all_physics_missing and str(Cap.ALL_PHYSICS_NOT_MEASURED) not in caps:
        caps.append(str(Cap.ALL_PHYSICS_NOT_MEASURED))
    if status in (Status.UNCERTAIN, Status.INVALID):
        return ConfidenceAssessment(Confidence.NONE, Confidence.NONE, thin, failed, caps)
    cell = cells.records.get(cell_of(status, angle_status, reason))
    return ConfidenceAssessment(apply_caps(level, caps), level, thin, failed, caps, cell)
