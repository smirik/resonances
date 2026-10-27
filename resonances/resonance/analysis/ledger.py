"""The decision ledger: every comparison on the path of a decision, with its numbers.

Decision code compares through `Ledger.compare(name, value, op, threshold, unit, column)`
instead of a bare `if value < threshold`. The entry keeps the value, the threshold, the CSV
column that holds the value and whether the threshold is an agreed convention. From the ledger
come the `decision_path` column, the verdict card of the combined figure and margin U: a
comparison whose threshold is a convention and whose value lies within `thin_band` (relative)
of it is a thin margin `U:<name>` — the decision would flip if the convention were a little
different. Thresholds derived from physics never make U.

Margin of an entry: NOT_MEASURED when the value is missing (None or NaN; the comparison is then
False), THIN as above, otherwise FULL. The margin measures closeness, not the outcome.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Collection, Iterator, Optional


class Margin(StrEnum):
    FULL = 'full'
    THIN = 'thin'
    FAIL = 'fail'
    NOT_MEASURED = 'not_measured'
    NOT_APPLICABLE = 'not_applicable'


class Op(StrEnum):
    LT = '<'
    LE = '<='
    GT = '>'
    GE = '>='
    EQ = '=='
    NE = '!='


_OPERATORS = {Op.LT: operator.lt, Op.LE: operator.le, Op.GT: operator.gt, Op.GE: operator.ge, Op.EQ: operator.eq, Op.NE: operator.ne}

MARKS = {Margin.FULL: '✓', Margin.THIN: '~', Margin.FAIL: '✗', Margin.NOT_MEASURED: '–', Margin.NOT_APPLICABLE: '–'}


@dataclass(frozen=True)
class LedgerEntry:
    """One comparison `value op threshold` (both in `unit`, the unit of `column`)."""

    name: str
    value: Optional[float] = None
    op: Op = Op.GE
    threshold: Optional[float] = None
    unit: str = ''
    column: str = ''
    setting: str = ''  # the setting that holds the threshold ('' = derived or fixed by physics)
    convention: bool = False
    result: Optional[bool] = None  # None: not compared (missing value or a name-only entry)
    margin: Margin = Margin.NOT_MEASURED
    relative_distance: Optional[float] = None  # |value/threshold - 1|; None for threshold 0 or no value

    def __bool__(self) -> bool:
        return bool(self.result)

    @classmethod
    def name_only(cls, name: str) -> 'LedgerEntry':
        """An entry known only by its name (a decision path read back from summary.csv)."""
        return cls(name=name)

    @property
    def mark(self) -> str:
        """✓ held with a full margin, ~ thin, ✗ did not hold, – not measured."""
        if self.result is None:
            return MARKS[Margin.NOT_MEASURED]
        if not self.result:
            return MARKS[Margin.FAIL]
        return MARKS[self.margin]

    def describe(self, digits: int = 3) -> str:
        """One line for the verdict card: `name: value unit op threshold mark`."""
        if self.result is None and self.value is None:
            return f'{self.name}: not measured'
        unit = f' {self.unit}' if self.unit else ''
        return f'{self.name}: {_format(self.value, digits)}{unit} {self.op} {_format(self.threshold, digits)} {self.mark}'


def _format(value: Optional[float], digits: int) -> str:
    if value is None:
        return '—'
    return f'{value:.{digits}g}'


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


@dataclass
class Ledger:
    """The comparisons of one decision, in the order they were made."""

    thin_band: float
    conventions: Collection[str] = frozenset()
    entries: list[LedgerEntry] = field(default_factory=list)

    @classmethod
    def from_settings(cls, settings) -> 'Ledger':
        """A ledger with the thin band and the convention names of an `AnalysisSettings`."""
        from resonances.resonance.analysis.settings import conventions

        return cls(thin_band=settings.confidence_thin_band, conventions=conventions(type(settings)))

    def compare(
        self,
        name: str,
        value: Optional[float],
        op: Op | str,
        threshold: float,
        unit: str,
        column: str,
        *,
        setting: str = '',
    ) -> LedgerEntry:
        """Compare, record and return the entry (truthy when the comparison holds).

        `setting` names the setting the threshold comes from; it is a convention when listed in
        `conventions`. A missing value (None, NaN) gives result None and margin NOT_MEASURED.
        """
        op = Op(op)
        convention = bool(setting) and setting in self.conventions
        if _missing(value):
            entry = LedgerEntry(name, None, op, threshold, unit, column, setting, convention, None, Margin.NOT_MEASURED, None)
        else:
            value = float(value)
            result = bool(_OPERATORS[op](value, threshold))
            distance = abs(value / threshold - 1.0) if threshold else None
            thin = convention and distance is not None and distance < self.thin_band
            entry = LedgerEntry(
                name, value, op, threshold, unit, column, setting, convention, result, Margin.THIN if thin else Margin.FULL, distance
            )
        self.entries.append(entry)
        return entry

    def compare_setting(
        self,
        name: str,
        value: Optional[float],
        op: Op | str,
        settings,
        setting: str,
        unit: str = '',
        column: str = '',
        *,
        named: bool = True,
    ) -> LedgerEntry:
        """`compare` against the threshold held by the setting `setting` of `settings`. `named` False records the
        threshold without its setting name: no convention the decision rests on, never a thin margin."""
        return self.compare(name, value, op, getattr(settings, setting), unit, column, setting=setting if named else '')

    def extend(self, other: 'Ledger') -> None:
        """Append the entries of another ledger (a sub-decision)."""
        self.entries.extend(other.entries)

    def thin_margins(self) -> list[str]:
        """`U:<name>` for every entry close to a convention, each name once, in order."""
        names = [entry.name for entry in self.entries if entry.margin == Margin.THIN]
        return [f'U:{name}' for name in dict.fromkeys(names)]

    def decision_path(self) -> str:
        """The `decision_path` column: the names in order, ';'-separated."""
        return ';'.join(entry.name for entry in self.entries)

    def columns(self) -> list[str]:
        """The CSV columns the decision read, each once, in order."""
        return list(dict.fromkeys(entry.column for entry in self.entries if entry.column))

    def __iter__(self) -> Iterator[LedgerEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)
