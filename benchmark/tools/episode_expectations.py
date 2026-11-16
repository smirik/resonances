"""The episodes the author asked for (benchmark/EPISODE_EXPECTATIONS.csv) checked on a reclassify run: the episode gate
of `make test-benchmark`.

    uv run python benchmark/tools/episode_expectations.py test-benchmark     # a run folder (name under benchmark/data or a path)
    uv run python benchmark/tools/episode_expectations.py v2-w22 --secular-zlk v2-w21a-secular-zlk

Each case is one record and a list of checks (JSON ops, `OPS`) on its accepted episodes (episodes.csv), its summary row
(summary.csv) and a few numbers (mmr_numbers.csv). The secular and Lidov-Kozai cases are not in an MMR run: they are
analysed here from their saved series (`reclassify_secular_zlk.analyse_one`, well under a second each), or read from a
secular/ZLK run folder with --secular-zlk. The verbatim quotes behind every case are in benchmark/EPISODE_QUOTES.json
(file, line, text); `sources` repeats their file:line.

Tiers: A the author's own words about the record's episodes, B a review conclusion he accepted, C an old author_verdict
that names an interval (a hint: the verdict's record may differ), D an analogy with his decision on another object. The
gate reads tiers A and B only, as KNOWN_DISAGREEMENTS.csv: a case that fails and is not `known_fail` is a problem (new),
so is a `known_fail` case that passes (resolved: set known_fail = 0) and a case whose record is not in the run
(missing). Tiers C and D are printed, never gate. `known_reason` is Claude's reading of a failure, not an author label.

The ops (times in yr; 'plain' episodes are the accepted ones not flagged as a forced wave, `count_forced` takes all;
under the minimal policy (W29, summary `policy` = minimal, episodes.csv `counted`) the plain episodes are the counted
ones: a vetoed or filter-artefact run is accepted but not counted, and the figure does not number it):
- n_accepted (min, max), single_covers (exactly one, both edges covered as last_covers_end reads the end), episode_near / single_episode_near (t0, t1,
  tol, optional min_tp and centre_deg within CENTRE_TOL_DEG), no_episode_in (t0, t1), no_plain_episode, all_forced;
- union (min) / no_union: geo_union_n_episodes of mmr_numbers.csv;
- return_cycle (t_a, t_b, tol): an accepted episode closed by its phase return at those crossings;
- exit_observed (optional value, default true) / entry_observed (mmr_numbers.csv), final_regime (value), angle (value,
  optional subtype);
- last_reaches_end / last_ends_before_end (the last episode's end gap <= 1 P or above), last_covers_end (the end as
  the package's `geometry.covers_edges` reads it: gap <= 1 P, or no turning point missed at the analysis window's end,
  window gap <= min(longest TP step, 1 P), with the filter's trim hiding <= 1 P; never with an observed exit),
  first_start_before (t, plus_P
  'max' or the median period: the first episode starts before t + P);
- boundary_in (t0, t1): a boundary between consecutive episodes overlaps the interval; n_runs_in (t0, t1, max): runs,
  accepted or not, starting there; max_half_amp (max_deg); episode_covers_t (t);
- period_in (lo, hi, which): which = 'decisive' (default) reads the decisive episode's period `ep_period_med_yr`
  (mmr_numbers.csv, or the summary row of a secular/ZLK run), 'any' passes when one episode's `period_med_yr` is in
  [lo, hi];
- episode_mark (t0, t1, tol, mark): a run of episodes.csv, accepted or not, carries the mark `mark` over an interval
  within tol of t0-t1. The `marks` column is ';'-separated items `name`, `name@t` or `name@t0:t1` [yr]; a mark
  without a time spans its run. A run without the column fails ('marks not written');
- core_near (t0, t1, tol): a plain episode whose core (episodes.csv core_t0_yr/core_t1_yr, its first and last measured
  turning points) lies within tol of t0-t1; residence_reaches_end: the last plain episode's inferred 1:1 residence is
  bounded by the record's edge (residence_bounded_by_end = 'edge': the stay goes on to the end, its continuation
  undetermined); residence_starts_before (t): the first plain episode's residence_t0_yr <= t (W28 C8; a run without the
  columns fails: 'core / residence not written');
- action (value, optional min_extend_to): the summary's `action`, and `extend_to_years` >= min_extend_to;
- flag (value, optional present = true): the flag is (or is not) in the summary's ';'-separated `flags`;
- field (column and one of value, contains, min): a summary column equals value (numerically when both are
  numbers), holds `contains` as an item of its ';'-separated list (rules_fired, caps, failed_support), or is >= min.
  A missing column fails.
- any_of (alternatives): a list of check lists, one of which passes in full.

Every check may take `scale` (default 'main'): the episode ops then read only the episodes.csv rows of that scale
(B10a writes scale='slow' rows; a run without the column, or an empty value, is the main scale). The summary ops
ignore it. Every check may also take `pool`: 'counted' (default) reads the plain episodes above, 'geometric' the
counted ones and the candidates (episodes.csv `candidate`: a run a measured veto or the subresonance member took out of
the count, drawn faint): the author's episodes are geometric, a veto changes the mechanism, not where the angle librates
(W34, the author's epcount A). Outside the minimal policy both pools are the plain episodes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_io as bio  # noqa: E402

CASES_CSV = bio.BENCH / 'EPISODE_EXPECTATIONS.csv'
QUOTES_JSON = bio.BENCH / 'EPISODE_QUOTES.json'
TIERS = ('A', 'B', 'C', 'D')
GATING_TIERS = ('A', 'B')
SCOPES = ('mmr', 'secular', 'zlk')
LONG_PERIOD_SCOPES = ('secular', 'zlk')
COUNTED, GEOMETRIC = 'counted', 'geometric'  # the episode pool a check reads (`Record`)
POOLS = (COUNTED, GEOMETRIC)
CENTRE_TOL_DEG = 30.0  # episode_near: the episode's centre within this of `centre_deg` (an island, not a position)
NAN_VALUE = float('nan')
MINIMAL = 'minimal'  # the summary `policy` whose episodes.csv `counted` says which runs are episodes (W29)
COVER_P = 1.0  # an edge gap of at most one period of the episode covers the edge (DECISIONS: LIBRATION_COVERAGE_PERIODS)
DECISIVE_PERIOD = 'ep_period_med_yr'  # period_in which='decisive': the period of the episode the verdict rests on
NUMBER_COLUMNS = ('example_id', 'resonance', 'geo_union_n_episodes', 'exit_observed', 'entry_observed', DECISIVE_PERIOD)
MAIN_SCALE = 'main'  # the episodes.csv scale of a row without one
PASS, FAIL, MISSING = 'PASS', 'FAIL', 'MISSING'


# ------------------------------------------------------------------ the cases


@dataclass(frozen=True)
class Quote:
    file: str  # repository-relative path
    line: int
    text: str
    who: str = 'author'
    transcript: str = ''


@dataclass(frozen=True)
class Case:
    case: str
    example_id: str
    tier: str
    scope: str
    mechanism: str
    expected: str
    checks: Tuple[dict, ...]
    sources: str
    known_fail: bool
    known_reason: str
    quotes: Tuple[Quote, ...] = ()

    @property
    def gates(self) -> bool:
        return self.tier in GATING_TIERS


def _read_commented_csv(path: Path) -> pd.DataFrame:
    lines = [line for line in Path(path).read_text().splitlines() if not line.startswith('#')]
    return pd.read_csv(StringIO('\n'.join(lines)), dtype=str, keep_default_na=False)


def load_cases(path: Path = CASES_CSV, quotes_path: Optional[Path] = QUOTES_JSON) -> List[Case]:
    """The cases of EPISODE_EXPECTATIONS.csv (leading '#' lines are its header comment), each with its quotes."""
    quotes: Dict[str, Tuple[Quote, ...]] = {}
    if quotes_path is not None and Path(quotes_path).exists():
        for entry in json.loads(Path(quotes_path).read_text()):
            quotes[entry['case']] = tuple(Quote(**q) for q in entry['quotes'])
    cases = []
    for row in _read_commented_csv(path).to_dict('records'):
        cases.append(
            Case(
                case=row['case'],
                example_id=row['example_id'],
                tier=row['tier'],
                scope=row['scope'],
                mechanism=row['mechanism'],
                expected=row['expected'],
                checks=tuple(json.loads(row['checks'])),
                sources=row['sources'],
                known_fail=row['known_fail'].strip() == '1',
                known_reason=row['known_reason'],
                quotes=quotes.get(row['case'], ()),
            )
        )
    return cases


def quote_line(file: str, text: str) -> Optional[int]:
    """The 1-based line where `text` starts in the repository file `file`; None when the file or the text is absent."""
    path = bio.REPO / file
    if not path.exists():
        return None
    raw = path.read_text(errors='replace')
    at = raw.find(text)
    return raw.count('\n', 0, at) + 1 if at >= 0 else None


def flatten(checks: Iterable[dict]) -> Iterator[dict]:
    """Every check, the alternatives of any_of included (any_of itself too)."""
    for check in checks:
        yield check
        for alternative in check.get('alternatives', ()):
            yield from flatten(alternative)


def long_period_ids(cases: Iterable[Case]) -> List[str]:
    return sorted({case.example_id for case in cases if case.scope in LONG_PERIOD_SCOPES})


# ------------------------------------------------------------------ a record of a run


def _ok(value) -> bool:
    if value is None:
        return False
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return str(value) != ''


def _fmt(value, digits: int = 2) -> str:
    return f'{float(value):.{digits}f}' if _ok(value) else 'nan'


def _flag(value) -> bool:
    return str(value) in ('True', 'true', '1', '1.0')


def _scales(frame: pd.DataFrame) -> pd.Series:
    """The scale of every episode row: the `scale` column, MAIN_SCALE where it is absent or empty."""
    if 'scale' not in frame:
        return pd.Series(MAIN_SCALE, index=frame.index, dtype=object)
    return frame['scale'].where(frame['scale'].map(_ok), MAIN_SCALE).astype(str)


class Record:
    """One example of a run at one scale: its summary row (None: not in the run), its episode rows of that scale
    (accepted or not) and the numbers the ops read. The plain episodes are the accepted ones without a forced-wave
    flag; under the minimal policy, with the episodes.csv column `counted`, the accepted counted ones (`counted_only`),
    and with `pool` GEOMETRIC the counted ones and the candidates. `at_scale` gives the same example at another scale or
    pool (the summary and the numbers are shared)."""

    def __init__(
        self,
        summary: Optional[Mapping],
        episodes: pd.DataFrame,
        numbers: Optional[Mapping] = None,
        scale: str = MAIN_SCALE,
        pool: str = COUNTED,
    ):
        self.summary = summary
        self.numbers = numbers
        self.scale = scale
        self.pool = pool
        self.all_episodes = episodes  # every scale
        self.marks_written = 'marks' in episodes
        self._other_scales: Dict[Tuple[str, str], 'Record'] = {}
        runs = episodes[(_scales(episodes) == scale).to_numpy()].copy()
        if 't_start_yr' in runs:
            runs = runs.sort_values('t_start_yr', kind='stable')
        self.runs = runs.reset_index(drop=True)
        accepted = self.runs['accepted'].map(_flag) if len(self.runs) else pd.Series(dtype=bool)
        self.accepted = self.runs[accepted] if len(self.runs) else self.runs
        forced = self.accepted['forced_wave'].map(_ok) if 'forced_wave' in self.accepted and len(self.accepted) else None
        self.forced = self.accepted[forced] if forced is not None else self.accepted.iloc[0:0]
        self.plain = self.accepted[~forced] if forced is not None else self.accepted
        if self.counted_only and len(self.accepted):
            kept = self.accepted['counted'].map(_flag)
            if pool == GEOMETRIC and 'candidate' in self.accepted:
                kept = kept | self.accepted['candidate'].map(_flag)
            self.plain = self.accepted[kept]

    @property
    def counted_only(self) -> bool:
        """The minimal policy's run (W29): the summary says so and episodes.csv has the `counted` column."""
        return self.summary is not None and str(self.summary.get('policy', '')) == MINIMAL and 'counted' in self.runs

    def at_scale(self, scale: str, pool: str = COUNTED) -> 'Record':
        """The same example with the episode rows of `scale`, its plain episodes read from `pool`."""
        if (scale, pool) == (self.scale, self.pool):
            return self
        if (scale, pool) not in self._other_scales:
            self._other_scales[scale, pool] = Record(self.summary, self.all_episodes, self.numbers, scale, pool)
        return self._other_scales[scale, pool]

    def number(self, key: str):
        return None if self.numbers is None else self.numbers.get(key)

    def field(self, key: str):
        return None if self.summary is None else self.summary.get(key)


@dataclass
class RunTables:
    """summary, episodes and numbers of a run, read once: `record` builds a case's `Record` from them."""

    summary: pd.DataFrame
    episodes: pd.DataFrame
    numbers: Optional[pd.DataFrame] = None

    def record(self, example_id: str) -> Optional[Record]:
        rows = self.summary[self.summary['example_id'] == example_id]
        if not len(rows):
            return None
        if 'parent' in rows:  # the catalogue row, not a member of its multiplet (subresonance_check)
            own = rows[rows['parent'].fillna('').astype(str) == '']
            rows = own if len(own) else rows
        summary = rows.iloc[0].to_dict()
        episodes = self.episodes[self.episodes['example_id'] == example_id]
        if 'resonance' in episodes and 'resonance' in summary:
            episodes = episodes[episodes['resonance'] == summary['resonance']]
        numbers = None
        if self.numbers is not None:
            own = self.numbers[self.numbers['example_id'] == example_id]
            if 'resonance' in own and 'resonance' in summary:
                own = own[own['resonance'] == summary['resonance']]
            numbers = own.iloc[0].to_dict() if len(own) else None
        return Record(summary, episodes, numbers)


def read_run(run: str | Path) -> RunTables:
    """A reclassify output folder (a name under benchmark/data or a path): summary.csv, episodes.csv and the columns of
    mmr_numbers.csv the ops read (when present)."""
    folder = run_folder(run)
    summary = pd.read_csv(folder / 'summary.csv', dtype=str, keep_default_na=False)
    episodes = pd.read_csv(folder / 'episodes.csv', low_memory=False)
    numbers = None
    if (folder / 'mmr_numbers.csv').exists():
        header = bio.read_header(folder / 'mmr_numbers.csv')
        numbers = pd.read_csv(folder / 'mmr_numbers.csv', usecols=[c for c in NUMBER_COLUMNS if c in header], low_memory=False)
    return RunTables(summary, episodes, numbers)


def run_folder(run: str | Path) -> Path:
    path = Path(run)
    return path if path.is_dir() else bio.DATA / str(run)


def analyse_long_period(example_ids: Sequence[str]) -> RunTables:
    """The secular and Lidov-Kozai examples analysed from their saved series, as reclassify_secular_zlk does (no
    figures): a `RunTables` with their summary and episode rows."""
    import reclassify_secular_zlk as szlk

    summaries, episodes = [], []
    for example_id in example_ids:
        result = szlk.analyse_one(example_id, False)
        if result['status'] != 'ok':
            raise RuntimeError(f'{example_id}: {result["error"]}\n{result.get("trace", "")}')
        summaries.extend(result['summary'])
        episodes.extend(result['episodes'])
    summary = pd.DataFrame(summaries, columns=sorted({k for row in summaries for k in row}) or ['example_id'])
    return RunTables(summary, pd.DataFrame(episodes, columns=sorted({k for row in episodes for k in row}) or ['example_id']))


def table_from_tuples(rows: Sequence[tuple], summary: Optional[Mapping] = None, numbers: Optional[Mapping] = None) -> Record:
    """A hand-made record (tests): one accepted episode per tuple (t_start, t_end, n_tp, P, start gap [P], end gap [P],
    forced wave or None), optionally followed by (half-amplitude [deg], centre [deg])."""
    columns = ['t_start_yr', 't_end_yr', 'n_tp', 'period_med_yr', 'coverage_start_gap_P', 'coverage_end_gap_P', 'forced_wave']
    records = []
    for row in rows:
        values = dict(zip(columns, row[:7]))
        values['half_amp_med_deg'] = row[7] if len(row) > 7 else float('nan')
        values['centre_mod_2pi_deg'] = row[8] if len(row) > 8 else float('nan')
        values['period_max_yr'] = values['period_med_yr']
        values['accepted'] = True
        records.append(values)
    frame = pd.DataFrame(records, columns=columns + ['half_amp_med_deg', 'centre_mod_2pi_deg', 'period_max_yr', 'accepted'])
    return Record({} if summary is None else dict(summary), frame, None if numbers is None else dict(numbers))


# ------------------------------------------------------------------ the ops


def _edge_covered(episode, side: str) -> bool:
    """One edge ('start' | 'end') covered as the package's `geometry.covers_edges` reads it: the gap <= 1 P, or no
    turning point missed at the analysis window's edge (window gap <= min(longest TP step, 1 P)) with the filter's trim
    hiding <= 1 P."""
    gap, window_gap, hidden = (
        episode[f'coverage_{side}_gap_P'],
        episode.get(f'window_{side}_gap_yr'),
        episode.get(f'window_{side}_hidden_P'),
    )
    if _ok(gap) and float(gap) <= COVER_P:
        return True
    if not (_ok(window_gap) and _ok(hidden)):
        return False
    step = min(float(episode.get('tp_step_max_yr', NAN_VALUE)), COVER_P * float(episode.get('period_med_yr', NAN_VALUE)))
    return float(window_gap) <= step and float(hidden) <= COVER_P


def _covers(episode) -> bool:
    return _edge_covered(episode, 'start') and _edge_covered(episode, 'end')


def _near(episode, check) -> bool:
    tol = check['tol']
    if abs(episode['t_start_yr'] - check['t0']) > tol or abs(episode['t_end_yr'] - check['t1']) > tol:
        return False
    if check.get('min_tp') is not None and episode['n_tp'] < check['min_tp']:
        return False
    if check.get('centre_deg') is not None:
        offset = abs((episode['centre_mod_2pi_deg'] - check['centre_deg'] + 180.0) % 360.0 - 180.0)
        if not offset <= CENTRE_TOL_DEG:
            return False
    return True


def _pool(rec: Record, check) -> pd.DataFrame:
    return rec.accepted if check.get('count_forced') else rec.plain


def _rows(frame: pd.DataFrame) -> List[Mapping]:
    return [row for _, row in frame.iterrows()]


def op_n_accepted(rec, c):
    k = len(_pool(rec, c))
    return c.get('min', 0) <= k <= c.get('max', math.inf), f'{k} accepted'


def op_single_covers(rec, c):
    pool = _pool(rec, c)
    if len(pool) != 1:
        return False, f'{len(pool)} accepted (need 1 covering)'
    episode = pool.iloc[0]
    return _covers(episode), f'gaps {_fmt(episode["coverage_start_gap_P"])}/{_fmt(episode["coverage_end_gap_P"])} P'


def op_episode_near(rec, c):
    hits = [e for e in _rows(_pool(rec, c)) if _near(e, c)]
    return bool(hits), 'match' if hits else f'none within {c["tol"]} yr of {c["t0"]:.0f}-{c["t1"]:.0f}'


def op_single_episode_near(rec, c):
    pool = _pool(rec, c)
    if len(pool) != 1:
        return False, f'{len(pool)} accepted (need one {c["t0"]:.0f}-{c["t1"]:.0f})'
    return op_episode_near(rec, c)


def op_no_episode_in(rec, c):
    bad = [e for e in _rows(_pool(rec, c)) if e['t_end_yr'] > c['t0'] and e['t_start_yr'] < c['t1']]
    return not bad, 'none' if not bad else '; '.join(f'{e["t_start_yr"]:.0f}-{e["t_end_yr"]:.0f}' for e in bad)


def op_no_plain_episode(rec, c):
    return len(rec.plain) == 0, f'{len(rec.plain)} accepted not forced'


def op_all_forced(rec, c):
    return len(rec.accepted) > 0 and len(rec.plain) == 0, f'{len(rec.forced)} forced / {len(rec.accepted)} accepted'


def op_union(rec, c):
    n = rec.number('geo_union_n_episodes')
    return _ok(n) and float(n) >= c['min'], f'union {n}'


def op_no_union(rec, c):
    n = rec.number('geo_union_n_episodes')
    return (not _ok(n)) or float(n) == 0, f'union {n}'


def op_return_cycle(rec, c):
    def closes(e):
        return (
            _flag(e.get('closed_by_return'))
            and _ok(e.get('return_t_a_yr'))
            and abs(e['return_t_a_yr'] - c['t_a']) <= c['tol']
            and abs(e['return_t_b_yr'] - c['t_b']) <= c['tol']
        )

    episodes = _rows(rec.accepted)
    got = [(round(e['return_t_a_yr']), round(e['return_t_b_yr'])) for e in episodes if _ok(e.get('return_t_a_yr'))]
    return any(closes(e) for e in episodes), f'returns {got}'


def op_exit_observed(rec, c):
    value = rec.number('exit_observed')
    return _flag(value) == bool(c.get('value', True)), f'exit_observed={value}'


def op_entry_observed(rec, c):
    value = rec.number('entry_observed')
    return _flag(value), f'entry_observed={value}'


def op_final_regime(rec, c):
    value = rec.field('final_regime')
    return value == c['value'], f'final_regime={value}'


def op_angle(rec, c):
    value, subtype = rec.field('angle_status'), rec.field('angle_subtype')
    return value == c['value'] and (c.get('subtype') is None or subtype == c['subtype']), f'{value}/{subtype}'


def op_last_reaches_end(rec, c):
    pool = _pool(rec, c)
    if not len(pool):
        return False, 'no episode'
    gap = pool.iloc[-1]['coverage_end_gap_P']
    return _ok(gap) and float(gap) <= COVER_P, f'end gap {_fmt(gap)} P'


def op_last_covers_end(rec, c):
    pool = _pool(rec, c)
    if not len(pool):
        return False, 'no episode'
    last = pool.iloc[-1]
    if _flag(last.get('exit_observed')):  # an observed exit is a regime the record shows after the episode
        return False, 'exit observed'
    gap, window_gap, hidden = last['coverage_end_gap_P'], last.get('window_end_gap_yr'), last.get('window_end_hidden_P')
    return _edge_covered(last, 'end'), f'end gap {_fmt(gap)} P; window gap {_fmt(window_gap, 0)} yr, hidden {_fmt(hidden)} P'


def op_last_ends_before_end(rec, c):
    pool = _pool(rec, c)
    if not len(pool):
        return False, 'no episode'
    gap = pool.iloc[-1]['coverage_end_gap_P']
    return _ok(gap) and float(gap) > COVER_P, f'end gap {_fmt(gap)} P'


def op_first_start_before(rec, c):
    pool = _pool(rec, c)
    if not len(pool):
        return False, 'no episode'
    first = pool.iloc[0]
    limit = c['t'] + (first['period_max_yr'] if c.get('plus_P') == 'max' else first['period_med_yr'])
    return first['t_start_yr'] <= limit, f'start {first["t_start_yr"]:.0f} vs {limit:.0f}'


def op_boundary_in(rec, c):
    pool = _rows(_pool(rec, c))
    bounds = [(left['t_end_yr'], right['t_start_yr']) for left, right in zip(pool, pool[1:])]
    hit = any(b0 <= c['t1'] and b1 >= c['t0'] for b0, b1 in bounds)
    return hit, f'boundaries {[(round(a), round(b)) for a, b in bounds]}'


def op_n_runs_in(rec, c):
    starts = rec.runs['t_start_yr'] if len(rec.runs) else pd.Series(dtype=float)
    k = int(((starts >= c['t0']) & (starts < c['t1'])).sum())
    return k <= c['max'], f'{k} runs start in {c["t0"]}-{c["t1"]}'


def op_max_half_amp(rec, c):
    pool = _pool(rec, c)
    largest = float(pool['half_amp_med_deg'].max()) if len(pool) else 0.0
    return (not len(pool)) or largest <= c['max_deg'], f'max half-amp {largest:.2f}'


def op_episode_covers_t(rec, c):
    hit = any(e['t_start_yr'] <= c['t'] <= e['t_end_yr'] for e in _rows(_pool(rec, c)))
    return hit, 'covers' if hit else f'no accepted episode at {c["t"]}'


def _number(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _items(value) -> List[str]:
    """The items of a ';'-separated summary list (flags, rules_fired, caps)."""
    return [item.strip() for item in str(value).split(';') if item.strip()] if _ok(value) else []


def op_period_in(rec, c):
    which = c.get('which', 'decisive')
    if which == 'decisive':
        if rec.scale != MAIN_SCALE:
            raise ValueError(f'bad check {c}: the decisive period is a main-scale number; use which=any')
        value = rec.number(DECISIVE_PERIOD)
        period = _number(value if _ok(value) else rec.field(DECISIVE_PERIOD))
        if period is None:
            return False, 'no decisive period'
        return c['lo'] <= period <= c['hi'], f'decisive P {period:.1f} yr'
    if which != 'any':
        raise ValueError(f'bad check {c}: which must be decisive or any')
    periods = [p for p in (_number(e['period_med_yr']) for e in _rows(_pool(rec, c))) if p is not None]
    hit = any(c['lo'] <= p <= c['hi'] for p in periods)
    return hit, f'periods {[round(p, 1) for p in periods]} yr'


def _marks(value) -> List[Tuple[str, Optional[float], Optional[float]]]:
    """(name, t0, t1) of every item `name`, `name@t` or `name@t0:t1` of a marks cell; no time: (name, None, None)."""
    marks = []
    for item in _items(value):
        name, _, span = item.partition('@')
        if not span:
            marks.append((name.strip(), None, None))
            continue
        t0, _, t1 = span.partition(':')
        marks.append((name.strip(), float(t0), float(t1 if t1 else t0)))
    return marks


def op_episode_mark(rec, c):
    if not rec.marks_written:
        return False, 'marks not written'
    found = []
    for run in _rows(rec.runs):
        for name, t0, t1 in _marks(run.get('marks')):
            if name == c['mark']:
                found.append((run['t_start_yr'] if t0 is None else t0, run['t_end_yr'] if t1 is None else t1))
    hit = any(abs(t0 - c['t0']) <= c['tol'] and abs(t1 - c['t1']) <= c['tol'] for t0, t1 in found)
    return hit, f'{c["mark"]} at {[(round(a), round(b)) for a, b in found]}'


RESIDENCE_EDGE = 'edge'  # geometry.RESIDENCE_EDGE: the record's edge bounds the residence


def _written(rec: Record, column: str) -> bool:
    return column in rec.runs


def op_core_near(rec, c):
    if not _written(rec, 'core_t0_yr'):
        return False, 'core not written'
    found = [(e['core_t0_yr'], e['core_t1_yr']) for e in _rows(_pool(rec, c)) if _ok(e['core_t0_yr'])]
    hit = any(abs(float(a) - c['t0']) <= c['tol'] and abs(float(b) - c['t1']) <= c['tol'] for a, b in found)
    return hit, f'cores {[(round(float(a)), round(float(b))) for a, b in found]}'


def op_residence_reaches_end(rec, c):
    if not _written(rec, 'residence_bounded_by_end'):
        return False, 'residence not written'
    pool = _pool(rec, c)
    if not len(pool):
        return False, 'no episode'
    last = pool.iloc[-1]
    bound, t1 = last['residence_bounded_by_end'], last.get('residence_t1_yr')
    return str(bound) == RESIDENCE_EDGE, f'residence to {_fmt(t1, 0)} yr, bounded by {bound if _ok(bound) else "none"}'


def op_residence_starts_before(rec, c):
    if not _written(rec, 'residence_t0_yr'):
        return False, 'residence not written'
    pool = _pool(rec, c)
    if not len(pool):
        return False, 'no episode'
    t0 = pool.iloc[0]['residence_t0_yr']
    return _ok(t0) and float(t0) <= c['t'], f'residence from {_fmt(t0, 0)} yr'


def op_action(rec, c):
    action, extend = rec.field('action'), rec.field('extend_to_years')
    good = action == c['value']
    if c.get('min_extend_to') is not None:
        good = good and _number(extend) is not None and _number(extend) >= c['min_extend_to']
    return good, f'action={action} extend_to={extend}'


def op_flag(rec, c):
    present = c['value'] in _items(rec.field('flags'))
    return present == c.get('present', True), f'{c["value"]} present={present}'


def op_field(rec, c):
    column = c['column']
    if rec.summary is None or column not in rec.summary:
        return False, f'no column {column}'
    value = rec.summary[column]
    if 'value' in c:
        want, got = _number(c['value']), _number(value)
        good = want == got if want is not None and got is not None else str(value) == str(c['value'])
    elif 'contains' in c:
        good = str(c['contains']) in _items(value)
    else:
        got = _number(value)
        good = got is not None and got >= c['min']
    return good, f'{column}={value}'


def op_any_of(rec, c):
    notes = []
    for alternative in c['alternatives']:
        results = [check(rec, x) for x in alternative]
        notes.append(' & '.join(note for _, note in results))
        if all(good for good, _ in results):
            return True, 'alt ok: ' + notes[-1]
    return False, ' | '.join(notes)


# op -> (function, required keys)
OPS: Dict[str, Tuple[Callable[[Record, dict], Tuple[bool, str]], Tuple[str, ...]]] = {
    'n_accepted': (op_n_accepted, ()),
    'single_covers': (op_single_covers, ()),
    'episode_near': (op_episode_near, ('t0', 't1', 'tol')),
    'single_episode_near': (op_single_episode_near, ('t0', 't1', 'tol')),
    'no_episode_in': (op_no_episode_in, ('t0', 't1')),
    'no_plain_episode': (op_no_plain_episode, ()),
    'all_forced': (op_all_forced, ()),
    'union': (op_union, ('min',)),
    'no_union': (op_no_union, ()),
    'return_cycle': (op_return_cycle, ('t_a', 't_b', 'tol')),
    'exit_observed': (op_exit_observed, ()),
    'entry_observed': (op_entry_observed, ()),
    'final_regime': (op_final_regime, ('value',)),
    'angle': (op_angle, ('value',)),
    'last_reaches_end': (op_last_reaches_end, ()),
    'last_covers_end': (op_last_covers_end, ()),
    'last_ends_before_end': (op_last_ends_before_end, ()),
    'first_start_before': (op_first_start_before, ('t',)),
    'boundary_in': (op_boundary_in, ('t0', 't1')),
    'n_runs_in': (op_n_runs_in, ('t0', 't1', 'max')),
    'max_half_amp': (op_max_half_amp, ('max_deg',)),
    'episode_covers_t': (op_episode_covers_t, ('t',)),
    'period_in': (op_period_in, ('lo', 'hi')),
    'episode_mark': (op_episode_mark, ('t0', 't1', 'tol', 'mark')),
    'core_near': (op_core_near, ('t0', 't1', 'tol')),
    'residence_reaches_end': (op_residence_reaches_end, ()),
    'residence_starts_before': (op_residence_starts_before, ('t',)),
    'action': (op_action, ('value',)),
    'flag': (op_flag, ('value',)),
    'field': (op_field, ('column',)),
    'any_of': (op_any_of, ('alternatives',)),
}
ONE_OF_KEYS = {'field': ('value', 'contains', 'min')}  # op -> exactly one of these keys


def missing_keys(c: dict) -> List[str]:
    """The keys an op needs that the check lacks (an unknown op lacks 'op'; a field check without exactly one of
    value/contains/min lacks 'value|contains|min')."""
    if c.get('op') not in OPS:
        return ['op']
    missing = [key for key in OPS[c['op']][1] if key not in c]
    one_of = ONE_OF_KEYS.get(c['op'])
    if one_of and sum(key in c for key in one_of) != 1:
        missing.append('|'.join(one_of))
    return missing


def check(rec: Record, c: dict) -> Tuple[bool, str]:
    """(passes, note) of one check on a record, at the check's scale (default main)."""
    if missing_keys(c):
        raise ValueError(f'bad check {c}: missing {missing_keys(c)}')
    if c.get('pool', COUNTED) not in POOLS:
        raise ValueError(f'bad check {c}: pool not in {POOLS}')
    good, note = OPS[c['op']][0](rec.at_scale(c.get('scale', MAIN_SCALE), c.get('pool', COUNTED)), c)
    return bool(good), note


def run_check(rec: Record, c: dict) -> bool:
    return check(rec, c)[0]


# ------------------------------------------------------------------ the gate


def describe(rec: Optional[Record]) -> str:
    """The record as the package read it: status/angle/subtype, final regime, the accepted episodes, the union."""
    if rec is None:
        return 'not in run'
    s = rec.summary
    head = f'{s.get("status")}/{s.get("angle_status")}/{s.get("angle_subtype")}; final={s.get("final_regime")}'
    parts = []
    for e in _rows(rec.accepted):
        tags = ''.join(
            tag
            for tag, on in (
                (' wave', _ok(e.get('forced_wave'))),
                (' psi', e.get('frame') == 'psi'),
                (' ret', _flag(e.get('closed_by_return'))),
            )
            if on
        )
        n_tp, half = _fmt(e.get('n_tp'), 0), _fmt(e.get('half_amp_med_deg'), 0)
        parts.append(f'{e["t_start_yr"] / 1e3:.2f}-{e["t_end_yr"] / 1e3:.2f}kyr({n_tp}TP,A{half}{tags})')
    union = rec.number('geo_union_n_episodes')
    tail = f' | union {int(float(union))}' if _ok(union) and float(union) > 0 else ''
    return head + ' | ' + ('; '.join(parts) if parts else 'no accepted episode') + tail


def evaluate(
    run: str | Path | RunTables,
    cases: Optional[Sequence[Case]] = None,
    long_period: Optional[RunTables] = None,
) -> pd.DataFrame:
    """One row per case: result PASS | FAIL | MISSING (the record is not in the run), the notes of every check, the
    package's reading. MMR cases read `run`; secular/ZLK cases read `long_period`, analysed here when not given (under
    the package's settings)."""
    cases = load_cases() if cases is None else list(cases)
    tables = run if isinstance(run, RunTables) else read_run(run)
    ids = long_period_ids(cases)
    if ids and long_period is None:
        long_period = analyse_long_period(ids)
    rows = []
    for case in cases:
        source = long_period if case.scope in LONG_PERIOD_SCOPES else tables
        rec = source.record(case.example_id) if source is not None else None
        if rec is None:
            result, notes = MISSING, 'example not in run'
        else:
            results = [check(rec, c) for c in case.checks]
            result = PASS if all(good for good, _ in results) else FAIL
            notes = '; '.join(f'{c["op"]}:{"ok" if good else "X"}({note})' for c, (good, note) in zip(case.checks, results))
        rows.append(
            {
                'case': case.case,
                'tier': case.tier,
                'example_id': case.example_id,
                'result': result,
                'known_fail': case.known_fail,
                'known_reason': case.known_reason,
                'notes': notes,
                'package': describe(rec),
            }
        )
    return pd.DataFrame(rows, columns=['case', 'tier', 'example_id', 'result', 'known_fail', 'known_reason', 'notes', 'package'])


def problems(table: pd.DataFrame) -> pd.DataFrame:
    """The gate's problems among tiers A and B: new (fails, not known_fail), resolved (known_fail, passes), missing."""
    gating = table[table['tier'].isin(GATING_TIERS)].copy()
    failed = gating['result'] == FAIL
    gating['problem'] = ''
    gating.loc[failed & ~gating['known_fail'], 'problem'] = 'new'
    gating.loc[(gating['result'] == PASS) & gating['known_fail'], 'problem'] = 'resolved'
    gating.loc[gating['result'] == MISSING, 'problem'] = 'missing'
    found = gating[gating['problem'] != '']
    return found[['problem', 'case', 'tier', 'example_id', 'result', 'notes', 'known_reason']].reset_index(drop=True)


def counts(table: pd.DataFrame) -> str:
    """'A: 43 PASS, 8 FAIL (8 known) | B: ...' per tier."""
    parts = []
    for tier, rows in table.groupby('tier', sort=True):
        tally = rows['result'].value_counts()
        text = ', '.join(f'{int(tally[r])} {r}' for r in (PASS, FAIL, MISSING) if r in tally)
        known = int((rows['known_fail'] & (rows['result'] != PASS)).sum())
        parts.append(f'{tier}: {text}' + (f' ({known} known)' if known else '') + ('' if tier in GATING_TIERS else ' (not gating)'))
    return ' | '.join(parts)


def format_table(table: pd.DataFrame) -> str:
    shown = table.assign(known=table['known_fail'].map({True: 'known', False: ''}))
    lines = [f'episode expectations: {counts(table)}']
    for row in shown.itertuples(index=False):
        lines.append(f'{row.case:4} {row.tier} {row.result:7} {row.known:5} {row.example_id:48} {row.notes[:140]}')
    return '\n'.join(lines)


def format_problems(found: pd.DataFrame) -> str:
    if found.empty:
        return 'episode expectations: no differences from the known failures'
    hint = {
        'new': f'fails and is not known_fail: fix the code, or mark it in {CASES_CSV.name} with a reason',
        'resolved': f'known_fail but passes now: set known_fail = 0 in {CASES_CSV.name}',
        'missing': 'the record is not in the run',
    }
    lines = [f'{len(found)} episode cases differ from {CASES_CSV.name}:']
    for problem, rows in found.groupby('problem', sort=False):
        lines.append(f'\n{problem} ({hint[problem]}):')
        lines.append(rows.drop(columns=['problem']).to_string(index=False, max_colwidth=90))
    return '\n'.join(lines)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', help='reclassify output: folder name under benchmark/data, or a path')
    ap.add_argument('--secular-zlk', help='read the secular/ZLK cases from this run folder instead of analysing them')
    args = ap.parse_args(argv)
    long_period = read_run(args.secular_zlk) if args.secular_zlk else None
    table = evaluate(args.run, long_period=long_period)
    print(format_table(table))
    found = problems(table)
    print(format_problems(found))
    return 1 if len(found) else 0


if __name__ == '__main__':
    raise SystemExit(main())
