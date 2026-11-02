"""A browsing tree of the benchmark's combined figures, sorted by a reclassification run's verdicts.

    uv run python benchmark/tools/sort_figures.py --run v2-w23 --dry-run --limit 20      # print what would be linked
    uv run python benchmark/tools/sort_figures.py --run v2-w23                           # confidence/status (default)
    uv run python benchmark/tools/sort_figures.py --run v2-w23 --preset status/confidence

Each row of the run's summary.csv (and of `<run>-secular-zlk/summary.csv` when that folder exists) puts its example's
figure, `data/examples/<example_id>/combined_v2.png` (a derived subresonance row: `combined_v2__<key>.png`; the MMR,
secular and ZLK figures have the same name), at

    data/figures/<preset>/<levels of the row>/<example_id>.png          ('/' of the preset written '-')

e.g. `data/figures/confidence-status/high/resonant/mmr2__12218__4M-7+0+3.png`; a derived row is
`<example_id>__<key>.png`. The levels are the package's own (resonances/simulation/layout.py: status, confidence,
type, resonance, body; an empty confidence is `none`), so the tree is what a package run with PLOT_LAYOUT=<preset>
would draw. Files are hard links to the examples' figures (a copy where a link cannot be made, e.g. across disks):
the tree costs no space and never writes into data/examples. Every run clears and rebuilds its own
data/figures/<preset>/ only. The folder is the run's verdict; the figure is whatever the last `reclassify.py
--figures` drew, which may be older than the run: a row whose figure is missing is listed, not linked.

Review mode (W28 card s-4): the figures the author re-checks after an iteration, with a CSV for his verdicts.

    uv run python benchmark/tools/sort_figures.py --review w28 --run v2-w28 --baseline v2-w26-final --dry-run

A row of the run (and of its `-secular-zlk` sibling; rows keyed by example_id and resonance) goes into

    data/figures/review-<NAME>/<discussed|changed|same_rule>/<status>/<confidence>/<example_id>.png

in this priority: `discussed`, its example is in --discussed (default benchmark/review/W28_DISCUSSED.csv, the
records the W28 cards named); `changed`, against --baseline: status, angle_status, confidence, reason or
libration_origin differ, the row is new, the number of accepted episodes of a scale differs, an episode has no partner
(episodes are paired by the largest overlap, ties by the smaller boundary shift, then by time: deterministic when the
counts differ), or a paired boundary moved by more than max(--shift-periods x P, --shift-fraction x record span)
(P the larger of the pair's libration periods; tool conventions, 1 and 0.02); `same_rule`, neither, but a rule of
iteration --rules-since (default W28, `resonances.resonance.analysis.rules.rules_since`) is in its `rules_fired`. A row
that is none of these is not in the folder. Next to the tree, `review-<NAME>.csv` (one line per row in the folder):
the old and new verdicts and episodes, why it changed, the rules fired ('name [doc id]'), the cards that discussed
it, the author's prior label (`reclassify.author_labels`) and the empty columns author_verdict and author_comment.
Like the presets, the tree is hard links rebuilt from scratch on every run and never writes into data/examples.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bench_io as bio  # noqa: E402
from resonances.resonance.analysis import rules as rule_registry  # noqa: E402
from resonances.resonance.analysis.verdict import split_list  # noqa: E402
from resonances.simulation import layout  # noqa: E402

FIGURE = 'combined_v2.png'
DEFAULT_PRESET = 'confidence/status'
DEFAULT_RUN = 'test-benchmark'  # what `make test-benchmark` writes; any run folder of reclassify.py
SECULAR_ZLK_SUFFIX = '-secular-zlk'  # reclassify_secular_zlk.py's runs are named <run>-secular-zlk by convention
TREE = 'figures'  # under the data root

# review mode (card s-4)
REVIEW_FOLDERS = ('discussed', 'changed', 'same_rule')  # in priority order: a row goes into the first that holds
REVIEW_PRESET = 'status/confidence'
REVIEW_PREFIX = 'review-'
DISCUSSED = bio.BENCH / 'review' / 'W28_DISCUSSED.csv'
DEFAULT_RULES_SINCE = 'W28'
VERDICT_FIELDS = ('status', 'angle_status', 'confidence', 'reason', 'libration_origin')
# a boundary moved by more than max(k P, f span) is a change; tool conventions of card s-4, not package settings
EPISODE_SHIFT_PERIODS = 1.0
EPISODE_SHIFT_RECORD_FRACTION = 0.02
MAIN_SCALE = 'main'  # the episodes.csv scale of a row without one (as benchmark/tools/episode_expectations.py)
KEY = ('example_id', 'resonance')
REVIEW_COLUMNS = [
    'folder',
    'example_id',
    'name',
    'resonance',
    'parent',
    'resonance_type',
    'changed',
    'change_why',
    *[f'{age}_{name}' for name in VERDICT_FIELDS for age in ('old', 'new')],
    'old_episodes',
    'new_episodes',
    'rules_fired',
    'rules_since',
    'cards',
    'prior_label',
    'prior_label_source',
    'figure',
    'author_verdict',
    'author_comment',
]


def run_dir(run: str, data: Path) -> Path:
    """A run folder: a path, or a name under the data root."""
    path = Path(run)
    return path if path.is_dir() else data / str(run)


def run_folders(runs: Iterable[str], data: Path) -> List[Path]:
    """The run folders to read: each one given and its `<run>-secular-zlk` sibling when it exists, each once."""
    folders = []
    for run in runs:
        folder = run_dir(run, data)
        if not (folder / 'summary.csv').exists():
            raise FileNotFoundError(f'no summary.csv in {folder}')
        sibling = folder.with_name(folder.name + SECULAR_ZLK_SUFFIX)
        for path in (folder, sibling):
            if (path / 'summary.csv').exists() and path.resolve() not in {f.resolve() for f in folders}:
                folders.append(path)
    return folders


def figure_of(row: dict, examples: Path) -> Path:
    """The example's figure of a summary row (a derived row: combined_v2__<key>.png, as reclassify._draw names it)."""
    folder = examples / str(row['example_id'])
    if str(row.get('parent') or '').strip():
        return folder / FIGURE.replace('.png', f'__{bio.safe_token(row["resonance"])}.png')
    return folder / FIGURE


def target_name(row: dict) -> str:
    suffix = f'__{bio.safe_token(row["resonance"])}' if str(row.get('parent') or '').strip() else ''
    return f'{row["example_id"]}{suffix}.png'


def preset_folder(preset: str) -> str:
    """The folder of a preset under data/figures/ ('confidence/status' -> 'confidence-status')."""
    return layout.layout_name(layout.parse_layout(preset)).replace('/', '-')


def plan(folders: List[Path], preset: str, data: Path, limit: Optional[int] = None) -> Tuple[List[Tuple[Path, Path]], List[Path]]:
    """(source figure, target) pairs of the tree and the missing figures, in the order of the runs' rows. A target
    met again (the same example and verdict in two runs) is kept once; a different verdict of the same example in
    another run gives it a second place."""
    levels = layout.parse_layout(preset)
    root = data / TREE / preset_folder(preset)
    examples = data / 'examples'
    pairs, missing, seen = [], [], set()
    for folder in folders:
        frame = pd.read_csv(folder / 'summary.csv', dtype=str, keep_default_na=False)
        for row in frame.to_dict('records'):
            if limit is not None and len(pairs) + len(missing) >= limit:
                return pairs, missing
            source = figure_of(row, examples)
            if not source.exists():
                missing.append(source)
                continue
            target = root.joinpath(*layout.layout_parts(levels, row), target_name(row))
            if target not in seen:
                seen.add(target)
                pairs.append((source, target))
    return pairs, missing


def link(source: Path, target: Path) -> str:
    """A hard link, else a copy; returns which."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return 'link'
    except OSError:
        shutil.copy2(source, target)
        return 'copy'


def build(pairs: List[Tuple[Path, Path]], root: Path, data: Path) -> dict:
    """Clear `root` (only a folder directly under data/figures/, neither of them a symlink) and write the tree."""
    tree = data / TREE
    if root.parent != tree or tree.is_symlink() or root.is_symlink() or root.name in ('', '.', '..'):
        raise ValueError(f'{root} is not a real folder directly under {tree}: refusing to clear it')
    if root.exists():
        shutil.rmtree(root)
    counts = {'link': 0, 'copy': 0}
    for source, target in pairs:
        counts[link(source, target)] += 1
    return counts


# ------------------------------------------------------------------ review mode


@dataclass(frozen=True)
class Episode:
    t0: float
    t1: float
    period: float  # libration period (yr), nan when the run did not measure one


@dataclass
class Tables:
    """summary rows by (example_id, resonance) and the accepted episodes of each key by scale, of one or more run
    folders (a run and its -secular-zlk sibling)."""

    rows: Dict[Tuple[str, str], dict]
    episodes: Dict[Tuple[str, str], Dict[str, List[Episode]]]
    spans: Dict[Tuple[str, str], float] = field(default_factory=dict)  # record span (yr) from smp_span_yr


def _flag(value) -> bool:
    return str(value).strip() in ('True', 'true', '1', '1.0')


def _number(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number


def _key(row: Mapping) -> Tuple[str, str]:
    return str(row.get('example_id', '')), str(row.get('resonance', ''))


def read_tables(folders: List[Path]) -> Tables:
    """The summary rows, accepted episodes and record spans of the run folders (the first row of a key wins)."""
    rows, episodes, spans = {}, {}, {}
    for folder in folders:
        frame = pd.read_csv(folder / 'summary.csv', dtype=str, keep_default_na=False)
        for row in frame.to_dict('records'):
            rows.setdefault(_key(row), row)
            span = _number(row.get('smp_span_yr'))
            if math.isfinite(span):
                spans.setdefault(_key(row), span)
        numbers = folder / 'mmr_numbers.csv'
        if numbers.exists() and 'smp_span_yr' in bio.read_header(numbers):
            for row in pd.read_csv(numbers, usecols=['example_id', 'resonance', 'smp_span_yr'], dtype=str).to_dict('records'):
                span = _number(row['smp_span_yr'])
                if math.isfinite(span):
                    spans.setdefault(_key(row), span)
        path = folder / 'episodes.csv'
        if not path.exists():
            continue
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        if 'accepted' not in table:
            continue
        for row in table[table['accepted'].map(_flag)].to_dict('records'):
            scale = str(row.get('scale') or '').strip() or MAIN_SCALE
            episode = Episode(_number(row.get('t_start_yr')), _number(row.get('t_end_yr')), _number(row.get('period_med_yr')))
            episodes.setdefault(_key(row), {}).setdefault(scale, []).append(episode)
    for by_scale in episodes.values():
        for scale in by_scale:
            by_scale[scale].sort(key=lambda e: (e.t0, e.t1))
    return Tables(rows, episodes, spans)


def record_span(key: Tuple[str, str], tables: Tables, data: Path) -> float:
    """The record span (yr): smp_span_yr of the run, else meta.json regime.span_yr of the example, else 0 (then only
    the period part of the shift threshold counts)."""
    if key in tables.spans:
        return tables.spans[key]
    meta = data / 'examples' / key[0] / 'meta.json'
    if meta.exists():
        span = _number((json.loads(meta.read_text()).get('regime') or {}).get('span_yr'))
        if math.isfinite(span):
            return span
    return 0.0


def _overlap(a: Episode, b: Episode) -> float:
    return min(a.t1, b.t1) - max(a.t0, b.t0)


def _shift(a: Episode, b: Episode) -> float:
    return max(abs(a.t0 - b.t0), abs(a.t1 - b.t1))


def pair_episodes(old: List[Episode], new: List[Episode]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """(pairs (old index, new index), unpaired old, unpaired new). Pairs are taken greedily by the largest positive
    overlap, ties by the smaller boundary shift, then by the indices (time order): the same episodes always give the
    same pairs, whatever their counts."""
    candidates = sorted(
        ((-_overlap(a, b), _shift(a, b), i, j) for i, a in enumerate(old) for j, b in enumerate(new) if _overlap(a, b) > 0),
    )
    pairs, used_old, used_new = [], set(), set()
    for _, _, i, j in candidates:
        if i not in used_old and j not in used_new:
            pairs.append((i, j))
            used_old.add(i)
            used_new.add(j)
    pairs.sort()
    return pairs, [i for i in range(len(old)) if i not in used_old], [j for j in range(len(new)) if j not in used_new]


def shift_limit(a: Episode, b: Episode, span: float, periods: float, fraction: float) -> float:
    """max(periods x P, fraction x span), P the larger finite libration period of the pair (0 when neither has one)."""
    finite = [p for p in (a.period, b.period) if math.isfinite(p)]
    return max(periods * (max(finite) if finite else 0.0), fraction * span)


def episode_changes(
    old: Mapping[str, List[Episode]], new: Mapping[str, List[Episode]], span: float, periods: float, fraction: float
) -> List[str]:
    """Why the accepted episodes changed, per scale: 'episodes@<scale>: count a -> b', 'unpaired', 'shift'."""
    why = []
    for scale in sorted(set(old) | set(new)):
        before, after = old.get(scale, []), new.get(scale, [])
        if len(before) != len(after):
            why.append(f'episodes@{scale}: count {len(before)} -> {len(after)}')
            continue
        pairs, lone_old, lone_new = pair_episodes(before, after)
        if lone_old or lone_new:
            why.append(f'episodes@{scale}: unpaired {len(lone_old)} old, {len(lone_new)} new')
            continue
        for i, j in pairs:
            a, b = before[i], after[j]
            limit = shift_limit(a, b, span, periods, fraction)
            if _shift(a, b) > limit:
                why.append(f'episodes@{scale}: shift {_shift(a, b):.0f} > {limit:.0f} yr ({b.t0:.0f}-{b.t1:.0f})')
    return why


def verdict_changes(old: Mapping, new: Mapping) -> List[str]:
    return [name for name in VERDICT_FIELDS if str(old.get(name, '')).strip() != str(new.get(name, '')).strip()]


def episodes_text(by_scale: Mapping[str, List[Episode]]) -> str:
    parts = []
    for scale in sorted(by_scale, key=lambda s: (s != MAIN_SCALE, s)):
        prefix = '' if scale == MAIN_SCALE else f'{scale}:'
        for e in by_scale[scale]:
            period = f' (P {e.period:.0f})' if math.isfinite(e.period) else ''
            parts.append(f'{prefix}{e.t0:.0f}-{e.t1:.0f}{period}')
    return '; '.join(parts)


def rules_text(names: Iterable[str]) -> str:
    """'name [doc id]' for each rule (a name outside the registry as it is)."""
    out = []
    for name in names:
        try:
            out.append(f'{name} [{rule_registry.RULES[rule_registry.Rule(name)].doc_id}]')
        except ValueError:
            out.append(name)
    return '; '.join(out)


def read_discussed(path: Optional[Path]) -> Dict[str, str]:
    """example_id -> the cards that discussed it (';'-separated); {} without a file."""
    if path is None or not Path(path).exists():
        return {}
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:  # an empty file (e.g. /dev/null): nothing discussed
        return {}
    return dict(zip(frame['example_id'], frame['cards'] if 'cards' in frame else [''] * len(frame)))


def prior_labels(bench: Optional[Path]) -> Dict[str, Tuple[str, str]]:
    """example_id -> ('status/confidence/angle_status', label source) of the author's newest label."""
    import reclassify

    frame = reclassify.author_labels(bench).fillna('')
    labels = {}
    for row in frame.to_dict('records'):
        text = '/'.join(str(row[c]) for c in ('author_status', 'author_confidence', 'author_angle_status'))
        labels[row['example_id']] = (text, str(row['label_source']))
    return labels


@dataclass
class ReviewRow:
    folder: str
    row: dict
    source: Path
    target: Path
    line: dict  # the CSV line


def review_plan(
    run: Tables,
    baseline: Tables,
    discussed: Mapping[str, str],
    since: str,
    data: Path,
    name: str,
    labels: Mapping[str, Tuple[str, str]],
    periods: float = EPISODE_SHIFT_PERIODS,
    fraction: float = EPISODE_SHIFT_RECORD_FRACTION,
) -> Tuple[List[ReviewRow], List[Tuple[str, str]]]:
    """The rows of the review folder (in the run's order) and the keys of the baseline missing from the run."""
    levels = layout.parse_layout(REVIEW_PRESET)
    root = data / TREE / f'{REVIEW_PREFIX}{name}'
    examples = data / 'examples'
    new_rules = {str(rule) for rule in rule_registry.rules_since(since)}
    out = []
    for key, row in run.rows.items():
        old = baseline.rows.get(key)
        before, after = baseline.episodes.get(key, {}), run.episodes.get(key, {})
        if old is None:
            why = ['new row']
        else:
            why = verdict_changes(old, row) + episode_changes(before, after, record_span(key, run, data), periods, fraction)
        fired = split_list(row.get('rules_fired'))
        since_fired = [r for r in fired if r in new_rules]
        if key[0] in discussed:
            folder = 'discussed'
        elif why:
            folder = 'changed'
        elif since_fired:
            folder = 'same_rule'
        else:
            continue
        source = figure_of(row, examples)
        target = root.joinpath(folder, *layout.layout_parts(levels, row), target_name(row))
        derived = bool(str(row.get('parent') or '').strip())
        label, label_source = ('', '') if derived else labels.get(key[0], ('', ''))
        old = old or {}
        line = {
            'folder': folder,
            **{c: row.get(c, '') for c in ('example_id', 'name', 'resonance', 'parent', 'resonance_type')},
            'changed': bool(why),
            'change_why': '; '.join(why),
            **{f'{age}_{n}': (old if age == 'old' else row).get(n, '') for n in VERDICT_FIELDS for age in ('old', 'new')},
            'old_episodes': episodes_text(before),
            'new_episodes': episodes_text(after),
            'rules_fired': rules_text(fired),
            'rules_since': '; '.join(since_fired),
            'cards': discussed.get(key[0], ''),
            'prior_label': label,
            'prior_label_source': label_source,
            'figure': target.relative_to(root).as_posix() if source.exists() else '',
            'author_verdict': '',
            'author_comment': '',
        }
        out.append(ReviewRow(folder, row, source, target, line))
    gone = [key for key in baseline.rows if key not in run.rows]
    return out, gone


def review(args, data: Path) -> int:
    name = args.review.strip()
    if not name or '/' in name or name in ('.', '..') or os.sep in name:
        raise ValueError(f'--review {args.review!r}: a plain folder name is needed')
    if not args.baseline:
        raise ValueError('--review needs --baseline')
    run = read_tables(run_folders([r.strip() for r in args.run.split(',') if r.strip()], data))
    baseline = read_tables(run_folders([r.strip() for r in args.baseline.split(',') if r.strip()], data))
    discussed = read_discussed(args.discussed)
    labels = prior_labels(args.bench)
    rows, gone = review_plan(run, baseline, discussed, args.rules_since, data, name, labels, args.shift_periods, args.shift_fraction)
    root = data / TREE / f'{REVIEW_PREFIX}{name}'
    counts = {folder: sum(r.folder == folder for r in rows) for folder in REVIEW_FOLDERS}
    missing = [r for r in rows if not r.source.exists()]
    print(f'review: {root} ({", ".join(f"{k} {v}" for k, v in counts.items())}; {len(missing)} figures missing)')
    for r in missing:
        print(f'  missing: {r.source}')
    for key in gone:
        print(f'  not in the run (baseline only): {key[0]} {key[1]}')
    if args.dry_run:
        for r in rows:
            print(f'  {r.target.relative_to(root)}  {r.line["change_why"] or r.line["rules_since"] or r.line["cards"]}')
        return 0
    result = build([(r.source, r.target) for r in rows if r.source.exists()], root, data)
    csv = root / f'{REVIEW_PREFIX}{name}.csv'
    csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.line for r in rows], columns=REVIEW_COLUMNS).to_csv(csv, index=False)
    print(f'written: {result["link"]} hard links, {result["copy"]} copies; {csv}')
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        '--run', default=DEFAULT_RUN, help=f'comma-separated run folders (names under the data root or paths; default {DEFAULT_RUN})'
    )
    ap.add_argument('--preset', default=DEFAULT_PRESET, help=f'layout: one of {", ".join(layout.PRESETS)} or levels joined by "/"')
    ap.add_argument('--limit', type=int, help='only the first N rows (a check)')
    ap.add_argument('--dry-run', action='store_true', help='print the plan, write nothing')
    ap.add_argument('--data', type=Path, default=bio.DATA, help='the data root (default: benchmark/data or RESONANCES_BENCHMARK_DATA)')
    review_args = ap.add_argument_group('review mode (card s-4)')
    review_args.add_argument('--review', metavar='NAME', help='build data/figures/review-NAME/ instead of a preset tree')
    review_args.add_argument('--baseline', help='the run to compare with (comma-separated, like --run)')
    review_args.add_argument('--discussed', type=Path, default=DISCUSSED, help='CSV example_id,cards of the discussed records')
    review_args.add_argument('--rules-since', default=DEFAULT_RULES_SINCE, help='the iteration whose rules fill same_rule/')
    review_args.add_argument('--shift-periods', type=float, default=EPISODE_SHIFT_PERIODS, help='boundary shift limit in libration periods')
    review_args.add_argument(
        '--shift-fraction', type=float, default=EPISODE_SHIFT_RECORD_FRACTION, help='boundary shift limit as a share of the record'
    )
    review_args.add_argument('--bench', type=Path, help='the folder with the author label files (default benchmark/)')
    args = ap.parse_args(argv)

    data = args.data.expanduser().resolve()
    if args.review is not None:
        return review(args, data)
    folders = run_folders([r.strip() for r in args.run.split(',') if r.strip()], data)
    pairs, missing = plan(folders, args.preset, data, args.limit)
    root = data / TREE / preset_folder(args.preset)
    print(f'runs: {", ".join(str(f) for f in folders)}')
    print(f'tree: {root} ({len(pairs)} figures, {len(missing)} missing)')
    for path in missing:
        print(f'  missing: {path}')
    if args.dry_run:
        for source, target in pairs:
            print(f'  {target.relative_to(root)}  <-  {source.relative_to(data)}')
        return 0
    counts = build(pairs, root, data)
    print(f'written: {counts["link"]} hard links, {counts["copy"]} copies')
    return 0


if __name__ == '__main__':
    sys.exit(main())
