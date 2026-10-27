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
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bench_io as bio  # noqa: E402
from resonances.simulation import layout  # noqa: E402

FIGURE = 'combined_v2.png'
DEFAULT_PRESET = 'confidence/status'
DEFAULT_RUN = 'test-benchmark'  # what `make test-benchmark` writes; any run folder of reclassify.py
SECULAR_ZLK_SUFFIX = '-secular-zlk'  # reclassify_secular_zlk.py's runs are named <run>-secular-zlk by convention
TREE = 'figures'  # under the data root


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


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        '--run', default=DEFAULT_RUN, help=f'comma-separated run folders (names under the data root or paths; default {DEFAULT_RUN})'
    )
    ap.add_argument('--preset', default=DEFAULT_PRESET, help=f'layout: one of {", ".join(layout.PRESETS)} or levels joined by "/"')
    ap.add_argument('--limit', type=int, help='only the first N rows (a check)')
    ap.add_argument('--dry-run', action='store_true', help='print the plan, write nothing')
    ap.add_argument('--data', type=Path, default=bio.DATA, help='the data root (default: benchmark/data or RESONANCES_BENCHMARK_DATA)')
    args = ap.parse_args(argv)

    data = args.data.expanduser().resolve()
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
