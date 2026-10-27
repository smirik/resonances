"""Folder layout of a run's output (docs/output.md).

Summary tables (summary.csv, episodes.csv, ...) and simulation.json stay at the top of `save_path`. The series
(data-<body>.csv and the other per-body or per-planet CSVs) go to `save_path/<data_folder>`; the figures to
`plot_path/<plot_figures_folder>` and the combined (verdict) figures to `plot_path/<plot_combined_folder>`, each split
further by a layout: a '/'-separated list of levels, outermost first, e.g. 'confidence/status' puts a figure in
`<confidence>/<status>/`. The same functions sort the benchmark's figures (benchmark/tools/sort_figures.py).
"""

import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple, Union

# A level and the summary.csv column it reads
LEVEL_COLUMNS = {
    'status': 'status',  # resonant, transient, near_separatrix, non_resonant, uncertain, invalid
    'confidence': 'confidence',  # certain, high, medium, low; none for a verdict without a level (uncertain, invalid)
    'type': 'resonance_type',  # mmr, secular, lidov_kozai
    'resonance': 'resonance',  # the resonance key, e.g. 4J-2S-1+0+0-1
    'body': 'name',  # the body's name
}
LEVELS = tuple(LEVEL_COLUMNS)

# Named layouts (any other combination of LEVELS is accepted too); 'flat' = no subfolder
PRESETS = (
    'flat',
    'status',
    'confidence',
    'confidence/status',
    'status/confidence',
    'type/status',
    'type/confidence/status',
    'resonance',
    'body',
)

# The folder of an empty value: a confidence of a verdict without a level, a row without a verdict
EMPTY_FOLDERS = {'confidence': 'none', 'status': 'unclassified'}
FLAT = ('', 'flat')

_UNSAFE = re.compile(r'[^A-Za-z0-9+\-.=_]')


def parse_layout(layout: Union[str, Sequence[str], None]) -> Tuple[str, ...]:
    """The levels of a layout ('confidence/status', a list of level names, or ''/'flat'/None for none).

    Raises ValueError on an unknown or repeated level."""
    if layout is None:
        return ()
    if isinstance(layout, str):
        text = layout.strip().lower()
        if text in FLAT:
            return ()
        levels = tuple(part.strip() for part in text.replace(',', '/').split('/') if part.strip())
    else:
        levels = tuple(str(part).strip().lower() for part in layout if str(part).strip())
    unknown = [level for level in levels if level not in LEVEL_COLUMNS]
    if unknown:
        raise ValueError(
            f"Unknown layout level(s) {', '.join(unknown)}: a layout is 'flat' or levels from {', '.join(LEVELS)} joined by '/'"
        )
    if len(set(levels)) != len(levels):
        raise ValueError(f"A layout level is repeated in {'/'.join(levels)}")
    return levels


def layout_name(levels: Iterable[str]) -> str:
    """The text of a layout ('flat' for none): what the config keeps and simulation.json stores."""
    return '/'.join(levels) or 'flat'


def safe_folder(value) -> str:
    """A value as one folder name: anything outside [A-Za-z0-9+-.=_] becomes '_' (no separators; '.' and '..' too)."""
    text = _UNSAFE.sub('_', '' if value is None else str(value).strip())
    return '_' * len(text) if text in ('.', '..') else text  # never the folder itself or its parent


def layout_parts(levels: Iterable[str], row: Mapping) -> list:
    """The subfolders of a row for these levels, outermost first. `row` has the summary.csv columns of LEVEL_COLUMNS
    (a summary row as it is, or a dict built from a verdict)."""
    parts = []
    for level in levels:
        value = row.get(LEVEL_COLUMNS[level])
        text = '' if value is None or (isinstance(value, float) and value != value) else str(value).strip()
        parts.append(safe_folder(text) if text else EMPTY_FOLDERS.get(level, 'unknown'))
    return parts


def layout_path(base: Union[str, Path], folder: str, levels: Iterable[str], row: Mapping) -> Path:
    """base/folder/<levels of the row>; an empty `folder` is `base` itself."""
    path = Path(base)
    if folder and folder.strip() not in ('', '.'):
        path = path / folder.strip()
    return path.joinpath(*layout_parts(levels, row))


def folder_setting(value) -> str:
    """A folder setting as given ('' = the parent folder itself; None = '')."""
    return '' if value is None else str(value).strip()
