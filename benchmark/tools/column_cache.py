"""A binary copy of the benchmark's CSV series for the reclassify runs and the tests (outside git, under
<data>/cache/columns/).

The package loader (`resonances.resonance.analysis.mmr.io`) parses series.csv.gz and planets.csv.gz as text, ~21 % of a
reclassify run (v2-w21a: 367 s of 1778 s). `install()` puts `read_columns` in place of the loader's `_read_columns` in
this process: the first read of a file stores its columns as float64 .npy files (one folder per CSV), later reads take
the requested columns from them (2026-10-23: 20 examples 23.8 s -> 2.9 s of loading; all 719, 7 workers: 411 s -> 34 s
of loading, wall 332 s -> 249 s, the same summary.csv; the arrays are bit-identical, tests/.../test_column_cache.py).
Most of the saving is the run planets file that many examples share: it was parsed again for each of them. The copy is
made by the loader's own reader, so the column renames and the empty-field NaNs stay the package's; a file whose mtime
or size changed is read again.

The planets' columns come back as read-only memory maps of their .npy files (W23 P5): clean file-backed pages are not
in the process' footprint, the OS drops them under pressure and a column nothing reads is never paged in (1M rows, 8
planets: 680 -> 313 MB after the load; 37087 at 1 Myr -471 MB at the peak). Nothing may write into them: a write raises.
The body's own columns are loaded into memory as before.

Only the columns the loader can ask for are kept (the series columns, the planets' `io.PLANET_FIELDS` and t_yr); a
request for another column reads the CSV. A file reached through a symlink (an example's planets.csv.gz) is cached
once, by its real path. With the default root only files under the data root are kept (a test's temporary CSV is
read directly); an explicit `root` keeps any file, under external/<hash>. The cache is ~7.4 GB for the 719 MMR
examples; delete <data>/cache/columns freely (the .npz files of format 1 are no longer read). BENCHMARK_COLUMN_CACHE=0
turns it off.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

ENV = 'BENCHMARK_COLUMN_CACHE'
VERSION = 2  # 1: one .npz per CSV; 2: a folder of .npy per CSV, the planets' columns memory-mapped
SOURCE_KEY = '__source__'  # [version, mtime_ns, size] of the CSV the copy was made from
SUFFIX = '.columns'  # <the CSV's path relative to the data root><SUFFIX>/ holds its columns

_original = None


def enabled() -> bool:
    return os.environ.get(ENV, '1').strip().lower() not in ('0', 'false', 'no', 'off')


def cache_root() -> Path:
    from resonances.resonance.analysis.mmr import io

    return io.data_root() / 'cache' / 'columns'


def _relative(path: Path) -> Optional[Path]:
    """The file's real path relative to the data root; None outside it."""
    from resonances.resonance.analysis.mmr import io

    try:
        return Path(path).resolve().relative_to(io.data_root().resolve())
    except ValueError:
        return None


def cache_path(path: Path, root: Optional[Path] = None) -> Path:
    """<root>/<the file's real path relative to the data root><SUFFIX> (a folder); a file outside the data root by a hash
    of its path."""
    root = Path(root) if root is not None else cache_root()
    relative = _relative(path)
    if relative is None:
        real = Path(path).resolve()
        relative = Path('external') / hashlib.sha1(str(real).encode()).hexdigest()[:16] / real.name
    return root / (str(relative) + SUFFIX)


def is_mapped(name: str) -> bool:
    """A planet's column (`<Planet>.<field>`): served as a read-only memory map."""
    return '.' in name


def _source(path: Path) -> np.ndarray:
    stat = Path(path).stat()
    return np.array([VERSION, stat.st_mtime_ns, stat.st_size], dtype=np.int64)


def is_cacheable(name: str) -> bool:
    """A column the copy keeps: a series column, or a planet's `io.PLANET_FIELDS` field."""
    from resonances.resonance.analysis.mmr import io

    return name in io.SERIES_COLUMNS or ('.' in name and name.split('.', 1)[1] in io.PLANET_FIELDS)


def _file_stem(name: str) -> str:
    """The column's file name without .npy: each capital letter is written as ^ and the lower-case letter, since a
    case-insensitive file system (macOS) takes Omega.npy and omega.npy for one file."""
    return ''.join(f'^{c.lower()}' if c.isupper() else c for c in name)


def _column_name(stem: str) -> str:
    """`_file_stem` undone."""
    parts = stem.split('^')
    return parts[0] + ''.join(part[:1].upper() + part[1:] for part in parts[1:])


def _column_file(target: Path, name: str) -> Path:
    return target / (_file_stem(name) + '.npy')


def _load(target: Path, source: np.ndarray, wanted: set) -> Optional[Dict[str, np.ndarray]]:
    """The wanted columns from a valid copy (those the CSV lacks are absent, as from the reader; a planet's column as a
    read-only memory map); None when there is no copy, it is stale or unreadable."""
    marker = _column_file(target, SOURCE_KEY)
    if not marker.exists():
        return None
    try:
        if not np.array_equal(np.load(marker, allow_pickle=False), source):
            return None
        present = {_column_name(p.name[: -len('.npy')]) for p in target.iterdir() if p.name.endswith('.npy') and not p.name.startswith('.')}
        columns = {}
        for name in wanted & (present - {SOURCE_KEY}):
            if is_mapped(name):  # np.asarray: a plain read-only ndarray over the map, not an np.memmap
                columns[name] = np.asarray(np.load(_column_file(target, name), mmap_mode='r', allow_pickle=False))
            else:
                columns[name] = np.load(_column_file(target, name), allow_pickle=False)
        return columns
    except (OSError, ValueError, KeyError, EOFError):  # a broken or half-written copy: read the CSV
        return None


def read_columns(path, wanted: Iterable[str], reader=None, root: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """`io._read_columns(path, wanted)` through the cache: the same keys and the same values, bit for bit. `reader` is
    the CSV reader (default: the loader's own); `root` the cache folder (default <data>/cache/columns, which keeps only
    the files under the data root)."""
    from resonances.resonance.analysis.mmr import io

    reader = reader or _original or io._read_columns
    wanted = set(wanted)
    if not all(is_cacheable(name) for name in wanted) or (root is None and _relative(path) is None):
        return reader(path, wanted)
    path = Path(path)
    source = _source(path)
    target = cache_path(path, root)
    columns = _load(target, source, wanted)
    if columns is not None:
        return columns
    import pandas as pd

    header = list(pd.read_csv(path, nrows=0).columns)
    names = [name for name in io._current_names(header).values() if is_cacheable(name)]
    full = reader(path, names)
    if _store(target, full, source):
        columns = _load(target, source, wanted)  # the first read is served as the later ones (maps)
        if columns is not None:
            return columns
    if set(full) <= wanted:
        return full
    return {name: np.array(full[name]) for name in wanted if name in full}  # copies: the rest of the block can go


def _store(target: Path, columns: Dict[str, np.ndarray], source: np.ndarray) -> bool:
    """Write the columns, then the source marker last (a copy is complete when its marker matches); False on an OS
    error. Each file is written to a temporary name and renamed: workers may write the same run's planets at once."""
    try:
        target.mkdir(parents=True, exist_ok=True)
        for name, values in [*columns.items(), (SOURCE_KEY, source)]:
            final = _column_file(target, name)
            tmp = target / f'.{final.name}.{os.getpid()}.tmp.npy'
            try:
                np.save(tmp, np.ascontiguousarray(values))
                os.replace(tmp, final)
            finally:
                tmp.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def install() -> bool:
    """Read the benchmark CSVs through the cache in this process (idempotent); False when BENCHMARK_COLUMN_CACHE=0."""
    global _original
    from resonances.resonance.analysis.mmr import io

    if not enabled():
        return False
    if _original is None:
        _original = io._read_columns
        io._read_columns = read_columns
    return True


def uninstall() -> None:
    global _original
    from resonances.resonance.analysis.mmr import io

    if _original is not None:
        io._read_columns = _original
        _original = None
