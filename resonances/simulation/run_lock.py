"""One run at a time per output folder: `<save_path>/.run.lock`, an exclusive flock held by the coordinating process.

Two runs into one folder (a resume found by the checkpoint, or a launch repeated by mistake) both append their batches
to summary.csv and the analysis tables and leave every row twice (2026-11-06: a clone run started twice wrote 3232
rows for 1616). `Simulation.run` takes the lock before it writes anything and releases it at the end, also on an error.
The lock is an flock on an open file: the system drops it when its process ends, however it ends, so a lock is never
stale and a reused pid cannot hold it. The batch workers hold a shared flock on `.run.workers.lock` while they run;
the coordinator refuses to start while any worker of an earlier run still holds it (workers outlive a killed parent
and still append their batch's rows). The file text (host, pid) is for the message only. flock is advisory and local:
it does not protect a folder on a network filesystem shared by several machines. Where the system has no flock
(Windows) the locks here and in `DataManager._locked` do nothing, as before 1.0: one run per folder is then the user's
care.
"""

import os
import socket
from pathlib import Path
from typing import Optional

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None

HAS_FLOCK = fcntl is not None
LOCK_SH, LOCK_EX, LOCK_NB, LOCK_UN = (fcntl.LOCK_SH, fcntl.LOCK_EX, fcntl.LOCK_NB, fcntl.LOCK_UN) if HAS_FLOCK else (1, 2, 4, 8)
LOCK_NAME = '.run.lock'
WORKERS_NAME = '.run.workers.lock'


def flock(fd: int, operation: int) -> None:
    """`fcntl.flock`, or nothing without it (Windows)."""
    if HAS_FLOCK:
        fcntl.flock(fd, operation)


def _try_flock(fd: int, mode: int) -> bool:
    try:
        flock(fd, mode | LOCK_NB)
        return True
    except BlockingIOError:
        return False


def acquire(folder) -> int:
    """Take the folder's run lock (an open descriptor, held until `release`); raise RuntimeError when another run, or a
    worker of an earlier run, holds the folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / LOCK_NAME
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    if not _try_flock(fd, LOCK_EX):
        holder = os.read(fd, 200).decode(errors='replace').strip() or 'unknown'
        os.close(fd)
        raise RuntimeError(f'another run ({holder}: host, pid) is writing into {folder}: wait for it to end ({path})')
    workers = os.open(folder / WORKERS_NAME, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if not _try_flock(workers, LOCK_EX):
            flock(fd, LOCK_UN)
            os.close(fd)
            raise RuntimeError(f'batch workers of an earlier run still write into {folder}: wait for them to end')
        flock(workers, LOCK_UN)
    finally:
        os.close(workers)
    os.ftruncate(fd, 0)
    os.write(fd, f'{socket.gethostname()} {os.getpid()}\n'.encode())
    return fd


def release(fd: Optional[int], folder=None) -> None:
    """Drop the lock taken by `acquire` and remove its files (a folder with nothing else stays as before the run)."""
    if fd is None:
        return
    if not HAS_FLOCK:  # Windows: nothing to unlock, and an open file cannot be removed
        os.close(fd)
        fd = None
    if folder is not None:
        for name in (WORKERS_NAME, LOCK_NAME):
            try:
                (Path(folder) / name).unlink()
            except OSError:
                pass
    if fd is None:
        return
    try:
        flock(fd, LOCK_UN)
    finally:
        os.close(fd)


def hold_as_worker(folder) -> Optional[int]:
    """A batch worker's shared flock on the folder's workers file (blocks nothing but a new coordinator); None when the
    folder does not exist."""
    folder = Path(folder)
    if not folder.is_dir():
        return None
    fd = os.open(folder / WORKERS_NAME, os.O_RDWR | os.O_CREAT, 0o644)
    flock(fd, LOCK_SH)
    return fd


def release_worker(fd: Optional[int]) -> None:
    if fd is not None:
        flock(fd, LOCK_UN)
        os.close(fd)
