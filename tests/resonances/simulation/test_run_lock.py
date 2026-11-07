"""One run per output folder (simulation/run_lock.py): an flock held by the coordinator, a shared one by its workers."""

import subprocess
import sys
import textwrap

import pytest

from resonances.simulation import run_lock


def _child(folder, call: str, ready):
    """A child process that takes a lock in `folder` (`call`), says so through the file `ready`, and sleeps."""
    code = textwrap.dedent(
        f"""
        import time
        from resonances.simulation import run_lock
        run_lock.{call}({str(folder)!r})
        open({str(ready)!r}, 'w').write('ok')
        time.sleep(60)
    """
    )
    child = subprocess.Popen([sys.executable, '-c', code])
    for _ in range(600):  # up to 30 s for the import
        if ready.exists():
            return child
        if child.poll() is not None:
            break
        subprocess.run(['sleep', '0.05'])
    child.kill()
    raise AssertionError('the child did not take its lock')


def test_lock_is_taken_and_released(tmp_path):
    folder = tmp_path / 'run'
    fd = run_lock.acquire(folder)
    assert (folder / run_lock.LOCK_NAME).exists()
    run_lock.release(fd, folder)
    assert not (folder / run_lock.LOCK_NAME).exists() and list(folder.iterdir()) == []


def test_a_live_run_holds_the_folder(tmp_path):
    child = _child(tmp_path, 'acquire', tmp_path.parent / f'{tmp_path.name}.ready')
    try:
        with pytest.raises(RuntimeError, match='another run'):
            run_lock.acquire(tmp_path)
    finally:
        child.kill()
        child.wait()
    fd = run_lock.acquire(tmp_path)  # the system dropped the dead child's lock: never stale
    run_lock.release(fd, tmp_path)


def test_a_worker_of_an_earlier_run_holds_the_folder(tmp_path):
    """Workers outlive a killed parent and still append rows: a new run waits for them."""
    child = _child(tmp_path, 'hold_as_worker', tmp_path.parent / f'{tmp_path.name}.ready')
    try:
        with pytest.raises(RuntimeError, match='workers'):
            run_lock.acquire(tmp_path)
    finally:
        child.kill()
        child.wait()
    run_lock.release(run_lock.acquire(tmp_path), tmp_path)


def test_a_coordinator_does_not_block_its_own_workers(tmp_path):
    fd = run_lock.acquire(tmp_path)
    hold = run_lock.hold_as_worker(tmp_path)
    run_lock.release_worker(hold)
    run_lock.release(fd, tmp_path)
