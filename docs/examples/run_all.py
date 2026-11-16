"""Run every code example of the documentation: `make docs-examples` (or `uv run python docs/examples/run_all.py`).

The pages include their code from the scripts in this folder (pymdownx.snippets), so what the reader sees is what
runs here. Each script runs in a fresh process at `nice -n 10`, in one temporary working directory shared by all
scripts (the planets, the MMR catalogues and the outputs go to its cache/; nothing is written into the repository).
Every *.py of this folder except this one is run (one script per page or group of pages). Short spans only; the
network is needed (NASA Horizons, AstDyS for the clones). The MPC examples read the 20-row sample
tests/fixtures/mpc/MPCORB_sample.DAT instead of downloading MPCORB (MPC_CATALOG, MPC_CACHE_DIR below).

    uv run python docs/examples/run_all.py                    # every script
    uv run python docs/examples/run_all.py --only core,examples --keep

Exits with 1 when any script fails.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MPC_SAMPLE = ROOT / 'tests' / 'fixtures' / 'mpc' / 'MPCORB_sample.DAT'


def scripts(only=None):
    found = sorted(path for path in HERE.glob('*.py') if path.name != Path(__file__).name)
    if only:
        wanted = {name.strip().removesuffix('.py') for name in only.split(',') if name.strip()}
        found = [path for path in found if path.stem in wanted]
    return found


def environment(workdir: Path) -> dict:
    env = dict(os.environ)
    env.update(
        {
            'MPLBACKEND': 'Agg',
            'MPC_CATALOG': MPC_SAMPLE.as_uri(),
            'MPC_CACHE_DIR': str(workdir / 'cache' / 'mpc'),
            'MPC_MAX_AGE_DAYS': '0',
            'CLONES_CACHE_DIR': str(workdir / 'cache' / 'clones'),
            'BATCH_N_CORES': '2',
            'OMP_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
        }
    )
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', help='comma-separated script names (without .py)')
    parser.add_argument('--keep', action='store_true', help='keep the temporary working directory')
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix='resonances-docs-examples-'))
    nice = ['nice', '-n', '10'] if shutil.which('nice') else []
    failed = []
    for script in scripts(args.only):
        start = time.time()
        print(f'--- {script.name}', flush=True)
        result = subprocess.run(nice + [sys.executable, str(script)], cwd=workdir, env=environment(workdir))
        elapsed = time.time() - start
        status = 'ok' if result.returncode == 0 else f'FAILED ({result.returncode})'
        print(f'--- {script.name}: {status} in {elapsed:.0f} s', flush=True)
        if result.returncode != 0:
            failed.append(script.name)
    if args.keep:
        print(f'outputs kept in {workdir}')
    else:
        shutil.rmtree(workdir, ignore_errors=True)
    if failed:
        print(f'failed: {", ".join(failed)}')
        return 1
    print('all examples ran')
    return 0


if __name__ == '__main__':
    sys.exit(main())
