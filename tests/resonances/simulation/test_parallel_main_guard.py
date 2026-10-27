"""A batch or clone run started from a script without `if __name__ == '__main__':` must say so.

The workers are spawned: each imports the script again, and an unguarded script starts the run again inside the
worker, which multiprocessing refuses while the worker bootstraps. The parent saw only a broken pool and blamed the
memory; it now names the missing guard (and does not retry)."""

import subprocess
import sys
import textwrap

import pytest

from resonances.simulation.parallel_executor import line_outside_main_guard

SCRIPT = textwrap.dedent(
    """\
    from resonances.simulation.parallel_executor import ParallelExecutor

    def main():
        return ParallelExecutor(2).execute_batches([(0, [])], {{}})

    {call}
    """
)


@pytest.mark.parametrize(
    'source, line, outside',
    [
        ("import x\nx.run()\n", 2, True),
        ("import x\nif __name__ == '__main__':\n    x.run()\n", 3, False),
        ("import x\nif '__main__' == __name__:\n    x.run()\n", 3, False),
        ("import x\nif __name__ == '__main__':\n    pass\nx.run()\n", 4, True),
        ("import x\nif __name__ != '__main__':\n    x.run()\n", 3, True),
        ("import x\nif __name__ == '__main__':\n    pass\nelse:\n    x.run()\n", 5, True),
    ],
)
def test_a_line_inside_the_guard_body_only_is_guarded(source, line, outside):
    assert line_outside_main_guard(source, line) is outside


def test_an_unguarded_script_is_told_about_the_guard(tmp_path):
    script = tmp_path / 'unguarded.py'
    script.write_text(SCRIPT.format(call='main()'))

    run = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300, cwd=tmp_path)

    assert run.returncode != 0
    assert "if __name__ == '__main__':" in run.stderr and f'{script}' in run.stderr and 'line 6' in run.stderr
    assert 'out-of-memory' not in run.stderr.splitlines()[-1]
