"""W5: the output step from OUTPUT_STEP is an upper bound and always gives a grid (codex astra W5 finding 4)."""

import pytest

from resonances.simulation.config import nout_for_step


@pytest.mark.parametrize(
    'span, step, rows',
    [
        (20000.0, 5.0, 4001),  # 4000 intervals of exactly 5 yr
        (20000.000000001, 5.0, 4001),  # tmax in whole REBOUND units: not one more row
        (1001.0, 2.5, 402),  # 400.4 -> 401 intervals of 2.4963 yr (round() gave 400 of 2.5025 yr, above the bound)
        (1.0, 2.5, 2),  # shorter than one step: one interval (round() gave Nout = 1, no step at all)
        (-500.0, 0.6, 835),  # backward: 833.3 -> 834 intervals
    ],
)
def test_nout_keeps_the_step_within_its_bound(span, step, rows):
    nout = nout_for_step(span, step)
    assert nout == rows
    assert abs(span) / (nout - 1) <= step * (1 + 1e-12)
