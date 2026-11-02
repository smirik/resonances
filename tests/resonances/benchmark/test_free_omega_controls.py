"""Control rows of the free-element anti-alias filter (W28 D1, cards g09-1 and g09-9).

Before W28 a decimation factor with a prime above 10 skipped the filter: 555527 (100 kyr at a 15.918-yr step, factor
31) kept its 15.9-yr grid, Jupiter's 11.86-yr term folded into the secular band, and the free varpi wound by whole
turns where that ripple outgrew the free circle (26 turns of excess in one 2-kyr window; 0.37 after the filter). The
kinematic rows below must not rest on such steps: their gate outcome stays kinematic (the author's labels for 591986
and 624; 1866's gate is kinematic while its episodic row keeps its transient) and no 2-kyr window of the free varpi
gains more than 0.75 turn beyond its median rate. Their factors (5 and 10 x 5) never had a leftover, so D1 leaves
them bit for bit; the test guards any later change of the filter. 555527's status itself is left to its own cards
(D2 measures its omega in the invariable plane); here only its steps are checked.

Local-only data (benchmark/data/examples): marker benchmark.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[3] / 'benchmark' / 'tools'
sys.path.insert(0, str(TOOLS))

import bench_io as bio  # noqa: E402

KINEMATIC_CONTROLS = ('zlk__591986__LK', 'zlk__591986__LK__4myr', 'zlk__624__LK', 'zlk__1866__LK')
WINDOW_YR = 2000.0
MAX_EXCESS_TURNS = 0.75  # card g09-9: a step is a gain of more than 0.75 turn in 2 kyr beyond the median rate


def _analysed(example_id):
    """(the body's FreeElements, the row's summary dict) after the classifier ran on the saved example."""
    import reclassify_secular_zlk as rz

    if not (bio.DATA / 'examples' / example_id).is_dir():
        pytest.skip(f'no saved example {example_id} in {bio.DATA}')
    example = bio.load_example(example_id, with_planets=False)
    sim, body, resonance = rz.build_simulation(example)
    sim.identify_librations()
    return body.free_elements, rz.summary_row(example_id, body, resonance)


def _largest_window_excess(elements):
    """The largest |gain - median gain| of the free varpi over 2-kyr windows, in turns."""
    times, varpi = elements.times[elements.keep_e], elements.varpi_free
    starts = np.arange(times[0], times[-1] - WINDOW_YR, WINDOW_YR)
    gains = (np.interp(starts + WINDOW_YR, times, varpi) - np.interp(starts, times, varpi)) / (2.0 * np.pi)
    return float(np.max(np.abs(gains - np.median(gains))))


@pytest.mark.benchmark
@pytest.mark.parametrize('example_id', KINEMATIC_CONTROLS)
def test_kinematic_controls_keep_their_gate_and_have_no_free_varpi_steps(example_id):
    elements, row = _analysed(example_id)

    assert row['zlk_gate'] == 'kinematic'
    assert not elements.resample_remainder
    assert _largest_window_excess(elements) < MAX_EXCESS_TURNS


@pytest.mark.benchmark
def test_555527_is_filtered_and_its_free_varpi_has_no_steps():
    elements, row = _analysed('zlk__555527__LK')

    assert elements.resample_steps == (31,)
    assert 'free_resample_remainder' in row['rules_fired']
    assert np.median(np.diff(elements.times)) == pytest.approx(31 * 15.918, rel=1e-3)
    assert _largest_window_excess(elements) < MAX_EXCESS_TURNS
