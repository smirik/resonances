"""W29: the action in the right column's plaque (the author on q05, 2026-11-02: TX1's 'extend' was missed in the
sentences, so it goes into the header): the secular/ZLK figure draws it as the header's chip. A time in its unit with 3
significant digits (2e5 yr = 200 kyr). The plaque tag and the extension sentences of the old column were removed with
it (2026-11-14)."""

from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from resonances.plotting import secular_plots as sp  # noqa: E402
from resonances.plotting import verdict_column as vc  # noqa: E402
from resonances.resonance.analysis.verdict import Action  # noqa: E402
from tests.resonances.plotting.test_w29_figures import _zlk  # noqa: E402


# --------------------------------------------------------------------------------------------- the plaque


def test_the_long_period_figure_draws_the_action_in_its_plaque():
    """The secular/ZLK figure draws the action as the header's one chip (`verdict_column`, 2026-11-04): 'extend to
    200 kyr' (2.0e5 yr), right of the status word on the plaque's line, and the column fits without growing."""
    analysis = _zlk(2.5)
    analysis = replace(analysis, verdict=replace(analysis.verdict, action=Action.EXTEND, extend_to_years=2.0e5))
    fig = sp.draw_long_period_combined(analysis)
    try:
        texts = {t.get_text(): t for t in fig.texts if t.get_gid() == 'verdict_column'}
        column = fig.text_stats['column']
        assert column.header.chip.kind == 'extend' and column.header.chip.text == 'extend to 200 kyr'
        assert vc.problems(column) == []  # the action once: the chip, never repeated in the text
        tag, badge = texts['extend to 200 kyr'], texts[column.header.word]
        assert tag.get_position()[0] > badge.get_position()[0]
        assert tag.get_position()[1] == pytest.approx(badge.get_position()[1], abs=0.1)  # one line (inches)
        assert fig.text_stats['overflow'] == 0 and fig.text_stats['grown_in'] == 0
    finally:
        plt.close(fig)
