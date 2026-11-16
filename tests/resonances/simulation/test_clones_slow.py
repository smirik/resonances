"""A real run with clones: 463 Lola in 4J-2S-1, 10 clones from AstDyS (downloaded now), 20 kyr at a 1-yr step.

463's orbit is known to sigma_a = 4.9e-9 AU, five orders below the resonance's width, and its record is regular over
20 kyr: every clone must get the nominal's verdict (clones of a good main-belt orbit differ only through chaos)."""

import pandas as pd
import pytest

import resonances


@pytest.mark.slow
def test_check_463_with_ten_clones(tmp_path):
    previous = resonances.config.get('CLONES_CACHE_DIR')
    resonances.config.set('CLONES_CACHE_DIR', str(tmp_path / 'clones-cache'))
    try:
        sim = resonances.check(
            463,
            '4J-2S-1',
            source='astdys',
            clones=10,
            integration_years=20_000,
            output_step=1.0,  # the step OUTPUT_STEP=auto takes for 463: no short period folded
            save_path=str(tmp_path / 'run'),
            plot_path=str(tmp_path / 'run'),
            save='nonzero',
            plot='all',
            plots=['clones'],
            save_summary=True,
            n_cores=1,
            clones_batch=5,
        )
        sim.run()
    finally:
        resonances.config.set('CLONES_CACHE_DIR', previous)

    row = pd.read_csv(tmp_path / 'run' / 'clones.csv', dtype={'name': str}).iloc[0]
    assert row['n_clones'] == 10 and row['n_classified'] == 10
    assert row['source'] == 'astdys' and row['product'] == 'eq1' and not row['propagated']
    assert row['nominal_offset_sigma'] < 3  # allnum.cat and the .eq1 record are the same AstDyS orbit
    assert row['modal_status'] == row['nominal_status'] and row['modal_share'] == 1.0
    summary = pd.read_csv(tmp_path / 'run' / 'summary.csv', dtype=str, keep_default_na=False)
    assert len(summary) == 11  # the nominal and its 10 clones
    for column in ('status', 'angle_status', 'reason'):  # 20 kyr is short for this libration: the verdict is the same
        assert summary[column].nunique() == 1, column
    assert row['nominal_agreement_lo'] == pytest.approx(10 / (10 + 1.959963984540054**2))  # Wilson's bound at 10/10
    assert len(list((tmp_path / 'run').rglob('463-4J-2S-1+0+0-1-clones.png'))) == 1
