"""The gallery's captions (docs/tools/build_figures.py) are written by hand; the level under each comes from the
package. A caption that states a level must agree with its row's verdict in the committed docs/img/gallery/verdicts.json
(what docs/physics/gallery.md prints), so a classifier change that moves a level fails here until the caption follows."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _build_figures():
    spec = importlib.util.spec_from_file_location('build_figures', ROOT / 'docs' / 'tools' / 'build_figures.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up while the module runs
    spec.loader.exec_module(module)
    return module


BF = _build_figures()


def test_the_stated_level_is_read_from_the_caption_or_the_case():
    case = BF.Case('a', 'mmr2__x', 'resonant', 't', 'Three cycles would give certain, so the confidence is high, not certain.')
    assert BF.caption_level(case) == 'high'  # the first 'confidence is', not a level named as a counterfactual
    assert BF.caption_level(BF.Case('a', 'x', 'g', 't', 'The identity fails, which makes the confidence low.')) == 'low'
    assert BF.caption_level(BF.Case('a', 'x', 'g', 't', 'Nothing caps the level, and 19 cycles give certain.')) == ''
    assert BF.caption_level(BF.Case('a', 'x', 'g', 't', '19 cycles give certain.', level='certain')) == 'certain'


def test_a_caption_that_contradicts_its_verdict_is_reported():
    case = BF.CASE_BY_ANCHOR['quick-start-463']
    assert BF.caption_level(case) == 'certain'
    assert BF.caption_mismatches({case.example_id: {'confidence': 'certain'}}) == []
    assert BF.caption_mismatches({case.example_id: {'confidence': 'high'}}) == [
        'quick-start-463: the caption says certain, the verdict is high'
    ]
    assert BF.caption_mismatches({}) == []  # a row not generated yet is not checked


def test_every_stated_caption_level_matches_the_committed_verdicts():
    verdicts = BF.load_verdicts()
    stated = [case for case in BF.CASES if BF.caption_level(case)]
    assert len(stated) >= 15 and all(case.example_id in verdicts for case in stated)
    assert BF.caption_mismatches(verdicts) == []
