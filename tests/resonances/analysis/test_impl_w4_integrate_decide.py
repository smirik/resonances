"""Step 5 of the status tree (SPEC §5.3): the four combinations of the force law (L4) and the potential with a (R0),
for every angle class that reaches it. The expected leaves are written out by hand from the SPEC, not derived from
the code: 5b (a_dot follows the force law, the energy disagrees with the potential from a) is uncertain/origin_unknown
for EVERY angle class, near_separatrix included (Codex W4 review, P1: it had been non_resonant there)."""

import pytest

from tests.resonances.analysis.builders import make_energy, make_kinematics, make_response, make_sampling
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import decide
from resonances.resonance.analysis.mmr.decide import AngleDecision
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, CheckValue, LibrationOrigin, Reason, Status

S = settings_from_config()
PASS, FAIL, NM = CheckValue.PASS, CheckValue.FAIL, CheckValue.NOT_MEASURED

LIBRATION = AngleDecision(AngleStatus.LIBRATION, AngleSubtype.STATIONARY, 'G2')
TRANSIENT = AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2')
NEAR_A = AngleDecision(AngleStatus.NEAR_SEPARATRIX, AngleSubtype.PATTERN_A, 'G3')
NEAR_B = AngleDecision(AngleStatus.NEAR_SEPARATRIX, AngleSubtype.PATTERN_B, 'G3')

UNKNOWN = (Status.UNCERTAIN, Reason.ORIGIN_UNKNOWN, LibrationOrigin.UNKNOWN)

# (angle, L4, R0) -> (status, reason, origin, rule), SPEC §5.3 steps 5a, 5a*, 5b, 5e
TABLE = [
    (LIBRATION, FAIL, FAIL, (*UNKNOWN, '5a')),
    (TRANSIENT, FAIL, FAIL, (*UNKNOWN, '5a')),
    (NEAR_A, FAIL, FAIL, (Status.NON_RESONANT, Reason.NO_RESONANT_POTENTIAL, LibrationOrigin.NONE, '5a')),
    (NEAR_B, FAIL, FAIL, (Status.NON_RESONANT, Reason.MODULATED_CIRCULATION, LibrationOrigin.NONE, '5a')),
    (LIBRATION, FAIL, NM, (*UNKNOWN, '5a*')),
    (NEAR_B, FAIL, NM, (Status.NON_RESONANT, Reason.MODULATED_CIRCULATION, LibrationOrigin.NONE, '5a*')),
    (LIBRATION, PASS, FAIL, (*UNKNOWN, '5b')),
    (TRANSIENT, PASS, FAIL, (*UNKNOWN, '5b')),
    (NEAR_A, PASS, FAIL, (*UNKNOWN, '5b')),
    (NEAR_B, PASS, FAIL, (*UNKNOWN, '5b')),
    (LIBRATION, NM, FAIL, (*UNKNOWN, '5e')),
    (NEAR_A, NM, FAIL, (Status.NON_RESONANT, Reason.NO_RESONANT_POTENTIAL, LibrationOrigin.NONE, '5e')),
]


def _response(l4):
    """A response whose B1 and slow-libration checks pass, so the tree reaches step 5."""
    return make_response(
        a_response=l4,
        a_amplitude=PASS,
        slow_libration=PASS,
        b1_ratio=0.5,
        slow_adot_r2=0.1,
        force_r2_h1=0.8,
        force_r2_h2=0.8,
        centre_rate_rad_yr=1e-6,
        centre_rate_err_rad_yr=1e-5,
        exact_commensurability=False,
        force_law='single',
        multiplet=None,
        phi0_max_dev_rad=0.1,
        phi0_tol_rad=0.5,
    )


@pytest.mark.parametrize('angle, l4, r0, expected', TABLE, ids=[f'{a.angle_status}-{a.subtype}-{l}-{r}' for a, l, r, _ in TABLE])
def test_step_5_leaves(angle, l4, r0, expected):
    energy = make_energy(with_slow=r0, ve_vf_agree_fraction=0.5)
    kinematics = make_kinematics(check=PASS, corr_rate=0.1, detail='')
    decision = decide.decide_status(
        angle, make_sampling(), None, energy, _response(l4), kinematics, None, None, None, S, Ledger.from_settings(S)
    )
    assert (decision.status, decision.reason, decision.origin, decision.rule) == expected
    assert ('force_law_disagrees' in decision.flags) == (expected[3] == '5b')
