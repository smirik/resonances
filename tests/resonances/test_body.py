import resonances


def test_checkers():
    body = resonances.Body()
    mmr = resonances.create_mmr('4J-2S-1')
    body.mmrs = [mmr]
    body.statuses[mmr.to_s()] = 2

    assert body.in_resonance(mmr) is True
    assert body.in_pure_resonance(mmr) is True

    body.statuses[mmr.to_s()] = 1
    assert body.in_resonance(mmr) is True
    assert body.in_pure_resonance(mmr) is False

    body.statuses[mmr.to_s()] = 0
    assert body.in_resonance(mmr) is False
    assert body.in_pure_resonance(mmr) is False


def test_setup():
    body = resonances.Body()
    mmr = resonances.create_mmr('4J-2S-1')
    body.mmrs = [mmr]
    body.setup_vars_for_simulation(5)
    assert 5 == len(body.axis)
    assert 5 == len(body.ecc)
    assert 5 == len(body.angles[mmr.to_s()])
    assert 5 == len(body.varpi)
    assert 5 == len(body.longitude)
