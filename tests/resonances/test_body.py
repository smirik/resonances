import resonances


def test_checkers():
    body = resonances.Body()
    body.status = 2

    assert body.in_resonance() is True
    assert body.in_pure_resonance() is True

    body.status = 1
    assert body.in_resonance() is True
    assert body.in_pure_resonance() is False

    body.status = 0
    assert body.in_resonance() is False
    assert body.in_pure_resonance() is False


def test_setup():
    body = resonances.Body()
    body.setup_vars_for_simulation(5)
    assert 5 == len(body.axis)
    assert 5 == len(body.ecc)
    assert 5 == len(body.angle)
    assert 5 == len(body.varpi)
    assert 5 == len(body.longitude)
