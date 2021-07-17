import resonances


def test_checkers():
    body = resonances.Body()
    body.status = 2

    assert True == body.in_resonance()
    assert True == body.in_pure_resonance()

    body.status = 1
    assert True == body.in_resonance()
    assert False == body.in_pure_resonance()

    body.status = 0
    assert False == body.in_resonance()
    assert False == body.in_pure_resonance()


def test_setup():
    body = resonances.Body()
    body.setup_vars_for_simulation(5)
    assert 5 == len(body.axis)
    assert 5 == len(body.ecc)
    assert 5 == len(body.angle)
    assert 5 == len(body.varpi)
    assert 5 == len(body.longitude)
