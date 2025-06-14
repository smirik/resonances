import resonances


def test_find_resonances():
    mmrs = resonances.find_resonances(a=2.39)

    mmrs_s = []
    for mmr in mmrs:
        mmrs_s.append(mmr.to_short())

    assert '4J-2S-1' in mmrs_s
