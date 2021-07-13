import resonances


def test_breaks():
    arr = [1, 2, 5]

    breaks = resonances.libration.find_breaks(arr, coeff=1)
    assert 0 == len(breaks[0])  # no breaks

    breaks = resonances.libration.find_breaks(arr, coeff=1, break_value=2)
    assert 1 == len(breaks[0])
    assert 2.0 == breaks[0][0]  # break at the second index
    assert 1 == breaks[1][0]
    assert 2 == breaks[2][0]
    assert 5 == breaks[3][0]

    breaks = resonances.libration.find_breaks(arr, coeff=10, break_value=2)
    assert 1 == len(breaks[0])
    assert 20.0 == breaks[0][0]  # break at the second index
    assert 1 == breaks[1][0]
    assert 2 == breaks[2][0]
    assert 5 == breaks[3][0]

    #       0  1  2  3  4   5   6   7
    arr = [-3, 3, 2, 3, -3, -2, -2, 2]  # breaks at the indexes 1, 4, 7
    breaks = resonances.libration.find_breaks(arr)
    assert 3 == len(breaks[0])

    assert 1.0 == breaks[0][0]
    assert 4.0 == breaks[0][1]
    assert 7.0 == breaks[0][2]

    assert 1 == breaks[1][0]
    assert -1 == breaks[1][1]
    assert 1 == breaks[1][2]

    assert -3 == breaks[2][0]
    assert 3 == breaks[3][0]
    assert 3 == breaks[2][1]

    assert -3 == breaks[3][1]
    assert -2 == breaks[2][2]
    assert 2 == breaks[3][2]

    # Breaks at the indexes 1, 7. Libration periods: from 0 to 1, from 1 to 7, from 7 to 8.
    #      0  1  2  3  4  5  6  7  8
    arr = [6, 1, 2, 3, 4, 5, 6.5, 1, 2]
    breaks = resonances.libration.find_breaks(arr)
    assert 2 == len(breaks[0])

    assert 1.0 == breaks[0][0]
    assert 6.0 == breaks[2][0]
    assert 1.0 == breaks[3][0]

    assert 7.0 == breaks[0][1]
    assert 6.5 == breaks[2][1]
    assert 1.0 == breaks[3][1]


def test_circulation():
    arr = [1, 2, 3]  # no breaks, full interval is a libration
    data = resonances.libration.circulation(arr)
    assert 0.0 == data[0][0]
    assert 2.0 == data[1][0]
    assert 2.0 == data[1][0]

    # Breaks at the indexes 1, 4, 7 but their direction are opposites. Therefore, it is apocentric libration; hence, the whole interval is a libration.
    #       0  1  2  3  4   5   6   7
    arr = [-3, 3, 2, 3, -3, -2, -2, 2]
    data = resonances.libration.circulation(arr)
    assert 1 == len(data[0])
    assert 0.0 == data[0][0]
    assert 8.0 == data[1][0]
    assert 8.0 == data[2][0]

    # Breaks at the indexes 1, 7. Libration periods: from 0 to to 7, from 7 to 8.
    #      0  1  2  3  4  5  6  7  8
    arr = [6, 1, 2, 3, 4, 5, 6, 1, 2]
    data = resonances.libration.circulation(arr)
    assert 2 == len(data[0])
    assert 0.0 == data[0][0]
    assert 7.0 == data[1][0]
    assert 7.0 == data[2][0]

    assert 7.0 == data[0][1]
    assert 9.0 == data[1][1]
    assert 2.0 == data[2][1]

    # Breaks at the indexes 1, 4, 11. However, at 4 - apocentric. Thus, the libration periods are from 0 to to 11, from 11 to 12 (don't ask!).
    #      0  1  2  3  4  5  6  7  8  9  10 11
    arr = [6, 1, 2, 1, 6, 1, 2, 3, 4, 5, 6, 1]
    data = resonances.libration.circulation(arr)
    assert 2 == len(data[0])
    assert 0.0 == data[0][0]
    assert 11.0 == data[1][0]
    assert 11.0 == data[2][0]

    assert 11.0 == data[0][1]
    assert 12.0 == data[1][1]
    assert 1.0 == data[2][1]
