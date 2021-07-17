import resonances


def test_find_breaks():
    x = [1, 2, 3]
    y = [1, 2, 5]

    breaks = resonances.libration.find_breaks(x, y)
    assert 0 == len(breaks[0])  # no breaks

    breaks = resonances.libration.find_breaks(x, y, break_value=2)
    assert 1 == len(breaks[0])
    assert 3 == breaks[0][0]  # break at x=3
    assert 1 == breaks[1][0]
    assert 2 == breaks[2][0]
    assert 5 == breaks[3][0]

    x = [2, 5, 10]
    breaks = resonances.libration.find_breaks(x, y, break_value=2)
    assert 1 == len(breaks[0])
    assert 10 == breaks[0][0]  # break at the second index
    assert 2 == breaks[2][0]
    assert 5 == breaks[3][0]

    #       0  1  2  3  4   5   6   7
    x = [3, 5, 7, 9, 12, 20, 40, 50]
    y = [-3, 3, 2, 3, -3, -2, -2, 2]  # breaks at the indexes 1, 4, 7
    breaks = resonances.libration.find_breaks(x, y)
    assert 3 == len(breaks[0])

    assert 5 == breaks[0][0]
    assert 12 == breaks[0][1]
    assert 50 == breaks[0][2]

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
    #    0  1  2  3  4  5   6   7  8
    x = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    y = [6, 1, 2, 3, 4, 5, 6.5, 1, 2]
    breaks = resonances.libration.find_breaks(x, y)
    assert 2 == len(breaks[0])

    assert 3 == breaks[0][0]
    assert 6.0 == breaks[2][0]
    assert 1.0 == breaks[3][0]

    assert 15 == breaks[0][1]
    assert 6.5 == breaks[2][1]
    assert 1.0 == breaks[3][1]


def test_circulation_and_circulation_metrics():
    x = [3, 5, 7]
    y = [1, 2, 3]  # no breaks, full interval is a libration
    data = resonances.libration.circulation(x, y)
    assert 3.0 == data[0][0]
    assert 7.0 == data[1][0]
    assert 4.0 == data[2][0]

    # Breaks at the indexes 1, 4, 7 but their direction are opposites. Therefore, it is apocentric libration; hence, the whole interval is a libration.
    #    0  1  2  3  4   5   6   7
    x = [0, 3, 4, 5, 6, 10, 11, 13]
    y = [-3, 3, 2, 3, -3, -2, -2, 2]
    data = resonances.libration.circulation(x, y)
    assert 1 == len(data[0])
    assert 0.0 == data[0][0]
    assert 13.0 == data[1][0]
    assert 13.0 == data[2][0]

    # Breaks at the indexes 1, 7. Libration periods: from 0 to to 7 (by index), from 7 to 8.
    #    0  1  2  3  4  5  6  7  8
    x = [1, 2, 5, 6, 7, 9, 10, 11, 15]
    y = [6, 1, 2, 3, 4, 5, 6, 1, 2]
    data = resonances.libration.circulation(x, y)
    assert 2 == len(data[0])
    assert 1.0 == data[0][0]
    assert 11.0 == data[1][0]
    assert 10.0 == data[2][0]

    assert 11.0 == data[0][1]
    assert 15.0 == data[1][1]
    assert 4.0 == data[2][1]

    metrics = resonances.libration.circulation_metrics(data)
    assert 2 == metrics['num_libration_periods']
    assert 10 == metrics['max_libration_length']

    # Breaks at the indexes 1, 4, 5, 11. However, at 4 and 5 - apocentric.
    # Thus, there is only one libration period from 0 to to 11.
    #      0  1  2  3  4  5  6  7  8  9  10 11
    x = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 15, 16]
    y = [6, 1, 2, 1, 6, 1, 2, 3, 4, 5, 6, 1]
    data = resonances.libration.circulation(x, y)
    assert 1 == len(data[0])
    assert 0 == data[0][0]
    assert 16.0 == data[1][0]
    assert 16.0 == data[2][0]

    metrics = resonances.libration.circulation_metrics(data)
    assert 1 == metrics['num_libration_periods']
    assert 16 == metrics['max_libration_length']


def test_overlap():
    assert 1.0 == resonances.libration.overlap([1, 3], [2, 4])
    assert 0.0 == resonances.libration.overlap([1, 2], [3, 4])
    assert 0.0 == resonances.libration.overlap([1, 2], [2, 3])
    assert 2.0 == resonances.libration.overlap([1, 10], [8, 12])

    assert 0 < resonances.libration.overlap([1, 2], [2, 3], delta=0.1)
    assert 0 == resonances.libration.overlap([1, 2], [3, 4], delta=0.1)
    assert 0 < resonances.libration.overlap([1, 2], [3, 4], delta=1.1)


def test_overlap_list():
    data = resonances.libration.overlap_list([[1, 3], [10, 11]], [[2, 4], [20, 21]])
    assert 1 == len(data)
    assert [1, 3] == data[0]

    data = resonances.libration.overlap_list([[1, 2], [10, 11]], [[3, 4], [20, 21]])
    assert 0 == len(data)

    data = resonances.libration.overlap_list([[1, 3], [10, 12]], [[2, 4], [11, 21]])
    assert 2 == len(data)
    assert [1, 3] == data[0]
    assert [10, 12] == data[1]

    data = resonances.libration.overlap_list([[1, 2], [10, 11]], [[2, 3], [20, 21]])
    assert 0 == len(data)
