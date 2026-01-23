from resonances.resonance.resolver import resolve_mmr_status


def test_resolve_mmr_status_full_libration_with_overlap():
    angle_peaks = {'position': [(9.0, 11.0)], 'peaks': [0]}
    axis_peaks = {'position': [(9.5, 10.5)], 'peaks': [0]}

    result = resolve_mmr_status(
        classification_status=2,
        angle_periodogram_peaks=angle_peaks,
        axis_periodogram_peaks=axis_peaks,
    )

    assert result['status'] == 2
    assert result['has_overlap'] is True
    assert result['n_angle_peaks'] == 1
    assert result['n_axis_peaks'] == 1
    assert len(result['overlapping_peaks']) == 1


def test_resolve_mmr_status_full_libration_no_overlap():
    angle_peaks = {'position': [(9.0, 11.0)], 'peaks': [0]}
    axis_peaks = {'position': [(20.0, 22.0)], 'peaks': [0]}

    result = resolve_mmr_status(
        classification_status=2,
        angle_periodogram_peaks=angle_peaks,
        axis_periodogram_peaks=axis_peaks,
    )

    assert result['status'] == -2
    assert result['has_overlap'] is False
    assert result['n_angle_peaks'] == 1
    assert result['n_axis_peaks'] == 1
    assert len(result['overlapping_peaks']) == 0


def test_resolve_mmr_status_partial_libration_with_overlap():
    angle_peaks = {'position': [(9.0, 11.0)], 'peaks': [0]}
    axis_peaks = {'position': [(9.5, 10.5)], 'peaks': [0]}

    result = resolve_mmr_status(
        classification_status=1,
        angle_periodogram_peaks=angle_peaks,
        axis_periodogram_peaks=axis_peaks,
    )

    assert result['status'] == 1
    assert result['has_overlap'] is True


def test_resolve_mmr_status_partial_libration_no_overlap():
    angle_peaks = {'position': [(9.0, 11.0)], 'peaks': [0]}
    axis_peaks = {'position': [(20.0, 22.0)], 'peaks': [0]}

    result = resolve_mmr_status(
        classification_status=1,
        angle_periodogram_peaks=angle_peaks,
        axis_periodogram_peaks=axis_peaks,
    )

    assert result['status'] == -1
    assert result['has_overlap'] is False


def test_resolve_mmr_status_no_libration():
    angle_peaks = {'position': [(9.0, 11.0)], 'peaks': [0]}
    axis_peaks = {'position': [(9.5, 10.5)], 'peaks': [0]}

    result = resolve_mmr_status(
        classification_status=0,
        angle_periodogram_peaks=angle_peaks,
        axis_periodogram_peaks=axis_peaks,
    )

    assert result['status'] == 0


def test_resolve_mmr_status_none_periodograms():
    # With None periodograms, should return negative status for libration
    result = resolve_mmr_status(
        classification_status=2,
        angle_periodogram_peaks=None,
        axis_periodogram_peaks=None,
    )

    assert result['status'] == -2
    assert result['has_overlap'] is False


def test_resolve_mmr_status_empty_peaks():
    angle_peaks = {'position': [], 'peaks': []}
    axis_peaks = {'position': [], 'peaks': []}

    result = resolve_mmr_status(
        classification_status=2,
        angle_periodogram_peaks=angle_peaks,
        axis_periodogram_peaks=axis_peaks,
    )

    assert result['status'] == -2
    assert result['has_overlap'] is False


def test_resolve_mmr_status_with_delta():
    # Peaks that don't overlap without delta
    angle_peaks = {'position': [(9.0, 10.0)], 'peaks': [0]}
    axis_peaks = {'position': [(10.5, 11.5)], 'peaks': [0]}

    # Without delta, no overlap
    result_no_delta = resolve_mmr_status(
        classification_status=2,
        angle_periodogram_peaks=angle_peaks,
        axis_periodogram_peaks=axis_peaks,
        overlap_delta=0,
    )
    assert result_no_delta['status'] == -2

    # With delta=0.5, they should overlap
    result_with_delta = resolve_mmr_status(
        classification_status=2,
        angle_periodogram_peaks=angle_peaks,
        axis_periodogram_peaks=axis_peaks,
        overlap_delta=0.5,
    )
    assert result_with_delta['status'] == 2
