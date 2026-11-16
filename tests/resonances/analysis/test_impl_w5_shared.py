"""W5 'shared' simplify/performance pass: the vectorised steps against plain per-item loops written here, and the centre
scale (a slow libration under faster waves, 2011 KT19 7N-9r6).

Every reference below is computed independently (by hand or by a loop over the items), not by the package."""

import numpy as np
import pytest
from scipy import integrate

from resonances.resonance.analysis import filtering
from resonances.resonance.analysis import geometry as geo
from resonances.resonance.analysis import pendulum as pend
from resonances.resonance.analysis.settings import settings_from_config

S = settings_from_config()


# --- filtering ---------------------------------------------------------------------------------


def test_linear_fit_by_hand():
    # x = 1 + 2 t exactly; and x = (2, 1, 4, 3): slope sum((t - 1.5)(x - 2.5)) / sum((t - 1.5)^2) = 3 / 5, value 2.5 - 0.6 x 1.5
    assert filtering.linear_fit([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0]) == pytest.approx((2.0, 1.0))
    assert filtering.linear_fit([0.0, 1.0, 2.0, 3.0], [2.0, 1.0, 4.0, 3.0]) == pytest.approx((0.6, 1.6))
    assert filtering.linear_fit([10.0, 11.0, 12.0, 13.0], [2.0, 1.0, 4.0, 3.0]) == pytest.approx((0.6, 1.6))  # value at t0
    assert all(np.isnan(filtering.linear_fit([1.0], [2.0])))


def test_linear_fit_agrees_with_polyfit_on_an_unwrapped_angle():
    t = np.arange(200_000) * 0.5
    x = 1e4 + 3e-3 * t + 0.4 * np.sin(2 * np.pi * t / 937.0)
    slope, intercept = filtering.linear_fit(t, x)
    expected = np.polyfit(t - t[0], x, 1)
    assert slope == pytest.approx(expected[0], rel=1e-10)
    assert intercept == pytest.approx(expected[1], rel=1e-12)


def test_lowpass_keeps_a_line_and_lowpass_or_none_refuses_bad_input():
    t = np.arange(5000) * 2.0
    line = 5.0 - 0.01 * t
    assert filtering.lowpass(t, line, 200.0, 2) == pytest.approx(line, abs=1e-9)
    assert filtering.lowpass_or_none(t, line, 3.0, S) is None  # not longer than two steps
    bent = t.copy()
    bent[10] += 0.5  # one step off by 25 %
    assert filtering.lowpass_or_none(bent, line, 200.0, S) is None
    assert filtering.lowpass_or_none(t, line, 200.0, S) == pytest.approx(line, abs=1e-9)


# --- geometry: revolutions, turn energies, block means ------------------------------------------


def _revolutions_loop(t, x, rate, window):
    """The turns by a plain loop: each turn's first sample at its lowest speed (np.argmin over [start, end])."""
    inside = np.flatnonzero((t >= window[0]) & (t <= window[1]))
    i0, i1 = inside[0], inside[-1]
    direction = np.sign(x[i1] - x[i0])
    n_turns = int(np.floor(abs(x[i1] - x[i0]) / (2 * np.pi)))
    progress = direction * (x[i0 : i1 + 1] - x[i0])
    edges = [i0]
    for k in range(1, n_turns + 1):
        edges.append(i0 + int(np.flatnonzero(progress >= 2 * np.pi * k)[0]))
    out = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        speed = direction * rate[start : end + 1]
        j = int(np.argmin(speed))
        out.append((t[start], t[end], x[start + j] % (2 * np.pi), speed[j] * (t[end] - t[start]) / (2 * np.pi)))
    return out


@pytest.mark.parametrize('direction', [1.0, -1.0])
def test_revolutions_match_a_loop_over_the_turns(direction):
    t = np.arange(0.0, 3000.0, 0.25)
    # a circulation slowed near 1 rad, with a beat so the slowest sample moves from turn to turn
    x = direction * (0.05 * t + 0.6 * np.sin(0.05 * t - 1.0) + 0.1 * np.sin(2 * np.pi * t / 173.0))
    rate = np.gradient(x, t)
    expected = _revolutions_loop(t, x, rate, (100.0, 2900.0))
    got = geo.revolutions(t, x, rate, (100.0, 2900.0))
    assert len(got) == len(expected) > 10
    for rev, (a, b, phase, depth) in zip(got, expected):
        assert (rev.t_start_yr, rev.t_end_yr) == (a, b)
        assert rev.phase_min_rate_rad == phase
        assert rev.rate_min_over_mean == pytest.approx(depth, rel=1e-12)


def test_revolutions_take_the_end_sample_and_the_first_of_equal_minima():
    t = np.arange(16.0)
    # one turn = 5.75 rate units: the turns are samples 0-6 and 6-14; turn 1 is slowest at its end sample (6),
    # turn 2 at two samples (8 and 10: the first counts)
    rate = np.array([1.0, 1, 1, 1, 1, 1, 0.6, 1, 0.4, 1, 0.4, 1, 1, 1, 1, 1])
    x = np.concatenate([[0.0], np.cumsum(rate[:-1])]) * (2 * np.pi / 5.75)
    got = geo.revolutions(t, x, rate, (0.0, 15.0))
    expected = _revolutions_loop(t, x, rate, (0.0, 15.0))
    assert [(r.t_start_yr, r.t_end_yr) for r in got] == [(0.0, 6.0), (6.0, 14.0)]
    assert [(r.t_start_yr, r.t_end_yr, r.phase_min_rate_rad) for r in got] == [e[:3] for e in expected]
    assert got[0].phase_min_rate_rad == pytest.approx(x[6] % (2 * np.pi))
    assert got[1].phase_min_rate_rad == pytest.approx(x[8] % (2 * np.pi))


def _energies_loop(revs, times, h):
    out = []
    for rev in revs:
        inside = np.flatnonzero((times >= rev.t_start_yr) & (times <= rev.t_end_yr) & np.isfinite(h))
        if not len(inside):
            out.append(np.nan)
            continue
        middle = (rev.t_start_yr + rev.t_end_yr) / 2
        out.append(h[inside[int(np.argmin(np.abs(times[inside] - middle)))]])
    return out


def test_turn_energies_match_a_loop_with_ties_and_unordered_cycles():
    revs = [geo.Revolution(a, a + 10.0, 0.0, 0.5) for a in np.arange(0.0, 200.0, 10.0)]
    rng = np.random.default_rng(3)
    times = np.concatenate([rng.uniform(0, 200, 40), [4.0, 6.0, 15.0, 15.0, 25.0, 35.0, 34.0, 36.0, 300.0]])
    h = rng.uniform(0.5, 2.0, len(times))
    h[[3, 7]] = np.nan
    order = rng.permutation(len(times))  # cycles in no particular order
    times, h = times[order], h[order]
    got = [rev.energy_h for rev in geo._with_energies(revs, times, h)]
    assert got == pytest.approx(_energies_loop(revs, times, h), nan_ok=True)


def test_raw_monotonic_reads_block_means():
    t = np.arange(100.0)
    rising = np.arange(100.0) + 3 * np.sin(np.arange(100.0))  # block means of 10 samples still rise
    mask = np.ones(100, dtype=bool)
    assert geo._raw_monotonic(t, rising, mask, 10.0, 1.0)
    assert not geo._raw_monotonic(t, rising, mask, 10.0, -1.0)
    wave = 5 * np.sin(2 * np.pi * t / 40.0)  # block means go up and down
    assert not geo._raw_monotonic(t, wave, mask, 10.0, 1.0)


# --- pendulum: per-window fits and cycle energies ----------------------------------------------


def _pendulum(omega0, amplitude, span, step, well):
    def rhs(_, y):
        return [y[1], -(omega0**2) * np.sin(y[0] - well)]

    t = np.arange(0.0, span, step)
    sol = integrate.solve_ivp(rhs, (0, span), [well + amplitude, 0.0], t_eval=t, rtol=1e-11, atol=1e-12)
    return t, sol.y[0], sol.y[1]


def _windows(t, angle, n):
    edges = np.linspace(t[0], t[-1], n + 1)
    return [(a, b) for a, b in zip(edges[:-1], edges[1:])]


def _window_wells_loop(t, angle, rate, windows, n_harmonics):
    """The well of V fitted on each window alone: least squares of rate^2 - mean on the harmonics - their means."""
    wells = []
    for a, b in windows:
        part = (t >= a) & (t <= b)
        columns = []
        for n in range(1, n_harmonics + 1):
            columns += [np.cos(n * angle[part]), np.sin(n * angle[part])]
        design = -2 * np.column_stack(columns)
        design -= design.mean(axis=0)
        target = rate[part] ** 2 - np.mean(rate[part] ** 2)
        coef = np.linalg.lstsq(design, target, rcond=None)[0]
        wells.append(pend.potential_from_coefficients(coef[0::2], coef[1::2]).well_rad)
    return np.array(wells)


@pytest.mark.parametrize('amplitude_deg', [120.0, 0.5])  # 0.5 deg: ill-conditioned windows, solved one by one
def test_window_fits_match_a_least_squares_loop(amplitude_deg):
    omega0, well = 2 * np.pi / 300.0, 2.0
    t, angle, rate = _pendulum(omega0, np.radians(amplitude_deg), 6000.0, 0.5, well)
    windows = _windows(t, angle, 20)
    potential, per_window = pend.fit_potential_from_rate(t, angle, rate, windows, S)
    assert len(per_window) == len(windows)
    expected = _window_wells_loop(t, angle, rate, windows, potential.n_harmonics)
    got = np.array([p.well_rad for p in per_window])
    assert np.abs(np.angle(np.exp(1j * (got - expected)))).max() < 1e-7
    assert got == pytest.approx(well, abs=2e-2)


def test_window_fits_skip_windows_with_too_few_samples():
    t, angle, rate = _pendulum(2 * np.pi / 300.0, 1.0, 3000.0, 0.5, 0.0)
    few = 2 * S.separatrix_harmonics_max + 1  # a window must hold more than this
    windows = [(0.0, 1000.0), (1500.0, 1500.0 + 0.5 * (few - 1)), (2000.0, 2999.0)]
    _, per_window = pend.fit_potential_from_rate(t, angle, rate, windows, S)
    assert len(per_window) == 2


def _cycle_energies_loop(t, angle, rate, bounds, potential):
    out = []
    for a, b, rotation in bounds:
        part = (t >= a) & (t <= b)
        if not part.any():
            continue
        energy = rate[part] ** 2 / 2 + potential.value(angle[part]) - potential.v_min
        h = float(np.nanmean(energy / potential.depth))
        ratio = np.nan if rotation else pend.action_ratio_in_potential(potential, h)
        period = pend.period_in_potential(potential, h)
        out.append((h, ratio, (b - a) / period if np.isfinite(period) and period > 0 else np.nan))
    return out


@pytest.mark.parametrize('harmonics', [1, 2])
def test_cycle_energies_match_a_loop_over_the_cycles(harmonics):
    t, angle, rate = _pendulum(2 * np.pi / 300.0, 2.0, 4000.0, 0.5, 1.0)
    cos, sin = ([-1.5e-4, -2e-5] if harmonics == 2 else [-1.5e-4]), ([1e-5, 0.0] if harmonics == 2 else [1e-5])
    potential = pend.potential_from_coefficients(cos, sin)
    edges = np.arange(0.0, 4000.0, 290.0)
    bounds = [(a, b, i % 3 == 2) for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]))] + [(5000.0, 5100.0, False)]
    got = pend.cycle_energies(t, angle, rate, bounds, potential)
    expected = _cycle_energies_loop(t, angle, rate, bounds, potential)
    assert len(got) == len(expected) == len(bounds) - 1  # the cycle without samples is skipped
    for cycle, (h, ratio, law) in zip(got, expected):
        assert cycle.h == pytest.approx(h, rel=1e-12)
        assert cycle.action_ratio == pytest.approx(ratio, rel=1e-9, nan_ok=True)
        assert cycle.period_law_ratio == pytest.approx(law, rel=1e-9, nan_ok=True)
        assert (cycle.h_mean, cycle.strength_rel_sq) == (pytest.approx(h, rel=1e-12), 1.0)


# --- centre scale: a slow libration under faster waves -----------------------------------------


def _slow_libration_with_waves():
    """sigma = 180 deg + 110 deg sin(2 pi t / 43 kyr) + 25 deg sin(2 pi t / 3.9 kyr) over 200 kyr, every 50 yr (the shape
    of 2011 KT19 7N-9r6 in the slow test: 3-4 kyr waves on a 40-43 kyr libration)."""
    t = np.arange(0.0, 200_000.0 + 1, 50.0)
    sigma = np.radians(180 + 110 * np.sin(2 * np.pi * t / 43_000.0) + 25 * np.sin(2 * np.pi * t / 3_900.0))
    return t, sigma


def _filtered(t, sigma, cutoff):
    angle_f = filtering.lowpass(t, sigma, cutoff, S.filter_order)
    return geo.FilteredAngle(
        t_yr=t,
        angle_raw=sigma,
        angle_f=angle_f,
        rate_f=np.gradient(angle_f, t),
        angle_f_half=filtering.lowpass(t, sigma, cutoff / 2, S.filter_order),
        angle_f_double=filtering.lowpass(t, sigma, 2 * cutoff, S.filter_order),
        slow_f=np.zeros_like(t),
        identity_coefficient=float('nan'),
        cutoff_yr=cutoff,
        step_yr=50.0,
        window=filtering.edge_window(t, cutoff),
    )


def test_the_centre_scale_turns_a_wavy_slow_libration_into_one_episode():
    t, sigma = _slow_libration_with_waves()
    fine_angle = _filtered(t, sigma, 1000.0)
    fine = geo.measure_geometry(fine_angle, np.radians(1.0), S)
    assert fine.period_cycle_yr < 10_000  # the fine turning points are the 3.9 kyr waves
    assert geo.covering_episode(fine.episodes, S) is None

    coarse_angle = geo.centre_scale_angle(fine_angle, fine.period_cycle_yr, S)
    assert coarse_angle.cutoff_yr == pytest.approx(S.libration_centre_cycles * fine.period_cycle_yr)
    assert coarse_angle.window == pytest.approx((coarse_angle.cutoff_yr, t[-1] - coarse_angle.cutoff_yr))
    coarse = geo.measure_geometry(coarse_angle, geo.centre_prominence(np.radians(1.0), S), S)
    episode = geo.covering_episode(coarse.episodes, S)
    assert episode is not None and episode.shape == geo.EpisodeShape.STATIONARY
    assert coarse.period_cycle_yr == pytest.approx(43_000.0, rel=0.03)
    assert np.degrees(episode.half_amp_med_rad) == pytest.approx(110.0, abs=8.0)
    assert np.degrees(episode.centre_mod_2pi_rad) == pytest.approx(180.0, abs=5.0)


def test_no_centre_scale_without_a_coarser_scale():
    t, sigma = _slow_libration_with_waves()
    angle = _filtered(t, sigma, 1000.0)
    assert geo.centre_scale_angle(angle, float('nan'), S) is None
    assert geo.centre_scale_angle(angle, 1000.0 / S.libration_centre_cycles, S) is None  # W_c = P_c
    assert geo.centre_scale_angle(angle, 150_000.0, S) is None  # W_c longer than half the record: no window


def test_centre_travel_reads_the_passes_of_any_object_with_a_centre():
    t = np.linspace(0, 100, 101)
    centre = np.where(t < 50, 0.04 * t, 2.0 - 0.1 * (t - 50))  # up by 2 rad, then down by 5 rad
    passes = [geo.MonotonePass(0.0, 50.0, 2.0), geo.MonotonePass(50.0, 100.0, -5.0)]

    class Holder:
        pass

    holder = Holder()
    holder.centre = Holder()
    holder.centre.t_yr, holder.centre.centre_rad, holder.centre.passes = t, centre, passes
    assert geo.centre_travel_in(holder, 0, 100) == pytest.approx(5.0)
    assert geo.centre_travel_in(holder, 40, 70) == pytest.approx(2.0)  # 0.4 up, then 2.0 down
    assert geo.centre_travel_in(holder, 200, 300) == 0.0


def test_cycle_energies_in_a_frame_keep_infinite_energies_like_nanmean():
    # m(t) = 0 over the second half: h = E / (m depth) is +-inf there; np.nanmean keeps inf (only NaN is left out)
    t, angle, rate = _pendulum(2 * np.pi / 300.0, 2.0, 2000.0, 0.5, 1.0)
    strength = np.where(t < 1000.0, 1.0, 0.0)
    frame = pend.ModulatedFrame(t, strength, np.zeros_like(t), np.zeros_like(t))
    potential = pend.potential_from_coefficients([-1.5e-4], [1e-5])
    bounds = [(a, a + 290.0, False) for a in np.arange(0.0, 1700.0, 290.0)]
    got = [c.h for c in pend.cycle_energies(t, angle, rate, bounds, potential, frame=frame)]
    expected = []
    with np.errstate(divide='ignore', invalid='ignore'):
        for a, b, _ in bounds:
            part = (t >= a) & (t <= b)
            m = strength[part]
            energy = rate[part] ** 2 / 2 + m * (potential.value(angle[part]) - potential.v_min)
            expected.append(float(np.nanmean(energy / (m * potential.depth))))
    assert np.isinf(expected).sum() >= 2
    assert got == pytest.approx(expected, rel=1e-12)
