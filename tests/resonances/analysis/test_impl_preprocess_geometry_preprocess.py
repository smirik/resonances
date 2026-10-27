from types import SimpleNamespace

import numpy as np
import pytest

from resonances.resonance.analysis.filtering import butterworth_gain, lowpass, window_mask
from resonances.resonance.analysis.mmr import preprocess as pre
from resonances.resonance.analysis.mmr.models import GreatInequality, MMRSeries, PlanetSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import settings_from_config

S = settings_from_config(mmr_libration_period_min_yr=0.0)  # the synodic bound alone
N_JUPITER = 2 * np.pi / 11.862  # rad/yr
MU_JUPITER = 9.5479e-4
GI = GreatInequality(937.0, np.radians(0.34), np.radians(0.83), False)
SYNODIC_2J1 = 1.5 * 11.862  # the cutoff of the 2J-1 series below: 17.793 yr


def test_cutoff_hilda_griqua_user_and_no_planets():
    # Hilda 3:2 without the floor: P_syn,J = 23.7 yr -> 1.5 x 23.7 = 35.55 yr; a 250-yr libration keeps > 94 %
    hilda = pre.choose_cutoff({'Jupiter': 23.7}, 0.1, S)
    assert hilda.binding == 'synodic' and hilda.cutoff_yr == pytest.approx(35.55)
    assert butterworth_gain(250.0, hilda.cutoff_yr, S.filter_order) >= 0.94
    # with the floor 150 yr: 150 / 2 = 75 yr > 35.55 yr, at any step (the step only counts samples)
    default = settings_from_config()
    for step in (0.1, 1.0, 15.9):
        cut = pre.choose_cutoff({'Jupiter': 23.7}, step, default)
        assert (cut.binding, cut.cutoff_yr, cut.floor_bound_yr) == ('floor', 75.0, 75.0)
        assert cut.samples_per_cutoff == pytest.approx(75.0 / step)
    # a 150-yr libration keeps G = 1 / (1 + (75/150)^4) = 16/17
    assert butterworth_gain(150.0, 75.0, S.filter_order) == pytest.approx(16 / 17)
    # Griqua 2J-1: synodic 1.5 x 11.9 = 17.85 yr, the floor bound 75 yr binds
    griqua = pre.choose_cutoff({'Jupiter': 11.9}, 1.0, default)
    assert griqua.binding == 'floor' and griqua.cutoff_yr == 75.0 and griqua.synodic_bound_yr == pytest.approx(17.85)
    user = pre.choose_cutoff({'Jupiter': 11.9}, 1.0, settings_from_config(mmr_filter_cutoff_yr=500.0))
    assert user.binding == 'user' and user.cutoff_yr == 500.0
    alone = pre.choose_cutoff({}, 2.0, S)  # no planet, no floor: nothing to filter by
    assert alone.synodic_bound_yr == 0.0 and alone.cutoff_yr == 0.0 and not pre.filter_realizable(alone.cutoff_yr, 2.0)
    # a Jupiter Trojan: floor 0.5 x 11.862 / sqrt(27/4 x 9.5479e-4) = 73.88 yr -> P_c = 36.94 yr
    floor = 0.5 * 11.862 / np.sqrt(6.75 * MU_JUPITER)
    trojan = pre.choose_cutoff({}, 15.9, default, floor)
    assert trojan.binding == 'floor' and trojan.cutoff_yr == pytest.approx(36.94, abs=0.01)


def test_the_filter_exists_while_the_cutoff_is_longer_than_two_steps():
    # Trojan P_c = 36.94 yr: dt = 18.4 yr can be filtered, 18.47 yr (= P_c / 2) and 20 yr cannot
    assert pre.filter_realizable(36.94, 18.4) and not pre.filter_realizable(36.94, 18.47) and not pre.filter_realizable(36.94, 20.0)
    # a 150-yr floor: P_c = 75 yr, the limit is 37.5 yr
    assert pre.filter_realizable(75.0, 37.4) and not pre.filter_realizable(75.0, 37.5)
    assert not pre.filter_realizable(float('nan'), 1.0)


@pytest.mark.parametrize(
    'key, planet',
    [
        ('1E-1+0+0', 'Earth'),  # lambda-part lambda_E - lambda: co-orbital with the Earth
        ('1J-1+0+0', 'Jupiter'),  # Jupiter Trojan
        ('3J+8S-3+0+0-8', None),  # k_J = -k, but n = n_J + 8 n_S / 3 (P ~ 5.7 yr): Jupiter's synodic wave is real
        ('2J-1+0-1', None),
        ('1J-3r0', None),  # retrograde
    ],
)
def test_coorbital_planet(key, planet):
    assert pre.coorbital_planet(ResonanceSpec.from_key(key)) == planet


def test_filter_guard():
    too_slow, gain = pre.filter_guard(150.0, 100.0, S)  # G = 1 / (1 + (100/150)^4) = 0.835 < 0.94
    assert too_slow and gain == pytest.approx(1 / (1 + (100 / 150) ** 4))
    assert pre.filter_guard(250.0, 100.0, S) == (False, pytest.approx(1 / (1 + 0.4**4)))
    too_slow, gain = pre.filter_guard(float('nan'), 100.0, S)
    assert too_slow is False and np.isnan(gain)


def test_filter_guard_reads_the_digital_gain_of_a_quantized_trojan_cycle():
    # 588 at dt = 15.92 yr, P_c = 73.89 yr (Codex, inv-15): a 9-step cycle of 143.3 yr is below 2 P_c = 147.8 yr, but
    # the digital gain 1 / (1 + (tan(pi dt / P) / tan(pi dt / P_c))^4) = 0.9596 is above 0.94: not too slow
    dt, cutoff, period = 15.918509, 73.88522, 9 * 15.918509
    expected = 1 / (1 + (np.tan(np.pi * dt / period) / np.tan(np.pi * dt / cutoff)) ** 4)
    assert expected == pytest.approx(0.9596, abs=1e-4) and period < 2 * cutoff
    too_slow, gain = pre.filter_guard(period, cutoff, S, dt)
    assert not too_slow and gain == pytest.approx(expected)
    # above Nyquist the gain is not defined: too slow
    too_slow, gain = pre.filter_guard(25.0, cutoff, S, dt)
    assert too_slow and np.isnan(gain)


def test_predicted_ripple_from_the_great_inequality():
    spec = ResonanceSpec.from_key('3J-1S-1')  # |k_J| = 3, |k_S| = 1
    expected = (3 * np.radians(0.34) + np.radians(0.83)) * butterworth_gain(937.0, 50.0, 2)
    assert pre.predicted_ripple(spec, GI, 50.0, 2) == pytest.approx(expected)
    assert pre.predicted_ripple(ResonanceSpec.from_key('9M-16'), GI, 50.0, 2) == 0.0
    assert pre.predicted_ripple(spec, None, 50.0, 2) == 0.0


def _series(planets: bool, key='2J-1'):
    t = np.arange(0, 9000, 1.0)
    jupiter_wave = np.radians(0.34) * np.sin(2 * np.pi * t / 937 + 0.3) + np.radians(0.05) * np.sin(2 * np.pi * t / 5.93)
    jupiter_longitude = N_JUPITER * t + jupiter_wave
    sigma_true = 1.0 + 2 * np.pi * t / 3000  # a slow circulation: no own power in the 300-3000 yr band
    spec = ResonanceSpec.from_key(key)
    k_jupiter = spec.planet_coefficients[0]
    sigma = sigma_true + k_jupiter * jupiter_wave
    ones = np.ones_like(t)
    planet_series = {}
    if planets:
        planet_series['Jupiter'] = PlanetSeries(
            'Jupiter', t, 5.2 * ones, 0.05 * ones, 0.0 * ones, 0.0 * ones, 0.0 * ones, np.mod(jupiter_longitude, 2 * np.pi)
        )
    series = MMRSeries(
        t_yr=t,
        a=3.27 + 1e-4 * np.sin(2 * np.pi * t / 400),
        e=0.2 * ones,
        inc=0.1 * ones,
        Omega=0.5 + 1e-4 * t,
        omega=2.0 - 2e-4 * t,
        varpi=np.mod(2.5 - 1e-4 * t, 2 * np.pi),
        angle_wrapped=np.mod(sigma, 2 * np.pi),
        resonance=spec,
        regime=Regime(1.0, 9000.0, len(t)),
        planets=planet_series,
    )
    context = SimpleNamespace(
        synodic_periods_yr={'Jupiter': 11.862},
        planet_mean_motion_rad_yr={'Jupiter': N_JUPITER},
        great_inequality=GI,
        identity_coefficient=-1.5 * -1 * (2 * np.pi * 3.27**-1.5) / 3.27,
    )
    sampling = SimpleNamespace(step_yr=1.0, angle_unwrapped=sigma)
    return series, context, sampling, sigma_true


def test_great_inequality_is_subtracted_exactly_when_planets_are_saved():
    series, context, sampling, sigma_true = _series(planets=True)
    prep = pre.preprocess(series, context, sampling, S)
    assert prep.cutoff.binding == 'synodic' and prep.cutoff.cutoff_yr == pytest.approx(SYNODIC_2J1)
    assert prep.ripple_removed and prep.ripple_corr > 0.99 and prep.flags == []
    assert prep.ripple_half_amp_rad == pytest.approx(2 * np.radians(0.34), rel=0.05)
    assert prep.unremoved_ripple_rad == 0.0
    inside = window_mask(prep.t_yr, prep.window)
    assert np.max(np.abs(prep.angle_f - sigma_true)[inside]) < 1e-3
    assert prep.rate_f[inside] == pytest.approx(2 * np.pi / 3000, rel=1e-2)
    assert prep.window == pytest.approx((SYNODIC_2J1, 8999.0 - SYNODIC_2J1))


def test_without_planets_the_ripple_stays_and_raises_the_prominence():
    series, context, sampling, sigma_true = _series(planets=False)
    prep = pre.preprocess(series, context, sampling, S)
    assert not prep.ripple_removed and prep.ripple_rad is None
    assert prep.flags == ['ripple_not_removed']
    gain = 1 / (1 + (np.tan(np.pi / 937.0) / np.tan(np.pi / SYNODIC_2J1)) ** 4)  # digital, dt = 1 yr
    assert prep.unremoved_ripple_rad == pytest.approx(2 * np.radians(0.34) * gain)
    inside = window_mask(prep.t_yr, prep.window)
    assert np.max(np.abs(prep.angle_f - sigma_true)[inside]) == pytest.approx(2 * np.radians(0.34), rel=0.05)
    columns = pre.preprocessed_columns(prep)
    assert columns['flt_cutoff_yr'] == pytest.approx(SYNODIC_2J1) and columns['flt_ripple_removed'] is False
    assert columns['flt_ripple_unremoved_deg'] == pytest.approx(0.68, rel=1e-3)


def test_coorbital_planet_does_not_bound_the_cutoff_and_keeps_its_ripple():
    # 1J-1: Jupiter's synodic period does not bound P_c; the floor 0.5 x 147.76 yr (tadpole period) / 2 does. The
    # planet's slow longitude is not subtracted though Jupiter is saved (the Trojan follows it): nothing unremoved
    series, context, sampling, _ = _series(planets=True, key='1J-1')
    context.synodic_periods_yr = {'Jupiter': 1e6}
    context.planet_mass = {'Jupiter': MU_JUPITER}
    prep = pre.preprocess(series, context, sampling, settings_from_config())
    tadpole = 11.862 / np.sqrt(6.75 * MU_JUPITER)
    assert tadpole == pytest.approx(147.76, abs=0.01)
    assert prep.cutoff.binding == 'floor' and prep.cutoff.cutoff_yr == pytest.approx(tadpole / 4)
    assert prep.period_floor_yr == pytest.approx(tadpole / 2)
    assert not prep.ripple_removed and prep.ripple_rad is None and prep.unremoved_ripple_rad == 0.0 and prep.flags == []
    assert np.array_equal(prep.angle_clean, sampling.angle_unwrapped)


def test_filtered_angle_and_slow_components():
    series, context, sampling, _ = _series(planets=True)
    prep = pre.preprocess(series, context, sampling, S)
    angle = pre.filtered_angle(prep, context, 1.0)
    assert angle.slow_f is prep.a_f and angle.cutoff_yr == pytest.approx(SYNODIC_2J1)
    assert angle.identity_coefficient == context.identity_coefficient
    slow = pre.slow_components(series, prep, 400.0, S)
    assert slow.window_yr == pytest.approx(1200.0)  # 3 cycles of 400 yr
    assert slow.window == pytest.approx((1200.0, 9000 - 1 - 1200.0))
    middle = (slow.t_yr > 2000) & (slow.t_yr < 7000)
    assert slow.varpi_slow[middle] == pytest.approx(2.5 - 1e-4 * slow.t_yr[middle], abs=1e-6)
    assert np.ptp(slow.a_slow[middle]) < 1e-5  # the 400-yr a wave is gone at W_c = 1200 yr
    assert pre.slow_components(series, prep, float('nan'), S).window_yr == pytest.approx(3 * SYNODIC_2J1)  # 3 P_c


def test_retrograde_slow_varpi_is_omega_minus_node():
    series, context, sampling, _ = _series(planets=False)
    retro = ResonanceSpec.from_key('1J-3r0')
    series = MMRSeries(**{**series.__dict__, 'resonance': retro})
    prep = pre.preprocess(series, context, sampling, S)
    slow = pre.slow_components(series, prep, 400.0, S)
    middle = (slow.t_yr > 2000) & (slow.t_yr < 7000)
    expected = lowpass(slow.t_yr, (2.0 - 2e-4 * slow.t_yr) - (0.5 + 1e-4 * slow.t_yr), 1200.0, 2)
    assert slow.varpi_slow[middle] == pytest.approx(expected[middle], abs=1e-9)
