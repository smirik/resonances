from resonances.resonance.secular import GeneralSecularResonance
import resonances.secular_finder


def test_secular_check_nu6():
    """Test that asteroid 759 shows libration in nu6 secular resonance."""

    sim = resonances.secular_finder.check(
        asteroids=[759, 1222, 760],
        secular_resonance='nu6',
        integration_years=200000,
        oscillations_cutoff=0.0005,
        plot='all',
        save='all',
    )

    sim.run(progress=True)
    print(sim.bodies[0].secular_resonances[0].to_s())
    summary = sim.data_manager.get_simulation_summary(sim.bodies)
    status759 = summary.loc[(summary['name'] == '759') & (summary['resonance'] == 'nu6_Saturn'), 'status'].iloc[0]
    status1222 = summary.loc[(summary['name'] == '1222') & (summary['resonance'] == 'nu6_Saturn'), 'status'].iloc[0]
    status760 = summary.loc[(summary['name'] == '760') & (summary['resonance'] == 'nu6_Saturn'), 'status'].iloc[0]
    assert 2 == abs(status759)
    assert 2 == abs(status1222)
    assert 0 == abs(status760)


# def test_secular_check_other_resonances():
#     resonance_5507 = GeneralSecularResonance(
#         coeffs={'varpi': [1, -2, 1]},  # [body_coeff, saturn_coeff, jupiter_coeff]
#         planet_names=['Saturn', 'Jupiter'],
#         resonance_name='g-2g6+g5',
#     )
#     sim = resonances.secular_finder.check(
#         asteroids=[5507],
#         secular_resonance=resonance_5507,
#         integration_years=200000,
#         oscillations_cutoff=0.0005,
#         plot='all',
#         save='all',
#     )

#     sim.run(progress=True)


# 2335 nu5
