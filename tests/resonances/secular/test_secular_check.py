from resonances.resonance.secular import GeneralSecularResonance
import resonances.secular_finder
import resonances
import numpy as np


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


def test_general_secular_resonance():
    """Test that asteroid 759 shows libration in nu6 secular resonance using GeneralSecularResonance with g-g6 formula."""

    general_nu6 = GeneralSecularResonance(formula='g-g6')
    assert isinstance(general_nu6, GeneralSecularResonance), f"Expected GeneralSecularResonance, got {type(general_nu6)}"
    print(f"Resonance formula: {general_nu6.to_s()}")

    resonance_5507 = GeneralSecularResonance(formula='g-2g6+g5')

    sim = resonances.Simulation(
        name="test_secular_check_general_nu6",
        tmax=int(200000 * 2 * np.pi),
        integrator='SABA(10,6,4)',
        dt=5.0,
        save='all',
        plot='all',
        oscillations_cutoff=0.0005,
        libration_period_min=10000,
        libration_period_critical=200000 * 0.2,
    )
    sim.create_solar_system()
    asteroids = [759, 760, 1222]
    for asteroid in asteroids:
        sim.add_body(asteroid, general_nu6, name=f"{asteroid}")
        print(f"Added asteroid {asteroid} with GeneralSecularResonance")
    sim.add_body(5507, [resonance_5507, general_nu6], name="5507")
    print("Added asteroid 5507 with GeneralSecularResonance")
    sim.run(progress=True)

    summary = sim.data_manager.get_simulation_summary(sim.bodies)
    for _idx, row in summary.iterrows():
        print(f"Asteroid {row['name']}: status = {row['status']}")

    resonance_names = summary['resonance'].unique()
    expected_resonance = 'g-g6'
    assert expected_resonance in resonance_names, f"Expected '{expected_resonance}' in resonance names: {resonance_names}"

    status759 = summary.loc[(summary['name'] == '759') & (summary['resonance'] == expected_resonance), 'status'].iloc[0]
    status1222 = summary.loc[(summary['name'] == '1222') & (summary['resonance'] == expected_resonance), 'status'].iloc[0]
    status760 = summary.loc[(summary['name'] == '760') & (summary['resonance'] == expected_resonance), 'status'].iloc[0]
    status5507_nu6 = summary.loc[(summary['name'] == '5507') & (summary['resonance'] == expected_resonance), 'status'].iloc[0]
    status5507_g = summary.loc[(summary['name'] == '5507') & (summary['resonance'] == 'g-2g6+g5'), 'status'].iloc[0]

    assert 2 == abs(status759), f"Expected |status759| = 2, got {abs(status759)}"
    assert 2 == abs(status1222), f"Expected |status1222| = 2, got {abs(status1222)}"
    assert 0 == abs(status760), f"Expected |status760| = 0, got {abs(status760)}"
    assert 2 == abs(status5507_g), f"Expected |status5507| in g-2g6+g5 = 2, got {abs(status5507_g)}"
    assert 2 != abs(status5507_nu6), f"Expected |status5507| in nu6 = 2, got {abs(status5507_nu6)}"
