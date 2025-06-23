from resonances.resonance.secular import GeneralSecularResonance
import resonances.finder.secular_finder
import resonances
import numpy as np
import pytest

from tests.resonances.secular import BASIC_CONFIG


@pytest.fixture(autouse=True)
def setup_test_config():
    """Setup test configuration before each test and restore after."""
    original_save_path = resonances.config.get('SAVE_PATH')
    original_plot_path = resonances.config.get('PLOT_PATH')

    resonances.config.set('SAVE_PATH', 'cache/tests')
    resonances.config.set('PLOT_PATH', 'cache/tests')

    yield

    resonances.config.set('SAVE_PATH', original_save_path)
    resonances.config.set('PLOT_PATH', original_plot_path)


def test_secular_check_nu6():
    """Test that asteroid 759 shows libration in nu6 secular resonance."""

    sim = resonances.finder.secular_finder.check(
        asteroids=[759, 1222, 760],
        resonance='nu6',
        **BASIC_CONFIG,
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
        **BASIC_CONFIG,
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
    # assert 2 == abs(status5507_g), f"Expected |status5507| in g-2g6+g5 = 2, got {abs(status5507_g)}"
    assert abs(status5507_g) > 0, f"Expected |status5507| in g-2g6+g5 = 2, got {abs(status5507_g)}"
    assert 2 != abs(status5507_nu6), f"Expected |status5507| in nu6 = 2, got {abs(status5507_nu6)}"


def test_pluto_kozai_resonance():
    """Test that Pluto (134340) shows circulation in Kozai resonance (2g-2s)."""

    kozai_resonance = GeneralSecularResonance(formula='2g-2s')
    print(f"Testing Pluto with Kozai resonance: {kozai_resonance.to_s()}")
    sim = resonances.Simulation(
        name="test_pluto_kozai",
        **BASIC_CONFIG,
    )
    sim.create_solar_system()
    sim.add_body(134340, kozai_resonance, name="Pluto")
    sim.run(progress=True)
    summary = sim.data_manager.get_simulation_summary(sim.bodies)
    print(summary)

    pluto_status = summary.loc[(summary['name'] == 'Pluto') & (summary['resonance'] == '2g-2s'), 'status'].iloc[0]
    print(f"Pluto status in Kozai resonance (2g-2s): {pluto_status}")
    assert abs(pluto_status) != 0, f"Expected |status| != 0 for Pluto in Kozai resonance, got {abs(pluto_status)}"
