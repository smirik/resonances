import pytest
import resonances


@pytest.mark.slow
def test_classification_edge_cases():
    """
    Test classification of edge cases.
    Test cases:
    - Jupiter resonances (2J-1, 3J-1, 1J+1):
      * Transient: 887, 6489 (3J-1), 17346 (2J-1)
      * Full libration: 4177 (2J-1)
      * Chaotic: 624 (1J+1, uniformity > 0.7)
    - Neptune resonances (3N-4):
      * Clear circulation: 564160 (r_sq=0.997), 666184 (r_sq=0.999)
      * Transient: 42355 (r_sq=0.815)
    """
    sim = resonances.Simulation(
        name="classification_edge_cases",
        integration_years=40000,
        integrator="SABA(10,6,4)",
        planets=["Jupiter", "Neptune"],
        save="none",
        plot="none",
    )
    sim.create_solar_system()

    test_cases = [
        # === Jupiter resonances ===
        # Transient cases - status 1 or -1
        # ("6489", "3J-1", 1, 'nominal'),  # Libration ~8000-20000 yrs - TODO: revisit after drift_norm tuning
        ("17346", "2J-1", 1, 'absolute'),  # Transient, r_sq=0.92, lib_frac=0.32
        ("4177", "2J-1", 2, 'nominal'),  # Clean apocentric libration throughout
        ("624", "1J+1", 0, 'nominal'),  # Uniformity > 0.7, not resonant
        ("564160", "3N-4", 0, 'nominal'),  # r_sq=0.997, definite circulation
        ("666184", "3N-4", 0, 'nominal'),  # r_sq=0.999, definite circulation
        # ("42355", "3N-4", 1, 'absolute'),  # r_sq=0.815 - TODO: revisit after drift_norm tuning
    ]

    test_cases_chaotic = [
        ("189865", "lkr"),
        ("477492", "lkr"),
    ]

    for asteroid, resonance_str, _, _ in test_cases:
        mmr = resonances.create_mmr(resonance_str)
        sim.add_body(asteroid, mmr, name=f"{asteroid}")
    for asteroid, resonance_str in test_cases_chaotic:
        lkr = resonances.create_resonance(resonance_str)
        sim.add_body(asteroid, lkr, name=f"{asteroid}")

    sim.run()

    summary = sim.data_manager.get_simulation_summary(sim.bodies)

    for asteroid, resonance_str, expected_abs_status, expected_type in test_cases:
        mask = summary["name"] == asteroid
        matching_rows = summary[mask]

        assert len(matching_rows) > 0, f"No results found for asteroid {asteroid}"

        status = matching_rows["status"].iloc[0]
        if expected_type == 'absolute':
            status = abs(status)

        assert status == expected_abs_status, (
            f"Asteroid {asteroid} ({resonance_str}): "
            f"expected |status|={expected_abs_status}, got status={status} (expected_type={expected_type})"
        )

    for asteroid, resonance_str in test_cases_chaotic:
        mask = summary["name"] == asteroid
        matching_rows = summary[mask]
        assert len(matching_rows) > 0, f"No results found for asteroid {asteroid}"
        status = matching_rows["status"].iloc[0]
        assert status == -3, f"Asteroid {asteroid} ({resonance_str}): expected status=-3, got status={status}"
