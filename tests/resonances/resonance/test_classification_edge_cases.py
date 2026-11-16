import astdys.util
import pytest
import resonances
from resonances.resonance.analysis.verdict import AngleStatus, Reason, Status


@pytest.mark.slow
def test_classification_edge_cases():
    """
    Test classification of edge cases.
    Test cases:
    - Jupiter resonances (2J-1, 3J-1, 1J+1):
      * Transient: 6489 (3J-1), 17346 (2J-1; libration only at the end)
      * Full apocentriclibration: 4177 (2J-1)
      * Not resonant: 624 (1J+1)
    - Neptune resonances (3N-4):
      * Near the separatrix: 564160 (circulates on the inner side of the island, approaches the saddle of the
        spatial model in the first ~30 kyr; author:2026-10-17)
      * Clear circulation: 666184
      (42355 3N-4 removed by the author 2026-10-26: chaotic, its status changes with the epoch; today non_resonant)
    The date is fixed at 2026-06-10, where every row was measured (W24; the rows were the same on 4 dates).
    The Lidov-Kozai cases 189865 and 477492 were removed by the author 2026-10-27 (W24 Q5b): their 'invalid' was a SABA
    blow-up on planet-crossing orbits; on ias15 they are non_resonant and resonant (the benchmark records
    zlk__189865__LK, zlk__477492__LK). The invalid path is checked on a real ejection below.
    """
    sim = resonances.Simulation(
        name="classification_edge_cases",
        source='nasa',
        date='2026-06-10',
        integration_years=50000,
        integrator="SABA(10,6,4)",
        planets=["Jupiter", "Neptune"],
        save="none",
        plot="none",
    )
    sim.create_solar_system()

    test_cases = [
        # === Jupiter resonances ===
        # ("6489", "3J-1", Status.TRANSIENT),  # Libration ~8000-20000 yrs
        ("17346", "2J-1", Status.TRANSIENT),  # libration only at the end (was |status| = 1: transient, sure or not)
        ("4177", "2J-1", Status.RESONANT),  # Clean apocentric libration throughout
        ("624", "1J+1", Status.NON_RESONANT),  #
        ("564160", "3N-4", Status.NEAR_SEPARATRIX),  # circulation along the separatrix (author:2026-10-17)
        ("666184", "3N-4", Status.NON_RESONANT),  # r_sq=0.999, definite circulation
    ]

    for asteroid, resonance_str, _ in test_cases:
        mmr = resonances.create_mmr(resonance_str)
        sim.add_body(asteroid, mmr, name=f"{asteroid}")

    sim.run()

    summary = sim.data_manager.get_simulation_summary(sim.bodies)

    # every case is checked before failing, so one run shows all the disagreements
    wrong = []
    for asteroid, resonance_str, expected_status in test_cases:
        matching_rows = summary[summary["name"] == asteroid]
        assert len(matching_rows) > 0, f"No results found for asteroid {asteroid}"
        status = matching_rows["status"].iloc[0]
        if status != expected_status:
            wrong.append(f"{asteroid} ({resonance_str}): expected {expected_status}, got {status}")

    assert not wrong, "; ".join(wrong)


# A body on a bound heliocentric orbit (a = 15.2 au, e = 0.77, q = 3.55 au) that meets Jupiter: it starts 0.5 au
# outside Jupiter on the Sun-Jupiter line, moving sunward at 0.3 (8.9 km/s) relative to Jupiter, aimed 0.015 au behind
# it. The flyby (closest approach 0.0072 au = 15 Jupiter radii, measured with REBOUND ias15 at 0.2-day sampling, no
# collision) adds the energy of a gravity assist: the orbit is hyperbolic after it (e 1.75, a -6.6 au, ten years
# later). Every aim from 0.01 to 0.03 au behind Jupiter (7 to 43 R_J) ejects it, so the result does not hang on the
# planets' positions to better than 10^6 km. Elements: heliocentric ecliptic J2000 at MJD 60000, from those
# Cartesian states and the Horizons planets of that date.
EJECTED = {
    'a': 15.223145494326948,
    'e': 0.7666831907828751,
    'inc': 0.022751653787939172,
    'Omega': 1.754361340551568,
    'omega': 6.199250440161631,
    'M': 6.123859548269845,
}


@pytest.mark.slow
def test_an_orbit_ejected_by_jupiter_is_invalid(tmp_path):
    """The unphysical path on a physical ejection (ias15 resolves the encounter), for an MMR and a ZLK row alike: the
    record starts bound, becomes hyperbolic, and no verdict is given on its angles (invalid/unphysical)."""
    sim = resonances.Simulation(
        name='ejected_by_jupiter',
        date=astdys.util.convert_mjd_to_datetime(60000),
        integrator='ias15',
        integration_years=1000,
        save=None,
        plot=None,
        save_summary=False,
        save_path=str(tmp_path),
        plot_path=str(tmp_path),
    )
    sim.create_solar_system()
    sim.add_body(EJECTED, ['1J-3', 'LK'], name='ejected')
    sim.run(progress=False)

    body = sim.bodies[0]
    assert 0 < body.axis[0] and body.ecc[0] < 1  # bound at the start
    assert body.axis[-1] < 0 and body.ecc[-1] > 1  # unbound at the end: ejected
    summary = sim.data_manager.get_simulation_summary(sim.bodies)
    assert sorted(summary['resonance']) == ['1J-3+0+2', 'LK']
    for _, row in summary.iterrows():
        assert (row['status'], row['reason'], row['angle_status']) == (Status.INVALID, Reason.UNPHYSICAL, AngleStatus.UNDEFINED), row
