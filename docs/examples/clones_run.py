"""Clones for docs/examples.md. The clones run in worker processes (spawned on macOS and Windows), so a script that
runs them needs the `if __name__ == '__main__':` guard."""

# --8<-- [start:clones]
import resonances


def main():
    solution = resonances.orbit_solution(463, source='astdys')  # nominal orbit + covariance (cached in cache/clones/)
    ensemble = solution.sample(n=10, seed=0)  # 10 clones and the nominal; 100-1000 for real shares

    sim = resonances.Simulation(name='clones-demo', date='2026-01-01', integration_years=2_000)
    sim.create_solar_system()
    sim.add_ensemble(ensemble, resonances=['4J-2S-1'])  # clones drawn at another epoch are integrated to the date first
    sim.run()
    shares = sim.clone_summary  # the rows of clones.csv: the share of each status with its 95 % Wilson interval
    print(shares[['resonance', 'n_clones', 'nominal_status', 'modal_status', 'modal_share', 'f_resonant_lo', 'f_resonant_hi']])


if __name__ == '__main__':  # the clones run in worker processes: the guard is required
    main()
# --8<-- [end:clones]
