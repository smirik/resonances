#!/usr/bin/env python3
"""
Final Improved Secular Resonance Analysis
=========================================

This script analyzes multiple secular resonance asteroids simultaneously using the SABA(10,6,4) integrator
and the standard plotting functionality provided by the resonances package.

Key improvements:
1. Simultaneous integration of all asteroids (more efficient)
2. Proper SABA integrator configuration
3. Standard plotting for all results
4. Robust error handling

Test asteroids:
1. Asteroid 1222 - ν₆ resonance (Tina family)
2. Asteroid 31 - ν₆ resonance
3. Asteroid 759 - ν₆ resonance
4. Asteroid 956 - ν₁₆ (z2) resonance

Integration: 1 million years with SABA(10,6,4) integrator
Plotting: Standard resonances.resonance.plot.body() function
"""

import numpy as np
import resonances
import resonances.resonance.plot
import time


def analyze_multiple_secular_asteroids(integration_years=1000000):  # noqa: C901
    """
    Analyze multiple asteroids for secular resonance simultaneously using SABA integrator.

    Parameters:
    -----------
    integration_years : int
        Integration time in years (default 1 million)
    """
    print("🌌 Final Improved Secular Resonance Analysis")
    print("=" * 60)
    print("Analyzing multiple secular resonance asteroids simultaneously")
    print("Integration time: 1,000,000 years")
    print("Integrator: SABA(10,6,4) - optimized for long-term secular dynamics")
    print("Plotting: Standard resonances.resonance.plot.body() function")
    print("=" * 60)

    # Define known secular resonance asteroids
    asteroids = [
        {'id': 1222, 'name': 'Tina', 'secular': 'nu6'},
        {'id': 31, 'name': 'Euphrosyne', 'secular': 'nu6'},
        {'id': 759, 'name': 'Vinifera', 'secular': 'nu6'},
        {'id': 956, 'name': 'Elisa', 'secular': 'nu16'},
    ]

    print(f"\n🔬 Setting up simultaneous analysis of {len(asteroids)} asteroids")

    # Create single simulation with SABA integrator
    sim = resonances.Simulation(
        name="multi_secular_final",
        tmax=int(integration_years * 2 * np.pi),
        integrator='SABA(10,6,4)',
        dt=5.0,
        save='all',
        plot='save',
    )

    # Set up solar system
    print("✓ Creating solar system...")
    sim.create_solar_system()

    # Add all asteroids to the same simulation
    print("✓ Adding asteroids:")
    for asteroid in asteroids:
        print(f"  - Asteroid {asteroid['id']} ({asteroid['name']}) with {asteroid['secular']} resonance")
        sim.add_body(asteroid['id'], asteroid['secular'], name=asteroid['name'])

    # Configure integration
    sim.Nout = 10000  # 10,000 points over 1 Myr = 100 year resolution

    print("✓ Integration setup:")
    print(f"  - Duration: {integration_years:,} years")
    print(f"  - Output points: {sim.Nout}")
    print(f"  - Time resolution: {integration_years/sim.Nout:.1f} years per point")
    print(f"  - Total bodies: {len(sim.bodies)} asteroids + 9 planets")

    # Manually configure the integrator to ensure SABA is used properly
    # This addresses the WHFast warning issue
    print("✓ Configuring SABA integrator...")

    # Run simulation with timing
    start_time = time.time()

    print("✓ Running simultaneous integration...")

    try:
        # Override integrator setup before running
        def configure_saba_properly():
            sim.integration_engine.sim.integrator = 'SABA(10,6,4)'
            sim.integration_engine.sim.dt = 5.0
            # Set N_active to include all bodies to avoid WHFast fallback
            total_bodies = len(sim.integration_engine.sim.particles) + len(sim.bodies)
            sim.integration_engine.sim.N_active = total_bodies
            sim.integration_engine.sim.ri_saba.safe_mode = 1
            sim.integration_engine.sim.move_to_com()
            print(f"  - Integrator: {sim.integration_engine.sim.integrator}")
            print(f"  - N_active: {sim.integration_engine.sim.N_active}")

        # Apply configuration before adding bodies to simulation
        sim.integration_engine.setup_integrator = configure_saba_properly

        sim.run(progress=True)

    except Exception as e:
        print(f"❌ Integration failed: {e}")
        return None

    end_time = time.time()
    execution_time = end_time - start_time
    performance = integration_years / execution_time if execution_time > 0 else 0

    print("✅ Simultaneous integration completed!")
    print(f"   ⏱️  Execution time: {execution_time:.2f} seconds")
    print(f"   🚀 Performance: {performance:,.0f} years/second")

    # Analyze each body
    results = []
    print("📊 Individual Asteroid Analysis:")

    for _, (body, asteroid_info) in enumerate(zip(sim.bodies, asteroids)):
        print(f"\n--- {asteroid_info['name']} ({asteroid_info['secular'].upper()}) ---")

        try:
            # Get the secular resonance
            secular_res = body.secular_resonances[0]

            # Get angle data for analysis
            angles = body.secular_angles[secular_res.to_s()]

            # Calculate angle statistics
            angle_range = np.max(angles) - np.min(angles)
            angle_mean = np.mean(angles)
            is_librating = angle_range < np.pi

            print(f"   Range: {np.degrees(angle_range):.1f}°")
            print(f"   Mean: {np.degrees(angle_mean):.1f}°")

            if is_librating:
                print("   🎯 POTENTIAL LIBRATION")
            else:
                print("   🔄 CIRCULATION")

            # Generate standard plot
            print("   📊 Generating plot...")
            resonances.resonance.plot.body(sim, body, secular_res, image_type='png')

            # Store results
            results.append(
                {
                    'name': asteroid_info['name'],
                    'asteroid_id': asteroid_info['id'],
                    'secular_type': asteroid_info['secular'],
                    'angle_range_deg': np.degrees(angle_range),
                    'is_librating': is_librating,
                    'angle_mean_deg': np.degrees(angle_mean),
                    'semi_major_axis_mean': np.mean(body.axis),
                    'eccentricity_mean': np.mean(body.ecc),
                }
            )

        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
            continue

    # Final summary
    print(f"\n{'='*60}")
    print("🎯 FINAL SUMMARY")
    print("=" * 60)

    total_librating = 0
    for result in results:
        status = "LIBRATING" if result['is_librating'] else "CIRCULATING"
        print(f"{result['name']} ({result['secular_type'].upper()}): {status}")
        print(f"  Angle range: {result['angle_range_deg']:.1f}°")
        print(f"  Semi-major axis: {result['semi_major_axis_mean']:.4f} AU")
        if result['is_librating']:
            total_librating += 1
        print()

    print(f"📊 Results: {total_librating}/{len(results)} asteroids show potential libration")
    print(f"⚡ Execution time: {execution_time:.2f} seconds")
    print(f"🚀 Performance: {performance:,.0f} years/second")

    print("💡 Key Improvements Implemented:")
    print("• ✅ Simultaneous integration of all asteroids")
    print("• ✅ Proper SABA(10,6,4) integrator configuration")
    print("• ✅ Robust error handling")
    print("• ✅ Standard plotting functionality")
    print("• ✅ Efficient single simulation setup")

    return results


if __name__ == "__main__":
    analyze_multiple_secular_asteroids()
