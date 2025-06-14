# Secular Resonances

Secular resonances occur when the precession frequency of an asteroid's orbital elements (perihelion or ascending node) matches the precession frequency of a planet's orbital elements. Unlike mean-motion resonances (MMRs) that involve orbital periods, secular resonances involve the slow precession of orbital orientations over much longer timescales.

The secular resonance system in this package provides tools for identifying and analyzing asteroids trapped in these long-period resonances. The implementation includes both predefined resonances (ν₆, ν₅, ν₁₆) and support for custom secular resonance configurations.

## Types of Secular Resonances

### Perihelion Precession Resonances

**ν₆ Resonance (nu6)**: The most important secular resonance in the asteroid belt. It occurs when an asteroid's perihelion precession frequency matches Saturn's perihelion precession frequency. This resonance creates a boundary around 2.0-2.1 AU that prevents asteroids from having low eccentricities in this region.

**ν₅ Resonance (nu5)**: Occurs when an asteroid's perihelion precession frequency matches Jupiter's perihelion precession frequency. This resonance affects asteroids primarily in the inner asteroid belt.

### Node Precession Resonances

**ν₁₆ Resonance (nu16)**: A nodal secular resonance where an asteroid's ascending node precession frequency matches Saturn's node precession frequency. This resonance affects orbital inclinations and is important for understanding the inclination distribution of asteroids.

## Basic Usage

### Predefined Secular Resonances

The simplest way to analyze secular resonances is using the predefined resonance types:

```python
import resonances

# Create simulation for ν₆ secular resonance analysis
sim = resonances.Simulation(
    name='nu6_analysis',
    tmax=628319,  # 100,000 years (minimum for secular analysis)
    integrator='SABA(10,6,4)',  # Recommended for long-term secular dynamics
    dt=5.0,
    save='all',
    plot='all'
)

sim.create_solar_system()

# Add asteroid 1747 Wright (known ν₆ resonance)
sim.add_body_with_secular(1747, 'nu6', name='Wright')

# Run the simulation
sim.run()

# Get results
body = sim.bodies[0]
secular_res = body.secular_resonances[0]
print(f"Secular resonance status: {body.secular_statuses[secular_res.to_s()]}")
```

### Multiple Secular Resonances

You can analyze multiple secular resonances simultaneously:

```python
import resonances

sim = resonances.Simulation(
    name='multi_secular_analysis',
    tmax=1256637,  # 200,000 years for comprehensive analysis
    integrator='SABA(10,6,4)',
    dt=5.0
)

sim.create_solar_system()

# Add asteroid with multiple secular resonances
sim.add_body_with_secular(463, ['nu6', 'nu5'], name='Lola')

sim.run()

# Analyze results for each resonance
body = sim.bodies[0]
for secular in body.secular_resonances:
    status = body.secular_statuses[secular.to_s()]
    print(f"{secular.to_s()}: status = {status}")
```

### Custom Orbital Elements

You can also analyze synthetic asteroids with specific orbital elements:

```python
import resonances

# Define custom orbital elements
test_elements = {
    'a': 2.08,      # Semi-major axis near ν₆ resonance
    'e': 0.15,      # Moderate eccentricity
    'inc': 0.1,     # Low inclination
    'Omega': 0.5,   # Longitude of ascending node
    'omega': 1.2,   # Argument of perihelion
    'M': 2.1        # Mean anomaly
}

sim = resonances.Simulation(
    name='custom_nu6_test',
    tmax=3141593,  # 500,000 years
    integrator='SABA(10,6,4)',
    dt=5.0
)

sim.create_solar_system()
sim.add_body_with_secular(test_elements, 'nu6', name='TestAsteroid')

sim.run()
```

## Advanced Usage

### Custom Secular Resonances

For research purposes, you can create custom secular resonances using the `GeneralSecularResonance` class:

```python
import resonances
from resonances.resonance.secular import GeneralSecularResonance

# Create a custom secular resonance
# Example: resonance involving both perihelion and node precession
custom_coeffs = {
    'varpi': [1, -1],  # Body perihelion - Planet perihelion
    'Omega': [1, -1]   # Body node - Planet node
}

custom_secular = GeneralSecularResonance(
    coeffs=custom_coeffs,
    planet_names=['Saturn'],
    resonance_name='custom_mixed'
)

sim = resonances.Simulation(
    name='custom_secular_analysis',
    tmax=1256637,  # 200,000 years
    integrator='SABA(10,6,4)',
    dt=5.0
)

sim.create_solar_system()

# Add body with custom secular resonance
body = resonances.Body()
body.initial_data = {
    'a': 2.1, 'e': 0.2, 'inc': 0.15,
    'Omega': 0.8, 'omega': 1.5, 'M': 3.0
}
body.name = 'CustomTest'
body.secular_resonances = [custom_secular]

# Setup planet indices
custom_secular.index_of_planets = sim.body_manager.get_index_of_planets(['Saturn'])

sim.body_manager.bodies.append(body)
sim.run()
```

### Factory Function Usage

You can also use the factory function for creating secular resonances:

```python
import resonances

# Create individual secular resonances
nu6 = resonances.create_secular_resonance('nu6')
nu5 = resonances.create_secular_resonance('nu5')
nu16 = resonances.create_secular_resonance('nu16')

# Create multiple resonances at once
secular_list = resonances.create_secular_resonance(['nu6', 'nu5'])

print(f"ν₆ resonance: {nu6.to_s()}")
print(f"ν₅ resonance: {nu5.to_s()}")
print(f"ν₁₆ resonance: {nu16.to_s()}")
```

## Integration Considerations

### Time Scales

Secular resonances require much longer integration times than MMRs:

- **Minimum**: 10⁴ years (100,000 simulation units)
- **Recommended**: 10⁵ - 10⁶ years (1-10 million simulation units)
- **Research quality**: 10⁶ - 10⁷ years (10-100 million simulation units)

```python
# Different time scales for different purposes
time_scales = {
    'quick_test': 62832,      # 10,000 years
    'standard': 628319,       # 100,000 years
    'detailed': 6283185,      # 1,000,000 years
    'research': 62831853      # 10,000,000 years
}

sim = resonances.Simulation(
    tmax=time_scales['detailed'],  # Choose appropriate scale
    integrator='SABA(10,6,4)',
    dt=5.0  # Larger timestep for long integrations
)
```

### Integrator Selection

For secular resonance analysis, use symplectic integrators designed for long-term stability:

```python
# Recommended integrators for secular analysis
integrators = {
    'fast': 'whfast',           # Fast, good for initial tests
    'standard': 'SABA(10,6,4)', # High-order, excellent long-term stability
    'research': 'SABA(10,6,4)'  # Same as standard, with smaller dt
}

# Example with different integrator settings
sim = resonances.Simulation(
    integrator='SABA(10,6,4)',
    dt=5.0,  # 5-year timestep for million-year integrations
    tmax=6283185  # 1 million years
)
```

## Analysis and Interpretation

### Resonance Status Codes

The libration analysis system provides status codes for secular resonances:

| Status | Description                                                |
| :----: | ---------------------------------------------------------- |
|   2    | Pure libration around 0 or π (confirmed resonance)         |
|   1    | Transient libration (likely resonance)                     |
|   0    | No libration (not in resonance)                            |
|   -1   | Requires manual verification (unclear behavior)            |
|   -2   | Requires manual verification (pure but no frequency match) |

### Accessing Results

After integration, you can access detailed results:

```python
sim.run()
body = sim.bodies[0]
secular_res = body.secular_resonances[0]

# Basic status
status = body.secular_statuses[secular_res.to_s()]
print(f"Resonance status: {status}")

# Libration metrics
if secular_res.to_s() in body.libration_metrics:
    metrics = body.libration_metrics[secular_res.to_s()]
    print(f"Max libration length: {metrics.get('max_libration_length', 0)} years")
    print(f"Number of libration periods: {metrics.get('num_libration_periods', 0)}")

# Pure libration flag
is_pure = body.libration_pure.get(secular_res.to_s(), False)
print(f"Pure libration: {is_pure}")

# Monotony metric
monotony = body.monotony.get(secular_res.to_s(), 0)
print(f"Monotony: {monotony:.3f}")
```

### Plotting and Visualization

Create custom plots for secular resonance analysis:

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_secular_analysis(body, secular_res, times):
    """Create comprehensive secular resonance analysis plot."""

    # Get data
    angles = body.secular_angles[secular_res.to_s()]
    times_years = times / (2 * np.pi)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'Secular Resonance Analysis: {body.name} - {secular_res.to_s()}')

    # Plot 1: Resonant angle evolution
    axes[0].plot(times_years, angles, 'b-', linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('Resonant Angle (rad)')
    axes[0].set_title('Resonant Angle Evolution')
    axes[0].grid(True, alpha=0.3)

    # Add reference lines
    for y_val, label in [(0, '0'), (np.pi, 'π'), (2*np.pi, '2π')]:
        axes[0].axhline(y=y_val, color='r', linestyle='--', alpha=0.5)
        axes[0].text(times_years[-1] * 0.02, y_val + 0.1, label, color='r')

    # Plot 2: Semi-major axis evolution
    axes[1].plot(times_years, body.axis, 'g-', linewidth=0.5, alpha=0.7)
    axes[1].set_ylabel('Semi-major Axis (AU)')
    axes[1].set_title('Semi-major Axis Evolution')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Eccentricity evolution
    axes[2].plot(times_years, body.ecc, 'r-', linewidth=0.5, alpha=0.7)
    axes[2].set_ylabel('Eccentricity')
    axes[2].set_xlabel('Time (years)')
    axes[2].set_title('Eccentricity Evolution')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Use the plotting function
sim.run()
body = sim.bodies[0]
secular_res = body.secular_resonances[0]
plot_secular_analysis(body, secular_res, sim.times)
```

## Performance Optimization

### Memory Management

For very long integrations, consider memory usage:

```python
# Optimize output frequency for long integrations
sim = resonances.Simulation(
    tmax=62831853,  # 10 million years
    integrator='SABA(10,6,4)',
    dt=10.0  # Larger timestep
)

# Set reasonable number of output points
sim.Nout = 10000  # 10,000 points over 10 Myr = 1,000 year resolution
```

### Parallel Analysis

For multiple asteroids, consider running separate simulations:

```python
def analyze_asteroid_secular(asteroid_id, secular_type, name):
    """Analyze single asteroid for secular resonance."""
    sim = resonances.Simulation(
        name=f"{name}_{secular_type}",
        tmax=1256637,  # 200,000 years
        integrator='SABA(10,6,4)',
        dt=5.0
    )

    sim.create_solar_system()
    sim.add_body_with_secular(asteroid_id, secular_type, name=name)
    sim.run()

    return sim

# Analyze multiple asteroids
asteroids = [
    (1747, 'Wright', 'nu6'),
    (1566, 'Icarus', 'nu6'),
    (1862, 'Apollo', 'nu5')
]

results = []
for asteroid_id, name, secular_type in asteroids:
    result = analyze_asteroid_secular(asteroid_id, secular_type, name)
    results.append(result)
```

## Summary Data

Get comprehensive summary of all secular resonance analyses:

```python
# Run simulation with summary enabled
sim = resonances.Simulation(
    save_summary=True,  # Enable summary generation
    save='all'
)

# ... add bodies and run simulation ...

# Get summary DataFrame
summary = sim.get_simulation_summary()

# Filter for secular resonances only
secular_summary = summary[summary['type'] == 'Secular']

print("Secular Resonance Summary:")
print(secular_summary[['name', 'resonance', 'status', 'pure', 'max_libration_length']])

# Save summary to file
secular_summary.to_csv(f"{sim.save_path}/secular_summary.csv", index=False)
```

## Best Practices

1. **Use appropriate time scales**: Minimum 10⁵ years for reliable secular resonance identification
2. **Choose stable integrators**: SABA(10,6,4) is recommended for long-term secular dynamics
3. **Monitor libration metrics**: Use the full libration analysis system for proper identification
4. **Visual verification**: Always plot resonant angle evolution for manual verification
5. **Consider computational cost**: Balance integration time with available computational resources
6. **Save intermediate results**: Use `save='all'` to preserve data for post-processing analysis

## References

For theoretical background on secular resonances, see:

1. **Murray, C. D. & Dermott, S. F.** Solar System Dynamics. Cambridge University Press (1999).
2. **Morbidelli, A.** Modern Celestial Mechanics: Aspects of Solar System Dynamics. Taylor & Francis (2002).
3. **Knežević, Z. & Milani, A.** Proper element catalogs and asteroid families. Astronomy & Astrophysics (2003).
