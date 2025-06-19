# Secular Resonances

This guide shows how to work with secular resonances in the resonances library. Examples assume familiarity with secular resonance theory and asteroid dynamics.

## Quick Start

### Creating Individual Secular Resonances

```python
import resonances

# Create specific secular resonances
nu6 = resonances.create_secular_resonance('nu6')      # ν₆ resonance (g-g₆)
nu5 = resonances.create_secular_resonance('nu5')      # ν₅ resonance (g-g₅)
nu16 = resonances.create_secular_resonance('nu16')    # ν₁₆ resonance (s-s₆)

# Create from mathematical formulas
nu6_formula = resonances.create_secular_resonance('g-g6')
nu5_formula = resonances.create_secular_resonance('g-g5')
nu16_formula = resonances.create_secular_resonance('s-s6')

# Create nonlinear resonances
nonlinear = resonances.create_secular_resonance('2g-g5-g6')
complex_res = resonances.create_secular_resonance('g+s-s7-g5')
```

### Building Multiple Resonances

```python
from resonances.matrix.secular_matrix import SecularMatrix

# Build specific resonances
resonances_list = SecularMatrix.build(['g-g5', 'g-g6', 's-s6'])

# Build all resonances of a specific order
order2_resonances = SecularMatrix.build(order=2)  # Linear resonances
order4_resonances = SecularMatrix.build(order=4)  # Nonlinear resonances of order 4

# Build all available resonances
all_resonances = SecularMatrix.build()
```

## Example: Asteroid 759 Vinifera (ν₆ Resonance)

```python
import resonances

# Create the ν₆ resonance
nu6 = resonances.create_secular_resonance('nu6')
# Set up simulation with Vinifera
sim = resonances.Simulation()
sim.add_body(759, nu6, name='Vinifera')
# Access the body
body = sim.bodies[0]
print(f"Added body: {body.name}")
print(f"Body has {len(body.secular_resonances)} secular resonances")

sim = resonances.secular_finder.check(
    asteroids=[759],
    resonance='nu6',
    integration_years=1000000,
    oscillations_cutoff=0.0005,
    plot='all',
    save='all',
)# Configure simulation
sim.run(progress=True)

print(f"Simulation completed for {sim.bodies[0].name}")
print(f"Resonance: {sim.bodies[0].secular_resonances[0].to_s()}")
```

## Nonlinear Secular Resonances

```python
import resonances

# High-order resonances
high_order = resonances.GeneralSecularResonance(formula='g+s-s7-g5')
kozai_type = resonances.GeneralSecularResonance(formula='2g-2s')

# Mixed frequency resonances
mixed = resonances.GeneralSecularResonance(formula='g-g5+s7-s6')
complex_mixed = resonances.GeneralSecularResonance(formula='2g-s7-s6')

# Special case with parentheses
special = resonances.GeneralSecularResonance(formula='2(g-g6)+(s-s6)')
```

## Custom Planetary Frequencies

You can customize planetary frequencies by creating a `.env` file in your project directory:

```bash
# .env file
g5=4.25749319
g6=28.24552984
g7=3.08675577
g8=0.67255084
s5=0.0
s6=-26.34496354
s7=-2.99266093
s8=-0.69251386
```

The frequencies will be loaded automatically when you import the resonances library.

```python
from resonances.matrix.secular_resonances import load_planetary_frequencies

# Load current frequencies
frequencies = load_planetary_frequencies()
print("Current planetary frequencies (arcsec/yr):")
for freq, value in frequencies.items():
    print(f"  {freq}: {value}")
```

## References

For theoretical background on secular resonances, see:

1. **Murray, C. D. & Dermott, S. F.** Solar System Dynamics. Cambridge University Press (1999).
2. **Morbidelli, A.** Modern Celestial Mechanics: Aspects of Solar System Dynamics. Taylor & Francis (2002).
3. **Knežević, Z. & Milani, A.** Proper element catalogs and asteroid families. Astronomy & Astrophysics (2003).
