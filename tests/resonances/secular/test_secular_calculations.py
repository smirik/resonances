#!/usr/bin/env python3
"""
Simple test to verify secular resonance coefficient parsing and angle calculations
for g-g6+s-s6 and 2g-2g6+s-s6 cases.
"""

import math
from resonances.resonance.secular import GeneralSecularResonance


class SimpleBody:
    """Simple body class with orbital elements."""
    def __init__(self, Omega, omega):
        self.Omega = Omega  # Longitude of ascending node
        self.omega = omega  # Argument of pericenter


def test_coefficient_parsing():
    """Test that formula parsing produces correct coefficients."""
    print("Testing coefficient parsing...\n")
    
    # Test case 1: g-g6+s-s6
    print("Case 1: g-g6+s-s6")
    res1 = GeneralSecularResonance(formula='g-g6+s-s6')
    coeffs1 = res1._parse_formula('g-g6+s-s6')
    print(f"Parsed coefficients: {coeffs1}")
    print(f"Expected: g=1, g6=-1, s=1, s6=-1")
    print(f"Resonance coefficients: {res1.coeffs}")
    print()
    
    # Test case 2: 2g-2g6+s-s6
    print("Case 2: 2g-2g6+s-s6")
    res2 = GeneralSecularResonance(formula='2g-2g6+s-s6')
    coeffs2 = res2._parse_formula('2g-2g6+s-s6')
    print(f"Parsed coefficients: {coeffs2}")
    print(f"Expected: g=2, g6=-2, s=1, s6=-1")
    print(f"Resonance coefficients: {res2.coeffs}")
    print()


def test_angle_calculations():
    """Test angle calculations with concrete values."""
    print("\nTesting angle calculations...\n")
    
    # Create test bodies with simple values
    asteroid = SimpleBody(Omega=0.5, omega=0.3)  # radians
    saturn = SimpleBody(Omega=0.2, omega=0.1)     # radians
    
    # Test case 1: g-g6+s-s6
    print("Case 1: g-g6+s-s6")
    res1 = GeneralSecularResonance(formula='g-g6+s-s6')
    angle1 = res1.calc_angle(asteroid, [saturn])
    
    # Manual calculation
    manual_angle1 = (
        1 * (asteroid.Omega + asteroid.omega) +     # g term
        -1 * (saturn.Omega + saturn.omega) +        # -g6 term
        1 * asteroid.Omega +                        # s term
        -1 * saturn.Omega                           # -s6 term
    )
    manual_angle1 = manual_angle1 % (2 * math.pi)
    
    print(f"Asteroid: Omega={asteroid.Omega}, omega={asteroid.omega}, varpi={asteroid.Omega + asteroid.omega}")
    print(f"Saturn: Omega={saturn.Omega}, omega={saturn.omega}, varpi={saturn.Omega + saturn.omega}")
    print(f"Calculated angle: {angle1:.6f} rad ({math.degrees(angle1):.2f}°)")
    print(f"Manual calculation: {manual_angle1:.6f} rad ({math.degrees(manual_angle1):.2f}°)")
    print(f"Match: {abs(angle1 - manual_angle1) < 1e-10}")
    
    # Detailed breakdown
    print("\nDetailed breakdown:")
    print(f"  g term (asteroid varpi): 1 * {asteroid.Omega + asteroid.omega:.3f} = {asteroid.Omega + asteroid.omega:.3f}")
    print(f"  -g6 term (Saturn varpi): -1 * {saturn.Omega + saturn.omega:.3f} = {-(saturn.Omega + saturn.omega):.3f}")
    print(f"  s term (asteroid Omega): 1 * {asteroid.Omega:.3f} = {asteroid.Omega:.3f}")
    print(f"  -s6 term (Saturn Omega): -1 * {saturn.Omega:.3f} = {-saturn.Omega:.3f}")
    print(f"  Total: {manual_angle1:.6f}")
    print()
    
    # Test case 2: 2g-2g6+s-s6
    print("Case 2: 2g-2g6+s-s6")
    res2 = GeneralSecularResonance(formula='2g-2g6+s-s6')
    angle2 = res2.calc_angle(asteroid, [saturn])
    
    # Manual calculation
    manual_angle2 = (
        2 * (asteroid.Omega + asteroid.omega) +     # 2g term
        -2 * (saturn.Omega + saturn.omega) +        # -2g6 term
        1 * asteroid.Omega +                        # s term
        -1 * saturn.Omega                           # -s6 term
    )
    manual_angle2 = manual_angle2 % (2 * math.pi)
    
    print(f"Calculated angle: {angle2:.6f} rad ({math.degrees(angle2):.2f}°)")
    print(f"Manual calculation: {manual_angle2:.6f} rad ({math.degrees(manual_angle2):.2f}°)")
    print(f"Match: {abs(angle2 - manual_angle2) < 1e-10}")
    
    # Detailed breakdown
    print("\nDetailed breakdown:")
    print(f"  2g term (asteroid varpi): 2 * {asteroid.Omega + asteroid.omega:.3f} = {2 * (asteroid.Omega + asteroid.omega):.3f}")
    print(f"  -2g6 term (Saturn varpi): -2 * {saturn.Omega + saturn.omega:.3f} = {-2 * (saturn.Omega + saturn.omega):.3f}")
    print(f"  s term (asteroid Omega): 1 * {asteroid.Omega:.3f} = {asteroid.Omega:.3f}")
    print(f"  -s6 term (Saturn Omega): -1 * {saturn.Omega:.3f} = {-saturn.Omega:.3f}")
    print(f"  Total: {manual_angle2:.6f}")


if __name__ == "__main__":
    test_coefficient_parsing()
    test_angle_calculations()