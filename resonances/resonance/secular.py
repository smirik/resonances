import rebound
from resonances.resonance.resonance import Resonance


class SecularResonance(Resonance):
    """
    Base class for secular resonances.

    Secular resonances occur when the precession frequency of an asteroid's
    orbital elements (perihelion or node) matches the precession frequency
    of a planet's orbital elements.
    """

    def __init__(self, resonance_type, planet_name):
        """
        Initialize a secular resonance.

        Parameters
        ----------
        resonance_type : str
            Type of secular resonance ('nu6', 'nu5', 'nu16', etc.)
        planet_name : str
            Name of the planet involved in the resonance
        """
        self.resonance_type = resonance_type
        self.planet_name = planet_name
        self.planets_names = [planet_name]
        self.index_of_planets = None

    def calc_angle(self, body, planets):
        """
        Calculate the secular resonance angle.

        This method should be overridden by specific resonance classes.
        """
        raise NotImplementedError("Subclasses must implement calc_angle")

    def to_s(self):
        """String representation of the secular resonance."""
        return f"{self.resonance_type}_{self.planet_name}"

    def to_short(self):
        """Short string representation of the secular resonance."""
        return self.resonance_type

    def order(self):
        """
        Order of the resonance (for secular resonances, typically 1).
        """
        return 1

    def get_planet_name_from_letter(self, letter):
        if letter == 'R':
            return 'Mercury'
        elif letter == 'V':
            return 'Venus'
        elif letter == 'E':
            return 'Earth'
        elif letter == 'M':
            return 'Mars'
        elif letter == 'J':
            return 'Jupiter'
        elif letter == 'S':
            return 'Saturn'
        elif letter == 'U':
            return 'Uranus'
        elif letter == 'N':
            return 'Neptune'
        raise Exception('Bad notation used. Only the following letter are available: R (for Mercury), V, E, M, J, S, U, N ')

    def get_letter_from_planet_name(self, planet_name: str) -> str:
        if 'Mercury' == planet_name:
            return 'R'
        return planet_name[0]


class Nu6Resonance(SecularResonance):
    """
    ν₆ secular resonance - perihelion precession with Saturn.

    The ν₆ resonance occurs when the precession frequency of an asteroid's
    perihelion (ϖ̇) equals the precession frequency of Saturn's perihelion.
    This resonance is particularly important for asteroids around 2 AU.
    """

    def __init__(self):
        super().__init__('nu6', 'Saturn')

    def calc_angle(self, body, planets):
        """
        Calculate the ν₆ resonant angle.

        For ν₆ resonance, the critical argument is the difference between
        the asteroid's longitude of perihelion and Saturn's longitude of perihelion.

        Parameters
        ----------
        body : object
            The asteroid/body with orbital elements
        planets : list
            List of planet objects (Saturn should be first for this resonance)

        Returns
        -------
        float
            The resonant angle in radians
        """
        saturn = planets[0]  # Saturn should be the first planet

        # Calculate longitude of perihelion (ϖ = Ω + ω) for both bodies
        body_varpi = body.Omega + body.omega
        saturn_varpi = saturn.Omega + saturn.omega

        # The secular resonance angle is the difference in longitudes of perihelion
        angle = rebound.mod2pi(body_varpi - saturn_varpi)

        return angle


class Nu5Resonance(SecularResonance):
    """
    ν₅ secular resonance - perihelion precession with Jupiter.

    The ν₅ resonance occurs when the precession frequency of an asteroid's
    perihelion (ϖ̇) equals the precession frequency of Jupiter's perihelion.
    """

    def __init__(self):
        super().__init__('nu5', 'Jupiter')

    def calc_angle(self, body, planets):
        """
        Calculate the ν₅ resonant angle.

        Parameters
        ----------
        body : object
            The asteroid/body with orbital elements
        planets : list
            List of planet objects (Jupiter should be first for this resonance)

        Returns
        -------
        float
            The resonant angle in radians
        """
        jupiter = planets[0]  # Jupiter should be the first planet

        # Calculate longitude of perihelion (ϖ = Ω + ω) for both bodies
        body_varpi = body.Omega + body.omega
        jupiter_varpi = jupiter.Omega + jupiter.omega

        # The secular resonance angle is the difference in longitudes of perihelion
        angle = rebound.mod2pi(body_varpi - jupiter_varpi)

        return angle


class Nu16Resonance(SecularResonance):
    """
    ν₁₆ secular resonance - node precession with Saturn.

    The ν₁₆ resonance occurs when the precession frequency of an asteroid's
    ascending node (Ω̇) equals the precession frequency of Saturn's ascending node.
    This resonance affects orbital inclinations.
    """

    def __init__(self):
        super().__init__('nu16', 'Saturn')

    def calc_angle(self, body, planets):
        """
        Calculate the ν₁₆ resonant angle.

        Parameters
        ----------
        body : object
            The asteroid/body with orbital elements
        planets : list
            List of planet objects (Saturn should be first for this resonance)

        Returns
        -------
        float
            The resonant angle in radians
        """
        saturn = planets[0]  # Saturn should be the first planet

        # For node resonances, we use the longitude of ascending node (Ω)
        angle = rebound.mod2pi(body.Omega - saturn.Omega)

        return angle


class GeneralSecularResonance(SecularResonance):
    """
    General secular resonance for custom combinations.

    This class allows for creating custom secular resonances by specifying
    coefficients for different orbital elements.
    """

    def __init__(self, coeffs, planet_names, resonance_name=None):
        """
        Initialize a general secular resonance.

        Parameters
        ----------
        coeffs : dict
            Dictionary with coefficients for different elements:
            {'varpi': [c1, c2], 'Omega': [c3, c4]}
            where c1, c2 are coefficients for body and planet longitude of perihelion
            and c3, c4 are coefficients for body and planet longitude of node
        planet_names : list
            List of planet names involved
        resonance_name : str, optional
            Custom name for the resonance
        """
        self.coeffs = coeffs
        self.planet_names = planet_names

        if resonance_name is None:
            resonance_name = "custom_secular"

        # Use the first planet as the primary planet
        super().__init__(resonance_name, planet_names[0])
        self.planets_names = planet_names

    def calc_angle(self, body, planets):
        """
        Calculate the general secular resonance angle.

        Parameters
        ----------
        body : object
            The asteroid/body with orbital elements
        planets : list
            List of planet objects

        Returns
        -------
        float
            The resonant angle in radians
        """
        angle = 0.0

        # Process longitude of perihelion coefficients
        if 'varpi' in self.coeffs:
            body_varpi = body.Omega + body.omega
            angle += self.coeffs['varpi'][0] * body_varpi

            for i, coeff in enumerate(self.coeffs['varpi'][1:]):
                planet_varpi = planets[i].Omega + planets[i].omega
                angle += coeff * planet_varpi

        # Process longitude of ascending node coefficients
        if 'Omega' in self.coeffs:
            angle += self.coeffs['Omega'][0] * body.Omega

            for i, coeff in enumerate(self.coeffs['Omega'][1:]):
                angle += coeff * planets[i].Omega

        return rebound.mod2pi(angle)


def create_secular_resonance(resonance_str):
    """
    Factory function to create secular resonance objects.

    Parameters
    ----------
    resonance_str : str
        String representation of the secular resonance

    Returns
    -------
    SecularResonance
        The appropriate secular resonance object

    Examples
    --------
    >>> create_secular_resonance('nu6')
    Nu6Resonance()
    >>> create_secular_resonance('nu5')
    Nu5Resonance()
    >>> create_secular_resonance('nu16')
    Nu16Resonance()
    """
    resonance_str = resonance_str.lower().strip()

    if resonance_str == 'nu6':
        return Nu6Resonance()
    elif resonance_str == 'nu5':
        return Nu5Resonance()
    elif resonance_str == 'nu16':
        return Nu16Resonance()
    else:
        raise ValueError(f"Unknown secular resonance: {resonance_str}")
