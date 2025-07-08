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

    @property
    def type(self) -> str:
        return 'secular'

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
        saturn = planets[0]
        body_varpi = body.Omega + body.omega
        saturn_varpi = saturn.Omega + saturn.omega
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
        jupiter = planets[0]
        body_varpi = body.Omega + body.omega
        jupiter_varpi = jupiter.Omega + jupiter.omega
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
        saturn = planets[0]
        angle = rebound.mod2pi(body.Omega - saturn.Omega)

        return angle


class GeneralSecularResonance(SecularResonance):
    """
    General secular resonance for custom combinations.

    This class allows for creating custom secular resonances by specifying
    coefficients for different orbital elements, or by providing a mathematical formula.
    """

    def __init__(self, coeffs=None, planet_names=None, resonance_name=None, formula=None):
        """
        Initialize a general secular resonance.

        Parameters
        ----------
        coeffs : dict, optional
            Dictionary with coefficients for different elements:
            {'varpi': [c1, c2], 'Omega': [c3, c4]}
            where c1, c2 are coefficients for body and planet longitude of perihelion
            and c3, c4 are coefficients for body and planet longitude of node
        planet_names : list, optional
            List of planet names involved
        resonance_name : str, optional
            Custom name for the resonance
        formula : str, optional
            Mathematical formula like 'g-g5', '2g-g5-g6', etc.
            If provided, coeffs and planet_names will be calculated automatically
        """
        if formula is not None:
            parsed_coeffs = self._parse_formula(formula)
            resonance_params = self._coeffs_to_resonance_params(parsed_coeffs, formula)

            self.coeffs = resonance_params['coeffs']
            self.planet_names = resonance_params['planet_names']
            resonance_name = formula
        else:
            if coeffs is None or planet_names is None:
                raise ValueError("Either 'formula' or both 'coeffs' and 'planet_names' must be provided")

            self.coeffs = coeffs
            self.planet_names = planet_names

            if resonance_name is None:
                resonance_name = "custom_secular"

        super().__init__(resonance_name, ','.join([p[0] for p in planet_names]) if planet_names else 'Unknown')
        self.planets_names = self.planet_names

    @staticmethod
    def _parse_formula(formula: str) -> dict:
        """
        Parse a secular resonance formula into coefficients.

        Parameters
        ----------
        formula : str
            Mathematical formula like 'g-g5', '2g-g5-g6', etc.

        Returns
        -------
        dict
            Dictionary with coefficients for different terms
        """
        import re

        # Initialize coefficients
        coeffs = {
            'g': 0.0,
            's': 0.0,
            'g4': 0.0,
            'g5': 0.0,
            'g6': 0.0,
            'g7': 0.0,
            'g8': 0.0,
            's4': 0.0,
            's5': 0.0,
            's6': 0.0,
            's7': 0.0,
            's8': 0.0,
        }

        # Handle special cases first
        if '2(g-g6)+(s-s6)' in formula:
            coeffs['g'] = 2.0
            coeffs['g6'] = -2.0
            coeffs['s'] = 1.0
            coeffs['s6'] = -1.0
            return coeffs

            # First normalize old format with * to new format
        formula = re.sub(r'(\d+)\*([a-zA-Z]\d*)', r'\1\2', formula)  # 2*g -> 2g

        # Normalize the formula by adding spaces around operators
        normalized = re.sub(r'([+-])', r' \1 ', formula)
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # Split into terms
        if not normalized.startswith('-') and not normalized.startswith('+'):
            normalized = '+' + normalized

        # Find all terms using regex
        terms = re.findall(r'[+-][^+-]+', normalized)

        for term in terms:
            term = term.strip()
            sign = 1 if term.startswith('+') else -1
            term = term[1:].strip()  # Remove the sign

            # Handle coefficients like "2g" or just "g"
            # First try to match coefficient pattern like "2g5", "3g6"
            coeff_match = re.match(r'^(\d+)([a-zA-Z]\d*)$', term)
            if coeff_match:
                coeff = float(coeff_match.group(1)) * sign
                var = coeff_match.group(2)
            else:
                coeff = sign
                var = term

            # Apply coefficient to the appropriate variable
            if var in coeffs:
                coeffs[var] += coeff
            else:
                # Handle special cases or unknown variables
                print(f"Warning: Unknown variable '{var}' in formula '{formula}'")
                raise ValueError(f"Unknown variable '{var}' in formula '{formula}'")

        return coeffs

    @staticmethod
    def _determine_planet_names(coeffs: dict) -> list:
        """Determine which planets are involved based on coefficients."""
        planet_names = []
        planet_mapping = {
            'g4': 'Mars',
            's4': 'Mars',
            'g5': 'Jupiter',
            's5': 'Jupiter',
            'g6': 'Saturn',
            's6': 'Saturn',
            'g7': 'Uranus',
            's7': 'Uranus',
            'g8': 'Neptune',
            's8': 'Neptune',
        }

        for freq in ['g4', 'g5', 'g6', 'g7', 'g8', 's4', 's5', 's6', 's7', 's8']:
            if coeffs[freq] != 0:
                planet = planet_mapping[freq]
                if planet not in planet_names:
                    planet_names.append(planet)

        # If no planets are involved, assume Jupiter and Saturn (most common)
        if not planet_names:
            planet_names = ['Jupiter', 'Saturn']

        return planet_names

    @staticmethod
    def _build_varpi_coeffs(coeffs: dict, planet_names: list) -> list:
        """Build longitude of perihelion coefficients."""
        varpi_coeffs = [coeffs['g']]  # Body coefficient
        for planet in planet_names:
            if planet == 'Mars':
                varpi_coeffs.append(coeffs['g4'])
            elif planet == 'Jupiter':
                varpi_coeffs.append(coeffs['g5'])
            elif planet == 'Saturn':
                varpi_coeffs.append(coeffs['g6'])
            elif planet == 'Uranus':
                varpi_coeffs.append(coeffs['g7'])
            elif planet == 'Neptune':
                varpi_coeffs.append(coeffs['g8'])
            else:
                varpi_coeffs.append(0.0)
        return varpi_coeffs

    @staticmethod
    def _build_omega_coeffs(coeffs: dict, planet_names: list) -> list:
        """Build longitude of ascending node coefficients."""
        omega_coeffs = [coeffs['s']]  # Body coefficient
        for planet in planet_names:
            if planet == 'Mars':
                omega_coeffs.append(coeffs['s4'])
            if planet == 'Jupiter':
                omega_coeffs.append(coeffs['s5'])
            elif planet == 'Saturn':
                omega_coeffs.append(coeffs['s6'])
            elif planet == 'Uranus':
                omega_coeffs.append(coeffs['s7'])
            elif planet == 'Neptune':
                omega_coeffs.append(coeffs['s8'])
            else:
                omega_coeffs.append(0.0)
        return omega_coeffs

    @staticmethod
    def _coeffs_to_resonance_params(coeffs: dict, formula: str) -> dict:
        """
        Convert parsed coefficients to GeneralSecularResonance parameters.

        Parameters
        ----------
        coeffs : dict
            Coefficients from _parse_formula
        formula : str
            Original formula string

        Returns
        -------
        dict
            Parameters for GeneralSecularResonance constructor
        """
        # Determine which planets are involved
        planet_names = GeneralSecularResonance._determine_planet_names(coeffs)

        # Build coefficients dictionary for GeneralSecularResonance
        resonance_coeffs = {}

        # Longitude of perihelion coefficients (varpi = Omega + omega)
        varpi_coeffs = GeneralSecularResonance._build_varpi_coeffs(coeffs, planet_names)
        if any(c != 0 for c in varpi_coeffs):
            resonance_coeffs['varpi'] = varpi_coeffs

        # Longitude of ascending node coefficients (Omega)
        omega_coeffs = GeneralSecularResonance._build_omega_coeffs(coeffs, planet_names)
        if any(c != 0 for c in omega_coeffs):
            resonance_coeffs['Omega'] = omega_coeffs

        return {'coeffs': resonance_coeffs, 'planet_names': planet_names, 'resonance_name': formula}

    def to_s(self):
        """String representation of the general secular resonance."""
        return self.resonance_type

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
