import numpy as np

from resonances.resonance.classify import ResonanceClassifyResult
from resonances.resonance.classify.models import ResonanceStatus
from resonances.resonance import cross_spectrum
from resonances.resonance.resonance import Resonance
from resonances.mmr.mmr import MMR
from resonances.secular.free_elements import DEFAULT_MASK_QUANTILE, free_elements
from resonances.secular.proper_angle import build_proper_angle_series
from resonances.secular.secular_resonance import SecularResonance
from resonances.lidov_kozai.lidov_kozai_resonance import LidovKozaiResonance
from .logger import logger
from typing import List, Union


class Body:
    def __init__(self, type='particle'):
        self.type = type

        self.initial_data = None
        self.name = ''
        self.mass = 0.0

        # Integration data
        self.times = None
        self.axis = None
        self.ecc = None
        self.inc = None
        self.Omega = None
        self.omega = None
        self.M = None
        self.longitude = None
        self.varpi = None

        # Mean-motion resonances data
        # keys are resonances (4J-2S-1-0-0-1), values are values

        self.mmrs: List[MMR] = []
        self.angles = {}  # For MMR angles
        self.angles_unwrapped = {}  # Wrapped by mod2pi angles
        self.angles_filtered = {}
        self.angles_filtered_unwrapped = {}
        self.statuses = {}

        # Secular resonances data
        self.secular_resonances: List[SecularResonance] = []
        self.secular_angles_osculating = {}
        self.secular_angles_proper = {}

        # Lidov-Kozai resonance data
        self.lidov_kozai_resonances: List[LidovKozaiResonance] = []

        # Libration and filtering data (shared between MMR and secular)
        self.librations: dict[str, List[ResonanceClassifyResult]] = {}
        self.libration_segments = {}

        self.periodogram_frequency = {}
        self.periodogram_power = {}
        self.periodogram_peaks = {}

        self.axis_filtered = None
        self.axis_periodogram_frequency = None
        self.axis_periodogram_power = None
        self.axis_periodogram_peaks = None

        self.eccentricity_periodogram_frequency = None
        self.eccentricity_periodogram_power = None
        self.eccentricity_periodogram_peaks = None

        self.periodogram_peaks_overlapping = {}

        # Cross-spectral coherence data
        # keys are resonance keys, values are {pair: PairAnalysis} / ZLKGate
        self.coherence = {}
        self.zlk_gates = {}

        # Free (proper) element decomposition — one per body, since it depends on the
        # orbit alone; the gate built on it is per resonance key.
        self.free_elements = None
        self.free_gates = {}

        # Simulation data
        self.index_in_simulation = None

    def __str__(self):
        s = f'Body(type={self.type}, name={self.name}, mass={self.mass})\n'
        if self.mmrs:
            s += 'MMR Resonances: '
            for mmr in self.mmrs:
                s += mmr.to_s() + ', '
            s += '\n'
        if self.secular_resonances:
            s += 'Secular Resonances: '
            for sec in self.secular_resonances:
                s += sec.to_s() + ', '
        return s

    def resonances(self) -> List[Resonance]:
        return self.mmrs + self.secular_resonances + self.lidov_kozai_resonances

    def keplerian_elements_to_dict(self) -> dict:
        df_data = {
            'a': self.axis,
            'e': self.ecc,
            'inc': self.inc,
            'Omega': self.Omega,
            'omega': self.omega,
            'M': self.M,
            'longitude': self.longitude,
            'varpi': self.varpi,
        }
        if self.axis_filtered is not None:
            df_data['a_filtered'] = self.axis_filtered
        return df_data

    def resonance_to_dict(self, resonance: Resonance):
        try:
            df_data = {
                resonance.to_s() + '_angle_unwrapped': self.angles_unwrapped[resonance.to_s()],
                resonance.to_s() + '_angle': self.angles[resonance.to_s()],
            }

            if self.angles_filtered.get(resonance.to_s()) is not None:
                df_data[resonance.to_s() + '_angle_filtered_unwrapped'] = self.angles_filtered_unwrapped[resonance.to_s()]
                df_data[resonance.to_s() + '_angle_filtered'] = self.angles_filtered[resonance.to_s()]

            if isinstance(resonance, SecularResonance):
                df_data[resonance.to_s() + "_angle_osculating"] = self.secular_angles_osculating[resonance.to_s()]
                df_data[resonance.to_s() + "_angle_proper"] = self.secular_angles_proper[resonance.to_s()]

        except Exception as e:
            error_text = f'Error in resonance_to_dict function for body={self.name} and resonance={resonance.to_s()}: {e}'
            logger.error(error_text)
            raise Exception(error_text)
        return df_data

    def setup_vars_for_simulation(self, times):
        self.times = times
        num = len(times)
        self.axis, self.ecc, self.inc, self.Omega, self.omega, self.M, self.longitude, self.varpi = (
            np.zeros(num),
            np.zeros(num),
            np.zeros(num),
            np.zeros(num),
            np.zeros(num),
            np.zeros(num),
            np.zeros(num),
            np.zeros(num),
        )
        # Setup MMR angles
        for resonance in self.resonances():
            self.angles_unwrapped[resonance.to_s()] = np.zeros(num)
            self.angles[resonance.to_s()] = np.zeros(num)

    def has_eccentricity_vector(self) -> bool:
        return self.ecc is not None and self.omega is not None and len(self.ecc) == len(self.omega) and len(self.ecc) > 0

    def eccentricity_vector(self):
        """Non-singular eccentricity components (k, h) = (e cos omega, e sin omega)."""
        if not self.has_eccentricity_vector():
            return None, None
        return cross_spectrum.eccentricity_vector(self.ecc, self.omega)

    def forced_free_eccentricity(self):
        """Forced and free eccentricity from the (k, h) scatter. See cross_spectrum module.

        Returns (None, None) when the body carries no integration data yet.
        """
        if not self.has_eccentricity_vector():
            return None, None
        return cross_spectrum.forced_free_eccentricity(self.ecc, self.omega)

    def build_free_elements(self, sampling_years=500.0, mask_quantile=DEFAULT_MASK_QUANTILE):
        """Split the eccentricity and inclination vectors into forced and free parts.

        Stored on the body rather than per resonance: the decomposition depends only on
        the orbit. See `secular.free_elements` for what it is and where it stops working.
        """
        if self.times is None or not self.has_eccentricity_vector() or self.Omega is None:
            return None
        try:
            self.free_elements = free_elements(
                times=np.asarray(self.times) / (2 * np.pi),
                ecc=self.ecc,
                inc=self.inc,
                Omega=self.Omega,
                omega=self.omega,
                sampling_years=sampling_years,
                mask_quantile=mask_quantile,
            )
        except Exception as exc:
            logger.warning(f"Failed to build free elements for the body {self.name}: {exc}")
            self.free_elements = None
        return self.free_elements

    def angle(self, resonance: Resonance) -> np.ndarray:
        """
        Get angle array for any supported resonance.
        """
        try:
            return self.angles[resonance.to_s()]
        except Exception:
            raise Exception(f"The angle for the resonance {resonance.to_s()} does not exist in the body {self.name}.")

    def angle_unwrapped(self, resonance: Resonance) -> np.ndarray:
        try:
            return self.angles_unwrapped[resonance.to_s()]
        except Exception:
            raise Exception(f"The angle for the resonance {resonance.to_s()} does not exist in the body {self.name}.")

    def angle_filtered(self, resonance: Resonance) -> np.ndarray:
        try:
            return self.angles_filtered[resonance.to_s()]
        except Exception:
            raise Exception(f"The filtered angle for the resonance {resonance.to_s()} does not exist in the body {self.name}.")

    def build_proper_angle(self, resonance: Resonance):
        if not isinstance(resonance, SecularResonance):
            raise Exception(
                f"You can build proper angle only for a secular resonance. Here resonance is {resonance.to_s()} for the body {self.name}"
            )

        try:
            existing = self.angles.get(resonance.to_s())
            if existing is not None:
                self.secular_angles_osculating[resonance.to_s()] = existing.copy()
            proper_angle = build_proper_angle_series(
                times=self.times,
                body=self,
                resonance=resonance,
                existing_angle=existing,
                cutoff_period_years=700_000.0,
            )
            self.secular_angles_proper[resonance.to_s()] = proper_angle
        except Exception as exc:
            logger.warning(f"Failed to build proper secular angle for the body {self.name} in the resonance {resonance.to_s()}: {exc}")

    def in_resonance(self, resonance: Union[MMR, SecularResonance, LidovKozaiResonance]):
        """
        Check if body is in resonance (works for both MMR and secular).
        """
        status = self.status(resonance)
        return status is not None and status > ResonanceStatus.NON_RESONANT

    def status(self, resonance: Union[MMR, SecularResonance, LidovKozaiResonance]):
        """
        Get resonance status
        """
        if resonance.to_s() in self.statuses:
            return self.statuses[resonance.to_s()]

        raise Exception(
            f"The status for the resonance {resonance.to_s()} does not exist "
            f"in the body {self.name}. Available resonance statuses: {self.statuses.keys()}"
        )

    def in_pure_resonance(self, resonance: Union[MMR, SecularResonance, LidovKozaiResonance]):
        """
        Check if body is in pure resonance (works for both MMR and secular).
        """
        return self.status(resonance) == ResonanceStatus.LIBRATION

    def is_particle(self):
        if 'particle' == self.type:
            return True
        return False
