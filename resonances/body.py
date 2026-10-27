import numpy as np

from resonances.resonance.analysis.verdict import Status, Verdict
from resonances.resonance import cross_spectrum
from resonances.resonance.resonance import Resonance
from resonances.mmr.mmr import MMR
from resonances.secular.free_elements import DEFAULT_MASK_QUANTILE, free_elements
from resonances.data.const import INVARIABLE_PLANE, PLANETARY_PHASES, SOLAR_SYSTEM_WITH_SUN
from resonances.secular.proper_angle import build_proper_angle_series, free_divisor, free_rates
from resonances.secular.secular_resonance import SecularResonance
from resonances.lidov_kozai.lidov_kozai_resonance import LidovKozaiResonance
from resonances.resonance.factory import detect_resonance_type
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
        # Years as read from a saved data file (restore): the analysis reads them rather than times / 2 pi, so a
        # restored record is bit for bit the one the run analysed (x / 2pi * 2pi / 2pi can differ in the last bit).
        # Set only by SimulationSerializer.restore; None in a live run, where the years are times / (2 pi)
        self.times_years = None
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
        # The verdict's status string of every resonance (`Status` value), written with `librations` by
        # `Simulation._finalise_verdict`; the save/plot filters read it.
        self.statuses: dict[str, str] = {}
        # MMR: the full analysis of every analysed angle (catalogue and derived rows): CSV rows, figure data
        self.mmr_analyses = {}
        # Secular and Lidov-Kozai: the `LongPeriodAnalysis` of every analysed angle (CSV rows, figure data)
        self.secular_analyses = {}
        # Derived (subresonance) rows added by the analysis: key -> parent key (depth 1, see add_derived_resonance)
        self.derived_resonances: dict[str, str] = {}

        # Secular resonances data
        self.secular_resonances: List[SecularResonance] = []
        self.secular_angles_osculating = {}
        self.secular_angles_proper = {}

        # Lidov-Kozai resonance data
        self.lidov_kozai_resonances: List[LidovKozaiResonance] = []

        # The verdict of every classified resonance (all types: `analysis.verdict.Verdict`)
        self.librations: dict[str, Verdict] = {}
        # Secular/ZLK only: every number their analysis measured (`LongPeriodAnalysis.numbers_row()`); no status here
        self.classifier_numbers: dict[str, dict] = {}

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
        self._free_rates = None  # (g, s) of the free vectors, arcsec/yr; see secular_free_divisor

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
        """The data-file columns of one resonance. An MMR writes only its raw wrapped angle and the analysis' unwrap:
        the filtered angle is reproduced from them with the cutoff of summary.csv (`flt_cutoff_yr`)."""
        try:
            df_data = {
                resonance.to_s() + '_angle_unwrapped': self.angles_unwrapped[resonance.to_s()],
                resonance.to_s() + '_angle': self.angles[resonance.to_s()],
            }

            if resonance.type != 'mmr' and self.angles_filtered.get(resonance.to_s()) is not None:
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
        # The integration fills the raw angles; `Simulation.prepare_angles` makes the wrapped ones from them
        for resonance in self.resonances():
            self.angles_unwrapped[resonance.to_s()] = np.zeros(num)

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

    def build_proper_angle(self, resonance: Resonance, epoch_years: float = 0.0, remove_forced: bool = True, planetary_longitudes=None):
        """The secular angle against the planets' linear mode longitudes g_j*t + phase_j, twice:

        - `secular_angles_osculating`: from the body's osculating varpi and Omega (Omega measured from
          the invariable plane) — what the angle looks like before anything is removed;
        - `secular_angles_proper`: the same, reduced when `remove_forced` (config `secular_remove_forced`):
          the body's vectors lose their forced terms at the fundamentals that are not in the angle
          (`proper_angle.removed_modes`). Without it the two are equal.

        `epoch_years` is J2000 -> t = 0. `planetary_longitudes` ({'g4': array, ...}, config
        `secular_planet_phases='measured'`) replaces the linear law for the modes it has
        (`planet_modes.measured_longitudes`). The angle of the integration itself, built from the
        planets' osculating longitudes, is not kept: those turn at their own mode only for Mercury's,
        Jupiter's and Saturn's varpi (docs/secular.md).
        """
        if not isinstance(resonance, SecularResonance):
            raise Exception(
                f"You can build proper angle only for a secular resonance. Here resonance is {resonance.to_s()} for the body {self.name}"
            )

        common = dict(
            times=self.times,
            body=self,
            resonance=resonance,
            planetary_phases=PLANETARY_PHASES,
            epoch_years=epoch_years,
            invariable_plane=INVARIABLE_PLANE,
            planetary_longitudes=planetary_longitudes,
        )
        try:
            osculating = build_proper_angle_series(**common)
            self.secular_angles_osculating[resonance.to_s()] = osculating
            self.secular_angles_proper[resonance.to_s()] = (
                build_proper_angle_series(**common, remove_forced=True) if remove_forced else osculating
            )
        except Exception as exc:
            logger.warning(f"Failed to build proper secular angle for the body {self.name} in the resonance {resonance.to_s()}: {exc}")

    def secular_free_divisor(self, resonance: SecularResonance):
        """The angle's divisor with the body's own free g and s (`proper_angle.free_divisor`), arcsec/yr.

        The free rates are measured once per body, from the whole series; None when they cannot be."""
        if self._free_rates is None:
            try:
                self._free_rates = free_rates(np.asarray(self.times) / (2 * np.pi), self.ecc, self.varpi, self.inc, self.Omega)
            except Exception as exc:
                logger.warning(f"Failed to measure the free rates of the body {self.name}: {exc}")
                self._free_rates = (float('nan'), float('nan'))
        value = free_divisor(resonance.formula.terms, *self._free_rates)
        return None if not np.isfinite(value) else value

    def in_resonance(self, resonance: Union[MMR, SecularResonance, LidovKozaiResonance]) -> bool:
        """Whether the verdict is resonant or transient (any resonance type)."""
        return self.status(resonance) in (Status.RESONANT, Status.TRANSIENT)

    def status(self, resonance: Union[MMR, SecularResonance, LidovKozaiResonance]) -> str:
        """The status string of the verdict (`Status` value: resonant, transient, near_separatrix, ...)."""
        if resonance.to_s() in self.statuses:
            return self.statuses[resonance.to_s()]

        raise Exception(
            f"The status for the resonance {resonance.to_s()} does not exist "
            f"in the body {self.name}. Available resonance statuses: {self.statuses.keys()}"
        )

    def verdict(self, resonance: Union[MMR, SecularResonance, LidovKozaiResonance, str]) -> Verdict | None:
        """The full verdict of a resonance (or its key), None when it was not classified in memory.

        A short MMR key without the d'Alembert tail ('4J-2S-1' for the stored '4J-2S-1+0+0-1') is accepted when it
        heads exactly one stored key; when it heads several, ValueError lists them."""
        key = resonance if isinstance(resonance, str) else resonance.to_s()
        if key in self.librations or not isinstance(resonance, str):
            return self.librations.get(key)
        matches = self._keys_headed_by(key)
        if len(matches) > 1:
            raise ValueError(f"'{key}' matches several resonances of {self.name}: {', '.join(matches)}; give the full key.")
        return self.librations[matches[0]] if matches else None

    def _keys_headed_by(self, short: str) -> list[str]:
        """Stored MMR keys that are `short` followed by their d'Alembert coefficients (the next character is a sign)."""
        if not short or not short[-1].isdigit() or detect_resonance_type(short) != 'mmr':
            return []
        return [k for k in self.librations if k.startswith(short) and len(k) > len(short) and k[len(short)] in '+-']

    def in_pure_resonance(self, resonance: Union[MMR, SecularResonance, LidovKozaiResonance]) -> bool:
        """Whether the verdict is resonant (a libration over the record, any resonance type)."""
        return self.status(resonance) == Status.RESONANT

    def add_derived_resonance(self, key: str, parent: str, angle_wrapped=None, angle_unwrapped=None) -> Resonance | None:
        """Add a row found by the analysis of `parent` (an accepted planar subresonance, depth 1).

        The resonance joins `mmrs` (after the catalogue ones), so it is saved, restored and listed like them;
        `derived_resonances` remembers its parent. The angles, when given (body time order), go to `angles`
        and `angles_unwrapped` so that `save_body` writes them. Adding an existing derived key only updates the
        angles. None when the key is one of the body's own (catalogue) resonances: that row has its own analysis.
        """
        from resonances.resonance.factory import create_mmr

        resonance = create_mmr(key)
        existing = next((r for r in self.mmrs if r.to_s() == resonance.to_s()), None)
        if existing is not None and existing.to_s() not in self.derived_resonances:
            return None
        resonance = existing if existing is not None else resonance
        if existing is None:
            resonance.index_of_planets = [SOLAR_SYSTEM_WITH_SUN.index(planet) for planet in resonance.planets_names]
            self.mmrs.append(resonance)
        self.derived_resonances[resonance.to_s()] = parent
        if angle_wrapped is not None:
            self.angles[resonance.to_s()] = np.asarray(angle_wrapped, dtype=float)
        if angle_unwrapped is not None:
            self.angles_unwrapped[resonance.to_s()] = np.asarray(angle_unwrapped, dtype=float)
        return resonance

    def remove_derived_resonance(self, key: str) -> None:
        """Drop a derived row and everything stored under its key (before its parent is analysed again)."""
        self.derived_resonances.pop(key, None)
        self.mmrs = [r for r in self.mmrs if r.to_s() != key]
        for store in (
            self.angles,
            self.angles_unwrapped,
            self.angles_filtered,
            self.angles_filtered_unwrapped,
            self.statuses,
            self.librations,
            self.mmr_analyses,
            self.periodogram_frequency,
            self.periodogram_power,
            self.periodogram_peaks,
            self.coherence,
        ):
            store.pop(key, None)

    def resonance_origin(self, key: str) -> tuple[str, str]:
        """(added_by, parent) of a row: ('subresonance_check', parent) for a derived row, ('', '') otherwise."""
        from resonances.resonance.analysis.mmr.models import ADDED_BY_SUBRESONANCE

        parent = self.derived_resonances.get(key, '')
        return (ADDED_BY_SUBRESONANCE, parent) if parent else ('', '')

    def is_particle(self):
        if 'particle' == self.type:
            return True
        return False
