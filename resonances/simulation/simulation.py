import numpy as np
from typing import List, Union

from resonances.body import Body
from resonances.resonance import coherence_analysis, omega_free_gate
from resonances.resonance.analysis import confidence as confidence_module
from resonances.resonance.analysis.mmr import analyse_mmr
from resonances.resonance.analysis.secular import analyse_secular
from resonances.resonance.analysis.verdict import CheckValue, ResonanceType, Status, Verdict
from resonances.resonance.analysis.zlk import analyse_zlk
from resonances.resonance.periodogram import Periodogram
from resonances.lidov_kozai.lidov_kozai_resonance import LidovKozaiResonance
from resonances.data.const import SOLAR_SYSTEM_WITH_SUN

from . import mmr_adapter
from .dry_run import prepare_run
from .config import SavePlanets, SimulationConfig
from resonances import mpcorb
from .body_manager import BodyManager
from .integration import IntegrationEngine
from .data_manager import DataManager
from .batch_manager import BatchManager
from .serializer import SimulationSerializer

from resonances.secular import planet_modes
from resonances.secular.secular_resonance import SecularResonance
from resonances.resonance.resonance import Resonance
from resonances.logger import logger
from resonances.resonance.filtering import filter_angle, wrap
from resonances.data.const import GAUSSIAN_YEAR_DAYS, J2000


class Simulation:
    """
    Main Simulation class with component-based architecture.

    This class orchestrates the various components to run resonance simulations.
    """

    def __init__(self, **kwargs):
        """Initialize the simulation with component-based architecture."""
        self.config = SimulationConfig(**kwargs)
        self.body_manager = BodyManager(self.config)
        self.integration_engine = IntegrationEngine(self.config)
        self.data_manager = DataManager(self.config)
        self.batch_manager = BatchManager(self.config)

        self.running_time = {
            "start": None,
            "adding_bodies_started": None,
            "integration_started": None,
            "integration_finished": None,
            "librations_identified": None,
            "bodies_saved": None,
            "stop": None,
        }

        self.running_time["start"] = logger.get_current_time()

        self.times = []
        # Clones (virtual asteroids): one `ensemble_run.EnsembleRun` per parent, run after the ordinary bodies; their
        # clones.csv rows after the run (`clone_summary`, a DataFrame)
        self.ensembles = []
        self.clone_summary = None

    @property
    def bodies(self) -> List[Body]:
        return self.body_manager.bodies

    def create_solar_system(self, force=False):
        """Create or load the Solar System simulation."""
        self.integration_engine.create_solar_system(force)

    def add_body(
        self, elem_or_num, resonance: Union[Resonance, str, list[Resonance], list[str]], name='asteroid', clones: int | None = None
    ):
        """Add a celestial body with any resonances.

        `clones` (None = CLONES of the config, 0 = none): that many virtual asteroids of the body, drawn from the orbit
        solution of CLONES_SOURCE (docs/clones.md) and run after the ordinary bodies, as `add_ensemble` does; the body
        itself is added as without them. Only for a body given by its number or designation."""
        n = int(self.config.clones if clones is None else clones)
        if n > 0 and clones is not None and isinstance(elem_or_num, dict):
            raise ValueError(f'{name}: clones need the body by its number or designation, not its elements')
        if n > 0 and not isinstance(elem_or_num, dict) and any(run.parent == str(name) for run in self.ensembles):
            raise ValueError(f'{name} has an ensemble in this run already')  # refused before the body is added
        self.body_manager.add_body(elem_or_num, resonance, name)
        if n <= 0 or isinstance(elem_or_num, dict):
            return
        from . import ensemble_run

        ensemble_run.add_clones_of_body(self, elem_or_num, str(name), n, self.bodies[-1].resonances(), self.bodies[-1].initial_data)

    def add_ensemble(self, ensemble, resonances=None, name: str | None = None, nominal: bool = True, **find_kwargs):
        """Add an ensemble of clones (`resonances.clones.Ensemble`, e.g. `orbit_solution(463).sample(100)`).

        Its nominal (when it has one and `nominal`) becomes an ordinary body `name` (default: the ensemble's name); the
        clones `<name>~c001`, ... run after the ordinary bodies in batches of CLONES_BATCH test particles. Drawn at
        another epoch than the run's date, the nominal and every clone are integrated to it first (IAS15, the run's
        planets). `resonances=None` takes the MMR candidates (`find_kwargs`: planets, sigma2, sigma3, sigma_retro of
        `candidate_mmrs`): the nominal its own, as without clones, the clones those of the whole cloud. Returns the
        `EnsembleRun` (None when no candidate was found).
        """
        from . import ensemble_run

        return ensemble_run.add_ensemble(self, ensemble, resonances, name, nominal, **find_kwargs)

    def add_bodies(self, bodies: List[str], resonance, prefix: str = None):
        """Add multiple celestial bodies to the simulation."""
        if prefix is None:
            prefix = ""
        else:
            prefix = f"{prefix}_"

        for body in bodies:
            self.add_body(body, resonance, f"{prefix}{body}")

    def run(self, progress=False):
        """Run the complete simulation with optional batching.

        First the static dry-run (`dry_run.prepare_run`, docs/dry-run.md): its warnings go to the log and
        OUTPUT_STEP='auto' becomes the step and Nout; once per simulation, never in batch workers, never raises.
        The MPC table loaded to add the bodies (source 'mpc', ~0.5 GB for MPCORB) is released first."""
        mpcorb.clear_cache()
        prepare_run(self)
        self.times = np.linspace(0.0, self.config.tmax, self.config.Nout)

        if self.batch_manager.should_batch(len(self.bodies)):
            self._run_batched(progress)
        else:
            self._run_single(progress)
        if self.ensembles:  # the clones, after the ordinary bodies (simulation/ensemble_run.py)
            from . import ensemble_run

            ensemble_run.run_ensembles(self, progress)
        self.running_time["stop"] = logger.get_current_time()

    def _run_single(self, progress=False):
        """Run single-batch execution (original behavior)."""
        self.running_time["adding_bodies_started"] = logger.get_current_time()
        self.body_manager.add_bodies_to_simulation(self.integration_engine.sim)
        self.running_time["integration_started"] = logger.get_current_time()
        self.integration_engine.run_integration(self.bodies, self.times, progress, planets=self.planets_needed())
        self.running_time["integration_finished"] = logger.get_current_time()
        self.prepare_angles()
        # One body at a time from its verdicts to its figures (`save_data`): the arrays of a body's analyses are
        # released before the next body is analysed, so a batch never holds the analyses of all its bodies.
        self.data_manager.save_data(self.bodies, self.times, self, analyse=self._analyse_body)
        self.running_time["librations_identified"] = self.running_time["bodies_saved"]

    def _analyse_body(self, body: Body):
        """The verdicts, then the periodograms of one body (the MMR periodogram reads the analysis' sigma_f)."""
        self.identify_librations(bodies=[body])
        self.build_periodograms(bodies=[body])

    def _run_batched(self, progress=False):
        """Run batched execution with multi-core support."""
        self.running_time["adding_bodies_started"] = logger.get_current_time()
        self.running_time["integration_started"] = logger.get_current_time()
        self.batch_manager.execute_batches(self, self.bodies, self.times, progress)
        self.running_time["integration_finished"] = logger.get_current_time()
        self.running_time["librations_identified"] = logger.get_current_time()
        self.running_time["bodies_saved"] = logger.get_current_time()
        self.running_time["stop"] = logger.get_current_time()
        # The workers' results live in their files: bring the derived rows and the statuses back to these
        # bodies (body.librations stays empty), then save simulation.json for all bodies.
        for name, rows in self.batch_manager.derived_resonances.items():
            body = next((b for b in self.bodies if str(b.name) == str(name)), None)
            for key, parent in rows if body is not None else ():
                body.add_derived_resonance(key, parent)
        if self.config.save_summary:
            SimulationSerializer.restore_statuses(self.bodies, f'{self.config.save_path}/summary.csv')
        else:
            logger.warning("Batched run without save_summary: the bodies in memory have no statuses")
        self.data_manager.save_configuration_details(self.bodies, self)

    def prepare_angles(self):
        """Unwrap and wrap every angle; build the proper angle of a secular resonance.

        No angle is filtered here: each analysis filters its own (an MMR at the record's cutoff, a secular or
        Lidov-Kozai angle at its type's cutoff, `analysis.secular.preprocess`) and fills `angles_filtered[key]`
        in `identify_librations`.

        A secular angle is the proper one by default (`secular_angle_mode='proper'`): the planets enter through
        their mode longitudes, not their osculating ones (see `Body.build_proper_angle`). The epoch of t = 0 is
        counted in the model's (Gaussian) years from J2000, the unit of the mode frequencies and of t.

        A series with NaN/inf (an orbit the integration lost) is kept unfiltered: the analysis gives such a body
        `invalid` (unphysical) or `uncertain` (unusable series).
        """
        date = self.config.date.replace(tzinfo=None) if self.config.date else J2000
        epoch_years = (date - J2000).total_seconds() / 86400.0 / GAUSSIAN_YEAR_DAYS  # the one conversion: model years
        longitudes = self.measured_mode_longitudes()
        for body in self.bodies:
            if not np.isfinite(body.axis).all():
                logger.warning(f"Non-finite orbit of {body.name}: its series are not filtered")
            body.axis_filtered = self._filter(body.axis)
            for resonance in body.resonances():
                res_key = resonance.to_s()
                # np.unwrap the raw integration angles and store back
                body.angles_unwrapped[res_key] = np.unwrap(body.angle_unwrapped(resonance))
                body.angles[res_key] = wrap(body.angle_unwrapped(resonance))
                if isinstance(resonance, SecularResonance):
                    body.build_proper_angle(
                        resonance, epoch_years=epoch_years, remove_forced=self.config.secular_remove_forced, planetary_longitudes=longitudes
                    )
                    if self.config.secular_angle_mode == "proper" and res_key in body.secular_angles_proper:
                        body.angles_unwrapped[res_key] = np.unwrap(body.secular_angles_proper[res_key])
                        body.angles[res_key] = body.secular_angles_proper[res_key]

    def _filter(self, series):
        """`filter_angle`, or a copy of the series when it has NaN/inf (filtfilt raises on them)."""
        if np.isfinite(series).all():
            return filter_angle(series, self.config)
        return np.array(series, dtype=float)

    def measured_mode_longitudes(self):
        """The planets' mode longitudes measured in this integration (`planet_modes`), for the modes the
        bodies' secular angles use, or None under the linear law (`secular_planet_phases='linear'`)."""
        if self.config.secular_planet_phases != 'measured':
            return None
        labels = {
            f'{t.mode}{t.index}' for body in self.bodies for r in body.secular_resonances for t in r.formula.terms if t.index is not None
        }
        if not labels:
            return None
        return planet_modes.measured_longitudes(
            self.times / (2 * np.pi), self.integration_engine.planets_data, labels, window=self.config.secular_phase_window
        )

    def build_periodograms(self, only: dict | None = None, bodies: List[Body] | None = None):
        """Periodograms of a, e and every filtered angle of `bodies` (default: all). `only` ({body name: resonance
        keys}, as in `identify_librations`) limits them to the angles of those bodies (a restore that recomputed them;
        the a and e spectra do not depend on the analysis and stay); a key whose new analysis has no filtered angle
        (stopped at G0) loses its old periodogram."""
        base_periodogram_config = {
            'integration_time_yrs': self.config.tmax_yrs,
            'Nout': self.config.Nout,
            'libration_period_min': self.config.libration_period_min,
            'minimum_frequency': self.config.periodogram_frequency_min,
            'maximum_frequency': self.config.periodogram_frequency_max,
            'threshold': self.config.periodogram_soft,
        }

        for body in self.bodies if bodies is None else bodies:
            keys = None if only is None else only.get(body.name, set())
            if keys is not None and not keys:
                continue
            if keys is None:
                (
                    body.axis_periodogram_frequency,
                    body.axis_periodogram_power,
                    body.axis_periodogram_peaks,
                ) = Periodogram.periodogram(
                    self.times,
                    body.axis_filtered,
                    label=f'{body.name}:a',
                    **base_periodogram_config,
                )
                (
                    body.eccentricity_periodogram_frequency,
                    body.eccentricity_periodogram_power,
                    body.eccentricity_periodogram_peaks,
                ) = Periodogram.periodogram(
                    self.times,
                    body.ecc,
                    label=f'{body.name}:e',
                    **base_periodogram_config,
                )

            for resonance in body.resonances():
                key = resonance.to_s()
                if keys is not None and key not in keys and body.derived_resonances.get(key) not in keys:
                    continue
                if body.angles_filtered.get(key) is None:  # an MMR whose analysis stopped at G0
                    for store in (body.periodogram_frequency, body.periodogram_power, body.periodogram_peaks):
                        store.pop(key, None)
                    continue
                (
                    body.periodogram_frequency[resonance.to_s()],
                    body.periodogram_power[resonance.to_s()],
                    body.periodogram_peaks[resonance.to_s()],
                ) = Periodogram.periodogram(
                    self.times,
                    body.angles_filtered[resonance.to_s()],  # if not filtered, easy to skip relevant frequencies
                    label=f'{body.name}:{resonance.to_s()}',
                    **base_periodogram_config,
                )

    def identify_librations(self, only: dict | None = None, bodies: List[Body] | None = None):
        """Give every resonance of every body (or of `bodies`) its verdict (`body.librations[key]`, `body.statuses[key]`).

        MMR: `analyse_mmr` on the body's series (`mmr_adapter`); the subresonance scan, the owner and the
        kinematics are parts of it, and an accepted planar subresonance comes back as a derived row of the body
        (`Body.add_derived_resonance`, depth 1). Secular and Lidov-Kozai: `analyse_secular` / `analyse_zlk` on the
        same shared geometry (turning points, episodes, the centre and its scale; the free-omega gate of omega is part
        of the ZLK analysis). The cross spectra and the e-i exchange are pure diagnostics (a check column).

        This is the one place every path goes through (single run, batch worker, and
        `SimulationSerializer.restore(recompute_librations=True)`, which re-analyses the saved series without
        replaying `prepare_angles`), so post-classification changes belong here, and `_finalise_verdict` is the
        only writer of `body.librations` and `body.statuses`.

        `only` ({body name: resonance keys}) limits the work to those pairs; restore passes the pairs whose series
        were saved. None classifies every resonance of every body. A derived row is produced again (or dropped) by
        its parent's analysis; only when its parent's series was not saved (SAVE=resonant keeps a resonant child of a
        non-resonant parent) is it analysed alone, at depth one, keeping its parent (`_analyse_derived_alone`).
        """
        settings = self.config.analysis_settings
        cells = confidence_module.cells_from_settings(settings)
        for body in self.bodies if bodies is None else bodies:
            keys = None if only is None else only.get(body.name, set())
            resonances = [r for r in body.resonances() if (keys is None or r.to_s() in keys) and r.to_s() not in body.derived_resonances]
            parents = body.derived_resonances
            orphans = [r for r in body.mmrs if keys is not None and r.to_s() in keys and parents.get(r.to_s(), r.to_s()) not in keys]
            for resonance in orphans:
                self._analyse_derived_alone(body, resonance, settings, cells)
            if not resonances:
                continue
            self._warn_inclination_crossing(body, resonances)
            # Welch spectra of a, e and i are shared by every resonance of a body.
            psd_cache = {} if self.config.coherence_enabled else None
            # The forced/free split depends on the orbit alone, so it is done once per
            # body, and only where it is read — the Lidov-Kozai angle omega = varpi - Omega.
            if self.config.free_elements_enabled and any(isinstance(r, LidovKozaiResonance) for r in resonances):
                body.build_free_elements(self.config.free_elements_sampling_years, self.config.free_gate_mask_quantile)
            for resonance in resonances:
                if resonance.type == 'mmr':
                    for row in self._analyse_mmr(body, resonance, settings, cells):
                        self._analyse_coherence(body, row, psd_cache)
                else:
                    self._analyse_long_period(body, resonance, settings, cells)
                    self._analyse_coherence(body, resonance, psd_cache)

    @staticmethod
    def _finalise_verdict(body: Body, res_key: str, verdict: Verdict) -> None:
        """The only writer of a verdict: `body.librations[key]` (summary.csv) and `body.statuses[key]` (the save/plot
        filters) are always set together, so the two can never disagree."""
        body.librations[res_key] = verdict
        body.statuses[res_key] = str(verdict.status)

    def _analyse_mmr(self, body: Body, resonance: Resonance, settings, cells) -> list:
        """Analyse one MMR of a body; returns the resonances it gave a verdict (the angle, then its derived rows).

        The derived rows of a previous analysis of this angle are dropped first: the new analysis adds back the ones
        it accepts. The analysis' series (predictive unwrap, sigma_f at the record's cutoff) replace the body's.
        """
        res_key = resonance.to_s()
        for derived_key in [key for key, parent in body.derived_resonances.items() if parent == res_key]:
            body.remove_derived_resonance(derived_key)
        order = mmr_adapter.time_order(body)
        analysis = analyse_mmr(mmr_adapter.series_from_body(body, resonance, self), settings, cells)
        mmr_adapter.store_analysis(body, res_key, analysis, order)
        body.mmr_analyses[res_key] = analysis
        self._finalise_verdict(body, res_key, analysis.verdict)
        rows = [resonance]
        for derived_key, derived in mmr_adapter.derived_rows(analysis):
            series = derived.derived_series
            derived_resonance = body.add_derived_resonance(derived_key, res_key)
            if derived_resonance is None:  # the body has this angle in its own list: its own analysis gives the row
                logger.info(f"{body.name}/{res_key}: subresonance {derived_key} is analysed as a resonance of the body")
                continue
            body_key = derived_resonance.to_s()
            if body_key != derived_key:
                logger.warning(f"{body.name}: derived row {derived_key} is {body_key} in the package notation")
            mmr_adapter.store_analysis(body, body_key, derived, order, angle_wrapped=None if series is None else series.angle_wrapped)
            body.mmr_analyses[body_key] = derived
            self._finalise_verdict(body, body_key, derived.verdict)
            self._release_unplotted(body, derived_resonance, derived)
            logger.info(f"{body.name}/{res_key}: subresonance {body_key} added as a row of its own")
            rows.append(derived_resonance)
        self._release_unplotted(body, resonance, analysis)  # after its derived rows: the release reaches them too
        return rows

    def _analyse_derived_alone(self, body: Body, resonance: Resonance, settings, cells) -> None:
        """A saved derived row whose parent was not saved (restore): its own angle analysed at depth one (no scans),
        as the figure re-runs a derived row, with its parent and the parent's multiplet status kept."""
        key = resonance.to_s()
        previous = body.librations.get(key)
        order = mmr_adapter.time_order(body)
        series = mmr_adapter.series_from_body(body, resonance, self)
        analysis = analyse_mmr(series, settings, cells, scan_subresonances=False, scan_owner=False, parent=body.derived_resonances[key])
        if previous is not None and previous.multiplet_status is not None:
            analysis.multiplet_status = previous.multiplet_status
        mmr_adapter.store_analysis(body, key, analysis, order)
        body.mmr_analyses[key] = analysis
        self._finalise_verdict(body, key, analysis.verdict)
        self._release_unplotted(body, resonance, analysis)

    def _release_unplotted(self, body: Body, resonance: Resonance, analysis) -> None:
        """Drop the arrays of an analysis whose row will not be plotted, right after its verdict: the body already holds
        the series the data files need (`store_analysis`) and the CSV rows are frozen by `drop_arrays`, so a batch
        keeps only the arrays of the rows it will draw (memory: many angles x 2e5 rows per body)."""
        if self.data_manager.should_plot_body(body, resonance):
            return
        if analysis.derived and any(self.data_manager.should_plot_body(body, row) for row in self._derived_of(body, resonance)):
            return  # a derived row to be drawn shares the release (`drop_arrays` reaches the derived analyses)
        analysis.drop_arrays()

    @staticmethod
    def _derived_of(body: Body, resonance: Resonance) -> list:
        key = resonance.to_s()
        return [r for r in body.resonances() if body.derived_resonances.get(r.to_s()) == key]

    def long_period_analysis(self, body: Body, resonance: Resonance):
        """The analysis of a secular/ZLK row with its arrays: the one in memory, or the same run again when its arrays
        were released (`drop_arrays`) or the row was restored without recomputation (figures)."""
        analysis = body.secular_analyses.get(resonance.to_s())
        if analysis is not None and analysis.angle is not None:
            return analysis
        settings = self.config.analysis_settings
        series = long_period_series(body, resonance)
        if isinstance(resonance, SecularResonance):
            return analyse_secular(series, settings)
        elements = body.free_elements if self.config.free_elements_enabled else None
        return analyse_zlk(series, settings, free_elements=elements, gate_config=_gate_config(self.config))

    def _analyse_long_period(self, body: Body, resonance: Resonance, settings, cells) -> None:
        """The verdict of a secular or Lidov-Kozai angle (`analysis.secular`, `analysis.zlk`).

        The analysis is kept in `body.secular_analyses[key]`, its numbers in `body.classifier_numbers[key]`, its
        filtered angle (at P_c) in `body.angles_filtered*[key]` (body time order) and, for ZLK, the free-omega gate in
        `body.free_gates[key]`.
        """
        res_key = resonance.to_s()
        series = long_period_series(body, resonance)
        if isinstance(resonance, SecularResonance):
            analysis = analyse_secular(series, settings, cells)
        else:
            elements = body.free_elements if self.config.free_elements_enabled else None
            analysis = analyse_zlk(series, settings, cells, free_elements=elements, gate_config=_gate_config(self.config))
            if analysis.gate is not None and self.config.free_elements_enabled:  # no free elements: the gate never ran
                gate = analysis.gate
                body.free_gates[res_key] = omega_free_gate.OmegaFreeGate(
                    omega_free_gate.GateOutcome(gate.outcome), list(gate.matches), gate.n_cycles
                )
        body.secular_analyses[res_key] = analysis
        body.classifier_numbers[res_key] = analysis.numbers_row()
        if analysis.angle is not None:
            fine = analysis.angle if analysis.fine_geometry is None else None
            filtered = fine.angle_f if fine is not None else _filtered_at(series, analysis.cutoff.cutoff_yr, settings)
            filtered = mmr_adapter.in_body_order(filtered, mmr_adapter.time_order(body), len(body.times))
            if filtered is not None:
                body.angles_filtered_unwrapped[res_key] = filtered
                body.angles_filtered[res_key] = wrap(filtered)
        self._finalise_verdict(body, res_key, analysis.verdict)
        if not self.data_manager.should_plot_body(body, resonance):
            analysis.drop_arrays()  # the CSV rows are frozen; a figure would re-run the analysis

    @staticmethod
    def _warn_inclination_crossing(body: Body, resonances):
        """Warn when the inclination crosses 90 deg in a body with MMRs: a retrograde angle is
        meaningful only for i > 90 deg and a prograde one only below. No status changes."""
        if body.inc is None or not any(r.type == 'mmr' for r in resonances):
            return
        inc = np.asarray(body.inc, dtype=float)
        inc = inc[np.isfinite(inc)]
        if len(inc) and inc.min() < np.pi / 2 < inc.max():
            logger.warning(
                f"{body.name}: the inclination crosses 90 deg ({np.degrees(inc.min()):.1f}..{np.degrees(inc.max()):.1f} deg); "
                "a retrograde MMR angle is meaningful only while i > 90 deg, a prograde one only while i < 90 deg."
            )

    def _analyse_coherence(self, body: Body, resonance: Resonance, psd_cache):
        """Run the cross spectrum for a body-resonance and record the e-i exchange flag.

        Diagnostic only: nothing here changes a status. See `coherence_analysis`.
        """
        if not self.config.coherence_enabled:
            return
        res_key = resonance.to_s()
        verdict = body.librations.get(res_key)
        if self.config.coherence_skip_non_resonant and verdict is not None and verdict.status == Status.NON_RESONANT:
            return

        analyses = coherence_analysis.analyse_resonance(body, resonance, self.config, psd_cache=psd_cache)
        body.coherence[res_key] = analyses

        if isinstance(resonance, LidovKozaiResonance):
            body.zlk_gates[res_key] = coherence_analysis.evaluate_ei_exchange(analyses, self.config)
            # the e-i flag is a check of the verdict (it changes no status)
            verdict = body.librations[res_key]
            verdict.checks['check_zlk_ei'] = _ei_check(body.zlk_gates[res_key])
            self._finalise_verdict(body, res_key, verdict)

    def planets_needed(self) -> list:
        """The planets this run keeps (`save_planets`): all, none, or those the analysis reads — the planets of the
        bodies' resonances, plus `mmr_owner_planets` when there is an MMR and the owner search is on.

        A batch worker gets the list of the whole run from its parent, so every worker keeps the same planets.
        """
        every = list(self.integration_engine.planets_without_sun)
        mode = self.config.save_planets
        if mode == SavePlanets.ALL:
            return every
        if mode == SavePlanets.NONE:
            return []
        if getattr(self.config, '_planets_needed', None) is not None:
            return [planet for planet in every if planet in set(self.config._planets_needed)]
        names = set()
        has_mmr = False
        for body in self.bodies:
            for resonance in body.resonances():
                has_mmr = has_mmr or resonance.type == 'mmr'
                names.update(getattr(resonance, 'planets_names', None) or [])
                if isinstance(resonance, SecularResonance):
                    names.update(SOLAR_SYSTEM_WITH_SUN[index] for index in resonance.index_of_planets)
        if has_mmr and self.config.mmr_owner_search != 'off':
            names.update(self.config.mmr_owner_planets)
        return [planet for planet in every if planet in names]

    def planets_saved(self) -> list:
        """The planets whose series this run holds (written to simulation.json): those integrated here, or, in the
        parent of a batched run (no integration of its own), the list its workers kept."""
        held = [name for name, data in self.integration_engine.planets_data.items() if data and len(data.get('a', ())) > 0]
        return held if held else self.planets_needed()

    def dry_run(self, level: str = 'static'):
        """Check the run before integrating: 'static' (elements only) or 'pilot' (a short fine integration per body).
        Returns a `DryRunReport` (summary, rows, bodies, run, apply, to_csv); see `simulation/dry_run.py`."""
        from .dry_run import dry_run

        return dry_run(self, level)


def long_period_series(body: Body, resonance: Resonance):
    """The `LongPeriodSeries` of a secular or Lidov-Kozai angle of a body, in increasing time: the angle the package
    classifies (`body.angles_unwrapped[key]`: the reduced proper angle of a secular resonance, omega for ZLK), the
    elements, and the secular angle's terms with the planetary frequencies."""
    from resonances.data.const import PLANETARY_FREQUENCIES
    from resonances.resonance.analysis.secular import LongPeriodSeries
    from resonances.resonance.analysis.secular.preprocess import terms_from_formula

    order = mmr_adapter.time_order(body)
    key = resonance.to_s()

    def column(values):
        return None if values is None else mmr_adapter.in_time_order(values, order)

    secular = isinstance(resonance, SecularResonance)
    return LongPeriodSeries(
        resonance_type=ResonanceType.SECULAR if secular else ResonanceType.LIDOV_KOZAI,
        key=key,
        name=str(body.name),
        t_yr=np.asarray(body.times, dtype=float)[order] / (2 * np.pi),
        angle=np.unwrap(column(body.angles_unwrapped[key])),
        a=column(body.axis),
        e=column(body.ecc),
        inc=column(body.inc),
        varpi=column(body.varpi),
        Omega=column(body.Omega),
        terms=terms_from_formula(resonance.formula.terms) if secular else (),
        planet_frequencies=PLANETARY_FREQUENCIES if secular else {},
    )


def _filtered_at(series, cutoff_yr: float, settings):
    """The angle of `series` filtered at `cutoff_yr` (the fine scale, for the data file and the periodogram)."""
    from resonances.resonance.analysis.filtering import lowpass

    try:
        return lowpass(series.t_yr, series.angle, cutoff_yr, settings.filter_order, grid_rtol=settings.filter_grid_rtol)
    except ValueError:
        return None


def _gate_config(config):
    """The gate's settings: off when the free elements are off (the gate cannot run, the status stands)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        free_gate_enabled=bool(config.free_elements_enabled and config.free_gate_enabled), free_gate_min_cycles=config.free_gate_min_cycles
    )


def _ei_check(gate) -> CheckValue:
    """check_zlk_ei from the e-i exchange flag `confirmed_ei` (it changes no status): pass, fail, or not_measured."""
    confirmed = None if gate is None else gate.confirmed_ei
    if confirmed is None:
        return CheckValue.NOT_MEASURED
    return CheckValue.PASS if confirmed else CheckValue.FAIL
