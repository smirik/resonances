import numpy as np
import pandas as pd
from pathlib import Path

from .config import SimulationConfig
from resonances.body import Body
from resonances.logger import logger
from resonances.secular.secular_resonance import SecularResonance
from resonances.lidov_kozai.lidov_kozai_resonance import LidovKozaiResonance, LidovKozaiParameters
from resonances.plotting import Plotter
from .serializer import SimulationSerializer


class DataManager:
    """Manages data saving and export functionality."""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def should_save_body(self, body: Body, resonance):
        """Check if body MMR data should be saved."""
        return self._process_status(body.statuses.get(resonance.to_s(), 0), self.config.save)

    def should_plot_body(self, body: Body, resonance):
        """Check if body MMR should be plotted."""
        return self._process_status(body.statuses.get(resonance.to_s(), 0), self.config.plot)

    def _process_status(self, status: int, mode) -> bool:
        """Process status against mode to determine if action should be taken."""
        if (mode is None) or (mode is False) or (isinstance(mode, str) and mode.lower() == 'none'):
            return False
        if mode == 'all':
            return True
        if mode == 'resonant' and status > 0:
            return True
        if mode == 'nonzero' and status != 0:
            return True
        if mode == 'candidates' and status < 0:
            return True
        return False

    def ensure_save_path_exists(self):
        """Ensure save and plot paths exist."""
        Path(self.config.save_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.plot_path).mkdir(parents=True, exist_ok=True)

    def save_data(self, bodies, times, simulation=None):
        """Save simulation data and plots."""
        if self.config.save_summary:
            self.save_simulation_summary(bodies)

        if simulation and simulation.integration_engine.planets_data is not None and self.config.save_planets:
            self.save_planets(times, simulation.integration_engine.planets_data)

        for body in bodies:
            self.save_body(body, times)
        if simulation:
            simulation.running_time["bodies_saved"] = logger.get_current_time()
        for body in bodies:
            self.plot_body(body, simulation)
        if simulation:
            simulation.running_time["stop"] = logger.get_current_time()

        self.save_configuration_details(bodies, simulation)

    def save_body(self, body: Body, times):
        """Save all resonance data for a body."""

        self.ensure_save_path_exists()
        save_mode = self.config.save
        if (save_mode is None) or (save_mode is False) or (isinstance(save_mode, str) and save_mode.lower() == 'none'):
            return

        body_data = body.keplerian_elements_to_dict()
        body_data["times"] = times / (2 * np.pi)
        for resonance in body.resonances():
            if self.should_save_body(body, resonance):
                body_data.update(body.resonance_to_dict(resonance))

        if body_data is not None:
            df = pd.DataFrame(data=body_data)
            df.to_csv(f'{self.config.save_path}/data-{body.name}.csv')

        self._save_periodogram_data(body)

    def _save_periodogram_data(self, body: Body):
        """Save periodogram data for a resonance."""
        # Save resonant angle periodogram
        df_data = {}

        if body.axis_periodogram_frequency is not None:
            freq = body.axis_periodogram_frequency
            power = body.axis_periodogram_power
            df_data = {'a_frequency': freq, 'a_power': power, 'a_period': 1.0 / freq}

        for resonance in body.resonances():
            if self.should_save_body(body, resonance):
                resonance_key = resonance.to_s()
                if body.periodogram_frequency.get(resonance_key) is not None:
                    freq = body.periodogram_frequency[resonance_key]
                    power = body.periodogram_power[resonance_key]

                    df_data[resonance_key + '_frequency'] = freq
                    df_data[resonance_key + '_power'] = power
                    df_data[resonance_key + '_period'] = 1.0 / freq

        df = pd.DataFrame(df_data)
        df.to_csv(f'{self.config.save_path}/data-{body.name}-periodograms.csv', index=False)

    def plot_body(self, body: Body, simulation=None):
        """Plot MMR data for a body."""
        config = self.config.plot_config if self.config.plot_config is not None else 'full'
        for resonance in body.resonances():
            if self.should_plot_body(body, resonance):
                plot_path = self._get_plot_path(body, resonance)
                plot_filename = f'{plot_path}/{body.name}-{resonance.to_s()}.{self.config.image_type}'
                plotter = Plotter.from_body(body, resonance, simulation).configure(config).plot()
                if self.config.plot_type in ["show", "both"]:
                    plotter.show()
                if self.config.plot_type in ["save", "both"]:
                    plotter.save(plot_filename)
                plotter.close()

    def _get_plot_path(self, body: Body, resonance) -> str:
        """
        Get the plot path, optionally with subfolder based on strategy.

        If plot_subfolder_strategy is 'status', creates subfolders:
        - 'resonant' for status=2
        - 'transient' for status=1
        - 'non-resonant' for status=0
        - 'controversial-transient' for status=-1
        - 'controversial-libration' for status=-2
        - 'chaotic' for status=-3 (integration failure, e > 1.1)
        """
        base_path = self.config.plot_path

        if self.config.plot_subfolder_strategy == 'status':
            status = body.statuses.get(resonance.to_s(), 0)
            status_folders = {
                2: 'resonant',
                1: 'transient',
                0: 'non-resonant',
                -1: 'controversial-transient',
                -2: 'controversial-libration',
                -3: 'chaotic',
            }
            subfolder = status_folders.get(status, 'non-resonant')
            plot_path = f'{base_path}/{subfolder}'
            Path(plot_path).mkdir(parents=True, exist_ok=True)
            return plot_path

        return base_path

    def save_planets(self, times, planets_data):
        """Save planetary data."""
        self.ensure_save_path_exists()

        for planet, data in planets_data.items():
            df = pd.DataFrame(data=data)
            df.to_csv(f'{self.config.save_path}/data-planet-{planet}.csv', index=False)

    def save_simulation_summary(self, bodies):
        """Save simulation summary."""
        self.ensure_save_path_exists()

        df = self.get_simulation_summary(bodies)
        summary_filename = f'{self.config.save_path}/summary.csv'

        summary_file = Path(summary_filename)
        if summary_file.exists():
            df.to_csv(summary_filename, mode='a', header=False, index=False)
        else:
            df.to_csv(summary_filename, mode='a', header=True, index=False)

        return df

    def get_simulation_summary(self, bodies):
        """Generate simulation summary dataframe."""
        data = []

        for body in bodies:
            for resonance in body.resonances():
                try:
                    res_type = 'MMR'
                    c1_value = None
                    c2_value = None
                    c_value = None

                    if isinstance(resonance, SecularResonance):
                        res_type = 'Secular'
                    elif isinstance(resonance, LidovKozaiResonance):
                        res_type = 'Lidov-Kozai'
                        c1_value, c2_value, c_value = LidovKozaiParameters.evaluate(
                            body.initial_data['e'],
                            body.initial_data['inc'],
                            body.initial_data['omega'],
                        )

                    libr = body.librations[resonance.to_s()]

                    data.append(
                        [
                            body.name,
                            resonance.to_s(),
                            res_type,
                            body.statuses.get(resonance.to_s(), 0),
                            libr['subtype'],
                            libr['metrics']['phi_deg'],
                            libr['metrics']['phi_rad'],
                            libr['metrics']['amplitude'],
                            libr['metrics']['revolutions'],
                            libr['metrics']['displacement'],
                            libr['metrics']['sign_dominance'],
                            libr['metrics']['frac_positive'],
                            libr['metrics']['mean_sigma_dot'],
                            libr['metrics']['std_sigma_dot'],
                            libr['metrics']['n_zero_crossings'],
                            libr['metrics']['cv_intervals'],
                            libr['metrics']['ls_period'],
                            libr['metrics']['ls_fap'],
                            libr['metrics']['ls_snr'],
                            libr['resolvers']['A1_amplitude'],
                            libr['resolvers']['A2_revolutions_low'],
                            libr['resolvers']['A3_revolutions_high'],
                            libr['resolvers']['A4_sign_dominance'],
                            libr['resolvers']['C1_periodogram'],
                            libr['resolvers']['global_verdict'],
                            libr['resolvers']['has_libration_window'],
                            libr['resolvers']['has_circulation_window'],
                            libr['resolvers']['is_stickiness'],
                            body.initial_data['a'],
                            body.initial_data['e'],
                            body.initial_data['inc'],
                            body.initial_data['Omega'],
                            body.initial_data['omega'],
                            body.initial_data['M'],
                            c1_value,
                            c2_value,
                            c_value,
                        ]
                    )
                except Exception as e:
                    logger.error(f"Error getting resonance summary for {body.name}: {e}")

        return pd.DataFrame(
            data,
            columns=[
                'name',
                'resonance',
                'type',
                'status',
                'subtype',
                'phi_deg',
                'phi_rad',
                'amplitude',
                'revolutions',
                'displacement',
                'sign_dominance',
                'frac_positive',
                'mean_sigma_dot',
                'std_sigma_dot',
                'n_zero_crossings',
                'cv_intervals',
                'ls_period',
                'ls_fap',
                'ls_snr',
                'A1_amplitude',
                'A2_revolutions_low',
                'A3_revolutions_high',
                'A4_sign_dominance',
                'C1_periodogram',
                'global_verdict',
                'has_libration_window',
                'has_circulation_window',
                'is_stickiness',
                'a',
                'e',
                'inc',
                'Omega',
                'omega',
                'M',
                'c1',
                'c2',
                'c',
            ],
        )

    def save_configuration_details(self, bodies, simulation):
        """Save configuration details to file."""
        SimulationSerializer.save_simulation_json(self.config, bodies, simulation)
