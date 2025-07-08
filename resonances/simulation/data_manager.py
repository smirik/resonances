import pandas as pd
from pathlib import Path

import resonances
from .config import SimulationConfig


class DataManager:
    """Manages data saving and export functionality."""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def should_save_body(self, body: resonances.Body, resonance: resonances.Resonance):
        """Check if body MMR data should be saved."""
        return self._process_status(body.statuses.get(resonance.to_s(), 0), self.config.save)

    def should_plot_body(self, body: resonances.Body, resonance: resonances.Resonance):
        """Check if body MMR should be plotted."""
        return self._process_status(body.statuses.get(resonance.to_s(), 0), self.config.plot)

    def _process_status(self, status: int, mode: str) -> bool:
        """Process status against mode to determine if action should be taken."""
        if mode is None:
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

        for body in bodies:
            for resonance in body.mmrs + body.secular_resonances:
                if self.should_save_body(body, resonance):
                    self.save_body(body, resonance, times)
                if self.should_plot_body(body, resonance):
                    self.plot_body(body, resonance, simulation)

    def save_body(self, body: resonances.Body, resonance: resonances.Resonance, times):
        """Save MMR data for a body."""
        self.ensure_save_path_exists()

        if isinstance(resonance, resonances.MMR):
            df_data = body.mmr_to_dict(resonance, times)
        elif isinstance(resonance, resonances.SecularResonance):
            df_data = body.secular_to_dict(resonance, times)
        else:
            raise ValueError(f"Unknown resonance type: {type(resonance)}")

        if df_data is not None:
            df = pd.DataFrame(data=df_data)
            df.to_csv(f'{self.config.save_path}/data-{body.name}-{resonance.to_s()}.csv')

        self._save_periodogram_data(body, resonance.to_s(), body.name)

    def plot_body(self, body: resonances.Body, resonance: resonances.Resonance, simulation=None):
        """Plot MMR data for a body."""
        self.ensure_save_path_exists()
        resonances.resonance.plot.body(simulation, body, resonance, image_type=self.config.image_type)

    def _save_periodogram_data(self, body: resonances.Body, resonance_key: str, body_name: str):
        """Save periodogram data for a resonance."""
        # Save resonant angle periodogram
        if body.periodogram_frequency.get(resonance_key) is not None:
            freq = body.periodogram_frequency[resonance_key]
            power = body.periodogram_power[resonance_key]

            periodogram_data = {'frequency': freq, 'power': power, 'period': 1.0 / freq}

            df = pd.DataFrame(periodogram_data)
            df.to_csv(f'{self.config.save_path}/data-{body_name}-{resonance_key}-periodogram-angle.csv', index=False)

        # Save semi-major axis periodogram
        if body.axis_periodogram_frequency is not None:
            freq = body.axis_periodogram_frequency
            power = body.axis_periodogram_power

            periodogram_data = {'frequency': freq, 'power': power, 'period': 1.0 / freq}

            df = pd.DataFrame(periodogram_data)
            df.to_csv(f'{self.config.save_path}/data-{body_name}-{resonance_key}-periodogram-axis.csv', index=False)

    def save_simulation_summary(self, bodies):
        """Save simulation summary."""
        self.ensure_save_path_exists()
        self.save_configuration_details(bodies)

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
            for resonance in body.mmrs + body.secular_resonances:
                try:
                    overlapping_str = ', '.join(
                        f'({left:.0f}, {right:.0f})' for left, right in body.periodogram_peaks_overlapping.get(resonance.to_s(), [])
                    )
                    data.append(
                        [
                            body.name,
                            resonance.to_s(),
                            ('MMR' if isinstance(resonance, resonances.MMR) else 'Secular'),
                            body.statuses.get(resonance.to_s(), 0),
                            body.libration_pure.get(resonance.to_s(), False),
                            body.libration_metrics.get(resonance.to_s(), {}).get('num_libration_periods', 0),
                            body.libration_metrics.get(resonance.to_s(), {}).get('max_libration_length', 0),
                            body.monotony.get(resonance.to_s(), 0),
                            overlapping_str,
                            body.initial_data['a'],
                            body.initial_data['e'],
                            body.initial_data['inc'],
                            body.initial_data['Omega'],
                            body.initial_data['omega'],
                            body.initial_data['M'],
                        ]
                    )
                except Exception as e:
                    resonances.logger.error(f"Error getting resonance summary for {body.name}: {e}")

            # for mmr in body.mmrs:
            #     try:
            #         overlapping_str = ', '.join(
            #             f'({left:.0f}, {right:.0f})' for left, right in body.periodogram_peaks_overlapping.get(mmr.to_s(), [])
            #         )
            #         data.append(
            #             [
            #                 body.name,
            #                 mmr.to_s(),
            #                 'MMR',
            #                 body.statuses.get(mmr.to_s(), 0),
            #                 body.libration_pure.get(mmr.to_s(), False),
            #                 body.libration_metrics.get(mmr.to_s(), {}).get('num_libration_periods', 0),
            #                 body.libration_metrics.get(mmr.to_s(), {}).get('max_libration_length', 0),
            #                 body.monotony.get(mmr.to_s(), 0),
            #                 overlapping_str,
            #                 body.initial_data['a'],
            #                 body.initial_data['e'],
            #                 body.initial_data['inc'],
            #                 body.initial_data['Omega'],
            #                 body.initial_data['omega'],
            #                 body.initial_data['M'],
            #             ]
            #         )
            #     except Exception as e:
            #         resonances.logger.error(f"Error getting MMR summary for {body.name}: {e}")
            # for secular in body.secular_resonances:
            #     try:
            #         overlapping_str = ', '.join(
            #             f'({left:.0f}, {right:.0f})' for left, right in body.periodogram_peaks_overlapping.get(secular.to_s(), [])
            #         )
            #         data.append(
            #             [
            #                 body.name,
            #                 secular.to_s(),
            #                 'Secular',
            #                 body.statuses.get(secular.to_s(), 0),
            #                 body.libration_pure.get(secular.to_s(), False),
            #                 body.libration_metrics.get(secular.to_s(), {}).get('num_libration_periods', 0),
            #                 body.libration_metrics.get(secular.to_s(), {}).get('max_libration_length', 0),
            #                 body.monotony.get(secular.to_s(), 0),
            #                 overlapping_str,
            #                 body.initial_data['a'],
            #                 body.initial_data['e'],
            #                 body.initial_data['inc'],
            #                 body.initial_data['Omega'],
            #                 body.initial_data['omega'],
            #                 body.initial_data['M'],
            #             ]
            #         )
            #     except Exception as e:
            #         resonances.logger.error(f"Error getting secular summary for {body.name}: {e}")

        return pd.DataFrame(
            data,
            columns=[
                'name',
                'resonance',
                'type',
                'status',
                'pure',
                'num_libration_periods',
                'max_libration_length',
                'monotony',
                'overlapping',
                'a',
                'e',
                'inc',
                'Omega',
                'omega',
                'M',
            ],
        )

    def save_configuration_details(self, bodies):
        """Save configuration details to file."""
        with open(f"{self.config.save_path}/simulation.cfg", "w") as f:
            f.write("Simulation Configuration\n")
            f.write("========================\n")
            f.write(f"Name: {self.config.name}\n")
            f.write(f"Date: {self.config.date}\n")
            f.write(f"Source: {self.config.source}\n")
            f.write(f"Number of bodies: {len(bodies)}\n")
            f.write("========================\n")
            f.write(f"Tmax: {self.config.tmax}\n")
            f.write(f"Integrator: {self.config.integrator}\n")
            f.write(f"dt: {self.config.dt}\n")
            f.write("========================\n")
            f.write("Libration analysis parameters\n")
            f.write(f"Cutoff: {self.config.oscillations_cutoff}\n")
            f.write(f"Filter order: {self.config.oscillations_filter_order}\n")
            f.write(f"Frequency min: {self.config.periodogram_frequency_min}\n")
            f.write(f"Frequency max: {self.config.periodogram_frequency_max}\n")
            f.write(f"Critical: {self.config.periodogram_critical}\n")
            f.write(f"Soft: {self.config.periodogram_soft}\n")
            f.write(f"Period critical: {self.config.libration_period_critical}\n")
