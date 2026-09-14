#!/usr/bin/env python3
"""
Tests for SimulationConfig Component
===================================

This module tests the SimulationConfig class.
"""

import pytest
import numpy as np

from resonances.simulation import SimulationConfig


class TestSimulationConfig:
    """Test the SimulationConfig component."""

    def test_default_config(self):
        """Test default configuration."""
        config = SimulationConfig()

        # Check that basic attributes are set
        assert config.name is not None
        assert config.date is not None
        assert config.source is not None
        assert config.integrator is not None
        assert config.dt > 0
        assert config.tmax > 0

    def test_custom_config(self):
        """Test custom configuration."""
        custom_name = "test_simulation"
        custom_tmax = 20000
        custom_integrator = "SABA(10,6,4)"

        config = SimulationConfig(name=custom_name, tmax=custom_tmax, integrator=custom_integrator)

        assert config.name == custom_name
        assert config.tmax == custom_tmax
        assert config.integrator == custom_integrator

    def test_tmax_property(self):
        """Test tmax property calculation."""
        config = SimulationConfig()
        tmax_value = 31415  # Arbitrary value
        config.tmax = tmax_value

        assert abs(config.tmax - tmax_value) < 1e-10
        assert abs(config.tmax_yrs - tmax_value / (2 * np.pi)) < 1e-10

    def test_libration_params_setup(self):
        """Test libration parameters setup."""
        config = SimulationConfig()

        # Check that libration parameters are set
        assert hasattr(config, 'oscillations_cutoff')
        assert hasattr(config, 'oscillations_filter_order')
        assert hasattr(config, 'periodogram_frequency_min')
        assert hasattr(config, 'periodogram_frequency_max')
        assert hasattr(config, 'libration_period_critical')

    def test_backward_integration_keeps_span_derived_params_positive(self):
        """A negative tmax (backward run) must not produce negative periodogram windows.

        With tmax = -100000 the span is 15915.5 yr, so the defaults are 5 % = 795.8 yr,
        10 % = 1592 yr, 20 % = 3183.1 yr and the minimum frequency is 1/15915.5 yr.
        Before the fix these came out negative and the periodogram trimmed a negative number
        of points, keeping only the tail of every series.
        """
        config = SimulationConfig(tmax=-100000)

        span_yrs = 100000 / (2 * np.pi)
        assert config.tmax == -100000
        assert config.Nout == 1000
        assert abs(config.libration_period_min - 0.05 * span_yrs) < 1e-9
        assert config.libration_period_critical == round(0.1 * span_yrs)
        assert abs(config.periodogram_critical - 0.2 * span_yrs) < 1e-9
        assert abs(config.periodogram_frequency_min - 1.0 / span_yrs) < 1e-15
        assert config.libration_period_min > 0
        assert config.periodogram_frequency_min > 0

        forward = SimulationConfig(tmax=100000)
        assert forward.libration_period_min == config.libration_period_min
        assert forward.periodogram_frequency_min == config.periodogram_frequency_min

    def test_classify_params_defaults(self):
        """Test classification parameters have correct defaults."""
        config = SimulationConfig()

        # Window settings
        assert config.classify_window_steps == [0.1, 0.2, 0.3]
        assert config.classify_window_step == 0.05

        # Global thresholds
        assert config.classify_rev_libration == 1.0
        assert config.classify_tto_pure_libration == 0.5
        assert config.classify_tto_partial_libration == 2.5
        assert config.classify_tto_non_resonant == 6.0
        assert config.classify_tto_transient_global == 3.0
        assert config.classify_tto_near_separatrix == 6.0

        # Segment quality thresholds
        assert config.classify_good_seg_max_rev == 1.0
        assert config.classify_good_seg_max_tto == 0.5
        assert config.classify_reasonable_seg_max_rev_1 == 2.0
        assert config.classify_reasonable_seg_max_tto_1 == 1.0
        assert config.classify_reasonable_seg_max_rev_2 == 1.0
        assert config.classify_reasonable_seg_max_tto_2 == 1.5

    def test_classify_params_custom(self):
        """Test classification parameters can be customized."""
        config = SimulationConfig(
            classify_tto_non_resonant=10.0,
            classify_good_seg_max_rev=0.5,
        )

        assert config.classify_tto_non_resonant == 10.0
        assert config.classify_good_seg_max_rev == 0.5
        # Other values should still be defaults
        assert config.classify_window_step == 0.05
        assert config.classify_rev_libration == 1.0


if __name__ == '__main__':
    pytest.main([__file__])
