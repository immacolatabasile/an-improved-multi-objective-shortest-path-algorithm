"""Centralized configuration for the MDA package."""

from dataclasses import dataclass


@dataclass
class MDAConfig:
    """Algorithm parameters for MDA."""
    parallel_threshold: int = 50
    parallel_dominance: bool = False
    num_objectives: int = 2
    source_node: int = 0
    show_plots: bool = False
    plot_dir: str = "plots"


@dataclass
class LoggingConfig:
    """Logging configuration defaults."""
    file_level: str = "INFO"
    console_level: str = "INFO"
    log_dir: str = "logs"
    file_enabled: bool = True
    console_enabled: bool = True


@dataclass
class TestConfig:
    """Test instance parameters."""
    grid_rows: int = 100
    grid_cols: int = 100
    random_seed: int = 42
    cost_min: int = 1
    cost_max: int = 10


# Singleton instances
mda_config = MDAConfig()
logging_config = LoggingConfig()
test_config = TestConfig()
