from __future__ import annotations

import numpy as np
import pytest

from polyneat.configs.neat.neat_config import NEATConfig


@pytest.fixture
def small_neat_config() -> NEATConfig:
    """Tiny config for fast structural tests (8 genomes, 2 inputs, 1 output)."""
    return NEATConfig(
        population_size=8,
        number_of_input_nodes=2,
        number_of_output_nodes=1,
        random_seed=42,
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
