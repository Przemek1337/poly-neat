"""The X-ray profiles must keep the extra preprocessing genes switched off.

The shared image path pads, resizes, augments and standardizes each batch
exactly once. A DeepNEAT genome that also carried a crop, a rescale, a flip or
a second variance normalization would process the batch twice, and a flip would
mirror a radiograph, which has a left and a right. So the pneumonia profiles
pin those genes to neutral values, and these tests check that initialization
and mutation both respect the pin rather than trusting the yaml comment.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import yaml

from examples.pediatric_pneumonia._protocol_lock import (
    ProtocolLockError,
    load_protocol_lock,
    validate_protocol_lock_for_full_run,
)
from polyneat.algorithms.deepneat.deepneat_algorithm import DeepNEATAlgorithm
from polyneat.configs.deepneat.deepneat_config import DeepNEATConfig

_PROFILE_PATHS = (
    Path("examples/pediatric_pneumonia/deepneat_smoke.yaml"),
    Path("examples/pediatric_pneumonia/random_search_smoke.yaml"),
)


def _algorithm_config(profile_path: Path) -> DeepNEATConfig:
    payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    return cast(DeepNEATConfig, DeepNEATConfig.from_dict(payload["algorithm"]))


@pytest.mark.parametrize("profile_path", _PROFILE_PATHS, ids=lambda path: path.stem)
class TestBlockedPreprocessingGenes:
    def test_the_profile_pins_every_extra_preprocessing_gene(
        self, profile_path: Path
    ) -> None:
        config = _algorithm_config(profile_path)
        assert config.global_cropped_image_size_min == 0
        assert config.global_cropped_image_size_max == 0
        assert config.global_spatial_scaling_min == 0.0
        assert config.global_spatial_scaling_max == 0.0
        assert config.available_horizontal_flip_options == (False,)
        assert config.available_variance_normalization_options == (False,)
        assert config.global_hue_shift_degrees_max == 0.0
        assert config.global_saturation_value_shift_max == 0.0
        assert config.global_saturation_value_scale_max == 0.0

    def test_no_initial_genome_carries_extra_preprocessing(
        self, profile_path: Path
    ) -> None:
        algorithm = DeepNEATAlgorithm.from_config(_algorithm_config(profile_path))
        population = algorithm.create_initial_population(np.random.default_rng(7))
        for genome in population.genomes:
            _assert_preprocessing_is_neutral(genome)

    def test_mutation_never_switches_extra_preprocessing_back_on(
        self, profile_path: Path
    ) -> None:
        algorithm = DeepNEATAlgorithm.from_config(_algorithm_config(profile_path))
        rng = np.random.default_rng(11)
        genome = algorithm.create_initial_population(rng).genomes[0]
        for _ in range(200):
            genome = algorithm.mutation.apply_to_genome(
                genome, rng, algorithm.innovation_tracker
            )
            _assert_preprocessing_is_neutral(genome)


def _assert_preprocessing_is_neutral(genome) -> None:
    """Every gene the shared image path owns must stay at its neutral value."""
    hyperparameters = genome.global_hyperparameters
    assert hyperparameters.cropped_image_size == 0
    assert hyperparameters.spatial_scaling == 0.0
    assert hyperparameters.uses_horizontal_flips is False
    assert hyperparameters.uses_variance_normalization is False
    assert hyperparameters.hue_shift_degrees == 0.0
    assert hyperparameters.saturation_value_shift == 0.0
    assert hyperparameters.saturation_value_scale == 0.0


class TestProtocolLockTemplate:
    def test_the_shipped_template_refuses_to_validate(self) -> None:
        """It is a list of decisions to make, not a runnable lock."""
        lock = load_protocol_lock(
            Path("examples/pediatric_pneumonia/protocol.lock.template.yaml")
        )
        with pytest.raises(ProtocolLockError) as raised:
            validate_protocol_lock_for_full_run(lock, manifest_sha256="a" * 64)
        message = str(raised.value)
        assert "undecided placeholder value at data.manifest_sha256" in message
        assert "undecided placeholder value at seeds.search_seeds" in message

    def test_the_template_names_every_required_section(self) -> None:
        from examples.pediatric_pneumonia._protocol_lock import REQUIRED_LOCK_KEYS

        lock = load_protocol_lock(
            Path("examples/pediatric_pneumonia/protocol.lock.template.yaml")
        )
        assert set(REQUIRED_LOCK_KEYS) <= set(lock.sections)
        for section_name, keys in REQUIRED_LOCK_KEYS.items():
            assert set(keys) <= set(lock.sections[section_name]), section_name
