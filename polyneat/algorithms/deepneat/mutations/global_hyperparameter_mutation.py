"""Evolution of DeepNEAT's chromosome-wide training/preprocessing genes."""

from __future__ import annotations

from dataclasses import replace

from numpy.random import Generator

from polyneat.algorithms.deepneat.deepneat_genome import (
    DeepNEATGenome,
    DeepNEATGlobalHyperparameters,
)
from polyneat.algorithms.deepneat.mutations.layer_hyperparameter_mutation import (
    _clip_gaussian,
    _draw_choice,
)
from polyneat.configs.deepneat.deepneat_config import DeepNEATConfig
from polyneat.core.component_protocols import InnovationTracker


def draw_global_hyperparameters(
    config: DeepNEATConfig, rng: Generator
) -> DeepNEATGlobalHyperparameters:
    """Draw an initial chromosome-wide hyperparameter vector."""
    crop_min = config.global_cropped_image_size_min
    crop_max = config.global_cropped_image_size_max
    cropped_image_size = (
        crop_min if crop_min == crop_max else int(rng.integers(crop_min, crop_max + 1))
    )
    return DeepNEATGlobalHyperparameters(
        learning_rate=float(
            rng.uniform(config.global_learning_rate_min, config.global_learning_rate_max)
        ),
        momentum=float(rng.uniform(config.global_momentum_min, config.global_momentum_max)),
        hue_shift_degrees=float(
            rng.uniform(
                config.global_hue_shift_degrees_min,
                config.global_hue_shift_degrees_max,
            )
        ),
        saturation_value_shift=float(
            rng.uniform(
                config.global_saturation_value_shift_min,
                config.global_saturation_value_shift_max,
            )
        ),
        saturation_value_scale=float(
            rng.uniform(
                config.global_saturation_value_scale_min,
                config.global_saturation_value_scale_max,
            )
        ),
        cropped_image_size=cropped_image_size,
        spatial_scaling=float(
            rng.uniform(
                config.global_spatial_scaling_min,
                config.global_spatial_scaling_max,
            )
        ),
        uses_horizontal_flips=bool(_draw_choice(config.available_horizontal_flip_options, rng)),
        uses_variance_normalization=bool(
            _draw_choice(config.available_variance_normalization_options, rng)
        ),
        uses_nesterov_momentum=bool(_draw_choice(config.available_nesterov_momentum_options, rng)),
    )


class GlobalHyperparameterMutation:
    """Gaussian-mutate one real global gene or flip one binary global gene."""

    def __init__(self, probability_of_application: float, config: DeepNEATConfig) -> None:
        self._probability_of_application = probability_of_application
        self._config = config

    def apply_to_genome(
        self,
        genome: DeepNEATGenome,
        rng: Generator,
        innovation_tracker: InnovationTracker,
    ) -> DeepNEATGenome:
        del innovation_tracker
        if rng.random() >= self._probability_of_application:
            return genome

        mutable_fields = [
            "learning_rate",
            "momentum",
            "hue_shift_degrees",
            "saturation_value_shift",
            "saturation_value_scale",
            "cropped_image_size",
            "spatial_scaling",
        ]
        option_fields = {
            "uses_horizontal_flips": self._config.available_horizontal_flip_options,
            "uses_variance_normalization": (self._config.available_variance_normalization_options),
            "uses_nesterov_momentum": (self._config.available_nesterov_momentum_options),
        }
        mutable_fields.extend(
            field_name for field_name, options in option_fields.items() if len(options) > 1
        )
        field_name = str(_draw_choice(tuple(mutable_fields), rng))
        current = genome.global_hyperparameters
        if field_name.startswith("uses_"):
            mutated = replace(current, **{field_name: not getattr(current, field_name)})
        else:
            minimum, maximum = self._bounds_for(field_name)
            value = _clip_gaussian(
                float(getattr(current, field_name)),
                minimum,
                maximum,
                self._config.gaussian_mutation_standard_deviation_fraction,
                rng,
            )
            if field_name == "cropped_image_size":
                value = int(round(value))
            mutated = replace(current, **{field_name: value})
        return replace(genome, global_hyperparameters=mutated)

    def _bounds_for(self, field_name: str) -> tuple[float, float]:
        prefix_by_field = {
            "learning_rate": "global_learning_rate",
            "momentum": "global_momentum",
            "hue_shift_degrees": "global_hue_shift_degrees",
            "saturation_value_shift": "global_saturation_value_shift",
            "saturation_value_scale": "global_saturation_value_scale",
            "cropped_image_size": "global_cropped_image_size",
            "spatial_scaling": "global_spatial_scaling",
        }
        prefix = prefix_by_field[field_name]
        return (
            float(getattr(self._config, f"{prefix}_min")),
            float(getattr(self._config, f"{prefix}_max")),
        )
