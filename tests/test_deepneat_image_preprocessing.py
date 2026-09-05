from __future__ import annotations

import torch

from polyneat.algorithms.deepneat.deepneat_genome import (
    DeepNEATGlobalHyperparameters,
)
from polyneat.evaluators.deepneat_image_preprocessing import (
    compute_training_feature_statistics,
    preprocess_deepneat_batch,
)


def test_evolved_crop_controls_the_phenotype_input_size() -> None:
    images = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
    mean, standard_deviation = compute_training_feature_statistics(images)
    result = preprocess_deepneat_batch(
        images,
        DeepNEATGlobalHyperparameters(cropped_image_size=6),
        mean,
        standard_deviation,
        training=False,
    )
    assert result.shape == (2, 3, 6, 6)
    assert torch.equal(result, images[:, :, 1:7, 1:7])


def test_variance_normalization_uses_training_statistics() -> None:
    images = torch.arange(4 * 3 * 4 * 4, dtype=torch.float32).reshape(4, 3, 4, 4)
    mean, standard_deviation = compute_training_feature_statistics(images)
    result = preprocess_deepneat_batch(
        images,
        DeepNEATGlobalHyperparameters(uses_variance_normalization=True),
        mean,
        standard_deviation,
        training=False,
    )
    assert torch.allclose(result.mean(dim=(0, 2, 3)), torch.zeros(3), atol=1e-6)
    assert torch.allclose(result.std(dim=(0, 2, 3), unbiased=False), torch.ones(3))


def test_hsv_and_spatial_augmentation_preserve_batch_shape_and_range() -> None:
    torch.manual_seed(3)
    images = torch.rand(5, 3, 8, 8)
    mean, standard_deviation = compute_training_feature_statistics(images)
    result = preprocess_deepneat_batch(
        images,
        DeepNEATGlobalHyperparameters(
            hue_shift_degrees=45.0,
            saturation_value_shift=0.5,
            saturation_value_scale=0.5,
            spatial_scaling=0.3,
            uses_horizontal_flips=True,
        ),
        mean,
        standard_deviation,
        training=True,
    )
    assert result.shape == images.shape
    assert torch.isfinite(result).all()
    assert 0.0 <= float(result.min()) <= float(result.max()) <= 1.0
