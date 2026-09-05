"""Differentiable batch preprocessing used by the DeepNEAT CIFAR-10 profile."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from polyneat.algorithms.deepneat.deepneat_genome import DeepNEATGlobalHyperparameters


def compute_training_feature_statistics(
    training_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return broadcastable training mean and standard deviation."""
    reduction_dimensions = (0, *range(2, training_features.ndim))
    mean = training_features.mean(dim=reduction_dimensions, keepdim=True)
    standard_deviation = training_features.std(
        dim=reduction_dimensions, keepdim=True, unbiased=False
    ).clamp_min(1e-6)
    return mean, standard_deviation


def preprocess_deepneat_batch(
    features: torch.Tensor,
    hyperparameters: DeepNEATGlobalHyperparameters,
    training_mean: torch.Tensor,
    training_standard_deviation: torch.Tensor,
    *,
    training: bool,
) -> torch.Tensor:
    """Apply the chromosome's evolved preprocessing to one minibatch."""
    processed = features
    if training and processed.ndim == 4:
        if processed.shape[1] == 3:
            processed = _random_hsv_perturbation(processed, hyperparameters)
        if hyperparameters.spatial_scaling > 0.0:
            processed = _random_spatial_scaling(processed, hyperparameters.spatial_scaling)
        if hyperparameters.uses_horizontal_flips:
            flip_mask = torch.rand(processed.shape[0], device=processed.device) < 0.5
            processed = processed.clone()
            processed[flip_mask] = torch.flip(processed[flip_mask], dims=(-1,))

    if processed.ndim == 4 and hyperparameters.cropped_image_size > 0:
        processed = _crop_square(
            processed,
            hyperparameters.cropped_image_size,
            random_crop=training,
        )

    if hyperparameters.uses_variance_normalization:
        processed = (processed - training_mean) / training_standard_deviation
    return processed


def _crop_square(images: torch.Tensor, crop_size: int, *, random_crop: bool) -> torch.Tensor:
    height, width = images.shape[-2:]
    crop_size = min(crop_size, height, width)
    if crop_size == height == width:
        return images
    if random_crop:
        top_offsets = torch.randint(
            0, height - crop_size + 1, (images.shape[0],), device=images.device
        )
        left_offsets = torch.randint(
            0, width - crop_size + 1, (images.shape[0],), device=images.device
        )
    else:
        top_offsets = torch.full(
            (images.shape[0],), (height - crop_size) // 2, device=images.device
        )
        left_offsets = torch.full(
            (images.shape[0],), (width - crop_size) // 2, device=images.device
        )
    return torch.stack(
        [
            image[:, top : top + crop_size, left : left + crop_size]
            for image, top, left in zip(
                images, top_offsets.tolist(), left_offsets.tolist(), strict=True
            )
        ]
    )


def _random_spatial_scaling(images: torch.Tensor, maximum_scaling: float) -> torch.Tensor:
    batch_size = images.shape[0]
    scale = 1.0 + torch.empty(batch_size, device=images.device).uniform_(
        -maximum_scaling, maximum_scaling
    )
    theta = torch.zeros(batch_size, 2, 3, device=images.device, dtype=images.dtype)
    theta[:, 0, 0] = 1.0 / scale
    theta[:, 1, 1] = 1.0 / scale
    grid = functional.affine_grid(theta, images.shape, align_corners=False)
    return functional.grid_sample(
        images, grid, mode="bilinear", padding_mode="reflection", align_corners=False
    )


def _random_hsv_perturbation(
    images: torch.Tensor, hyperparameters: DeepNEATGlobalHyperparameters
) -> torch.Tensor:
    hue, saturation, value = _rgb_to_hsv(images.clamp(0.0, 1.0))
    sample_shape = (images.shape[0], 1, 1, 1)
    hue_delta = torch.empty(sample_shape, device=images.device).uniform_(
        -hyperparameters.hue_shift_degrees / 360.0,
        hyperparameters.hue_shift_degrees / 360.0,
    )
    shift = hyperparameters.saturation_value_shift
    scale = hyperparameters.saturation_value_scale
    saturation_shift = torch.empty(sample_shape, device=images.device).uniform_(-shift, shift)
    value_shift = torch.empty(sample_shape, device=images.device).uniform_(-shift, shift)
    saturation_scale = torch.empty(sample_shape, device=images.device).uniform_(
        1.0 - scale, 1.0 + scale
    )
    value_scale = torch.empty(sample_shape, device=images.device).uniform_(1.0 - scale, 1.0 + scale)
    hue = (hue + hue_delta) % 1.0
    saturation = (saturation * saturation_scale + saturation_shift).clamp(0.0, 1.0)
    value = (value * value_scale + value_shift).clamp(0.0, 1.0)
    return _hsv_to_rgb(hue, saturation, value)


def _rgb_to_hsv(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    red, green, blue = images[:, 0:1], images[:, 1:2], images[:, 2:3]
    maximum, maximum_indices = images.max(dim=1, keepdim=True)
    minimum = images.min(dim=1, keepdim=True).values
    delta = maximum - minimum
    safe_delta = delta.clamp_min(1e-12)
    hue_red = ((green - blue) / safe_delta) % 6.0
    hue_green = (blue - red) / safe_delta + 2.0
    hue_blue = (red - green) / safe_delta + 4.0
    hue = torch.where(
        maximum_indices == 0,
        hue_red,
        torch.where(maximum_indices == 1, hue_green, hue_blue),
    )
    hue = torch.where(delta == 0.0, torch.zeros_like(hue), hue / 6.0)
    saturation = torch.where(maximum == 0.0, torch.zeros_like(delta), delta / maximum)
    return hue, saturation, maximum


def _hsv_to_rgb(hue: torch.Tensor, saturation: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    sector = torch.floor(hue * 6.0).to(torch.int64) % 6
    fraction = hue * 6.0 - torch.floor(hue * 6.0)
    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)
    choices = torch.stack(
        (
            torch.cat((value, t, p), dim=1),
            torch.cat((q, value, p), dim=1),
            torch.cat((p, value, t), dim=1),
            torch.cat((p, q, value), dim=1),
            torch.cat((t, p, value), dim=1),
            torch.cat((value, p, q), dim=1),
        ),
        dim=1,
    )
    gather_index = sector.expand(-1, 3, -1, -1).unsqueeze(1)
    return torch.gather(choices, dim=1, index=gather_index).squeeze(1)
