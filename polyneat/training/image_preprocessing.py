"""One image path shared by every model that is trained from scratch.

The order of operations is fixed and executed exactly once per batch, because
running it twice silently doubles the augmentation and re-normalizes already
normalized pixels:

1. pad symmetrically to a square with a constant value; an odd difference puts
   the extra pixel on the bottom or the right;
2. resize to the configured side with bilinear interpolation and antialiasing;
3. cast to ``float32`` in ``[0, 1]`` - casting alone is not normalization, so
   the scale conversion is explicit;
4. during training only, one affine transform (rotation, translation, scale)
   followed by a contrast change around the image mean;
5. standardize with a single mean and standard deviation fitted on the
   *unaugmented* images of the actual training split.

Horizontal and vertical flips are deliberately absent: a chest radiograph has a
left and a right, and mirroring one changes what it shows. Cropping, CLAHE and
lung segmentation are equally absent; they would be separate, planned ablations
rather than part of the shared path.

Fitted statistics belong to the preprocessor instance, and inference never
fits: :meth:`ImagePreprocessor.apply` raises rather than quietly fitting on
whatever batch it was handed. The state travels with a checkpoint through
:meth:`ImagePreprocessor.state_dict`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)

PREPROCESSING_STATE_SCHEMA_VERSION = "1.0"


class PreprocessingStateError(RuntimeError):
    """Raised when preprocessing is used before, or inconsistently with, its fit."""


@dataclass(frozen=True)
class ImagePreprocessingConfig:
    """The deterministic part of the image path.

    Attributes:
        target_side: Side length every image is resized to after padding.
        pad_value: Constant used to pad a non-square image to a square.
        use_antialias: Whether the resize low-pass filters before sampling.
            Kept explicit because it changes the pixels and therefore the
            fitted statistics.
        standardization_epsilon: Lower bound on the fitted standard deviation,
            so a constant training split cannot divide by zero.
    """

    target_side: int = 128
    pad_value: float = 0.0
    use_antialias: bool = True
    standardization_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.target_side < 1:
            raise ValueError(f"target_side must be >= 1, got {self.target_side}")
        if self.standardization_epsilon <= 0.0:
            raise ValueError(
                f"standardization_epsilon must be > 0, got {self.standardization_epsilon}"
            )


@dataclass(frozen=True)
class ImageAugmentationConfig:
    """The random part of the image path, applied during training only.

    One affine transform and one contrast change, both drawn per image. The
    defaults are the ranges this benchmark froze; they are parameters of the
    protocol, not a recommendation of the dataset authors.

    Attributes:
        maximum_rotation_degrees: Rotation is drawn from ``U(-x, x)`` degrees.
        maximum_translation_fraction: Each axis is shifted by ``U(-x, x)`` of
            the image side.
        minimum_scale: Lower end of the uniform scale factor.
        maximum_scale: Upper end of the uniform scale factor.
        minimum_contrast_factor: Lower end of the uniform contrast factor,
            applied around the mean of that image.
        maximum_contrast_factor: Upper end of the same factor.
        fill_value: Value sampled outside the source image.
    """

    maximum_rotation_degrees: float = 7.0
    maximum_translation_fraction: float = 0.03
    minimum_scale: float = 0.95
    maximum_scale: float = 1.05
    minimum_contrast_factor: float = 0.9
    maximum_contrast_factor: float = 1.1
    fill_value: float = 0.0

    def __post_init__(self) -> None:
        if self.minimum_scale <= 0.0 or self.maximum_scale < self.minimum_scale:
            raise ValueError(
                f"scale range must satisfy 0 < minimum <= maximum, got "
                f"({self.minimum_scale}, {self.maximum_scale})"
            )
        if self.maximum_contrast_factor < self.minimum_contrast_factor:
            raise ValueError(
                f"contrast range must satisfy minimum <= maximum, got "
                f"({self.minimum_contrast_factor}, {self.maximum_contrast_factor})"
            )
        if self.maximum_rotation_degrees < 0.0 or self.maximum_translation_fraction < 0.0:
            raise ValueError("rotation and translation limits must be non-negative")


def pad_to_square(images: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
    """Pad an ``NCHW`` batch to a square, symmetrically.

    Args:
        images: ``(batch, channels, height, width)`` tensor.
        pad_value: Constant written into the padding.

    Returns:
        A square batch. When the size difference is odd the extra pixel goes to
        the bottom or the right, so the rule is stated once here instead of
        depending on a library's rounding.

    Raises:
        ValueError: If ``images`` is not 4-dimensional.
    """
    if images.ndim != 4:
        raise ValueError(f"pad_to_square expects an NCHW batch, got shape {tuple(images.shape)}")
    height, width = images.shape[-2:]
    if height == width:
        return images
    target_side = max(height, width)
    vertical_padding = target_side - height
    horizontal_padding = target_side - width
    padding = (
        horizontal_padding // 2,
        horizontal_padding - horizontal_padding // 2,
        vertical_padding // 2,
        vertical_padding - vertical_padding // 2,
    )
    return functional.pad(images, padding, mode="constant", value=pad_value)


def resize_images(
    images: torch.Tensor, target_side: int, *, use_antialias: bool = True
) -> torch.Tensor:
    """Resize an ``NCHW`` batch to ``target_side`` with bilinear interpolation."""
    if images.shape[-2] == target_side and images.shape[-1] == target_side:
        return images
    return functional.interpolate(
        images,
        size=(target_side, target_side),
        mode="bilinear",
        align_corners=False,
        antialias=use_antialias,
    )


def apply_affine_and_contrast_augmentation(
    images: torch.Tensor,
    augmentation: ImageAugmentationConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply one affine transform and one contrast change per image.

    The random draws are taken on the CPU from ``generator`` and only then
    moved to the batch device. That keeps the augmentation stream independent
    of the device, of the model size and of how many numbers the model drew
    while initializing - which is exactly the property the protocol needs when
    it says the data RNG must not depend on the network.

    Args:
        images: ``(batch, channels, side, side)`` float tensor in ``[0, 1]``.
        augmentation: Ranges to draw from.
        generator: Explicit CPU generator owning the augmentation stream.

    Returns:
        The augmented batch, clamped back into ``[0, 1]``.
    """
    batch_size = images.shape[0]
    if batch_size == 0:
        return images

    def _uniform(low: float, high: float) -> torch.Tensor:
        draws = torch.rand(batch_size, generator=generator, dtype=torch.float32)
        return (low + (high - low) * draws).to(device=images.device, dtype=images.dtype)

    rotation_radians = _uniform(
        -augmentation.maximum_rotation_degrees, augmentation.maximum_rotation_degrees
    ) * (math.pi / 180.0)
    scale = _uniform(augmentation.minimum_scale, augmentation.maximum_scale)
    # affine_grid works in normalized [-1, 1] coordinates, where the full image
    # side is 2 units wide, so a shift of f image-fractions is 2f units.
    horizontal_shift = 2.0 * _uniform(
        -augmentation.maximum_translation_fraction, augmentation.maximum_translation_fraction
    )
    vertical_shift = 2.0 * _uniform(
        -augmentation.maximum_translation_fraction, augmentation.maximum_translation_fraction
    )

    cosine = torch.cos(rotation_radians) / scale
    sine = torch.sin(rotation_radians) / scale
    affine_matrices = torch.zeros(batch_size, 2, 3, device=images.device, dtype=images.dtype)
    affine_matrices[:, 0, 0] = cosine
    affine_matrices[:, 0, 1] = -sine
    affine_matrices[:, 0, 2] = horizontal_shift
    affine_matrices[:, 1, 0] = sine
    affine_matrices[:, 1, 1] = cosine
    affine_matrices[:, 1, 2] = vertical_shift

    sampling_grid = functional.affine_grid(affine_matrices, list(images.shape), align_corners=False)
    if augmentation.fill_value != 0.0:
        images = images - augmentation.fill_value
    transformed = functional.grid_sample(
        images, sampling_grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    if augmentation.fill_value != 0.0:
        transformed = transformed + augmentation.fill_value

    contrast_factor = _uniform(
        augmentation.minimum_contrast_factor, augmentation.maximum_contrast_factor
    ).view(batch_size, 1, 1, 1)
    per_image_mean = transformed.mean(dim=(1, 2, 3), keepdim=True)
    contrasted = per_image_mean + contrast_factor * (transformed - per_image_mean)
    return contrasted.clamp(0.0, 1.0)


class ImagePreprocessor:
    """The image path, with its fitted statistics as owned state.

    One instance belongs to one trained model. It is fitted once, on the
    unaugmented training images of that stage, and then travels with the
    checkpoint. Inference reuses the stored statistics and never refits, so a
    saved model reproduces its own predictions.

    Track A fits on ``train`` alone. Track B fits new statistics on
    ``train + search_validation``, because that is what its recipe trains on.
    Neither ever sees ``threshold_validation`` or the official test set.
    """

    def __init__(
        self,
        config: ImagePreprocessingConfig | None = None,
        augmentation: ImageAugmentationConfig | None = None,
    ) -> None:
        """Create an unfitted preprocessor.

        Args:
            config: Deterministic part of the path. Defaults to the protocol's
                128-pixel configuration.
            augmentation: Random part, used during training only. ``None``
                disables augmentation entirely, which is what an
                evaluation-only preprocessor wants.
        """
        self._config = config or ImagePreprocessingConfig()
        self._augmentation = augmentation
        self._fitted_mean: float | None = None
        self._fitted_standard_deviation: float | None = None

    @property
    def config(self) -> ImagePreprocessingConfig:
        """The deterministic configuration this instance was built with."""
        return self._config

    @property
    def augmentation(self) -> ImageAugmentationConfig | None:
        """The augmentation ranges, or ``None`` when augmentation is off."""
        return self._augmentation

    @property
    def is_fitted(self) -> bool:
        """Whether standardization statistics have been fitted."""
        return self._fitted_mean is not None

    def geometrically_normalize(self, images: torch.Tensor) -> torch.Tensor:
        """Run the deterministic steps: pad, resize, scale to ``[0, 1]``.

        Args:
            images: ``NCHW`` batch of any dtype. Integer input is taken to be
                8-bit and divided by 255; floating point input is taken to be
                in ``[0, 1]`` already and only cast.

        Returns:
            ``(batch, channels, side, side)`` ``float32`` tensor in ``[0, 1]``.

        Raises:
            ValueError: If the batch is not 4-dimensional.
        """
        if images.ndim != 4:
            raise ValueError(
                f"preprocessing expects an NCHW batch, got shape {tuple(images.shape)}"
            )
        if images.is_floating_point():
            scaled = images.to(torch.float32)
        else:
            scaled = images.to(torch.float32) / 255.0
        padded = pad_to_square(scaled, self._config.pad_value)
        return resize_images(
            padded, self._config.target_side, use_antialias=self._config.use_antialias
        )

    def fit_standardization(self, training_images: torch.Tensor) -> None:
        """Fit the single mean and standard deviation on unaugmented images.

        Args:
            training_images: The training split of this stage, raw. It is run
                through the deterministic steps here - padding included, as the
                protocol requires - but never through augmentation.

        Raises:
            ValueError: If the batch is empty.
        """
        if training_images.shape[0] == 0:
            raise ValueError("cannot fit standardization on an empty training split")
        normalized = self.geometrically_normalize(training_images)
        self._fitted_mean = float(normalized.mean())
        self._fitted_standard_deviation = max(
            float(normalized.std(unbiased=False)), self._config.standardization_epsilon
        )
        logger.info(
            "Preprocessing fitted on %d training images: mean=%.6f std=%.6f",
            training_images.shape[0],
            self._fitted_mean,
            self._fitted_standard_deviation,
        )

    def use_identity_standardization(self) -> None:
        """Mark the preprocessor fitted with a mean of zero and a scale of one.

        Some models bring their own normalization and must not receive a
        second one. A pretrained ImageNet backbone is the case this exists for:
        its features were calibrated against the ImageNet channel statistics,
        so standardizing with statistics fitted on chest X-rays first would
        hand it an input distribution it has never seen.

        Calling this is a deliberate, recorded choice rather than a way to skip
        fitting: the resulting state still travels with the checkpoint and
        still says what was applied.
        """
        self._fitted_mean = 0.0
        self._fitted_standard_deviation = 1.0
        logger.info(
            "Preprocessing set to identity standardization; the model is expected to "
            "normalize its own input"
        )

    def standardize(self, images: torch.Tensor) -> torch.Tensor:
        """Apply the fitted statistics.

        Raises:
            PreprocessingStateError: If called before fitting. Fitting lazily on
                whatever batch arrived would make inference depend on the batch
                it was handed, which is the leak this class exists to prevent.
        """
        if self._fitted_mean is None or self._fitted_standard_deviation is None:
            raise PreprocessingStateError(
                "preprocessing statistics have not been fitted; call fit_standardization on the "
                "training split of this stage before training or predicting"
            )
        return (images - self._fitted_mean) / self._fitted_standard_deviation

    def apply(
        self,
        images: torch.Tensor,
        *,
        training: bool,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Run the whole path exactly once over one batch.

        Args:
            images: ``NCHW`` batch straight from the dataset.
            training: When ``True`` and augmentation is configured, one affine
                transform and one contrast change run between the
                deterministic steps and standardization. Evaluation passes
                ``False`` and is therefore deterministic.
            generator: Generator owning the augmentation stream. Required
                whenever augmentation would actually run, so the stream is an
                explicit, recorded input rather than global torch state.

        Returns:
            The standardized batch.

        Raises:
            PreprocessingStateError: If augmentation would run without a
                generator, or if the statistics were never fitted.
        """
        normalized = self.geometrically_normalize(images)
        if training and self._augmentation is not None:
            if generator is None:
                raise PreprocessingStateError(
                    "augmentation needs an explicit torch.Generator so the data stream stays "
                    "independent of model initialization; pass the augmentation generator"
                )
            normalized = apply_affine_and_contrast_augmentation(
                normalized, self._augmentation, generator
            )
        return self.standardize(normalized)

    def state_dict(self) -> dict:
        """Return the state a checkpoint has to carry for this preprocessor."""
        return {
            "schema_version": PREPROCESSING_STATE_SCHEMA_VERSION,
            "target_side": self._config.target_side,
            "pad_value": self._config.pad_value,
            "use_antialias": self._config.use_antialias,
            "standardization_epsilon": self._config.standardization_epsilon,
            "fitted_mean": self._fitted_mean,
            "fitted_standard_deviation": self._fitted_standard_deviation,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore statistics saved by :meth:`state_dict`.

        Raises:
            PreprocessingStateError: If the state was written by another schema
                version, or describes a different deterministic configuration.
                Restoring statistics fitted at another resolution would produce
                predictions that silently disagree with the saved ones.
        """
        stored_version = state.get("schema_version")
        if stored_version != PREPROCESSING_STATE_SCHEMA_VERSION:
            raise PreprocessingStateError(
                f"preprocessing state has schema version {stored_version!r}, this build reads "
                f"{PREPROCESSING_STATE_SCHEMA_VERSION!r}"
            )
        if int(state["target_side"]) != self._config.target_side:
            raise PreprocessingStateError(
                f"preprocessing state was fitted at side {state['target_side']} but this "
                f"preprocessor is configured for {self._config.target_side}"
            )
        self._fitted_mean = None if state["fitted_mean"] is None else float(state["fitted_mean"])
        self._fitted_standard_deviation = (
            None
            if state["fitted_standard_deviation"] is None
            else float(state["fitted_standard_deviation"])
        )
