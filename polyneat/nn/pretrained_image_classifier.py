"""A pretrained ImageNet backbone adapted to single-channel medical images.

This baseline answers a question none of the evolved models can: how much of
the task is solved by features somebody else already learned. It is reported
separately for a reason the protocol states outright - the cost of the original
pretraining is not counted anywhere, so its budget is not comparable with a
search that started from nothing.

Two adaptations are needed and both are explicit rather than incidental. The
grayscale image is repeated across three channels, which is the one documented
exception to the single-channel path every from-scratch model uses; and the
normalization is the one the pinned weights were trained with, not the
statistics fitted on this dataset, because the wrong statistics quietly degrade
pretrained features.

The weights identifier and file name are recorded so a later run can prove it
used the same starting point. The unfreezing policy is a parameter, frozen
before the series, because "fine-tuned a ResNet" describes a family of
experiments rather than one.

References:
    He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for
        Image Recognition. *CVPR 2016*. arXiv:1512.03385.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import torch
from torch import nn

from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)

# The normalization the ImageNet weights were trained with. Replacing it with
# statistics fitted on the X-rays would feed the pretrained features inputs
# they were never calibrated for.
IMAGENET_CHANNEL_MEANS: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_CHANNEL_STANDARD_DEVIATIONS: tuple[float, float, float] = (0.229, 0.224, 0.225)


class UnfreezingPolicy(StrEnum):
    """Which parameters of the backbone are allowed to move.

    Attributes:
        HEAD_ONLY: Only the new classifier trains; the backbone is a fixed
            feature extractor.
        HEAD_AND_LAST_BLOCK: The classifier and the final residual stage train.
        FULL: Everything trains.
    """

    HEAD_ONLY = "head_only"
    HEAD_AND_LAST_BLOCK = "head_and_last_block"
    FULL = "full"


@dataclass(frozen=True)
class PretrainedClassifierConfig:
    """The frozen transfer-learning setup.

    Attributes:
        weights_identifier: Which pretrained weights to load, recorded in the
            protocol lock and in every artifact directory.
        number_of_classes: Output width of the replaced classifier.
        unfreezing_policy: Which parameters may move.
        uses_imagenet_normalization: Whether to apply the normalization the
            weights were trained with. Turning it off is a deliberate ablation,
            not a default.
    """

    weights_identifier: str = "torchvision/resnet18/IMAGENET1K_V1"
    number_of_classes: int = 2
    unfreezing_policy: UnfreezingPolicy = UnfreezingPolicy.HEAD_AND_LAST_BLOCK
    uses_imagenet_normalization: bool = True


class PretrainedResNetClassifier(nn.Module):
    """A ResNet-18 backbone with a fresh binary head, taking one-channel input.

    Satisfies the same trainable-model contract as the evolved phenotypes, so
    the shared trainer, the shared inference path and the shared threshold
    selection all apply unchanged. What differs is only the preprocessing it
    performs internally and where its initial parameters came from.
    """

    def __init__(self, config: PretrainedClassifierConfig | None = None) -> None:
        """Load the pinned weights and attach a fresh classifier.

        Raises:
            ImportError: If torchvision is not installed. It lives in the
                optional ``benchmark`` extra because the core library does not
                need it, and its version is pinned against torch there.
        """
        super().__init__()
        self._config = config or PretrainedClassifierConfig()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as missing_torchvision:
            raise ImportError(
                "the transfer-learning baseline needs torchvision; install the benchmark extra, "
                "which pins it against the torch build in use"
            ) from missing_torchvision

        weights = ResNet18_Weights.IMAGENET1K_V1
        self._backbone = resnet18(weights=weights)
        self._weights_file_name = weights.url.rsplit("/", 1)[-1]
        self._backbone.fc = nn.Linear(
            self._backbone.fc.in_features, self._config.number_of_classes
        )
        self._apply_unfreezing_policy()

        self.register_buffer(
            "_channel_means",
            torch.tensor(IMAGENET_CHANNEL_MEANS, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_channel_standard_deviations",
            torch.tensor(IMAGENET_CHANNEL_STANDARD_DEVIATIONS, dtype=torch.float32).view(
                1, 3, 1, 1
            ),
        )
        logger.info(
            "Transfer-learning baseline ready: %s (%s), unfreezing %s, %d trainable parameters",
            self._config.weights_identifier,
            self._weights_file_name,
            self._config.unfreezing_policy,
            self.trainable_parameter_count,
        )

    @property
    def config(self) -> PretrainedClassifierConfig:
        """The frozen setup this instance was built from."""
        return self._config

    @property
    def weights_file_name(self) -> str:
        """File name of the pretrained checkpoint, recorded in the artifacts."""
        return self._weights_file_name

    @property
    def trainable_parameter_count(self) -> int:
        """Parameters the unfreezing policy actually lets move."""
        return sum(
            int(parameter.numel())
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _apply_unfreezing_policy(self) -> None:
        """Freeze everything the policy does not name."""
        if self._config.unfreezing_policy is UnfreezingPolicy.FULL:
            return
        for parameter in self._backbone.parameters():
            parameter.requires_grad = False
        for parameter in self._backbone.fc.parameters():
            parameter.requires_grad = True
        if self._config.unfreezing_policy is UnfreezingPolicy.HEAD_AND_LAST_BLOCK:
            for parameter in self._backbone.layer4.parameters():
                parameter.requires_grad = True

    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Repeat the single channel to three, normalize, and classify.

        Args:
            input_tensor: ``(batch, 1, side, side)`` batch that has already been
                through the shared geometric path. A batch that already has
                three channels is passed through unchanged.

        Returns:
            Raw class logits.
        """
        if input_tensor.shape[1] == 1:
            input_tensor = input_tensor.repeat(1, 3, 1, 1)
        if self._config.uses_imagenet_normalization:
            # Registered buffers are typed as Tensor | Module on the module, so
            # the narrowing is stated once here rather than at every use.
            channel_means = cast(torch.Tensor, self._channel_means)
            channel_standard_deviations = cast(torch.Tensor, self._channel_standard_deviations)
            input_tensor = (input_tensor - channel_means) / channel_standard_deviations
        return self._backbone(input_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``nn.Module`` compatibility shim - delegates to :meth:`forward_pass`."""
        return self.forward_pass(x)

    def reset_recurrent_state(self) -> None:
        """No recurrent state to reset; present for the phenotype contract."""

    def reinitialize_parameters(self) -> None:
        """Reinitialize the classifier head only, keeping the pretrained backbone.

        Reinitializing the backbone would delete the pretrained features, which
        is the entire point of this baseline. Track B therefore does not apply
        its shared initialization to this model, and the report says so.
        """
        from polyneat.training.parameter_initialization import initialize_module_parameters

        generator = torch.Generator()
        generator.manual_seed(int(torch.randint(0, 2**62, (1,)).item()))
        initialize_module_parameters(self._backbone.fc, generator)
