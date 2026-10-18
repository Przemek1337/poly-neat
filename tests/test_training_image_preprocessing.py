"""Tests for the shared image path, class weights and random streams."""

from __future__ import annotations

import pytest
import torch

from polyneat.training.class_weights import compute_balanced_class_weights
from polyneat.training.image_preprocessing import (
    ImageAugmentationConfig,
    ImagePreprocessingConfig,
    ImagePreprocessor,
    PreprocessingStateError,
    pad_to_square,
    resize_images,
)
from polyneat.training.random_streams import (
    RandomStreamRole,
    TrainingRandomStreams,
    create_torch_generator,
    derive_stream_seed,
)


def _rectangular_batch(batch_size: int = 4, height: int = 9, width: int = 5) -> torch.Tensor:
    values = torch.arange(batch_size * height * width, dtype=torch.float32)
    return (values / values.max()).reshape(batch_size, 1, height, width)


class TestGeometricSteps:
    def test_padding_is_symmetric_and_puts_the_odd_pixel_bottom_right(self) -> None:
        images = torch.ones(1, 1, 4, 7)
        padded = pad_to_square(images, pad_value=0.0)
        assert padded.shape == (1, 1, 7, 7)
        # 3 rows of padding: 1 on top, 2 at the bottom.
        assert torch.equal(padded[0, 0, 0], torch.zeros(7))
        assert torch.equal(padded[0, 0, 5], torch.zeros(7))
        assert torch.equal(padded[0, 0, 6], torch.zeros(7))
        assert torch.all(padded[0, 0, 1:4] == 1.0)

    def test_already_square_batch_is_returned_unchanged(self) -> None:
        images = torch.rand(2, 1, 6, 6)
        assert pad_to_square(images) is images

    def test_resize_reaches_the_target_side(self) -> None:
        resized = resize_images(torch.rand(3, 1, 20, 20), 8)
        assert resized.shape == (3, 1, 8, 8)

    def test_uint8_input_is_scaled_to_unit_range(self) -> None:
        preprocessor = ImagePreprocessor(ImagePreprocessingConfig(target_side=8))
        images = torch.full((2, 1, 8, 8), 255, dtype=torch.uint8)
        normalized = preprocessor.geometrically_normalize(images)
        assert normalized.dtype == torch.float32
        assert torch.allclose(normalized, torch.ones_like(normalized))

    def test_non_nchw_input_is_rejected(self) -> None:
        preprocessor = ImagePreprocessor()
        with pytest.raises(ValueError, match="NCHW"):
            preprocessor.geometrically_normalize(torch.rand(4, 64))


class TestStandardizationOwnership:
    def test_predicting_before_fitting_raises(self) -> None:
        preprocessor = ImagePreprocessor(ImagePreprocessingConfig(target_side=8))
        with pytest.raises(PreprocessingStateError, match="not been fitted"):
            preprocessor.apply(_rectangular_batch(), training=False)

    def test_fitted_statistics_come_from_the_given_split_only(self) -> None:
        preprocessor = ImagePreprocessor(ImagePreprocessingConfig(target_side=8))
        training_images = torch.full((5, 1, 8, 8), 0.25)
        preprocessor.fit_standardization(training_images)
        other_images = torch.full((5, 1, 8, 8), 0.9)
        standardized = preprocessor.apply(other_images, training=False)
        # std of a constant split is clamped to epsilon, so the shift is what matters
        assert torch.all(standardized > 0.0)
        assert preprocessor.is_fitted

    def test_padding_is_included_in_the_fitted_statistics(self) -> None:
        preprocessor = ImagePreprocessor(ImagePreprocessingConfig(target_side=8))
        preprocessor.fit_standardization(torch.ones(4, 1, 8, 4))
        state = preprocessor.state_dict()
        assert state["fitted_mean"] < 1.0, "zero padding must pull the fitted mean down"

    def test_empty_training_split_raises(self) -> None:
        with pytest.raises(ValueError, match="empty training split"):
            ImagePreprocessor().fit_standardization(torch.empty(0, 1, 8, 8))

    def test_state_roundtrips_and_refuses_a_different_resolution(self) -> None:
        fitted = ImagePreprocessor(ImagePreprocessingConfig(target_side=8))
        fitted.fit_standardization(torch.rand(6, 1, 12, 9))
        restored = ImagePreprocessor(ImagePreprocessingConfig(target_side=8))
        restored.load_state_dict(fitted.state_dict())
        batch = _rectangular_batch()
        assert torch.allclose(
            fitted.apply(batch, training=False), restored.apply(batch, training=False)
        )
        wrong_side = ImagePreprocessor(ImagePreprocessingConfig(target_side=16))
        with pytest.raises(PreprocessingStateError, match="fitted at side"):
            wrong_side.load_state_dict(fitted.state_dict())

    def test_unknown_schema_version_is_refused(self) -> None:
        preprocessor = ImagePreprocessor()
        state = preprocessor.state_dict() | {"schema_version": "0.9"}
        with pytest.raises(PreprocessingStateError, match="schema version"):
            preprocessor.load_state_dict(state)


class TestAugmentation:
    @staticmethod
    def _fitted_preprocessor(augment: bool = True) -> ImagePreprocessor:
        preprocessor = ImagePreprocessor(
            ImagePreprocessingConfig(target_side=16),
            ImageAugmentationConfig() if augment else None,
        )
        preprocessor.fit_standardization(torch.rand(8, 1, 20, 14))
        return preprocessor

    def test_augmentation_runs_in_training_and_not_in_evaluation(self) -> None:
        preprocessor = self._fitted_preprocessor()
        batch = torch.rand(6, 1, 20, 14)
        generator = torch.Generator().manual_seed(7)
        trained_view = preprocessor.apply(batch, training=True, generator=generator)
        evaluated_view = preprocessor.apply(batch, training=False)
        assert not torch.allclose(trained_view, evaluated_view)
        assert torch.allclose(
            evaluated_view, preprocessor.apply(batch, training=False)
        ), "evaluation must be deterministic"

    def test_training_without_a_generator_is_refused(self) -> None:
        preprocessor = self._fitted_preprocessor()
        with pytest.raises(PreprocessingStateError, match="explicit torch.Generator"):
            preprocessor.apply(torch.rand(2, 1, 20, 14), training=True)

    def test_augmentation_stream_does_not_depend_on_other_torch_draws(self) -> None:
        preprocessor = self._fitted_preprocessor()
        batch = torch.rand(5, 1, 20, 14)

        torch.manual_seed(0)
        first = preprocessor.apply(
            batch, training=True, generator=torch.Generator().manual_seed(99)
        )

        torch.manual_seed(0)
        # Stand in for a bigger model drawing more numbers while initializing.
        torch.nn.Linear(512, 512)
        torch.rand(10_000)
        second = preprocessor.apply(
            batch, training=True, generator=torch.Generator().manual_seed(99)
        )
        assert torch.allclose(first, second)

    def test_augmentation_never_mirrors_the_image(self) -> None:
        preprocessor = self._fitted_preprocessor()
        # A bright stripe on the left half must stay on the left half: the
        # protocol switches flips off, and 7 degrees of rotation cannot move it.
        batch = torch.zeros(32, 1, 16, 16)
        batch[:, :, :, :4] = 1.0
        generator = torch.Generator().manual_seed(5)
        augmented = preprocessor.apply(batch, training=True, generator=generator)
        left_mass = augmented[:, :, :, :8].sum(dim=(1, 2, 3))
        right_mass = augmented[:, :, :, 8:].sum(dim=(1, 2, 3))
        assert torch.all(left_mass > right_mass)

    def test_no_augmentation_config_means_training_equals_evaluation(self) -> None:
        preprocessor = self._fitted_preprocessor(augment=False)
        batch = torch.rand(4, 1, 20, 14)
        assert torch.allclose(
            preprocessor.apply(batch, training=True),
            preprocessor.apply(batch, training=False),
        )


class TestClassWeights:
    def test_balanced_split_gets_unit_weights(self) -> None:
        weights = compute_balanced_class_weights(torch.tensor([0, 1, 0, 1]), 2)
        assert torch.allclose(weights, torch.ones(2))

    def test_imbalanced_split_follows_the_protocol_formula(self) -> None:
        labels = torch.tensor([0] * 2 + [1] * 6)
        weights = compute_balanced_class_weights(labels, 2)
        assert torch.allclose(weights, torch.tensor([8 / (2 * 2), 8 / (2 * 6)]))

    def test_missing_class_is_an_error_not_a_zero_weight(self) -> None:
        with pytest.raises(ValueError, match="do not occur in this split"):
            compute_balanced_class_weights(torch.tensor([1, 1, 1]), 2)

    def test_empty_split_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="empty split"):
            compute_balanced_class_weights(torch.empty(0, dtype=torch.long), 2)

    def test_label_outside_the_class_range_is_an_error(self) -> None:
        with pytest.raises(ValueError, match=r"labels must be in"):
            compute_balanced_class_weights(torch.tensor([0, 1, 2]), 2)


class TestRandomStreams:
    def test_roles_get_different_streams_from_one_root_seed(self) -> None:
        seeds = {
            role: derive_stream_seed(role=role, root_seed=17) for role in RandomStreamRole
        }
        assert len(set(seeds.values())) == len(RandomStreamRole)

    def test_derivation_is_deterministic(self) -> None:
        first = derive_stream_seed(
            role=RandomStreamRole.BATCH_ORDER, root_seed=3, evaluation_id="gen1/cand4", track="A"
        )
        second = derive_stream_seed(
            role=RandomStreamRole.BATCH_ORDER, root_seed=3, evaluation_id="gen1/cand4", track="A"
        )
        assert first == second

    def test_evaluation_id_and_track_separate_streams(self) -> None:
        base = dict(role=RandomStreamRole.PARAMETER_INITIALIZATION, root_seed=3)
        assert derive_stream_seed(**base, evaluation_id="a", track="A") != derive_stream_seed(
            **base, evaluation_id="b", track="A"
        )
        assert derive_stream_seed(**base, evaluation_id="a", track="A") != derive_stream_seed(
            **base, evaluation_id="a", track="B"
        )

    def test_generators_reproduce_the_same_draws(self) -> None:
        arguments = dict(
            role=RandomStreamRole.AUGMENTATION, root_seed=11, evaluation_id="c7", track="B"
        )
        first = torch.rand(5, generator=create_torch_generator(**arguments))
        second = torch.rand(5, generator=create_torch_generator(**arguments))
        assert torch.equal(first, second)

    def test_training_streams_are_independent_of_each_other(self) -> None:
        streams = TrainingRandomStreams.derive(root_seed=2, evaluation_id="e1", track="A")
        draws = [
            torch.rand(4, generator=streams.parameter_initialization),
            torch.rand(4, generator=streams.batch_order),
            torch.rand(4, generator=streams.augmentation),
        ]
        assert not torch.equal(draws[0], draws[1])
        assert not torch.equal(draws[1], draws[2])

    def test_stream_state_roundtrips_for_a_resumable_checkpoint(self) -> None:
        streams = TrainingRandomStreams.derive(root_seed=2, evaluation_id="e1", track="A")
        torch.rand(3, generator=streams.batch_order)
        captured_state = streams.state_dict()
        expected_next = torch.rand(3, generator=streams.batch_order)

        restored = TrainingRandomStreams.derive(root_seed=999, evaluation_id="other", track="B")
        restored.load_state_dict(captured_state)
        assert torch.equal(torch.rand(3, generator=restored.batch_order), expected_next)
