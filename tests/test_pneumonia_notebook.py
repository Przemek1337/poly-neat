from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from benchmarks.export_pneumonia_notebook_data import (
    CurveSplit,
    EpochObservingRecipe,
    LearningCurveRecorder,
)
from benchmarks.pneumonia_notebook import (
    calibration_bins,
    classifier_feature_sources,
    extended_metrics,
    grad_cam,
    image_skips_into_classifier,
    occlusion_sensitivity,
    preprocessing_stages,
    select_examples,
    top_fraction_overlap,
)
from benchmarks.retrain_selected_pneumonia import StabilizedTrainingRecipe
from polyneat.algorithms.deepneat.deepneat_genome import (
    DeepNEATGenome,
    LayerNodeGene,
    TensorEdgeGene,
)
from polyneat.algorithms.deepneat.layer_shape_propagation import TensorShape
from polyneat.algorithms.deepneat.torch_layer_stack_phenotype import TorchLayerStackPhenotype
from polyneat.training.image_preprocessing import ImagePreprocessingConfig, ImagePreprocessor
from polyneat.training.supervised_trainer import SupervisedTrainer


def _edge(innovation_id: int, source: int, target: int) -> TensorEdgeGene:
    return TensorEdgeGene(
        innovation_id=innovation_id, source_node_id=source, target_node_id=target, is_enabled=True
    )


def _phenotype(genome: DeepNEATGenome, side: int) -> TorchLayerStackPhenotype:
    return TorchLayerStackPhenotype(
        genome=genome,
        input_shape=TensorShape.spatial(channels=1, height=side, width=side),
        number_of_classes=2,
        maximum_total_parameter_count=None,
        device_for_computation=torch.device("cpu"),
    )


def _two_branch_genome_with_image_skip() -> DeepNEATGenome:
    """Two conv maps of different size and the raw image, all flattened into the output."""
    return DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
            LayerNodeGene(
                node_id=2,
                layer_type="conv",
                number_of_filters=4,
                kernel_size=3,
                is_followed_by_max_pooling=True,
            ),
            LayerNodeGene(
                node_id=3, layer_type="conv", number_of_filters=6, kernel_size=3, dropout_rate=0.3
            ),
        ),
        edge_genes=(_edge(0, 0, 2), _edge(1, 0, 3), _edge(2, 2, 1), _edge(3, 3, 1), _edge(4, 0, 1)),
    )


def _dense_only_genome() -> DeepNEATGenome:
    return DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
            LayerNodeGene(node_id=2, layer_type="dense", number_of_units=8),
        ),
        edge_genes=(_edge(0, 0, 2), _edge(1, 2, 1)),
    )


def test_extended_metrics_match_a_hand_counted_confusion_matrix() -> None:
    labels = np.array([1, 1, 1, 0, 0, 1, 0, 0])
    probabilities = np.array([0.9, 0.8, 0.3, 0.6, 0.1, 0.7, 0.2, 0.4])
    # Positive at >= 0.5: TP = 3, FN = 1, FP = 1, TN = 3.

    metrics = extended_metrics(labels, probabilities, 0.5)

    assert (metrics["true_positives"], metrics["false_negatives"]) == (3, 1)
    assert (metrics["false_positives"], metrics["true_negatives"]) == (1, 3)
    assert metrics["sensitivity"] == metrics["specificity"] == 0.75
    assert metrics["precision"] == metrics["negative_predictive_value"] == 0.75
    assert metrics["false_positive_rate"] == metrics["false_negative_rate"] == 0.25
    assert math.isclose(metrics["mcc"], (9 - 1) / 16)
    assert math.isclose(metrics["f1"], 0.75)
    # 14 of the 16 positive-negative pairs are ranked correctly; 0.3 beats only 0.1 and 0.2.
    assert math.isclose(metrics["roc_auc"], 14 / 16)
    assert math.isclose(metrics["brier"], float(np.mean((probabilities - labels) ** 2)))


def test_extended_metrics_report_an_empty_denominator_as_none() -> None:
    metrics = extended_metrics(np.array([0, 1]), np.array([0.2, 0.4]), 0.5)

    assert metrics["precision"] is None
    assert metrics["mcc"] is None
    assert metrics["sensitivity"] == 0.0


def test_calibration_bins_skip_empty_bins_and_count_every_row() -> None:
    bins = calibration_bins(np.array([0, 0, 1, 1]), np.array([0.05, 0.15, 0.95, 1.0]), 10)

    assert [entry["count"] for entry in bins] == [1, 1, 2]
    assert bins[-1]["positive_rate"] == 1.0


def test_grad_cam_uses_every_conv_branch_and_leaves_the_image_skip_out() -> None:
    torch.manual_seed(0)
    model = _phenotype(_two_branch_genome_with_image_skip(), side=16)
    image = torch.randn(1, 1, 16, 16)

    assert classifier_feature_sources(model) == [2, 3]
    assert image_skips_into_classifier(model)
    saliency = grad_cam(model, image)

    assert saliency.heatmap.shape == (16, 16)
    assert 0.0 <= float(saliency.heatmap.min()) and float(saliency.heatmap.max()) <= 1.0
    with torch.no_grad():
        logits = model.forward_pass(image)
    assert saliency.explained_class == int(logits.argmax(dim=1)[0])
    assert math.isclose(
        saliency.positive_probability, float(torch.softmax(logits, dim=1)[0, 1]), rel_tol=1e-5
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_grad_cam_refuses_a_network_without_convolutions() -> None:
    model = _phenotype(_dense_only_genome(), side=8)

    with pytest.raises(ValueError, match="no convolutional feature map"):
        grad_cam(model, torch.zeros(1, 1, 8, 8))


class _TopLeftModel(torch.nn.Module):
    """Pneumonia logit = sum of the top-left quadrant; the rest of the image is ignored."""

    def forward_pass(self, images: torch.Tensor) -> torch.Tensor:
        evidence = images[:, :, :8, :8].sum(dim=(1, 2, 3))
        return torch.stack([torch.zeros_like(evidence), evidence], dim=1)


def test_occlusion_finds_the_only_region_the_decision_depends_on() -> None:
    saliency = occlusion_sensitivity(
        _TopLeftModel(), torch.ones(1, 1, 16, 16), patch_size=4, stride=2, batch_size=7
    )

    assert saliency.explained_class == 1
    assert float(saliency.signed[:8, :8].min()) > 0.0
    assert float(np.abs(saliency.signed[10:, 10:]).max()) == 0.0
    assert saliency.heatmap.max() == 1.0


def test_top_fraction_overlap_is_one_for_equal_maps_and_zero_for_disjoint_ones() -> None:
    first = np.arange(16, dtype=float).reshape(4, 4)

    assert top_fraction_overlap(first, first, 0.25) == 1.0
    assert top_fraction_overlap(first, -first, 0.25) == 0.0


def test_preprocessing_stages_end_where_the_evaluation_path_ends() -> None:
    pixels = np.random.default_rng(3).integers(0, 256, size=(20, 30), dtype=np.uint8)
    preprocessor = ImagePreprocessor(ImagePreprocessingConfig(target_side=16))
    preprocessor.fit_standardization(torch.from_numpy(pixels.copy())[None, None])

    stages = preprocessing_stages(pixels, preprocessor)

    assert stages["padded"].shape == (30, 30)
    assert stages["resized"].dtype == np.uint8 and stages["resized"].shape == (16, 16)
    expected = preprocessor.apply(torch.from_numpy(stages["resized"])[None, None], training=False)
    assert np.allclose(stages["standardized"], expected[0, 0].numpy())


def test_example_selection_is_seeded_and_counts_each_category() -> None:
    entries = (
        [
            {"example_id": f"train/NORMAL/IM-{i}.jpeg", "label": 0, "target_split": "train"}
            for i in range(6)
        ]
        + [
            {
                "example_id": f"train/PNEUMONIA/person{i}_virus_1.jpeg",
                "label": 1,
                "target_split": "train",
            }
            for i in range(3)
        ]
        + [{"example_id": "test/NORMAL/IM-9.jpeg", "label": 0, "target_split": "official_test"}]
    )
    counts = {"NORMAL": 2, "virus": 1}

    first = select_examples(entries, split="train", counts=counts, seed=7)

    assert first == select_examples(entries, split="train", counts=counts, seed=7)
    assert [entry["label"] for entry in first] == [0, 0, 1]
    assert all(entry["target_split"] == "train" for entry in first)
    with pytest.raises(ValueError, match="requested"):
        select_examples(entries, split="train", counts={"bacteria": 1}, seed=7)


def _build_model() -> TorchLayerStackPhenotype:
    return _phenotype(_two_branch_genome_with_image_skip(), side=16)


def _train(recipe, images, labels, preprocessor, build=_build_model) -> TorchLayerStackPhenotype:
    torch.manual_seed(11)
    model = build()
    SupervisedTrainer(
        recipe=recipe, preprocessor=preprocessor, device_for_computation=torch.device("cpu")
    ).train(
        model,
        images=images,
        labels=labels,
        batch_order_generator=torch.Generator().manual_seed(1),
        augmentation_generator=torch.Generator().manual_seed(2),
    )
    return model


def test_recording_curves_does_not_change_the_training_trajectory() -> None:
    images = torch.randint(0, 256, (24, 1, 16, 16), dtype=torch.uint8)
    labels = torch.tensor([0, 1] * 12)
    preprocessor = ImagePreprocessor(ImagePreprocessingConfig(target_side=16))
    preprocessor.fit_standardization(images)
    settings = {"batch_size": 8, "number_of_epochs": 3, "warmup_steps": 2}
    recorder = LearningCurveRecorder(
        splits=[CurveSplit("train", images, labels), CurveSplit("validation", images, labels)],
        preprocessor=preprocessor,
        class_weights=torch.ones(2),
        device=torch.device("cpu"),
        batch_size=5,
    )
    observed = EpochObservingRecipe(**settings, epoch_observer=recorder.observe)

    plain_model = _train(StabilizedTrainingRecipe(**settings), images, labels, preprocessor)
    observed_model = _train(
        observed, images, labels, preprocessor, build=recorder.remember(_build_model)
    )

    assert [row["completed_epochs"] for row in recorder.epochs] == [0, 1, 2]
    assert set(recorder.epochs[0]) == {"completed_epochs", "train", "validation"}
    assert recorder.epochs[0]["train"]["loss"] > 0.0
    for name, tensor in plain_model.state_dict().items():
        assert torch.equal(tensor, observed_model.state_dict()[name]), name
    assert (
        observed.to_serializable_dict()
        == StabilizedTrainingRecipe(**settings).to_serializable_dict()
    )
