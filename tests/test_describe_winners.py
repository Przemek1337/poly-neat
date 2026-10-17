from __future__ import annotations

import json

import torch

from benchmarks.describe_winners import (
    describe_deepneat,
    describe_exact,
    describe_fixed_cnn,
    main,
)
from polyneat.algorithms.deepneat.deepneat_genome import (
    DeepNEATGenome,
    LayerNodeGene,
    TensorEdgeGene,
)
from polyneat.algorithms.deepneat.layer_shape_propagation import TensorShape
from polyneat.algorithms.deepneat.torch_layer_stack_phenotype import TorchLayerStackPhenotype
from polyneat.algorithms.exact.exact_genome import (
    ConvolutionEdgeGene,
    EXACTGenome,
    FilterNodeGene,
)
from polyneat.algorithms.exact.exact_image_input_adapter import FlattenedImageInputAdapter
from polyneat.algorithms.exact.torch_convolutional_phenotype import TorchConvolutionalPhenotype
from polyneat.nn.fixed_convolutional_network import (
    FixedConvolutionalNetwork,
    FixedConvolutionalNetworkConfig,
)
from polyneat.training.model_checkpoint import capture_model_checkpoint


def _edge(innovation_id: int, source: int, target: int, enabled: bool = True) -> TensorEdgeGene:
    return TensorEdgeGene(
        innovation_id=innovation_id,
        source_node_id=source,
        target_node_id=target,
        is_enabled=enabled,
    )


def _branching_genome() -> DeepNEATGenome:
    """Two conv branches of different resolution merged, then a dense head.

    Node 7 is a dead end and edge 0 -> 1 is disabled, so both must be absent
    from the executed network.
    """
    return DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
            LayerNodeGene(
                node_id=2,
                layer_type="conv",
                number_of_filters=8,
                kernel_size=3,
                uses_batch_normalization=True,
                is_followed_by_max_pooling=True,
            ),
            LayerNodeGene(
                node_id=3,
                layer_type="conv",
                number_of_filters=4,
                kernel_size=5,
                dropout_rate=0.25,
            ),
            LayerNodeGene(
                node_id=4,
                layer_type="conv",
                number_of_filters=16,
                kernel_size=3,
                is_followed_by_max_pooling=True,
            ),
            LayerNodeGene(node_id=6, layer_type="dense", number_of_units=32, dropout_rate=0.5),
            LayerNodeGene(node_id=7, layer_type="conv", number_of_filters=4, kernel_size=3),
        ),
        edge_genes=(
            _edge(0, 0, 1, enabled=False),
            _edge(1, 0, 2),
            _edge(2, 0, 3),
            _edge(3, 2, 4),
            _edge(4, 3, 4),
            _edge(5, 4, 6),
            _edge(6, 6, 1),
            _edge(7, 0, 7),
        ),
    )


def _phenotype(genome: DeepNEATGenome, channels: int, side: int) -> TorchLayerStackPhenotype:
    return TorchLayerStackPhenotype(
        genome=genome,
        input_shape=TensorShape.spatial(channels=channels, height=side, width=side),
        number_of_classes=2,
        maximum_total_parameter_count=None,
        device_for_computation=torch.device("cpu"),
    )


def _observed_output_shapes(model: torch.nn.Module, input_tensor: torch.Tensor) -> dict[int, str]:
    """Run the model once and record the shape every per-node module produced."""
    shapes: dict[int, str] = {}
    prefix = "_layer_modules_by_node_id."
    handles = []
    for name, module in model.named_modules():
        if name.startswith(prefix) and "." not in name[len(prefix) :]:
            node_id = int(name[len(prefix) :])

            def record(_module, _inputs, output, node_id=node_id) -> None:
                shapes[node_id] = "x".join(str(size) for size in output.shape[1:])

            handles.append(module.register_forward_hook(record))
    model.eval()
    with torch.no_grad():
        model(input_tensor)
    for handle in handles:
        handle.remove()
    return shapes


def test_deepneat_description_matches_the_executed_phenotype() -> None:
    genome = _branching_genome()
    phenotype = _phenotype(genome, channels=3, side=16)
    description = describe_deepneat(
        genome.to_serializable_dict(), phenotype.state_dict(), input_side=16
    )

    assert description.input_shape == "3x16x16"
    assert description.total_parameters == phenotype.total_parameter_count
    assert sum(row.parameters for row in description.layers) == phenotype.total_parameter_count
    observed = _observed_output_shapes(phenotype, torch.zeros(2, 3, 16, 16))
    assert {row.node_id: row.output_shape for row in description.layers if row.node_id} == observed

    rows = {row.node_id: row for row in description.layers}
    assert 7 not in rows
    assert sorted(description.edges) == [(0, 2), (0, 3), (2, 4), (3, 4), (4, 6), (6, 1)]
    assert description.genome_disabled_edge_count == 1
    assert rows[2].max_pooling == "applied" and rows[2].uses_batch_normalization
    assert rows[3].max_pooling == "none" and rows[3].dropout_rate == 0.25
    assert rows[4].merge == "concat 2, max-pool to 8x8"
    assert rows[6].merge == "flatten 256"
    assert description.max_pooling_count == 2
    assert description.dropout_count == 2


def test_pooling_flag_on_a_one_pixel_map_is_reported_as_skipped() -> None:
    genome = DeepNEATGenome(
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
        ),
        edge_genes=(_edge(0, 0, 2), _edge(1, 2, 1)),
    )
    phenotype = _phenotype(genome, channels=1, side=1)
    description = describe_deepneat(
        genome.to_serializable_dict(), phenotype.state_dict(), input_side=1
    )

    assert {row.node_id: row.max_pooling for row in description.layers}[2] == "skipped"
    assert description.max_pooling_count == 0


def test_input_channels_are_not_read_off_a_layer_that_concatenates_the_input() -> None:
    genome = DeepNEATGenome(
        node_genes=(
            LayerNodeGene(node_id=0, layer_type="input"),
            LayerNodeGene(node_id=1, layer_type="output"),
            LayerNodeGene(node_id=2, layer_type="conv", number_of_filters=4, kernel_size=3),
            LayerNodeGene(node_id=3, layer_type="conv", number_of_filters=8, kernel_size=3),
        ),
        # Node 3 sees the image and node 2 concatenated (1 + 4 channels) and its
        # edge from the input comes first.
        edge_genes=(_edge(0, 0, 3), _edge(1, 0, 2), _edge(2, 2, 3), _edge(3, 3, 1)),
    )
    phenotype = _phenotype(genome, channels=1, side=8)
    description = describe_deepneat(
        genome.to_serializable_dict(), phenotype.state_dict(), input_side=8
    )

    assert description.input_shape == "1x8x8"
    assert description.total_parameters == phenotype.total_parameter_count


def test_exact_description_counts_kernels_and_normalization() -> None:
    genome = EXACTGenome(
        node_genes=(
            FilterNodeGene(
                node_id=0, node_type="input", filter_height=8, filter_width=8, depth=0.0
            ),
            FilterNodeGene(
                node_id=1, node_type="output", filter_height=1, filter_width=1, depth=1.0
            ),
            FilterNodeGene(
                node_id=2, node_type="output", filter_height=1, filter_width=1, depth=1.0
            ),
            FilterNodeGene(
                node_id=3, node_type="hidden", filter_height=4, filter_width=4, depth=0.5
            ),
        ),
        edge_genes=(
            ConvolutionEdgeGene(
                innovation_id=0, source_node_id=0, target_node_id=1, is_enabled=True
            ),
            ConvolutionEdgeGene(
                innovation_id=1, source_node_id=0, target_node_id=2, is_enabled=False
            ),
            ConvolutionEdgeGene(
                innovation_id=2, source_node_id=0, target_node_id=3, is_enabled=True
            ),
            ConvolutionEdgeGene(
                innovation_id=3, source_node_id=3, target_node_id=2, is_enabled=True
            ),
            ConvolutionEdgeGene(
                innovation_id=4, source_node_id=3, target_node_id=1, is_enabled=True
            ),
        ),
    )
    model = FlattenedImageInputAdapter(
        TorchConvolutionalPhenotype(
            exact_genome=genome,
            input_image_height=8,
            input_image_width=8,
            leaky_relu_negative_slope=0.1,
            activation_clamp_maximum=5.5,
            device_for_computation=torch.device("cpu"),
            use_batch_normalization=True,
        )
    )
    description = describe_exact(genome.to_serializable_dict(), model.state_dict())

    expected_parameters = sum(parameter.numel() for parameter in model.parameters())
    assert description.total_parameters == expected_parameters
    assert sum(row.parameters for row in description.layers) == expected_parameters
    rows = {row.node_id: row for row in description.layers}
    assert rows[3].kernel == "5x5" and rows[3].uses_batch_normalization
    assert rows[3].parameters == 25 + 2
    assert rows[1].kernel == "8x8, 4x4" and rows[1].merge == "sum 2"
    assert len(description.edges) == 4
    assert description.genome_disabled_edge_count == 1


def test_fixed_cnn_description_splits_parameters_per_block() -> None:
    config = FixedConvolutionalNetworkConfig(
        input_channels=3, number_of_classes=2, convolution_channels=(8, 16), kernel_size=3
    )
    model = FixedConvolutionalNetwork(config)
    payload = {
        "convolution_channels": list(config.convolution_channels),
        "kernel_size": config.kernel_size,
        "uses_batch_normalization": config.uses_batch_normalization,
        "dropout_probability": config.dropout_probability,
    }
    description = describe_fixed_cnn(payload, model.state_dict(), input_side=15)

    assert description.total_parameters == model.total_parameter_count
    assert sum(row.parameters for row in description.layers) == model.total_parameter_count
    assert [row.output_shape for row in description.layers] == ["3x15x15", "8x8x8", "16x4x4", "2"]
    assert description.max_pooling_count == 2


def test_series_walk_writes_tables_diagrams_and_reports_missing_checkpoints(tmp_path) -> None:
    genome = _branching_genome()
    phenotype = _phenotype(genome, channels=1, side=16)
    run_directory = tmp_path / "pneumonia/series-v2/runs/deepneat/seed_101"
    checkpoint = capture_model_checkpoint(
        phenotype,
        model_id="deepneat_track_a_seed101",
        stage="track_a",
        genome_kind="DeepNEATGenome",
        genome_payload=genome.to_serializable_dict(),
        preprocessing_state={"target_side": 16},
        metadata={"parameter_count": phenotype.total_parameter_count},
    )
    checkpoint.write_file(run_directory / "checkpoints" / "deepneat_track_a_seed101.pt")
    report = {
        "frozen_models": {
            "deepneat_track_a_seed101": {
                "track": "track_a",
                "details": {"parameter_count": phenotype.total_parameter_count},
            },
            "deepneat_track_b_seed201": {"track": "track_b", "details": {}},
        },
        "official_test": {
            "deepneat_track_a_seed101": {
                "track": "track_a",
                "metrics_at_reference_threshold": {"auroc": 0.9},
                "metrics_at_selected_threshold": {"sensitivity": 0.8, "specificity": 0.7},
            },
            "deepneat_track_b_seed201": {
                "track": "track_b",
                "metrics_at_reference_threshold": {"auroc": 0.8},
            },
        },
        "effective_configuration": {"image_side": 16},
    }
    (run_directory / "run_report.json").write_text(json.dumps(report), encoding="utf-8")
    missing_run = tmp_path / "pneumonia/series-v2/runs/exact/seed_101"
    missing_run.mkdir(parents=True)
    (missing_run / "run_report.json").write_text(
        json.dumps(
            {
                "frozen_models": {
                    "exact_track_a_seed101": {
                        "track": "track_a",
                        "details": {"parameter_count": 1234},
                    }
                },
                "official_test": {},
            }
        ),
        encoding="utf-8",
    )

    main(["--results-root", str(tmp_path), "--output-directory", str(tmp_path / "out")])

    markdown = (tmp_path / "out/winners.md").read_text(encoding="utf-8")
    assert "| deepneat | 101 |" in markdown
    assert "0.9000" in markdown and "0.8000" in markdown
    assert "Checkpoints missing for 1 model(s)" in markdown
    stem = tmp_path / "out/pneumonia-series-v2/deepneat_seed101"
    for suffix in (".json", ".pdf", ".png", ".dot"):
        assert stem.with_suffix(suffix).is_file()
    assert (tmp_path / "out/pneumonia-series-v2/exact_seed101.json").is_file()
