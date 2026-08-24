from __future__ import annotations

import polyneat as pn

_EXPECTED_DEEPNEAT_EXPORTS = (
    "DeepNEATAlgorithm",
    "DeepNEATGenome",
    "LayerNodeGene",
    "TensorEdgeGene",
    "InvalidDeepNEATGenomeError",
    "DeepNEATCrossover",
    "DeepNEATSpeciator",
    "DeepNEATPhenotypeDecoder",
    "TorchLayerStackPhenotype",
    "DeepNEATCompositeMutation",
    "AddLayerNodeMutation",
    "AddTensorEdgeMutation",
    "ToggleTensorEdgeMutation",
    "LayerHyperparameterMutation",
    "DeepNEATInnovationTracker",
    "build_deepneat_initial_population",
    "DeepNEATConfig",
    "TrainedNetworkAccuracyEvaluator",
)


def test_deepneat_classes_are_exported_at_package_level() -> None:
    for export_name in _EXPECTED_DEEPNEAT_EXPORTS:
        assert hasattr(pn, export_name), f"polyneat.{export_name} missing"
        assert export_name in pn.__all__, f"{export_name} not in polyneat.__all__"


def test_deepneat_algorithm_subpackage_reexports_its_classes() -> None:
    from polyneat.algorithms import deepneat

    for export_name in (
        "DeepNEATAlgorithm",
        "DeepNEATGenome",
        "LayerNodeGene",
        "TensorEdgeGene",
        "InvalidDeepNEATGenomeError",
        "DeepNEATCrossover",
        "DeepNEATSpeciator",
        "DeepNEATPhenotypeDecoder",
        "TorchLayerStackPhenotype",
        "DeepNEATCompositeMutation",
        "AddLayerNodeMutation",
        "AddTensorEdgeMutation",
        "ToggleTensorEdgeMutation",
        "LayerHyperparameterMutation",
        "DeepNEATInnovationTracker",
        "build_deepneat_initial_population",
    ):
        assert hasattr(deepneat, export_name)


def test_deepneat_config_subpackage_reexports_its_config() -> None:
    from polyneat.configs.deepneat import DeepNEATConfig

    assert DeepNEATConfig.__name__ == "DeepNEATConfig"
