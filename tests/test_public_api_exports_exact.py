from __future__ import annotations

import polyneat as pn

_EXPECTED_EXACT_EXPORTS = (
    "EXACTConfig",
    "EXACTAlgorithm",
    "EXACTGenome",
    "FilterNodeGene",
    "ConvolutionEdgeGene",
    "InvalidEXACTGenomeError",
    "EXACTCrossover",
    "EXACTCompositeMutation",
    "AddConvolutionNodeMutation",
    "SingleSpeciesSpeciator",
    "EXACTPhenotypeDecoder",
    "TorchConvolutionalPhenotype",
    "EXACTBackpropagationTrainer",
    "EXACTTrainingHyperparameters",
    "SimplexHyperparameterOptimizer",
    "build_minimal_cnn_initial_population",
)


def test_exact_classes_are_exported_at_package_level() -> None:
    for export_name in _EXPECTED_EXACT_EXPORTS:
        assert hasattr(pn, export_name), f"polyneat.{export_name} missing"
        assert export_name in pn.__all__, f"{export_name} not in polyneat.__all__"


def test_exact_algorithm_subpackage_reexports_its_classes() -> None:
    from polyneat.algorithms import exact

    for export_name in (
        "EXACTAlgorithm",
        "EXACTGenome",
        "EXACTCrossover",
        "EXACTCompositeMutation",
        "AddConvolutionNodeMutation",
        "SingleSpeciesSpeciator",
        "EXACTPhenotypeDecoder",
        "TorchConvolutionalPhenotype",
        "EXACTBackpropagationTrainer",
        "EXACTTrainingHyperparameters",
        "SimplexHyperparameterOptimizer",
        "build_minimal_cnn_initial_population",
    ):
        assert hasattr(exact, export_name)


def test_exact_config_subpackage_reexports_its_config() -> None:
    from polyneat.configs.exact import EXACTConfig

    assert EXACTConfig.__name__ == "EXACTConfig"
