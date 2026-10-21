from __future__ import annotations

import polyneat as pn

_EXPECTED_LNEAT_EXPORTS = (
    "LNEATConfig",
    "LNEATAlgorithm",
    "BackpropagationWeightTrainer",
    "TrainableTorchFeedForwardPhenotype",
    "RecognizerEnsemblePhenotype",
)


def test_lneat_symbols_are_package_exported() -> None:
    for symbol_name in _EXPECTED_LNEAT_EXPORTS:
        assert symbol_name in pn.__all__, f"{symbol_name} missing from polyneat.__all__"
        assert getattr(pn, symbol_name) is not None


def test_lneat_algorithm_export_is_the_algorithm_class() -> None:
    from polyneat.algorithms.lneat.lneat_algorithm import LNEATAlgorithm

    assert pn.LNEATAlgorithm is LNEATAlgorithm


def test_binary_recognizer_evaluator_is_importable_by_module_path() -> None:
    # dataset-specific evaluators are reached by module path, as
    # ClassificationAccuracyEvaluator and XORWithDistractorsEvaluator are
    from polyneat.evaluators.binary_recognizer_evaluator import (
        BinaryRecognizerFitnessEvaluator,
    )

    assert BinaryRecognizerFitnessEvaluator is not None
