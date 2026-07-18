from __future__ import annotations

# Import the example's importable helpers directly (no MNIST download involved).
import importlib.util
import math
from pathlib import Path

import numpy as np
import torch

import polyneat as pn
from polyneat.evaluators.classification_accuracy_evaluator import (
    ClassificationAccuracyEvaluator,
)
from polyneat.evaluators.softmax_likelihood_evaluator import (
    SoftmaxLikelihoodFitnessEvaluator,
)

_EXAMPLE_PATH = Path(__file__).parent.parent / "examples" / "mnist_hyperneat.py"
_spec = importlib.util.spec_from_file_location("mnist_hyperneat_example", _EXAMPLE_PATH)
assert _spec is not None and _spec.loader is not None
mnist_hyperneat_example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mnist_hyperneat_example)

_GRID = 4
_NUM_CLASSES = 3


def _small_config() -> pn.HyperNEATConfig:
    return pn.HyperNEATConfig(
        population_size=12,
        substrate_input_layer_size=_GRID * _GRID,
        substrate_hidden_layer_sizes=(),
        substrate_output_layer_size=_NUM_CLASSES,
        substrate_node_activation_function="identity",
        weight_expression_threshold=0.1,
        random_seed=0,
    )


def _synthetic_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(1)
    features = rng.standard_normal((30, _GRID * _GRID)).astype(np.float32)
    labels = rng.integers(0, _NUM_CLASSES, size=30).astype(np.int64)
    return torch.from_numpy(features), torch.from_numpy(labels)


def test_example_config_loads_as_hyperneat_config():
    config_path = Path(__file__).parent.parent / "examples" / "mnist_hyperneat.yaml"
    config = pn.HyperNEATConfig.load_from_yaml_file(config_path)
    assert config.number_of_input_nodes == 4  # CPPN coordinate inputs
    assert config.substrate_output_layer_size == 10
    assert config.substrate_node_activation_function == "identity"


def test_sandwich_substrate_has_grid_inputs_and_output_row_and_bias():
    substrate = pn.build_grid_sandwich_substrate(
        input_grid_shape=(_GRID, _GRID),
        output_grid_shape=(1, _NUM_CLASSES),
        coordinate_range_min=-1.0,
        coordinate_range_max=1.0,
        include_bias_node=True,
        output_sheet_shares_input_plane=False,
    )
    assert len(substrate.input_layer.nodes) == _GRID * _GRID
    assert len(substrate.output_layer.nodes) == _NUM_CLASSES
    assert substrate.hidden_layers == ()
    assert substrate.bias_node is not None
    # contiguous ids, input-first, bias-last
    all_ids = [node.node_id for node in substrate.all_nodes()]
    assert all_ids == list(range(len(all_ids)))
    assert substrate.bias_node.node_id == all_ids[-1]


def test_short_run_completes_and_classifier_is_evaluable():
    features, labels = _synthetic_dataset()
    algorithm = mnist_hyperneat_example.build_mnist_hyperneat_algorithm(
        _small_config(), grid_side=_GRID, number_of_classes=_NUM_CLASSES
    )
    runner = pn.EvolutionRunner(
        algorithm=algorithm,
        fitness_evaluator=SoftmaxLikelihoodFitnessEvaluator(features, labels),
        termination_criterion=pn.MaxGenerationsTermination(max_generations=3),
        callbacks=[],
        random_seed=0,
    )
    result = runner.run_evolution()
    assert math.isfinite(result.best_fitness_ever_achieved)
    # softmax-likelihood is a probability in (0, 1)
    assert 0.0 < result.best_fitness_ever_achieved <= 1.0

    best_phenotype = algorithm.phenotype_decoder.build_phenotype_from_genome(
        result.best_genome_ever_found
    )
    logits = best_phenotype.forward_pass(features)
    assert logits.shape == (features.shape[0], _NUM_CLASSES)
    accuracy = ClassificationAccuracyEvaluator(
        features, labels
    ).evaluate_single_phenotype(best_phenotype)
    assert 0.0 <= accuracy <= 1.0
