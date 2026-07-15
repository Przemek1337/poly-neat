from __future__ import annotations

import math
from pathlib import Path

import polyneat as pn
from polyneat.evaluators.xor_evaluator import XORFitnessEvaluator

_EXAMPLE_CONFIG_PATH = (
    Path(__file__).parent.parent / "examples" / "xor_hyperneat.yaml"
)


def test_example_config_loads_as_hyperneat_config():
    config = pn.HyperNEATConfig.load_from_yaml_file(_EXAMPLE_CONFIG_PATH)
    assert config.number_of_input_nodes == 4
    assert config.substrate_input_layer_size == 2
    assert config.substrate_output_layer_size == 1


def test_short_hyperneat_run_completes_with_finite_fitness():
    config = pn.HyperNEATConfig(
        population_size=20,
        substrate_input_layer_size=2,
        substrate_hidden_layer_sizes=(3,),
        substrate_output_layer_size=1,
        random_seed=0,
    )
    algorithm = pn.HyperNEATAlgorithm.from_config(config)
    runner = pn.EvolutionRunner(
        algorithm=algorithm,
        fitness_evaluator=XORFitnessEvaluator(),
        termination_criterion=pn.MaxGenerationsTermination(max_generations=5),
        callbacks=[],
        random_seed=config.random_seed,
    )
    result = runner.run_evolution()
    assert math.isfinite(result.best_fitness_ever_achieved)
    # XOR fitness is in [0, 4]; a non-degenerate run scores strictly positive
    assert result.best_fitness_ever_achieved > 0.0
    # >= 5 rather than == 5: the runner is known to breed one generation past
    # termination (accepted debt, see docs/architecture.md "Known debts").
    assert len(result.full_generation_history) >= 5
