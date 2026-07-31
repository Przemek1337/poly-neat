from __future__ import annotations

import math
from pathlib import Path

import polyneat as pn
from examples.visual_discrimination import (
    hyperneat as visual_discrimination_example,
)
from polyneat.evaluators.visual_discrimination_evaluator import (
    VisualDiscriminationFitnessEvaluator,
    generate_visual_discrimination_trials,
)

_FIELD_SIDE = 5


def _small_config() -> pn.HyperNEATConfig:
    return pn.HyperNEATConfig(
        population_size=16,
        substrate_input_layer_size=_FIELD_SIDE * _FIELD_SIDE,
        substrate_hidden_layer_sizes=(),
        substrate_output_layer_size=_FIELD_SIDE * _FIELD_SIDE,
        substrate_node_activation_function="identity",
        random_seed=0,
    )


def test_example_config_loads_as_hyperneat_config():
    config_path = (
        Path(__file__).parent.parent
        / "examples"
        / "visual_discrimination"
        / "hyperneat.yaml"
    )
    config = pn.HyperNEATConfig.load_from_yaml_file(config_path)
    assert config.number_of_input_nodes == 4  # CPPN coordinate inputs
    assert config.substrate_input_layer_size == config.substrate_output_layer_size
    assert "gaussian" in config.available_activation_functions


def test_sandwich_substrate_has_two_equal_grids_and_bias():
    substrate = pn.build_grid_sandwich_substrate(
        input_grid_shape=(_FIELD_SIDE, _FIELD_SIDE),
        output_grid_shape=(_FIELD_SIDE, _FIELD_SIDE),
        coordinate_range_min=-1.0,
        coordinate_range_max=1.0,
        include_bias_node=True,
        output_sheet_shares_input_plane=True,
    )
    assert len(substrate.input_layer.nodes) == _FIELD_SIDE * _FIELD_SIDE
    assert len(substrate.output_layer.nodes) == _FIELD_SIDE * _FIELD_SIDE
    assert substrate.hidden_layers == ()
    assert substrate.bias_node is not None
    all_ids = [node.node_id for node in substrate.all_nodes()]
    assert all_ids == list(range(len(all_ids)))


def test_ideal_receptive_field_scores_near_perfect():
    """The paper's solution (accumulate activation in a local neighborhood) must
    be representable on this substrate and score near 1.0 - proving the task and
    fitness are set up so that HyperNEAT has a regularity to find."""
    import torch

    trial_fields, true_rows, true_cols = (
        generate_visual_discrimination_trials(
            field_side=_FIELD_SIDE,
            big_object_side=3,
            small_object_side=1,
            number_of_trials=40,
            seed=123,
        )
    )
    evaluator = VisualDiscriminationFitnessEvaluator(
        trial_fields, true_rows, true_cols, field_side=_FIELD_SIDE
    )

    class _IdealReceptiveField:
        def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
            batch = input_tensor.shape[0]
            fields = input_tensor.view(batch, _FIELD_SIDE, _FIELD_SIDE)
            outputs = torch.zeros(batch, _FIELD_SIDE * _FIELD_SIDE)
            for row in range(_FIELD_SIDE):
                for col in range(_FIELD_SIDE):
                    r_lo, r_hi = max(0, row - 1), min(_FIELD_SIDE, row + 2)
                    c_lo, c_hi = max(0, col - 1), min(_FIELD_SIDE, col + 2)
                    outputs[:, row * _FIELD_SIDE + col] = fields[
                        :, r_lo:r_hi, c_lo:c_hi
                    ].sum(dim=(1, 2))
            return outputs

        def reset_recurrent_state(self) -> None:
            return None

    reward = evaluator.evaluate_single_phenotype(_IdealReceptiveField())
    assert reward > 0.95


def test_short_run_completes_and_improves_over_generations():
    trial_fields, true_rows, true_cols = (
        generate_visual_discrimination_trials(
            field_side=_FIELD_SIDE,
            big_object_side=3,
            small_object_side=1,
            number_of_trials=40,
            seed=123,
        )
    )
    algorithm = visual_discrimination_example.build_visual_discrimination_algorithm(
        _small_config(), field_side=_FIELD_SIDE
    )
    evaluator = VisualDiscriminationFitnessEvaluator(
        trial_fields, true_rows, true_cols, field_side=_FIELD_SIDE
    )
    runner = pn.EvolutionRunner(
        algorithm=algorithm,
        fitness_evaluator=evaluator,
        termination_criterion=pn.MaxGenerationsTermination(max_generations=4),
        callbacks=[],
        random_seed=0,
    )
    result = runner.run_evolution()
    assert math.isfinite(result.best_fitness_ever_achieved)
    # reward is a normalized distance score in [0, 1]
    assert 0.0 <= result.best_fitness_ever_achieved <= 1.0
