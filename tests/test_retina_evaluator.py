from __future__ import annotations

import torch

from polyneat.evaluators.retina_evaluator import (
    MAXIMUM_RETINA_FITNESS,
    RetinaProblemEvaluator,
    build_all_retina_patterns,
    compute_expected_outputs,
)


class _ConstantPhenotype:
    def __init__(self, output_value: float) -> None:
        self._output_value = output_value

    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.full((input_tensor.shape[0], 2), self._output_value)

    def reset_recurrent_state(self) -> None:
        return None


class _PerfectPhenotype:
    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return compute_expected_outputs(input_tensor)

    def reset_recurrent_state(self) -> None:
        return None


def test_there_are_256_distinct_patterns_of_8_pixels() -> None:
    patterns = build_all_retina_patterns()
    assert patterns.shape == (256, 8)
    assert set(patterns.flatten().tolist()) == {0.0, 1.0}
    assert len({tuple(row.tolist()) for row in patterns}) == 256


def test_expected_outputs_are_binary_and_two_wide() -> None:
    expected = compute_expected_outputs(build_all_retina_patterns())
    assert expected.shape == (256, 2)
    assert set(expected.flatten().tolist()) == {0.0, 1.0}


def test_exactly_eight_of_sixteen_half_patterns_are_objects() -> None:
    # Kashtan & Alon: "three or more black pixels or one or two black pixels in
    # the left column only" selects 5 + 3 = 8 of the 16 patterns.
    patterns = build_all_retina_patterns()
    expected = compute_expected_outputs(patterns)
    left_half_to_answer = {
        tuple(row[:4].tolist()): float(out[0].item())
        for row, out in zip(patterns, expected, strict=True)
    }
    assert len(left_half_to_answer) == 16
    assert sum(left_half_to_answer.values()) == 8

    right_half_to_answer = {
        tuple(row[4:].tolist()): float(out[1].item())
        for row, out in zip(patterns, expected, strict=True)
    }
    assert len(right_half_to_answer) == 16
    assert sum(right_half_to_answer.values()) == 8


def test_three_or_more_black_pixels_always_make_an_object() -> None:
    patterns = build_all_retina_patterns()
    expected = compute_expected_outputs(patterns)
    for row, out in zip(patterns, expected, strict=True):
        if float(row[:4].sum().item()) >= 3.0:
            assert float(out[0].item()) == 1.0
        if float(row[4:].sum().item()) >= 3.0:
            assert float(out[1].item()) == 1.0


def test_one_or_two_black_pixels_count_only_in_the_outer_column() -> None:
    expected = compute_expected_outputs(
        torch.tensor(
            [
                # left block: [outer-top, outer-bottom, inner-top, inner-bottom]
                [1, 0, 0, 0, 0, 0, 0, 0],  # one black, outer column  -> object
                [1, 1, 0, 0, 0, 0, 0, 0],  # two black, outer column  -> object
                [0, 0, 1, 0, 0, 0, 0, 0],  # one black, inner column  -> no
                [1, 0, 1, 0, 0, 0, 0, 0],  # two black, mixed columns -> no
                [0, 0, 0, 0, 0, 0, 0, 1],  # right block, outer column -> object
                [0, 0, 0, 0, 1, 0, 0, 0],  # right block, inner column -> no
            ],
            dtype=torch.float32,
        )
    )
    assert [float(v) for v in expected[:, 0][:4]] == [1.0, 1.0, 0.0, 0.0]
    assert float(expected[4, 1]) == 1.0
    assert float(expected[5, 1]) == 0.0


def test_empty_half_is_never_an_object() -> None:
    expected = compute_expected_outputs(torch.zeros(1, 8))
    assert float(expected[0, 0]) == 0.0
    assert float(expected[0, 1]) == 0.0


def test_left_answer_ignores_the_right_half_and_vice_versa() -> None:
    # The property that makes the task modular.
    patterns = build_all_retina_patterns()
    expected = compute_expected_outputs(patterns)
    left_seen: dict[tuple[float, ...], float] = {}
    right_seen: dict[tuple[float, ...], float] = {}
    for row, out in zip(patterns, expected, strict=True):
        left_key, right_key = tuple(row[:4].tolist()), tuple(row[4:].tolist())
        assert left_seen.setdefault(left_key, float(out[0])) == float(out[0])
        assert right_seen.setdefault(right_key, float(out[1])) == float(out[1])


def test_the_two_hemispheres_use_mirror_symmetric_rules() -> None:
    # Reversing all eight pixels maps the left block onto the right one, outer
    # column onto outer column, so it must swap the two answers.
    patterns = build_all_retina_patterns()
    expected = compute_expected_outputs(patterns)
    mirrored_expected = compute_expected_outputs(torch.flip(patterns, dims=[1]))
    assert torch.equal(mirrored_expected[:, 0], expected[:, 1])
    assert torch.equal(mirrored_expected[:, 1], expected[:, 0])


def test_task_is_not_linearly_separable() -> None:
    # A single-layer perceptron cannot solve it, so the substrate must use its
    # hidden layer. Contradiction: w_a >= t and w_b + w_c < t while
    # w_b + w_c + w_d >= t and w_c + w_d < t cannot hold together.
    expected = compute_expected_outputs(
        torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],  # outer-top alone      -> object
                [0, 0, 1, 1, 0, 0, 0, 0],  # both inner           -> no
                [0, 1, 1, 0, 0, 0, 0, 0],  # outer-bottom + inner -> no
                [0, 1, 1, 1, 0, 0, 0, 0],  # three black          -> object
            ],
            dtype=torch.float32,
        )
    )
    assert [float(v) for v in expected[:, 0]] == [1.0, 0.0, 0.0, 1.0]


def test_perfect_network_scores_the_maximum() -> None:
    evaluator = RetinaProblemEvaluator()
    assert (
        abs(evaluator.evaluate_single_phenotype(_PerfectPhenotype()) - MAXIMUM_RETINA_FITNESS)
        < 1e-4
    )


def test_constant_network_scores_below_the_maximum_but_above_zero() -> None:
    evaluator = RetinaProblemEvaluator()
    fitness = evaluator.evaluate_single_phenotype(_ConstantPhenotype(0.5))
    assert 0.0 < fitness < MAXIMUM_RETINA_FITNESS


def test_fitness_never_goes_negative() -> None:
    evaluator = RetinaProblemEvaluator()
    assert evaluator.evaluate_single_phenotype(_ConstantPhenotype(-50.0)) >= 0.0


def test_fitness_never_exceeds_the_maximum() -> None:
    evaluator = RetinaProblemEvaluator()
    assert evaluator.evaluate_single_phenotype(_ConstantPhenotype(50.0)) <= MAXIMUM_RETINA_FITNESS


def test_batch_evaluation_preserves_order() -> None:
    evaluator = RetinaProblemEvaluator()
    scores = evaluator.evaluate_batch_of_phenotypes(
        [_ConstantPhenotype(0.5), _PerfectPhenotype(), _ConstantPhenotype(0.5)]
    )
    assert len(scores) == 3
    assert scores[1] > scores[0]
    assert scores[0] == scores[2]
