"""Plateau detection and the two convergence figures."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_sweep.convergence import (
    describe_plateau,
    find_plateau_generation,
    mean_curve,
    plot_cell_convergence,
    plot_dataset_convergence,
    running_best_curve,
)


def test_running_best_curve_never_decreases() -> None:
    assert running_best_curve([0.1, 0.5, 0.3, 0.4]) == [0.1, 0.5, 0.5, 0.5]


def test_find_plateau_generation_returns_where_improvement_stops() -> None:
    assert find_plateau_generation([0.1, 0.5, 0.9, 0.9, 0.9]) == 2


def test_find_plateau_generation_sees_through_a_dip() -> None:
    # generation 3 is worse than generation 2, but the best-so-far already
    # reached its final value at generation 2.
    assert find_plateau_generation([0.1, 0.5, 0.9, 0.4, 0.9]) == 2


def test_find_plateau_generation_reports_the_last_generation_when_still_climbing() -> None:
    curve = [0.1, 0.2, 0.3, 0.4]

    assert find_plateau_generation(curve) == len(curve) - 1


def test_find_plateau_generation_honours_the_tolerance_fraction() -> None:
    # Total improvement is 0.7002. At 1 percent the remaining 0.0002 counts as
    # flat, so the plateau is generation 1; at a millionth of a percent it does
    # not, so the run is still improving at the end.
    curve = [0.10, 0.80, 0.8001, 0.8002]

    assert find_plateau_generation(curve, tolerance_fraction=0.01) == 1
    assert find_plateau_generation(curve, tolerance_fraction=1e-8) == 3


def test_find_plateau_generation_is_scale_free() -> None:
    """The same shape scaled down by 100x must give the same answer.

    This is what an absolute tolerance cannot do, and it is the whole reason
    the tolerance is a fraction of the run's own improvement.
    """
    large = [0.0, 0.40, 0.80, 0.80, 0.80]
    small = [0.0, 0.004, 0.008, 0.008, 0.008]

    assert find_plateau_generation(large) == find_plateau_generation(small) == 2


def test_find_plateau_generation_of_a_flat_run_is_generation_zero() -> None:
    assert find_plateau_generation([0.5, 0.5, 0.5]) == 0


def test_describe_plateau_refuses_to_read_a_truncated_curve() -> None:
    """One short run drags the mean curve back; a plateau read off it is an artefact."""
    _, caption = describe_plateau(
        [0.1, 0.5, 0.9], number_of_runs_at_final_generation=1, total_runs=5
    )

    assert "truncated" in caption
    assert "not read" in caption


def test_describe_plateau_captions_a_settled_run() -> None:
    generation, caption = describe_plateau(
        [0.1, 0.5, 0.9, 0.9, 0.9], number_of_runs_at_final_generation=5, total_runs=5
    )

    assert generation == 2
    assert caption == "plateau at generation 2"


def test_describe_plateau_captions_a_run_still_climbing() -> None:
    _, caption = describe_plateau(
        [0.1, 0.2, 0.3], number_of_runs_at_final_generation=5, total_runs=5
    )

    assert "still improving" in caption


def test_find_plateau_generation_returns_none_for_an_empty_curve() -> None:
    assert find_plateau_generation([]) is None


def test_mean_curve_averages_point_wise_and_truncates_to_the_shortest() -> None:
    assert mean_curve([[0.0, 1.0, 2.0], [1.0, 2.0]]) == pytest.approx([0.5, 1.5])


def test_mean_curve_of_nothing_is_empty() -> None:
    assert mean_curve([]) == []


def test_plot_cell_convergence_writes_a_png(tmp_path: Path) -> None:
    output_path = tmp_path / "cell.png"

    plot_cell_convergence(
        [[0.1, 0.4, 0.6, 0.6], [0.2, 0.3, 0.7, 0.7]], output_path, title="tiny/neat"
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_cell_convergence_tolerates_having_no_curves(tmp_path: Path) -> None:
    output_path = tmp_path / "empty.png"

    plot_cell_convergence([], output_path, title="tiny/neat")

    assert not output_path.exists()


def test_plot_dataset_convergence_writes_a_png(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset.png"

    plot_dataset_convergence(
        {"neat": [0.1, 0.4, 0.6], "fsneat": [0.2, 0.3, 0.5]},
        output_path,
        dataset_key="tiny",
    )

    assert output_path.exists()
