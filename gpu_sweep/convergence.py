"""Convergence curves, and the generation at which a run stops improving.

Every run stores its per-generation best fitness. Those curves answer the
question the sweep exists to ask about budgets: is five generations too few,
and if not, where does the curve flatten? A generation's best fitness can dip
below an earlier one, so "has it stopped improving" is asked of the running
best rather than of the raw curve.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

DEFAULT_PLATEAU_TOLERANCE_FRACTION = 0.01
"""Remaining improvement, as a fraction of the run's own total improvement,
below which the curve counts as flat.

An absolute tolerance does not work here. Fitness lives in ``[0, 1]`` but the
*scale of movement* differs by orders of magnitude between an easy dataset and
a hard one, so a fixed 1e-4 answers "still improving" for almost every run and
can never express the case that matters most: flat for twenty-five generations,
then a late jump. Measuring the remainder against the run's own improvement
makes the answer scale-free."""


def running_best_curve(generation_best_fitnesses: list[float]) -> list[float]:
    """Return the best-so-far value at each generation.

    Args:
        generation_best_fitnesses: Best fitness of each generation, in order.

    Returns:
        A non-decreasing curve of the same length.
    """
    best_so_far: list[float] = []
    current_best = float("-inf")
    for value in generation_best_fitnesses:
        current_best = max(current_best, float(value))
        best_so_far.append(current_best)
    return best_so_far


def find_plateau_generation(
    generation_best_fitnesses: list[float],
    *,
    tolerance_fraction: float = DEFAULT_PLATEAU_TOLERANCE_FRACTION,
) -> int | None:
    """Return the generation index after which fitness stops meaningfully improving.

    Works on the running best, because a generation's best can dip below an
    earlier one. The plateau is the first generation whose running best is
    already within ``tolerance_fraction`` of the run's *total* improvement from
    its final value - a scale-free test, so an easy dataset that climbs 0.4 and
    a hard one that climbs 0.02 are judged on the same footing.

    A run that never improved at all plateaus at generation 0.

    Args:
        generation_best_fitnesses: Best fitness of each generation, in order.
        tolerance_fraction: Share of the run's own improvement that counts as
            no further improvement.

    Returns:
        The 0-based generation index, or ``None`` for an empty curve. A return
        value equal to ``len(curve) - 1`` means the run was still improving
        when its budget ran out - see :func:`describe_plateau`.
    """
    best_so_far = running_best_curve(generation_best_fitnesses)
    if not best_so_far:
        return None
    total_improvement = best_so_far[-1] - best_so_far[0]
    if total_improvement <= 0.0:
        return 0
    absolute_tolerance = tolerance_fraction * total_improvement
    final_best = best_so_far[-1]
    for generation_index, value in enumerate(best_so_far):
        if final_best - value <= absolute_tolerance:
            return generation_index
    return len(best_so_far) - 1


def describe_plateau(
    curve: list[float],
    *,
    number_of_runs_at_final_generation: int,
    total_runs: int,
    tolerance_fraction: float = DEFAULT_PLATEAU_TOLERANCE_FRACTION,
) -> tuple[int | None, str]:
    """The single source of truth for what a figure and its caption both say.

    Both the figure title and its legend entry are derived from this one call,
    so a cell can never ship a picture whose title claims a plateau while its
    legend says the run was still improving.

    Args:
        curve: The mean curve being plotted.
        number_of_runs_at_final_generation: How many runs actually reached the
            last plotted generation.
        total_runs: How many runs the cell has.
        tolerance_fraction: Passed to :func:`find_plateau_generation`.

    Returns:
        ``(plateau_generation, caption)``. The caption warns when the mean
        curve was truncated, because a single short run drags the whole curve
        back and a plateau read off it would be an artefact of the truncation
        rather than a property of the search.
    """
    plateau_generation = find_plateau_generation(curve, tolerance_fraction=tolerance_fraction)
    if plateau_generation is None:
        return None, "no generations recorded"
    truncated = number_of_runs_at_final_generation < total_runs
    if truncated:
        return plateau_generation, (
            f"curve truncated to the shortest of {total_runs} runs "
            f"({len(curve)} generations); plateau not read"
        )
    if plateau_generation == len(curve) - 1:
        return plateau_generation, f"still improving at generation {plateau_generation}"
    return plateau_generation, f"plateau at generation {plateau_generation}"


def mean_curve(curves: list[list[float]]) -> list[float]:
    """Average several curves point-wise, truncated to the shortest one.

    Runs that ended early - a stagnation stop, a shorter per-class evolution -
    make the curves ragged. Truncating keeps every averaged point backed by the
    same number of runs, so the mean curve never jumps where a run drops out.

    Args:
        curves: One curve per run.

    Returns:
        The averaged curve, empty when there are no curves.
    """
    non_empty_curves = [curve for curve in curves if curve]
    if not non_empty_curves:
        return []
    shortest_length = min(len(curve) for curve in non_empty_curves)
    return [
        sum(curve[index] for curve in non_empty_curves) / len(non_empty_curves)
        for index in range(shortest_length)
    ]


def plot_cell_convergence(
    curves: list[list[float]],
    output_path: Path,
    *,
    title: str,
    tolerance_fraction: float = DEFAULT_PLATEAU_TOLERANCE_FRACTION,
) -> None:
    """Plot every run, its running best, and the mean, marking the plateau.

    Title and legend both come from one :func:`describe_plateau` call, so the
    two can never contradict each other. Nothing is written when there are no
    curves to draw.

    Args:
        curves: One per-generation best-fitness curve per run.
        output_path: Target ``.png`` file; parents are created.
        title: Figure title, normally ``"<dataset>/<algorithm>"``.
        tolerance_fraction: Passed to :func:`describe_plateau`.
    """
    non_empty_curves = [curve for curve in curves if curve]
    if not non_empty_curves:
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    pyplot.figure(figsize=(8, 5))
    # Per-run raw curves, faint - they show the spread and the dips.
    for curve in non_empty_curves:
        pyplot.plot(range(len(curve)), curve, color="#cde8f5", linewidth=0.8, alpha=0.7)
    # Per-run running best, still faint but solid - this is the quantity the
    # plateau is measured on, so it has to be visible. Drawing only the raw
    # curves would put the plateau line where nothing observable happens.
    for curve in non_empty_curves:
        best = running_best_curve(curve)
        pyplot.plot(range(len(best)), best, color="#8ecae6", linewidth=0.9, alpha=0.8)

    averaged = mean_curve([running_best_curve(curve) for curve in non_empty_curves])
    runs_reaching_final_generation = sum(
        1 for curve in non_empty_curves if len(curve) >= len(averaged)
    )
    pyplot.plot(
        range(len(averaged)),
        averaged,
        color="#023047",
        linewidth=2.0,
        label=f"mean running best of {len(non_empty_curves)} runs",
    )

    plateau_generation, caption = describe_plateau(
        averaged,
        number_of_runs_at_final_generation=runs_reaching_final_generation,
        total_runs=len(non_empty_curves),
        tolerance_fraction=tolerance_fraction,
    )
    if plateau_generation is not None and "not read" not in caption:
        pyplot.axvline(
            plateau_generation,
            color="#fb8500" if "still improving" in caption else "#d62828",
            linestyle="--",
            linewidth=1.2,
            label=caption,
        )
    elif plateau_generation is not None:
        pyplot.plot([], [], " ", label=caption)

    pyplot.title(f"{title} — {caption}")
    pyplot.xlabel("generation")
    pyplot.ylabel("best fitness")
    pyplot.legend(loc="lower right", fontsize=8)
    pyplot.grid(alpha=0.3)
    pyplot.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pyplot.savefig(output_path, format="png", dpi=150)
    pyplot.close()


def plot_dataset_convergence(
    mean_curve_by_algorithm: dict[str, list[float]],
    output_path: Path,
    *,
    dataset_key: str,
) -> None:
    """Plot one dataset's algorithms together, one mean running-best curve each.

    The three fitness formulas in this sweep (one-hot MSE, per-class indicator
    MSE, and L-NEAT's equation 1) share the range ``[0, 1]`` but not a meaning,
    so this figure is for comparing the *shape* of convergence - when a curve
    rises and when it flattens - and never the heights. The axis label says so.

    Args:
        mean_curve_by_algorithm: Averaged curve of each algorithm.
        output_path: Target ``.png`` file; parents are created.
        dataset_key: Dataset the figure covers.
    """
    drawable = {name: curve for name, curve in mean_curve_by_algorithm.items() if curve}
    if not drawable:
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    pyplot.figure(figsize=(8, 5))
    for algorithm_name, curve in sorted(drawable.items()):
        pyplot.plot(range(len(curve)), curve, linewidth=1.6, label=algorithm_name)
    pyplot.title(f"{dataset_key}: mean best fitness per generation")
    pyplot.xlabel("generation")
    pyplot.ylabel("best fitness so far (three different formulas — compare shape, not height)")
    pyplot.legend(fontsize=8)
    pyplot.grid(alpha=0.3)
    pyplot.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pyplot.savefig(output_path, format="png", dpi=150)
    pyplot.close()
