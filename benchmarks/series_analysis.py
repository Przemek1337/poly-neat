"""Statistics behind the result-series notebook, kept out of the notebook so they are tested.

The notebook ``benchmarks/analiza_wynikow.ipynb`` only calls into this module,
renders tables and draws figures. Every number that goes into the thesis is
computed here.

The unit of replication is the **search seed**: five independent searches per
method and problem. Retrainings in track B are repeats *within* one search and
are aggregated per seed before any comparison between methods; treating fifteen
retrainings as fifteen independent samples would overstate the evidence.

Statistical choices, each stated once:

* Between methods, per problem: two-sided Mann-Whitney U on the five per-seed
  values, Holm-adjusted over all pairs of that problem (Holm, 1979), with the
  Vargha-Delaney A12 effect size (Vargha and Delaney, 2000), which is the
  recommended effect size for comparing randomized algorithms (Arcuri and
  Briand, 2011). Welch's t-test is reported as a secondary, parametric view.
* Track A against track B, per method: paired by search seed (the same
  topology), paired t-test and a t confidence interval of the mean difference,
  Holm-adjusted over the methods of a problem. The exact Wilcoxon signed-rank
  test is reported too, with the caveat that at n = 5 its smallest attainable
  two-sided p-value is 0.0625, so it can never reach 0.05.
* Within one seed, the uncertainty that comes from the finite test set: the
  bootstrap intervals the protocol already stored for AUROC, and a Wilson score
  interval for accuracy, which the multiclass protocol does not bootstrap.
* A track B retraining is ``diverged`` when the protocol recorded non-finite
  predictions for it, and ``collapsed`` when it finished but scores within
  ``COLLAPSE_MARGIN`` of chance. The margin is fixed here, before looking at the
  per-retraining values, so it cannot be tuned to the result.

References:
    Arcuri, A., Briand, L. (2011). A practical guide for using statistical tests
        to assess randomized algorithms in software engineering. ICSE 2011.
    Holm, S. (1979). A simple sequentially rejective multiple test procedure.
        Scandinavian Journal of Statistics, 6(2), 65-70.
    Vargha, A., Delaney, H. D. (2000). A critique and improvement of the CL
        common language effect size statistics of McGraw and Wong. Journal of
        Educational and Behavioral Statistics, 25(2), 101-132.
    Wilson, E. B. (1927). Probable inference, the law of succession, and
        statistical inference. JASA, 22(158), 209-212.
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from scipy import stats

from benchmarks.summarize_series import SERIES, SeriesSpec

METHOD_ORDER = ("deepneat", "exact", "random_search", "fixed_cnn")

# Categorical slots 1-4 of the reference palette, one fixed colour per method so
# a method looks the same in every figure. Validated on white: all checks pass;
# aqua and yellow sit below 3:1 contrast, so every figure names methods on its
# axis and the notebook prints the numbers as a table next to it.
METHOD_COLORS = {
    "deepneat": "#2a78d6",
    "exact": "#eb6834",
    "random_search": "#1baf7a",
    "fixed_cnn": "#eda100",
}

CHANCE_LEVEL = {"auroc": 0.5, "accuracy": 0.1}
COLLAPSE_MARGIN = 0.05

# Official test-set sizes. The frozen profiles run on the whole official split
# (maximum_test_samples: null), which ``load_series`` checks.
OFFICIAL_TEST_SIZE = {"MNIST": 10_000, "CIFAR-10": 10_000}

CONFIDENCE_LEVEL = 0.95


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Retraining:
    """One track B retraining of one selected topology."""

    retraining_seed: int
    value: float | None
    status: str  # "ok", "collapsed", "diverged" or "missing"


@dataclass(frozen=True)
class SeedResult:
    """Everything the analysis needs from one finished run."""

    method: str
    search_seed: int
    selection: float
    track_a: float
    retrainings: tuple[Retraining, ...]
    parameter_count: float
    run_directory: Path
    report: dict

    @property
    def track_b_mean(self) -> float | None:
        """Mean over retrainings that produced a score, as the protocol reports it."""
        values = [r.value for r in self.retrainings if r.value is not None]
        return statistics.mean(values) if values else None

    @property
    def track_b_median(self) -> float | None:
        values = [r.value for r in self.retrainings if r.value is not None]
        return statistics.median(values) if values else None


@dataclass(frozen=True)
class ProblemResults:
    """All runs of one problem's series, grouped by method in ``METHOD_ORDER``."""

    spec: SeriesSpec
    by_method: dict[str, list[SeedResult]]

    @property
    def methods(self) -> list[str]:
        return list(self.by_method)

    def track_a(self, method: str) -> list[float]:
        return [result.track_a for result in self.by_method[method]]


def _retrainings(report: dict, metric: str) -> tuple[Retraining, ...]:
    metric_values = report["summary"]["metric_values"]
    diverged = {
        entry["retraining_seed"] for entry in report.get("track_b_failed_retrainings", [])
    }
    retrainings = []
    for seed in report["effective_configuration"]["retraining_seeds"]:
        value = metric_values.get(f"track_b_seed{seed}_test_{metric}")
        if value is None:
            status = "diverged" if seed in diverged else "missing"
        elif value <= CHANCE_LEVEL[metric] + COLLAPSE_MARGIN:
            status = "collapsed"
        else:
            status = "ok"
        retrainings.append(Retraining(seed, value, status))
    return tuple(retrainings)


def load_series(results_root: Path, spec: SeriesSpec) -> ProblemResults:
    """Read every finished run of one series."""
    by_method: dict[str, list[SeedResult]] = {}
    series_directory = results_root / spec.series_directory
    for path in sorted(series_directory.glob("runs/*/seed_*/run_report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        configuration = report["effective_configuration"]
        if spec.problem in OFFICIAL_TEST_SIZE and configuration.get("maximum_test_samples"):
            raise ValueError(
                f"{path}: ran on a test subset, so OFFICIAL_TEST_SIZE does not apply"
            )
        metric_values = report["summary"]["metric_values"]
        seed = configuration["search_seed"]
        by_method.setdefault(report["method"], []).append(
            SeedResult(
                method=report["method"],
                search_seed=seed,
                selection=metric_values[spec.selection_metric],
                track_a=metric_values[f"track_a_seed{seed}_test_{spec.test_metric}"],
                retrainings=_retrainings(report, spec.test_metric),
                parameter_count=metric_values["selected_parameter_count"],
                run_directory=path.parent,
                report=report,
            )
        )
    ordered = {
        method: sorted(by_method[method], key=lambda result: result.search_seed)
        for method in sorted(by_method, key=_method_rank)
    }
    return ProblemResults(spec=spec, by_method=ordered)


def load_all(results_root: Path) -> dict[str, ProblemResults]:
    """Every configured series that has at least one finished run."""
    problems = {}
    for spec in SERIES:
        results = load_series(results_root, spec)
        if results.by_method:
            problems[spec.problem] = results
    return problems


def _method_rank(method: str) -> tuple[int, str]:
    return (METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER), method)


# --------------------------------------------------------------------- statistics


def mean_confidence_interval(
    values: list[float], level: float = CONFIDENCE_LEVEL
) -> tuple[float, float, float]:
    """Mean and its Student-t confidence interval; the interval is the mean alone for n < 2."""
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, mean, mean
    half_width = stats.t.ppf((1 + level) / 2, len(values) - 1) * statistics.stdev(values) / (
        math.sqrt(len(values))
    )
    return mean, mean - half_width, mean + half_width


def vargha_delaney_a12(first: list[float], second: list[float]) -> float:
    """Probability that a draw from ``first`` beats one from ``second``, ties counting half."""
    wins = sum((x > y) + 0.5 * (x == y) for x, y in itertools.product(first, second))
    return wins / (len(first) * len(second))


def a12_magnitude(a12: float) -> str:
    """Vargha-Delaney thresholds on the distance from 0.5."""
    distance = abs(a12 - 0.5)
    if distance < 0.06:
        return "pomijalny"
    if distance < 0.14:
        return "mały"
    if distance < 0.21:
        return "średni"
    return "duży"


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running_maximum = 0.0
    for rank, index in enumerate(order):
        running_maximum = max(running_maximum, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running_maximum)
    return adjusted


def wilson_interval(
    proportion: float, sample_size: int, level: float = CONFIDENCE_LEVEL
) -> tuple[float, float]:
    """Wilson score interval of a binomial proportion such as accuracy."""
    if not 0.0 <= proportion <= 1.0 or sample_size < 1:
        raise ValueError(f"need a proportion in [0, 1] and n >= 1, got {proportion}, {sample_size}")
    z = stats.norm.ppf((1 + level) / 2)
    denominator = 1 + z**2 / sample_size
    centre = (proportion + z**2 / (2 * sample_size)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / sample_size + z**2 / (4 * sample_size**2))
        / denominator
    )
    return centre - half_width, centre + half_width


def welch_difference_interval(
    first: list[float], second: list[float], level: float = CONFIDENCE_LEVEL
) -> tuple[float, float]:
    """Welch-Satterthwaite t interval of ``mean(first) - mean(second)``.

    A non-significant test is not evidence of equivalence; this interval says
    how large a difference the data still allow, which is what a claim such as
    "method X is not better than Y" has to be read against.
    """
    difference = statistics.mean(first) - statistics.mean(second)
    first_term = statistics.variance(first) / len(first)
    second_term = statistics.variance(second) / len(second)
    standard_error = math.sqrt(first_term + second_term)
    if standard_error == 0:
        return difference, difference
    degrees_of_freedom = (first_term + second_term) ** 2 / (
        first_term**2 / (len(first) - 1) + second_term**2 / (len(second) - 1)
    )
    half_width = stats.t.ppf((1 + level) / 2, degrees_of_freedom) * standard_error
    return difference - half_width, difference + half_width


@dataclass(frozen=True)
class MethodComparison:
    first: str
    second: str
    mean_difference: float
    difference_interval: tuple[float, float]
    a12: float
    mann_whitney_p: float
    holm_p: float
    welch_p: float

    @property
    def significant(self) -> bool:
        return self.holm_p < 1 - CONFIDENCE_LEVEL


def compare_methods(results: ProblemResults) -> list[MethodComparison]:
    """Every pair of methods on track A, Holm-adjusted within the problem."""
    pairs = list(itertools.combinations(results.methods, 2))
    raw = []
    for first, second in pairs:
        x, y = results.track_a(first), results.track_a(second)
        raw.append(
            (
                first,
                second,
                statistics.mean(x) - statistics.mean(y),
                welch_difference_interval(x, y),
                vargha_delaney_a12(x, y),
                # Enumerate label assignments: exact also when accuracy has ties.
                float(stats.mannwhitneyu(
                    x, y, alternative="two-sided",
                    method=stats.PermutationMethod(n_resamples=math.inf),
                ).pvalue),
                float(stats.ttest_ind(x, y, equal_var=False).pvalue),
            )
        )
    adjusted = holm_adjust([row[5] for row in raw])
    return [
        MethodComparison(first, second, difference, interval, a12, p, holm, welch)
        for (first, second, difference, interval, a12, p, welch), holm in zip(
            raw, adjusted, strict=True
        )
    ]


@dataclass(frozen=True)
class TrackComparison:
    method: str
    pairs: tuple[tuple[float, float], ...]  # (track A, track B mean) per seed with a track B
    seeds_without_track_b: int
    mean_difference: float  # track B minus track A
    difference_interval: tuple[float, float]
    seeds_where_b_is_higher: int
    paired_t_p: float | None
    holm_p: float | None
    wilcoxon_p: float | None


def compare_tracks(results: ProblemResults) -> list[TrackComparison]:
    """Track B against track A per method, paired by search seed, Holm across methods."""
    rows = []
    for method, seed_results in results.by_method.items():
        pairs = tuple(
            (result.track_a, result.track_b_mean)
            for result in seed_results
            if result.track_b_mean is not None
        )
        differences = [b - a for a, b in pairs]
        mean, lower, upper = mean_confidence_interval(differences)
        paired_t = wilcoxon = None
        if len(differences) >= 2 and statistics.pstdev(differences) > 0:
            paired_t = float(stats.ttest_rel([b for _, b in pairs], [a for a, _ in pairs]).pvalue)
            wilcoxon = float(stats.wilcoxon(differences).pvalue)
        rows.append(
            (
                method,
                pairs,
                len(seed_results) - len(pairs),
                mean,
                (lower, upper),
                sum(d > 0 for d in differences),
                paired_t,
                wilcoxon,
            )
        )
    testable = [row[6] for row in rows if row[6] is not None]
    adjusted = iter(holm_adjust(testable))
    return [
        TrackComparison(
            method=method,
            pairs=pairs,
            seeds_without_track_b=missing,
            mean_difference=mean,
            difference_interval=interval,
            seeds_where_b_is_higher=higher,
            paired_t_p=paired_t,
            holm_p=None if paired_t is None else next(adjusted),
            wilcoxon_p=wilcoxon,
        )
        for method, pairs, missing, mean, interval, higher, paired_t, wilcoxon in rows
    ]


def retraining_status_counts(results: ProblemResults) -> dict[str, dict[str, int]]:
    """Per method, how many track B retrainings were ok, collapsed, diverged or missing."""
    counts = {}
    for method, seed_results in results.by_method.items():
        method_counts = {"ok": 0, "collapsed": 0, "diverged": 0, "missing": 0}
        for result in seed_results:
            for retraining in result.retrainings:
                method_counts[retraining.status] += 1
        counts[method] = method_counts
    return counts


def official_test_interval(result: SeedResult, spec: SeriesSpec) -> tuple[float, float] | None:
    """Test-set sampling interval of one track A model.

    Bootstrap (stored by the protocol) for AUROC, Wilson for accuracy.
    """
    if spec.test_metric == "accuracy":
        return wilson_interval(result.track_a, OFFICIAL_TEST_SIZE[spec.problem])
    model_id = f"{result.method}_track_a_seed{result.search_seed}"
    interval = (
        result.report.get("official_test", {})
        .get(model_id, {})
        .get("confidence_intervals", {})
        .get(spec.test_metric)
    )
    if not interval or interval.get("lower_bound") is None:
        return None
    return interval["lower_bound"], interval["upper_bound"]


@dataclass(frozen=True)
class OperatingPoint:
    method: str
    search_seed: int
    threshold: float
    sensitivity: float
    specificity: float
    sensitivity_at_half: float
    specificity_at_half: float


def operating_points(results: ProblemResults) -> list[OperatingPoint]:
    """Track A sensitivity and specificity at the chosen threshold and at 0.5 (binary only)."""
    points = []
    for method, seed_results in results.by_method.items():
        for result in seed_results:
            model_id = f"{method}_track_a_seed{result.search_seed}"
            test = result.report["official_test"][model_id]
            chosen = test["metrics_at_selected_threshold"]
            reference = test["metrics_at_reference_threshold"]
            points.append(
                OperatingPoint(
                    method=method,
                    search_seed=result.search_seed,
                    threshold=result.report["frozen_models"][model_id]["threshold"],
                    sensitivity=chosen["sensitivity"],
                    specificity=chosen["specificity"],
                    sensitivity_at_half=reference["sensitivity"],
                    specificity_at_half=reference["specificity"],
                )
            )
    return points


def load_test_predictions(result: SeedResult) -> tuple[list[int], list[float]] | None:
    """Labels and positive-class probabilities of the track A model on the official test."""
    path = (
        result.run_directory
        / "predictions"
        / f"{result.method}_track_a_seed{result.search_seed}_official_test.json"
    )
    if not path.is_file():
        return None
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    return (
        [record["label"] for record in records],
        [record["positive_class_probability"] for record in records],
    )


# ------------------------------------------------------------------------ tables


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """A Markdown table; floats are printed with four decimals."""

    def cell(value: object) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def format_p(p_value: float | None) -> str:
    """p-values below 0.001 as ``<0.001``, otherwise three decimals."""
    if p_value is None:
        return "-"
    return "<0.001" if p_value < 0.001 else f"{p_value:.3f}"
