# PolyNEAT — Design Document

**Date:** 2026-06-30
**Status:** Approved (pending user review of this written form)
**Author:** Przemek (with brainstorming assistant)

---

## 1. Overview

PolyNEAT is a Python library for **neuroevolution algorithms**. The first
implemented algorithm is vanilla NEAT (Stanley & Miikkulainen, 2002), used as a
baseline. The library is the foundation for a master's thesis: additional
algorithms (variants of NEAT, and possibly deep neuroevolution methods) will be
added incrementally on top of the same abstractions.

The library prioritizes:

1. **Clean, composable abstractions** — algorithms are built by composing
   protocol-conforming components, not by inheriting from a deep hierarchy.
2. **Debug-friendly code** — explicit naming (no `x`, `g`, `mut`),
   per-step traceability, descriptive error messages.
3. **Reproducibility for thesis-quality experiments** — best-effort RNG
   seeding, run artifacts (config snapshot, best genome, topology renders)
   bundled per run.
4. **Extensibility** — adding a new algorithm should mean implementing a small
   set of protocols and composing existing components, not rewriting the core.

## 2. Goals

- Implement vanilla NEAT as the first concrete algorithm.
- Validate it on XOR (the standard NEAT baseline) using a `XORFitnessEvaluator`.
- Provide an abstraction surface that comfortably accommodates:
  - NEAT variants (HyperNEAT, ES-HyperNEAT, NEAT-LSTM, novelty search) —
    definitely planned.
  - Deep neuroevolution methods (Uber Deep GA, ES on fixed DNN topologies, PBT)
    — must remain possible without redesigning the core.
- Parallel fitness evaluation from MVP (multiprocessing via joblib).
- TensorBoard live monitoring + best-genome export + network topology
  visualization out of the box.
- Reproducible environments via `uv` + committed lockfile.

## 3. Non-goals (for v0.1.0)

- **Testing infrastructure beyond directory placeholder.** Tests are deferred;
  operators are still designed to be testable (pure functions with explicit
  RNG), but no test suite ships with MVP.
- **Bit-identical reproducibility from a single seed.** Best-effort only;
  thesis results will be averaged over multiple seeds (mean ± std).
- **Strict mypy / coverage gates.** Ruff format/lint only at start.
- **Persistent population checkpointing for resume after crash.** Only best
  genome is exported. Full population checkpointing is a future addition.
- **Structured per-generation logs (JSONL/CSV).** TensorBoard event files
  serve as the structured log; offline analysis happens by reading TB events.
- **CI pipeline, packaging to PyPI, public API stability guarantees.**
- **Distributed (multi-machine) parallelism.** Single-host multi-core only in
  MVP.

## 4. Foundational decisions

| Area | Decision |
|---|---|
| Language | Python 3.11 |
| Package management | `uv` + `pyproject.toml` + committed `uv.lock` |
| Project layout | `src/` layout |
| Phenotype backend | PyTorch (unified, GPU-capable from day one) |
| First algorithm | Vanilla NEAT (Stanley & Miikkulainen 2002) |
| Extensibility scope | NEAT variants (certain); deep neuroevolution (must be possible) |
| Fitness interface | `FitnessEvaluator.evaluate_batch_of_phenotypes` (batch-first) |
| First benchmark | XOR via `XORFitnessEvaluator` |
| Future benchmark | Gymnasium adapter (optional dependency) |
| Parallelism | `ParallelFitnessEvaluatorWrapper` (joblib) from MVP |
| Reproducibility | Best-effort global seed; thesis results averaged over N seeds |
| User API | `algorithm.advance_one_generation()` as foundation + `EvolutionRunner` with callbacks |
| Configuration | `@dataclass` configs with strict `from_yaml`/`from_dict` loaders |
| Output artifacts | Per-gen stats (console+TB) + best genome (JSON+pickle) + topology renders (PNG/SVG) |
| Architectural style | Protocols + Composition (no deep inheritance) |
| Tooling | `ruff` (format + lint); pytest/mypy deferred |

## 5. Architecture — Core abstractions

The library's core defines a small set of `Protocol`s. Algorithms compose
these; new algorithms = new compositions, not new class hierarchies.

### 5.1 Naming conventions

Followed throughout the codebase:

- Method names use **descriptive verbs**: `compute_*`, `apply_*`, `build_*`,
  `select_*`, `assign_*`. Generic verbs like `do`, `run`, `process` are avoided.
- Arguments use **full descriptive names**: `input_tensor` not `x`, `genome`
  not `g`, `parent_genome_a` not `a`.
- Dataclass fields use **full names**: `generation_number` not `gen`,
  `species_assignments` not `species`.
- All dataclasses get `__repr__` for free; non-dataclass classes define
  explicit `__repr__` listing field values.
- Every protocol/class has a docstring describing its **contract**:
  preconditions, postconditions, whether it mutates input.

### 5.2 Type aliases (`polyneat/core/type_aliases.py`)

```python
SpeciesId = int
InnovationId = int
FitnessValue = float
```

These prevent anonymous `int`/`float` in signatures; a reader of
`select_parents(..., n_parents: int)` immediately knows what `int` represents.

### 5.3 Protocols (`polyneat/core/protocols.py`)

```python
from typing import Protocol, runtime_checkable
from numpy.random import Generator
import torch

@runtime_checkable
class Genome(Protocol):
    """Immutable genotype. All operators return a new Genome instead of
    mutating in place — this makes it easy to log/diff before and after."""
    def clone_genome(self) -> "Genome": ...
    def to_serializable_dict(self) -> dict: ...
    @classmethod
    def from_serializable_dict(cls, payload: dict) -> "Genome": ...

@runtime_checkable
class Phenotype(Protocol):
    """Executable neural network built from a Genome."""
    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor: ...
    def reset_recurrent_state(self) -> None: ...   # no-op for feed-forward

@runtime_checkable
class PhenotypeBuilder(Protocol):
    """Maps Genome → Phenotype. Different builders for different encodings
    (direct NEAT, indirect HyperNEAT/CPPN, ...)."""
    def build_phenotype_from_genome(self, genome: Genome) -> Phenotype: ...

@runtime_checkable
class MutationOperator(Protocol):
    """A single mutation, as a pure function (deterministic given RNG).
    Separating operators from composition makes stack traces meaningful:
    you see `AddNodeMutation.apply_to_genome`, not generic `mutate`."""
    def apply_to_genome(self, genome: Genome, rng: Generator,
                        innovation_tracker: "InnovationTracker") -> Genome: ...

@runtime_checkable
class CrossoverOperator(Protocol):
    def apply_to_parents(self, fitter_parent: Genome, less_fit_parent: Genome,
                          rng: Generator) -> Genome: ...

@runtime_checkable
class ParentSelection(Protocol):
    def select_parents(self, candidate_genomes: list[Genome],
                       candidate_fitnesses: list[FitnessValue],
                       number_of_parents_to_select: int,
                       rng: Generator) -> list[Genome]: ...

@runtime_checkable
class Speciator(Protocol):
    """Assigns each genome to a species (used by NEAT and variants)."""
    def assign_genomes_to_species(self, genomes: list[Genome]) -> list[SpeciesId]: ...

@runtime_checkable
class FitnessEvaluator(Protocol):
    """Measures fitness. Batch is the primary interface (enables parallel
    and vectorized implementations)."""
    def evaluate_batch_of_phenotypes(self,
                                      phenotypes: list[Phenotype]
                                      ) -> list[FitnessValue]: ...

@runtime_checkable
class NeuroevolutionAlgorithm(Protocol):
    """The algorithm — composes the protocols above."""
    def create_initial_population(self, rng: Generator) -> "Population": ...
    def advance_one_generation(self,
                                current_population: "Population",
                                fitnesses_of_current_population: list[FitnessValue],
                                rng: Generator
                                ) -> tuple["Population", "GenerationStatistics"]: ...
    @property
    def phenotype_builder(self) -> PhenotypeBuilder: ...
```

### 5.4 Shared data types (`polyneat/core/population.py`, `statistics.py`)

```python
@dataclass(frozen=True)
class Population:
    genomes: list[Genome]
    species_assignments: list[SpeciesId] | None       # None when algorithm doesn't use speciation
    generation_number: int
    def size(self) -> int: ...

@dataclass(frozen=True)
class GenerationStatistics:
    generation_number: int
    best_fitness: FitnessValue
    mean_fitness: FitnessValue
    median_fitness: FitnessValue
    number_of_species: int | None
    number_of_genomes_evaluated: int
    elapsed_seconds: float
    extra_metrics: dict[str, float]                   # algorithm-specific extras
```

### 5.5 Design rationale

1. **Explicit `rng: Generator` on every stochastic operator.** Prepares the
   ground for component-level deterministic tests, and is good hygiene even
   under best-effort global seeding.
2. **`Population` as data, not behavior.** Structure is universal; the
   operations on populations are what differs between algorithms.
3. **Batch-only evaluator interface.** Forces batch-first thinking; sequential
   implementations can iterate internally.
4. **`Algorithm.advance_one_generation` returns `GenerationStatistics`.** The
   algorithm knows best what stats are meaningful; callbacks read them rather
   than introspecting the algorithm's internals.
5. **`PhenotypeBuilder` separate from `Genome` and `Phenotype`.** HyperNEAT
   reuses NEAT's `Genome` class but with a different builder (CPPN-substrate).
   Decoupling "what" from "how to interpret" is the key to NEAT variants.
6. **`@runtime_checkable`** allows `isinstance(x, MutationOperator)` for debug
   convenience.

## 6. NEAT implementation (`polyneat/algorithms/neat/`)

### 6.1 Genome (`genome.py`)

```python
@dataclass(frozen=True)
class NodeGene:
    node_id: int
    node_type: Literal["input", "hidden", "output", "bias"]
    activation_function_name: str            # "sigmoid", "tanh", "relu", ...

@dataclass(frozen=True)
class ConnectionGene:
    innovation_id: InnovationId
    source_node_id: int
    target_node_id: int
    weight: float
    is_enabled: bool

@dataclass(frozen=True)
class NEATGenome:
    """Direct encoding: tuple of nodes + tuple of connections."""
    node_genes: tuple[NodeGene, ...]
    connection_genes: tuple[ConnectionGene, ...]

    def clone_genome(self) -> "NEATGenome": ...
    def get_connection_by_innovation_id(self, innovation_id: InnovationId) -> ConnectionGene | None: ...
    def get_node_by_id(self, node_id: int) -> NodeGene | None: ...
    def to_serializable_dict(self) -> dict: ...
    @classmethod
    def from_serializable_dict(cls, payload: dict) -> "NEATGenome": ...
```

### 6.2 Innovation tracker (`innovation_tracker.py`)

```python
class InnovationTracker:
    """Issues globally-unique InnovationIds for new structural changes.
    Two genomes that get the same structural mutation in the same generation
    receive the same InnovationId (required for NEAT crossover to work)."""
    def get_or_assign_innovation_id_for_connection(
        self, source_node_id: int, target_node_id: int
    ) -> InnovationId: ...
    def reset_for_new_generation(self) -> None: ...
```

The tracker is an **instance** (not a global/static), so multiple parallel
runs don't share state.

### 6.3 Mutations (`mutations.py`)

Each mutation is its own class — stack traces show *which* mutation is
running, not a generic `mutate`.

```python
class AddNodeMutation:
    def __init__(self, probability_of_application: float): ...
    def apply_to_genome(self, genome: NEATGenome, rng: Generator,
                        innovation_tracker: InnovationTracker) -> NEATGenome: ...

class AddConnectionMutation: ...
class PerturbWeightsMutation: ...
class ToggleConnectionEnabledMutation: ...

class CompositeNEATMutation:
    """Applies the configured individual mutations in order (with probabilities)."""
    def __init__(self, individual_mutations: list[MutationOperator]): ...
    def apply_to_genome(self, genome, rng, innovation_tracker): ...
```

### 6.4 Crossover (`crossover.py`)

```python
class NEATCrossover:
    """Stanley's NEAT crossover using disjoint/excess/matching genes
    aligned by InnovationId."""
    def __init__(self, probability_of_inheriting_from_fitter_parent_for_matching_genes: float): ...
    def apply_to_parents(self, fitter_parent: NEATGenome,
                          less_fit_parent: NEATGenome,
                          rng: Generator) -> NEATGenome: ...
```

### 6.5 Speciation (`speciation.py`)

```python
class CompatibilityDistanceSpeciator:
    """Stanley's compatibility distance:
       distance = c1 * E / N + c2 * D / N + c3 * W̄
       where E=excess, D=disjoint, W̄=mean weight diff for matching genes,
       N=size of larger genome (or 1 for small genomes)."""
    def __init__(self,
                 coefficient_excess_c1: float,
                 coefficient_disjoint_c2: float,
                 coefficient_weight_difference_c3: float,
                 compatibility_threshold: float,
                 representative_genomes_from_previous_generation: list[NEATGenome] | None = None): ...
    def compute_compatibility_distance(self, genome_a: NEATGenome, genome_b: NEATGenome) -> float: ...
    def assign_genomes_to_species(self, genomes: list[NEATGenome]) -> list[SpeciesId]: ...
```

### 6.6 Parent selection (`selection.py`)

```python
class TournamentSelection:
    def __init__(self, tournament_size: int): ...
    def select_parents(self, candidate_genomes, candidate_fitnesses,
                       number_of_parents_to_select, rng): ...
```

### 6.7 Phenotype (`phenotype.py`)

```python
class TorchFeedForwardPhenotype(nn.Module):
    """Topologically-sorted forward pass built from a NEATGenome.
    Recurrent connections will be handled by a separate
    `TorchRecurrentPhenotype` class added later."""
    def __init__(self, neat_genome: NEATGenome, device: torch.device): ...
    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor: ...
    def forward(self, x):                        # nn.Module compatibility shim
        return self.forward_pass(x)
    def reset_recurrent_state(self) -> None:
        pass

class NEATPhenotypeBuilder:
    def __init__(self, device: torch.device = torch.device("cpu")): ...
    def build_phenotype_from_genome(self, genome: NEATGenome) -> TorchFeedForwardPhenotype: ...
```

`TorchFeedForwardPhenotype` deliberately subclasses `nn.Module` (even though
the `Phenotype` protocol only requires the interface) so it gets `.to(device)`,
`.state_dict()`, and `.parameters()` for free.

### 6.8 NEAT algorithm composition (`algorithm.py`)

```python
@dataclass
class NEATAlgorithm:
    config: NEATConfig
    mutation: MutationOperator
    crossover: CrossoverOperator
    parent_selection: ParentSelection
    speciator: Speciator
    innovation_tracker: InnovationTracker
    _phenotype_builder: NEATPhenotypeBuilder

    def create_initial_population(self, rng: Generator) -> Population: ...
    def advance_one_generation(self, current_population, fitnesses, rng): ...

    @property
    def phenotype_builder(self) -> PhenotypeBuilder:
        return self._phenotype_builder

    @classmethod
    def from_config(cls, config: NEATConfig,
                    device: torch.device = torch.device("cpu")) -> "NEATAlgorithm":
        """Default wiring of all components from config. Advanced users can
        construct NEATAlgorithm directly with custom components."""
        ...
```

## 7. Runner + callbacks (`polyneat/runner/`)

### 7.1 Callback protocol

```python
@runtime_checkable
class EvolutionCallback(Protocol):
    def on_run_started(self, context: "RunContext") -> None: ...
    def on_generation_started(self, context, population) -> None: ...
    def on_population_evaluated(self, context, population, fitnesses) -> None: ...
    def on_generation_completed(self, context, new_population, statistics) -> None: ...
    def on_new_best_genome_found(self, context, best_genome, best_fitness) -> None: ...
    def on_run_completed(self, context, final_result) -> None: ...

class BaseEvolutionCallback:
    """Convenience base with no-op defaults; subclass and override only what you need."""
    def on_run_started(self, context): pass
    # ... (all methods as no-op)
```

### 7.2 Context and result

```python
@dataclass
class RunContext:
    run_id: str
    run_started_at: datetime
    current_generation_number: int
    total_generations_planned: int
    history_of_generation_statistics: list[GenerationStatistics]
    current_best_genome: Genome | None
    current_best_fitness: FitnessValue | None
    configuration_snapshot: dict

@dataclass(frozen=True)
class EvolutionResult:
    final_population: Population
    best_genome_ever_found: Genome
    best_fitness_ever_achieved: FitnessValue
    full_generation_history: list[GenerationStatistics]
    total_runtime_seconds: float
    termination_reason: Literal["max_generations_reached", "target_fitness_reached",
                                  "stagnation_limit_reached", "manually_stopped"]
```

### 7.3 Termination

```python
@runtime_checkable
class TerminationCriterion(Protocol):
    def should_terminate_evolution(self, context: RunContext) -> bool: ...
    @property
    def termination_reason_label(self) -> str: ...

class MaxGenerationsTermination: ...
class TargetFitnessTermination: ...
class FitnessStagnationTermination: ...
class CompositeTermination:
    """OR of multiple criteria — terminates when any criterion fires."""
    def __init__(self, criteria: list[TerminationCriterion]): ...
```

### 7.4 Evolution runner

```python
class EvolutionRunner:
    """Convenience wrapper around algorithm.advance_one_generation().
    Users who need a custom loop should call advance_one_generation directly
    rather than wrap this runner."""
    def __init__(self,
                 algorithm: NeuroevolutionAlgorithm,
                 fitness_evaluator: FitnessEvaluator,
                 termination_criterion: TerminationCriterion,
                 callbacks: list[EvolutionCallback] | None = None,
                 random_seed: int | None = None): ...
    def run_evolution(self) -> EvolutionResult: ...
```

### 7.5 Built-in callbacks (`builtin_callbacks.py`)

- `ConsoleStatisticsLogger` — pretty per-generation stats (rich library).
- `TensorBoardLogger(log_directory, run_name=None)` — scalars per generation.
- `BestGenomePersister(output_directory, save_every_new_best=True,
  save_on_run_completed=True)` — incremental save of best genome (JSON + pickle).
- `NetworkTopologyVisualizer(output_directory, render_every_n_generations=None,
  render_on_run_completed=True)` — PNG/SVG topology renders.

## 8. Configuration (`polyneat/config/`)

### 8.1 Base config

```python
@dataclass
class AlgorithmConfig:
    population_size: int = 150
    number_of_input_nodes: int = 2
    number_of_output_nodes: int = 1
    random_seed: int | None = None
    device_for_phenotype_evaluation: str = "cpu"

    def validate(self) -> None: ...
    def __post_init__(self) -> None: self.validate()

    @classmethod
    def load_from_yaml_file(cls, yaml_file_path: Path) -> "AlgorithmConfig": ...
    @classmethod
    def from_dict(cls, data: dict) -> "AlgorithmConfig":
        """Strict — unknown keys raise ConfigurationError (catches typos)."""
        ...
    def save_to_yaml_file(self, yaml_file_path: Path) -> None: ...
    def to_dict(self) -> dict: ...
```

### 8.2 NEAT config

```python
@dataclass
class NEATConfig(AlgorithmConfig):
    # Mutation rates
    probability_of_add_node_mutation: float = 0.03
    probability_of_add_connection_mutation: float = 0.05
    probability_of_weight_perturbation: float = 0.8
    probability_of_weight_replacement: float = 0.1
    probability_of_toggle_connection_enabled: float = 0.01

    # Weight initialization & perturbation
    initial_weight_range_min: float = -1.0
    initial_weight_range_max: float = 1.0
    weight_perturbation_strength_sigma: float = 0.5

    # Speciation (compatibility distance)
    compatibility_distance_coefficient_excess_c1: float = 1.0
    compatibility_distance_coefficient_disjoint_c2: float = 1.0
    compatibility_distance_coefficient_weight_difference_c3: float = 0.4
    compatibility_distance_threshold: float = 3.0

    # Species management
    species_elitism_count: int = 1
    species_stagnation_generations_limit: int = 15
    minimum_species_size_for_elitism: int = 5

    # Crossover
    probability_of_crossover_vs_mutation_only: float = 0.75
    probability_of_inheriting_from_fitter_parent_for_matching_genes: float = 0.5

    # Selection
    tournament_size_for_parent_selection: int = 3

    # Activation functions
    available_activation_functions: tuple[str, ...] = ("sigmoid", "tanh", "relu")
    default_activation_function_for_hidden_nodes: str = "sigmoid"
    default_activation_function_for_output_nodes: str = "sigmoid"
```

### 8.3 Errors

```python
class ConfigurationError(ValueError):
    """Configuration is invalid. Message always names the field, the value,
    and the reason."""
```

### 8.4 Example YAML

```yaml
# experiments/xor_baseline.yaml
population_size: 150
number_of_input_nodes: 2
number_of_output_nodes: 1
random_seed: 42
probability_of_add_node_mutation: 0.05
probability_of_add_connection_mutation: 0.08
compatibility_distance_threshold: 3.0
species_stagnation_generations_limit: 20
```

## 9. End-to-end data flow for one generation

```
EvolutionRunner.run_evolution()
│
├─[0] (gen 0 only) algorithm.create_initial_population(rng)
│       creates minimal NEAT genomes: bias + inputs → outputs, no hidden
│
├─[1] callback hook: on_generation_started
│
├─[2] BUILD PHENOTYPES — for each genome, phenotype_builder.build_phenotype_from_genome
│       Each phenotype is a TorchFeedForwardPhenotype (nn.Module) on the
│       configured device.
│
├─[3] EVALUATE — fitness_evaluator.evaluate_batch_of_phenotypes(phenotypes)
│       ParallelFitnessEvaluatorWrapper splits phenotypes into chunks across
│       worker processes; each worker calls the wrapped evaluator on its
│       chunk and returns FitnessValues in original order.
│
├─[4] callback hook: on_population_evaluated
│
├─[5] ADVANCE GENERATION — algorithm.advance_one_generation
│       (a) speciation        — Speciator.assign_genomes_to_species
│       (b) shared fitness    — adjusted_fitness[i] = fitness[i] / |species_of(i)|
│       (c) offspring counts  — per species, proportional to mean adjusted fitness
│       (d) handle stagnation — kill species without improvement for N gens
│       (e) reproduce         — per species: elitism → crossover-or-mutation-only
│                                → mutate
│       (f) build new Population(genomes, species_assignments, generation+1)
│       (g) build GenerationStatistics with extra_metrics
│       (h) innovation_tracker.reset_for_new_generation()
│
├─[6] UPDATE CONTEXT — generation_number, history, best (if new_best)
│
├─[7] callback hooks:
│       if is_new_best: on_new_best_genome_found  (BestGenomePersister saves)
│       on_generation_completed                    (loggers log, possibly render)
│
├─[8] TERMINATION CHECK
│
└─ on_run_completed (final save, final render, summary log)
```

Key observations:

1. **Each step is a separable responsibility.** Bugs localize quickly.
2. **Phenotypes are built before evaluation** (step 2, not inside step 3) so
   parallel workers receive easy-to-pickle `nn.Module` objects, not
   builder dependencies.
3. **`is_new_best` triggers before `on_generation_completed`** — guarantees
   that `BestGenomePersister`'s file is up-to-date when other loggers run.
4. **`innovation_tracker.reset_for_new_generation()`** in step 5h ensures that
   identical structural mutations within one generation get the same
   `InnovationId` — required for NEAT crossover correctness.
5. **`extra_metrics: dict[str, float]`** keeps the `GenerationStatistics`
   contract stable while letting algorithms attach their own metrics.

## 10. Error handling

- **`ConfigurationError`** — raised at config construction; message names the
  field, the bad value, and why.
- **`InvalidGenomeError`** — raised when a structurally-invalid genome is
  produced (e.g. a connection referencing a missing node). Should never happen
  internally; defensive check at boundary of `from_serializable_dict`.
- **No silent corrections.** Failure mode depends on cause:
  - Mutation has **no valid choice** to make (e.g. `AddConnectionMutation`
    on a fully-connected graph) → return input genome unchanged, emit
    `DEBUG`-level log. This is a normal runtime condition, not an error.
  - Mutation would produce a **structurally-invalid genome** (e.g. introducing
    a cycle in a feed-forward network, dangling connection) → raise
    `InvalidGenomeError`. This indicates a bug in the operator or a
    contract violation by the caller.
- **Logger usage.** All components use `logging.getLogger(__name__)`; the
  application configures handlers. Default log level is `INFO` for
  user-facing messages, `DEBUG` for component traces.

## 11. Output artifact layout

Each run writes to `runs/<run_id>/`:

```
runs/<run_id>/
├── events.out.tfevents.*           # TensorBoard event file
├── config.yaml                     # snapshot of the configuration used
├── best_genome.json                # human-readable best genome
├── best_genome.pkl                 # exact-state best genome (for fast reload)
└── topology/
    ├── gen_0050_best.png
    ├── gen_0100_best.png
    └── final_best.svg
```

`run_id` is `f"{timestamp}_{short_uuid}"` so concurrent runs don't collide.

## 12. Project layout

```
poly-neat/
├── pyproject.toml
├── uv.lock                              # committed (reproducible env)
├── README.md
├── .gitignore                           # .venv/, runs/, __pycache__, *.pkl, *.egg-info
├── .python-version                      # "3.11"
│
├── src/polyneat/
│   ├── __init__.py                      # re-exports public API
│   │
│   ├── core/
│   │   ├── protocols.py
│   │   ├── population.py
│   │   ├── statistics.py
│   │   └── type_aliases.py
│   │
│   ├── algorithms/
│   │   └── neat/
│   │       ├── algorithm.py
│   │       ├── genome.py
│   │       ├── mutations.py
│   │       ├── crossover.py
│   │       ├── selection.py
│   │       ├── speciation.py
│   │       ├── innovation_tracker.py
│   │       └── phenotype.py
│   │
│   ├── nn/
│   │   ├── activations.py
│   │   └── topology.py                  # topo sort, cycle detection
│   │
│   ├── evaluators/
│   │   ├── base.py                      # SequentialFitnessEvaluator base
│   │   ├── parallel.py                  # ParallelFitnessEvaluatorWrapper
│   │   ├── xor.py                       # XORFitnessEvaluator
│   │   └── gymnasium_adapter.py         # (later, optional)
│   │
│   ├── config/
│   │   ├── base.py
│   │   ├── neat_config.py
│   │   └── errors.py
│   │
│   ├── runner/
│   │   ├── evolution_runner.py
│   │   ├── context.py
│   │   ├── termination.py
│   │   ├── callbacks.py
│   │   └── builtin_callbacks.py
│   │
│   ├── viz/
│   │   └── network_topology.py
│   │
│   └── utils/
│       ├── rng.py
│       └── serialization.py
│
├── examples/
│   ├── xor_baseline.py
│   └── xor_baseline.yaml
│
├── docs/superpowers/
│   ├── specs/
│   └── plans/
│
├── runs/                                # gitignored
│
└── tests/
    ├── __init__.py
    └── conftest.py                      # placeholder; pytest infra ready to extend
```

## 13. Dependencies (`pyproject.toml`)

```toml
[project]
name = "polyneat"
version = "0.1.0"
description = "Neuroevolution library for variable-topology and deep evolutionary algorithms"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "numpy>=1.26",
    "pyyaml>=6.0",
    "joblib>=1.3",
    "tensorboard>=2.15",
    "matplotlib>=3.8",
    "networkx>=3.2",
    "tqdm>=4.66",
    "rich>=13.7",
]

[project.optional-dependencies]
gymnasium = ["gymnasium>=0.29"]
dev = ["ruff>=0.4"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

## 14. First-run workflow

```bash
uv venv
uv pip install -e ".[dev]"
uv run python examples/xor_baseline.py
# output lands in runs/<run_id>/, with TensorBoard logs, best genome,
# and topology renders
```

Success criterion for MVP: `examples/xor_baseline.py` reaches `best_fitness ≥
3.95` (out of 4.0) on at least 3 of 5 random seeds within 300 generations.

## 15. Future work (out of scope for v0.1.0, but informed by this design)

- HyperNEAT — implements `NeuroevolutionAlgorithm` reusing `NEATGenome` with
  a different `PhenotypeBuilder` (CPPN-substrate decoding).
- Novelty search — reuses NEAT but replaces `FitnessEvaluator` with a
  novelty-archive-based one.
- Deep neuroevolution (e.g. Uber Deep GA) — new `Genome` (flat parameter
  vector), new `PhenotypeBuilder` (fixed-architecture DNN), reuses
  `EvolutionRunner` and most evaluator infrastructure.
- Population checkpointing — `PopulationPersister` callback; resume by
  loading population + innovation tracker state.
- pytest test suite — unit tests for operators (deterministic with seeded RNG)
  + integration smoke test (XOR convergence on N seeds) + property-based
  tests for genome invariants.
- Gymnasium integration via the `gymnasium` optional extra.

## 16. Open questions / risks

- **Joblib + PyTorch interaction.** Some `torch` versions misbehave when
  fork-pickled with CUDA tensors. Mitigation: CPU-only phenotypes for MVP
  (XOR baseline); revisit when introducing GPU-bound experiments.
- **Innovation tracker correctness across parallel workers.** Workers should
  receive immutable phenotype objects; mutation (which uses the tracker)
  happens *after* evaluation in the main process. The tracker is never
  shared across processes.
- **Pickle compatibility for `best_genome.pkl`.** Tied to Python/PyTorch
  versions. JSON is the canonical format; pickle is convenience only.

---

## 17. Approval and next steps

This design was developed through structured brainstorming on 2026-06-30.
After review of this written form, the next step is to invoke the
`superpowers:writing-plans` skill to produce an implementation plan under
`docs/superpowers/plans/`.
