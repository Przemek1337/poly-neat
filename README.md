# PolyNEAT

A Python library for **neuroevolution algorithms**, built around a full implementation of NEAT (Stanley & Miikkulainen, 2002). Variants subclass `NEATAlgorithm` and override only what they change, so the core is written once and shared.

Implemented algorithms:

| Algorithm | Class | Idea | Reference |
|---|---|---|---|
| NEAT | `NEATAlgorithm` | Evolves network topology and weights from a minimal start | Stanley & Miikkulainen, 2002 |
| FS-NEAT | `FSNEATAlgorithm` | Starts each genome with a single random input connection, so evolution selects the relevant input features | Whiteson et al., 2005 |
| HyperNEAT | `HyperNEATAlgorithm` | Evolves a CPPN that paints the weights of a fixed substrate as a function of geometry | Stanley, D'Ambrosio & Gauci, 2009 |
| NEAT-DBM | `NEATDBMAlgorithm` | Recombines each child's weights from three donor genomes, differential-evolution style | Stanovov et al., 2021 |
| C-NEAT | `CNEATAlgorithm` | Scores each organism on one class only and keeps a container of the best recognizer per class | Alfaham et al., 2024 |
| L-NEAT | `LNEATAlgorithm` | Interleaves evolution with Lamarckian backpropagation sessions on a fixed learning subset | Chen & Alahakoon, 2006 |

---

## Installation

Requires Python 3.11+. The recommended package manager is [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync                      # library + test dependencies
uv pip install -e ".[dev]"   # optional: ruff for linting
```

Alternatively with standard `pip`:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -e ".[dev]"
```

---

## Quick start

Examples live in `examples/<task>/<algorithm>.py`, each with a YAML config next to it. Run them as modules from the repository root:

```bash
uv run python -m examples.xor.neat            # NEAT on XOR
uv run python -m examples.iris.cneat          # C-NEAT on Iris
uv run python -m examples.mnist.hyperneat     # HyperNEAT on down-pooled MNIST
```

All ten examples:

| Task | NEAT | FS-NEAT | HyperNEAT | NEAT-DBM | C-NEAT | L-NEAT |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| `xor` | ✓ | ✓ | ✓ | ✓ | | |
| `iris` | | | | ✓ | ✓ | ✓ |
| `mnist` | ✓ | | ✓ | | | |
| `visual_discrimination` | | | ✓ | | | |

Artifacts (best genome as JSON and pickle, topology renders, TensorBoard event files) are written to `examples/<task>/artifacts/<algorithm>/`.

Every example exposes the same contract: a `CONFIG_FILE_PATH` constant and a `run_experiment(device, random_seed, artifacts_directory)` function returning an `ExperimentReport` (metrics, generation count, runtime). The registry of all runnable examples is `EXAMPLE_REGISTRY` in `examples/_experiment.py`.

---

## Datasets in the examples

Every example package has a `dataset.py` exposing a top-level `load_*` function that returns a named bundle — never a bare tuple you have to unpack by memory. The two classification datasets share one type, `ClassificationDataset` (in `examples/_datasets.py`), with `train_features`, `train_labels`, `test_features`, `test_labels`, `number_of_classes` and a `number_of_features` property.

```python
from examples.iris.dataset import load_iris
from examples.mnist.dataset import load_mnist

iris = load_iris(random_seed=42)                       # 99/51 split of the 150 UCI samples
mnist = load_mnist(random_seed=42, grid_side=7)        # 28x28 images average-pooled to 7x7
mnist_full = load_mnist(random_seed=42, grid_side=28,  # full-resolution, no subset caps
                        max_train_samples=None, max_test_samples=None)
```

Knobs are plain parameters: `train_fraction`, `grid_side` (any divisor of 28), `max_train_samples` / `max_test_samples`. The building blocks are public too — `load_mnist_features_and_labels(grid_side)` returns the whole 70,000-sample set as one `(features, labels)` pair, and `pool_features_to_grid(images, grid_side)` re-pools raw uint8 images by hand. Raw files download once into `examples/<task>/data/` and are cached.

The two synthetic tasks follow the same convention: `examples.xor.dataset.load_xor()` returns the four patterns, and `examples.visual_discrimination.dataset.load_visual_discrimination_trials(...)` returns a `VisualDiscriminationTrials` bundle.

---

## Benchmarks

`benchmarks/run_benchmark.py` runs one example several times with consecutive seeds and records the spread:

```bash
uv run python -m benchmarks.run_benchmark iris/cneat --repeats 5 [--cpu | --gpu] [--base-seed 0]
```

Each invocation writes one JSON document to `benchmarks/results/` holding every run, a mean/std summary, and the full YAML config that produced it — so after "edit the yaml, re-run, compare" the result file still records what produced it.

---

## Monitoring with TensorBoard

Examples that attach a `TensorBoardLogger` write event files to `examples/<task>/artifacts/<algorithm>/tensorboard/`:

```bash
uv run tensorboard --logdir examples/xor/artifacts/neat/tensorboard
# or point at everything at once:
uv run tensorboard --logdir examples
```

Logged scalars: `fitness/best`, `fitness/mean`, `fitness/median`, `population/num_species`, plus algorithm-specific metrics under `extra/*` (e.g. `extra/innovation_id_high_water_mark`). To log your own runs, add the callback to the runner:

```python
pn.TensorBoardLogger(log_directory=Path("runs/tensorboard"), run_name="my-run")
```

---

## Choosing the evaluation device (`--cpu` / `--gpu`)

All example scripts accept two mutually exclusive flags:

- `--cpu` - force phenotype evaluation on the CPU,
- `--gpu` - force CUDA; if CUDA is not available the script prints a clear
  error and exits with code 1 (there is deliberately **no** silent GPU→CPU
  fallback, so benchmark numbers are never quietly produced on the wrong
  device),
- no flag - the `device_for_phenotype_evaluation` value from the example's
  YAML config is used.

In library code the same mechanism is
`Algorithm.from_config(config, device_for_phenotype_computation=device)`.

---

## Project structure

The core of the library **is NEAT**: the full implementation lives in `polyneat/core/neat/`, and derived algorithms subclass `NEATAlgorithm`, overriding only what they change.

```
poly-neat/
├── polyneat/                    main library package
│   ├── __init__.py              public API - everything we export
│   ├── configs/                 config dataclasses, one subpackage per algorithm
│   ├── core/                    protocols, data types, and the NEAT core
│   │   └── neat/                full NEAT implementation + mutations
│   ├── algorithms/
│   │   ├── neat/                thin entry point (re-exports NEATAlgorithm)
│   │   ├── fsneat/              FS-NEAT initial population override
│   │   ├── hyperneat/           CPPN decoder, substrates, activation mutation
│   │   ├── neatdbm/             difference-based weight mutation
│   │   ├── cneat/               per-class containers and ensemble phenotype
│   │   └── lneat/               backpropagation trainer, trainable phenotype
│   ├── nn/                      activation functions + topology utilities
│   ├── evaluators/              fitness evaluators (sequential, parallel, per-task)
│   ├── runner/                  evolution loop, callbacks, termination criteria
│   ├── logging_utils/           custom colored logger
│   ├── viz/                     network topology rendering
│   └── utils/                   helpers (RNG, serialization)
├── examples/
│   ├── _datasets.py             shared dataset bundle + download/split helpers
│   ├── _example_cli.py          shared --cpu/--gpu CLI helper
│   ├── _experiment.py           example contract + registry
│   ├── xor/                     neat, fsneat, hyperneat, neatdbm
│   ├── iris/                    cneat, lneat, neatdbm
│   ├── mnist/                   neat, hyperneat
│   └── visual_discrimination/   hyperneat
├── benchmarks/                  repeat-runner + results
└── tests/
```

---

## What each module is responsible for

### `polyneat/configs/`

| File | Responsibility |
|---|---|
| `algorithm_config.py` | Base dataclass `AlgorithmConfig` - parameters shared by every algorithm (`population_size`, `number_of_input_nodes`, `random_seed`, etc.). Provides `load_from_yaml_file` and `from_dict` (strict - unknown keys raise `ConfigurationError`). |
| `configuration_errors.py` | `ConfigurationError` - raised when a configuration is invalid; the message always names the field, the value, and the reason. |
| `neat/neat_config.py` | `NEATConfig` - all NEAT hyperparameters: mutation probabilities, compatibility-distance coefficients, selection parameters. |
| `hyperneat/hyperneat_config.py` | `HyperNEATConfig(NEATConfig)` - substrate geometry, CPPN activation set, weight expression threshold. |
| `neatdbm/neatdbm_config.py` | `NEATDBMConfig(NEATConfig)` - difference-based mutation parameters. |
| `cneat/cneat_config.py` | `CNEATConfig(NEATConfig)` - number of class labels for the container. |
| `lneat/lneat_config.py` | `LNEATConfig(NEATConfig)` - learning interval, backpropagation session parameters, learning subset size. |

### `polyneat/core/`

The heart of the library. Defines the **protocols** (Go/Rust-style interfaces) every algorithm must implement.

| File | Responsibility |
|---|---|
| `component_protocols.py` | `@runtime_checkable Protocol`s: `Genome`, `Phenotype`, `PhenotypeDecoder`, `MutationOperator`, `CrossoverOperator`, `ParentSelection`, `Speciator`, `FitnessEvaluator`, `InitialPopulationStrategy`, `NeuroevolutionAlgorithm`, `InnovationTracker`. None of them is a base class - implementing the right methods is enough. |
| `population.py` | Frozen dataclass `Population(genomes, species_assignments, generation_number)`. Immutable - every generation creates a new object. |
| `generation_statistics.py` | Frozen dataclass `GenerationStatistics` - statistics of one generation (best and mean fitness, species count, timing). |
| `type_aliases.py` | Type aliases: `FitnessValue = float`, `InnovationId = int`, `SpeciesId = int`. They make signatures easier to read. |

### `polyneat/core/neat/`

The classic NEAT implementation. Every aspect of the algorithm lives in its own file. `polyneat/algorithms/` (below) holds the derived algorithms, each overriding only what it changes.

| File | Responsibility |
|---|---|
| `neat_genome.py` | Frozen dataclasses `NodeGene` and `ConnectionGene`, plus `NEATGenome`. A genome cannot be modified in place - mutation always returns a new object. `__post_init__` validates that there are no duplicate nodes or innovations. |
| `global_innovation_tracker.py` | Assigns global `InnovationId`s to new connections. Within one generation the same `(source, target)` pair gets the same id (deduplication), which lets crossover align structurally identical genes. After each generation the dedup table is cleared, but the counter is never reset. |
| `initial_population.py` | Generation-0 strategies: `fully_connected` (vanilla NEAT minimal start) and `fs_neat` (single random input→output connection per genome), plus a name→strategy registry extensible via `register_initial_population_strategy`. |
| `mutations/add_node_mutation.py` | Picks one enabled connection, disables it, inserts a new hidden node, adds two new connections: input→node (weight 1.0) and node→output (the original weight). |
| `mutations/add_connection_mutation.py` | Tries to add a new connection between unconnected nodes. Checks for duplicates, self-loops, cycles (BFS from the target - if the source is reachable, a cycle exists) and topological constraints (input nodes cannot be targets, output nodes cannot be sources). |
| `mutations/weight_modification_mutation.py` | Per connection: with probability `p_perturb` adds Gaussian noise, with probability `p_replace` redraws the weight from the initialization range. |
| `mutations/toggle_connection_enabled_mutation.py` | Flips the `is_enabled` flag of a random connection. Before re-enabling a disabled connection it checks that no cycle is created - a key fix, because `AddConnectionMutation` checks cycles only on the currently enabled connections. |
| `mutations/composite_neat_mutation.py` | Applies the mutations in order: WeightModification → AddConnection → AddNode → Toggle. Each mutation is optional (has its own probability). |
| `neat_crossover.py` | Crossover aligned by `innovation_id`. Matching genes (both parents share the id) are inherited randomly with the configured probability. Non-matching genes (disjoint/excess) come from the fitter parent. After assembling the child, `_resolve_enabled_connection_cycles` removes any cycles - they can appear when a gene was disabled in one parent and the re-enable chance closes a loop. |
| `compatibility_distance_speciator.py` | Computes the compatibility distance δ = c₁·E/N + c₂·D/N + c₃·W̄ (excess/disjoint genes + weight difference) and assigns genomes to species by comparison with each species' representative. After every speciation pass the representative of each species is **resampled** from the current members (as in the original paper; a frozen representative caused artificial species fragmentation). |
| `tournament_parent_selection.py` | Tournament: samples `tournament_size` individuals, returns the best. Repeats for every requested parent. |
| `torch_feedforward_phenotype.py` | The neural network as a PyTorch `nn.Module`. At construction it computes a topological order of the nodes (Kahn's algorithm); `forward_pass` then iterates the nodes in that order, sums weighted inputs and applies the activation function. Supports batched inputs `[batch, n_inputs]`. |
| `neat_phenotype_decoder.py` | Builds a `TorchFeedForwardPhenotype` from a `NEATGenome`. Passes the device (CPU/GPU) through. |
| `neat_algorithm.py` | `NEATAlgorithm` - the main class. `from_config()` builds every component from `NEATConfig` via overridable `_build_*` factory methods (template method - subclasses replace only what they change). `create_initial_population()` builds minimal genomes (inputs + bias → outputs). `advance_one_generation()` runs the full cycle: speciation → adjusted fitness → offspring allocation (proportional to the **sum** of each species' adjusted fitness) → elitism → survival threshold (only the best 20% of a species become parents) → reproduction (with rare interspecies mating, p=0.001) → innovation tracker reset. |

### `polyneat/algorithms/`

| Package | Responsibility |
|---|---|
| `neat/` | Thin entry point: re-exports `NEATAlgorithm` from `polyneat/core/neat/`. |
| `fsneat/` | `FSNEATAlgorithm(NEATAlgorithm)` - overrides only `create_initial_population`; every genome starts with a single random input→output connection, so evolution itself selects the relevant input features. Any `initial_population_strategy` from YAML is deliberately ignored. |
| `hyperneat/` | `HyperNEATAlgorithm` evolves CPPNs; `substrate.py` builds substrates (`build_grid_sandwich_substrate` for two-sheet sandwiches); `HyperNEATPhenotypeDecoder` queries the CPPN for every substrate connection and thresholds the result; `AddNodeWithRandomActivationMutation` gives new CPPN nodes a random activation from the configured set. |
| `neatdbm/` | `NEATDBMAlgorithm` - after standard reproduction, `DifferenceBasedWeightMutation` recombines each child's weights from three donors at shared innovation ids. |
| `cneat/` | `CNEATAlgorithm` - organisms are scored on one assigned class; `ClassGenomeContainer` keeps the best recognizer per class; `ContainerEnsemblePhenotype` classifies by argmax over the container networks; progress/update callbacks maintain the container during the run. |
| `lneat/` | `LNEATAlgorithm` - every `learning_interval_generations`, non-Type-1 offspring get a backpropagation session (`BackpropagationWeightTrainer`) on a fixed learning subset, and trained weights are inherited (Lamarckian). `TrainableTorchFeedForwardPhenotype` makes the phenotype's weights torch parameters; `RecognizerEnsemblePhenotype` assembles per-class recognizers into an argmax ensemble. |

### `polyneat/nn/`

| File | Responsibility |
|---|---|
| `activation_functions.py` | `sigmoid`, `steepened_sigmoid`, `tanh`, `relu`, `leaky_relu`, `identity` as PyTorch callables. Dictionary `ACTIVATION_FUNCTION_NAME_TO_CALLABLE` + `resolve_activation_function_by_name(name)` which raises `ConfigurationError` for unknown names. |
| `topology_utilities.py` | `compute_topological_order_of_node_ids` - Kahn's algorithm, raises `ValueError` on a cycle. `would_directed_edge_create_cycle` - BFS from the candidate target; if the source is reachable, the edge would create a cycle. |

### `polyneat/evaluators/`

| File | Responsibility |
|---|---|
| `sequential_evaluator_base.py` | `SequentialFitnessEvaluator` - base class for evaluating one phenotype at a time. Overriding `evaluate_single_phenotype` is enough. |
| `parallel_evaluator_wrapper.py` | `ParallelFitnessEvaluatorWrapper` wraps any evaluator and evaluates in parallel via `joblib`. Defaults to `prefer="threads"`. |
| `class_indexed_evaluator_base.py` | Base for evaluators that score an organism against one assigned class label (used by C-NEAT). |
| `xor_evaluator.py` | `XORFitnessEvaluator` - evaluates a phenotype on the four XOR patterns. Fitness = Σ(1 − (expected − actual)²), max = 4.0. Solved-XOR threshold: ≥ 3.95. |
| `xor_with_distractors_evaluator.py` | XOR plus noise distractor inputs - the FS-NEAT feature-selection benchmark. |
| `classification_accuracy_evaluator.py` | `ClassificationAccuracyEvaluator` - fraction of samples whose argmax output matches the label. |
| `softmax_likelihood_evaluator.py` | `SoftmaxLikelihoodFitnessEvaluator` - mean softmax probability of the correct class; a smooth training signal for many-class problems. |
| `binary_recognizer_evaluator.py` | Scores a single-output recognizer network for one target class (L-NEAT's per-class runs). |
| `multiclass_dataset_evaluator.py` | Scores each organism only on its assigned class over a labelled dataset (C-NEAT's training signal). |
| `visual_discrimination_evaluator.py` | Trial generator + evaluator for locating the larger of two squares in a 2-D field (Stanley et al. 2009, section 4). |

### `polyneat/runner/`

| File | Responsibility |
|---|---|
| `evolution_runner.py` | `EvolutionRunner` - the main loop: builds phenotypes → evaluates fitness → tracks the best genome → calls `advance_one_generation` → checks the termination criterion. Returns an `EvolutionResult`. |
| `run_context.py` | `RunContext` - the current run state (id, start time, generation number, statistics history, best genome so far). Passed to all callbacks. |
| `termination_criteria.py` | `MaxGenerationsTermination`, `TargetFitnessTermination`, `FitnessStagnationTermination`, `CompositeTermination` (OR logic). |
| `evolution_callback_protocol.py` | The `EvolutionCallback` protocol with six hooks: `on_run_started`, `on_generation_started`, `on_population_evaluated`, `on_generation_completed`, `on_new_best_genome_found`, `on_run_completed`. `BaseEvolutionCallback` provides empty default implementations. |
| `builtin_evolution_callbacks.py` | `ConsoleStatisticsLogger` (rich table), `TensorBoardLogger`, `BestGenomePersister` (JSON + pickle), `NetworkTopologyVisualizer`. |

### `polyneat/logging_utils/`

| File | Responsibility |
|---|---|
| `custom_logger.py` | The only allowed path to a logger: `get_logger(__name__)`. Registers `CustomLogger` as the logger class via `logging.setLoggerClass` at import time. Every logger attaches its own handler - there is no propagation. |
| `colored_level_formatter.py` | `ColoredLevelFormatter` - colors the message body (not the whole line) with `colorama` colors. DEBUG=blue, INFO=green, WARNING=yellow, ERROR=red, CRITICAL=dark red. |
| `logging_config.py` | `LoggingConfig` - log level, message format, optional directory for file logs. |

### `polyneat/viz/`

`network_topology_renderer.py` - `render_genome_topology(genome, output_path)`. Uses `matplotlib` and `networkx`; `matplotlib.use("Agg")` guarantees it works without a display (server, CI). Input/bias/hidden/output nodes get different colors; disabled connections are dashed.

### `polyneat/utils/`

| File | Responsibility |
|---|---|
| `random_generator_factory.py` | `create_seeded_random_generator(seed)` - returns a `numpy.random.Generator`. One function, one point of seed control. |
| `artifact_serialization.py` | `save_as_json`, `load_from_json`, `save_as_pickle`, `load_from_pickle`. |

---

## How NEAT works - the algorithm step by step

### Where the implementation comes from

NEAT is based on the original publication:

> Stanley, K. O. & Miikkulainen, R. (2002). **Evolving Neural Networks through Augmenting Topologies**. *Evolutionary Computation*, 10(2), 99–127.

The article is freely available on the author's website. Every implementation decision in the code (the compatibility distance formula, the crossover rules, offspring allocation) traces back to this document.

---

### The problem NEAT solves

Classic neural network evolution has three fundamental problems:

**1. The competing conventions problem**
If evolution independently discovers the same hidden node in two genomes, they are structurally identical but the nodes carry different numbers. Crossing over such genomes produces offspring with duplicated nodes - a mess.

*NEAT's solution:* every new structural edge gets a global `InnovationId`. If the same edge `(A→B)` appears in several genomes within the same generation, they all get the same id. Crossover aligns genes by `InnovationId`, not by index.

**2. The problem of protecting structural innovation**
A freshly added hidden node perturbs the network - fitness drops. Without protection, a lineage carrying a topological innovation dies before it can optimize it.

*NEAT's solution:* speciation. Structurally similar genomes (small compatibility distance) form one species. Selection happens within species, and fitness is divided by the species size (shared fitness). Each species competes with itself.

**3. The problem of minimal dimensionality**
Randomly initialized networks have as much freedom as large ones but are harder to evolve. Large weight spaces mean slow convergence.

*NEAT's solution:* start from a minimal network (inputs + bias → outputs, no hidden nodes). Topological complexity grows gradually through the `AddNode` and `AddConnection` mutations.

---

### Genome encoding

```
NodeGene:
  node_id              : int       - unique identifier
  node_type            : str       - "input" | "hidden" | "output" | "bias"
  activation_function  : str       - "sigmoid" | "tanh" | "relu"

ConnectionGene:
  innovation_id        : int       - global innovation number
  source_node_id       : int
  target_node_id       : int
  weight               : float
  is_enabled           : bool      - AddNodeMutation disables the split connection
```

`NodeGene`, `ConnectionGene` and `NEATGenome` are all **frozen dataclasses** - no operator modifies an existing genome. Mutation always returns a new object.

---

### One generation cycle (`advance_one_generation`)

```
Population t
    │
    ▼
[1] Speciation
    CompatibilityDistanceSpeciator compares every genome
    with each species' representative.
    δ = c₁·E/N + c₂·D/N + c₃·W̄
      E - excess genes
      D - disjoint genes
      W̄ - mean weight difference of matching genes
      N - normalization (size of the larger genome)
    If δ < threshold → same species.
    After assignment the representative of each species is
    resampled from the members of the current generation.
    │
    ▼
[2] Adjusted fitness
    For each individual i in species s (size |s|):
    adjusted_fitness[i] = raw_fitness[i] / |s|
    (fitness is "shared" across the species)
    │
    ▼
[3] Stagnation
    If a species has not improved its best raw fitness
    for species_stagnation_generations_limit generations → removed.
    The species containing the globally best genome always survives.
    │
    ▼
[4] Offspring allocation
    Each species receives offspring_slots proportional to the
    SUM of its members' adjusted fitness (= the species' mean raw
    fitness - as in the original paper). Largest-remainder
    rounding, so the slot total equals population_size.
    │
    ▼
[5] Elitism
    If a species has ≥ minimum_species_size_for_elitism members,
    its top species_elitism_count genomes pass through unchanged.
    │
    ▼
[6] Reproduction
    Survival threshold: only the best
    species_survival_fraction_for_reproduction (20%) of a species
    can become parents, minimum 2 individuals.
    For each open slot:
      - 75% chance: crossover of two tournament-selected parents;
        with probability probability_of_interspecies_mating
        (0.001) the second parent comes from the whole population
        (interspecies mating)
      - 25% chance: cloning a single parent
      Result → composite mutations
    │
    ▼
[7] Innovation tracker reset (fresh dedup table for the new generation)
    │
    ▼
Population t+1
```

---

### Crossover

Crossover aligns genes by `InnovationId`:

```
Parent 1 (fitter): [1][2][3][ ][5][6][ ][8]
Parent 2 (worse):  [1][2][ ][4][ ][6][7][ ]
                    ↑  ↑        ↑
                 matching    matching
                  (both)      (both)

Matching genes (1,2,6): a copy is drawn from the fitter or the worse parent
Disjoint genes (3,5,8 - only in the fitter): inherited from the fitter
Disjoint genes (4,7 - only in the worse): discarded
```

If a gene was disabled in either parent, the offspring has a 75% chance of inheriting it disabled (protection against topological chaos). After the whole child genome is assembled, `_resolve_enabled_connection_cycles` runs - it removes any cycles produced by the combination of enable states.

---

### Configuration

Every hyperparameter lives in the example's YAML file (`examples/xor/neat.yaml` for the XOR baseline) and maps 1:1 onto a `NEATConfig` field. The defaults follow the original paper where it specifies values (population 150, compatibility coefficients 1.0/1.0/0.4, threshold 3.0, interspecies mating 0.001) and are commented where they deviate. Unknown keys raise `ConfigurationError`, so a typo cannot silently fall back to a default.

---

## The XOR benchmark

### Why XOR

XOR (exclusive or) is **the standard NEAT benchmark** from the original paper. It is useful for several reasons:

- It is **not linearly separable** - a network without hidden nodes cannot solve it. The algorithm must evolve the right topology.
- It is **trivially verifiable** - 4 input patterns, zero ambiguity.
- Stanley and Miikkulainen used XOR to validate NEAT in 2002 - it provides a reference point.

| Input | Expected output |
|---|---|
| (0, 0) | 0 |
| (0, 1) | 1 |
| (1, 0) | 1 |
| (1, 1) | 0 |

### Fitness function

```python
fitness = sum(1.0 - (expected - actual)²)   # over each of the 4 patterns
```

**Maximum = 4.0** (all patterns perfect).  
**Solved threshold: ≥ 3.95**.

Squared error is used instead of absolute error because absolute error gives a flat gradient around the "3 patterns correct" local optimum (fitness = 3.0 no matter how badly the network misses the fourth pattern). Squared error penalizes large errors more and small errors less, so there is a gradient encouraging the network to reduce the error on the fourth pattern: a network that outputs 0.5 for pattern (1,1) gets fitness 3.75, not 3.5.

Configuration file: `examples/xor/neat.yaml`.

### Success criterion - a note on comparability with the original paper

The original paper counts a network as solving XOR when **all four outputs land on the correct side of 0.5** (correct classification). The `fitness ≥ 3.95` threshold with squared error is **much stricter**: it requires each output to be on average within ~0.11 of its target. A network the paper would count as a solution (e.g. outputs 0.3/0.7/0.7/0.3) has fitness of only 3.64 here. Generation counts measured against the two criteria are therefore not comparable - both are reported (helper: `XORFitnessEvaluator.classifies_all_patterns_correctly`).

### Benchmark results

With the shipped configuration, NEAT solves XOR reliably:

| Seed | Best fitness | Generations to fitness ≥ 3.95 | Generations to the paper's criterion |
|---:|---:|---:|---:|
| 0 | 3.9831 | 31 | 17 |
| 1 | 3.9779 | 23 | 17 |
| 2 | 3.9931 | 34 | 25 |
| 7 | 3.9945 | 39 | 29 |
| 13 | 3.9701 | 31 | 25 |
| 42 | 3.9819 | 33 | 10 |
| 100 | 3.9859 | 28 | 28 |
| 200 | 3.9600 | 26 | 15 |
| 300 | 3.9790 | 26 | 26 |
| 400 | 3.9756 | 60 | 57 |

**10/10 seeds; on average 33.1 generations to fitness ≥ 3.95 and 24.9 generations to the paper's criterion (the paper reports an average of 32).**

An earlier version of the implementation needed ~165 generations on average to reach fitness ≥ 3.95. The speedup comes from (ablation on the same 10 seeds):

1. **Survival threshold + interspecies mating + representative resampling + corrected offspring allocation** - ~165 → 60.1 generations. The dominant contribution is the survival threshold (selection pressure); the resampling and allocation fixes alone do not change the pace on XOR, but they remove the species fragmentation that matters in longer runs.
2. **Steepened sigmoid (4.9) instead of the standard one** - 60.1 → 33.1 generations.

---

## Using the library

### Logger - the only allowed path

```python
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)
logger.info("Message")
logger.debug("Debug with %s", "an argument")
```

Never `logging.getLogger` directly. Configure once at startup:

```python
import logging
from polyneat import LoggingConfig, set_logging_config

set_logging_config(LoggingConfig(
    log_level=logging.DEBUG,
    file_log_directory="runs/logs",  # None = no file logs
))
```

### A custom fitness function

```python
from polyneat.evaluators.sequential_evaluator_base import SequentialFitnessEvaluator
from polyneat.core.component_protocols import Phenotype
from polyneat.core.type_aliases import FitnessValue

class MyEvaluator(SequentialFitnessEvaluator):
    def evaluate_single_phenotype(self, phenotype: Phenotype) -> FitnessValue:
        import torch
        outputs = phenotype.forward_pass(torch.tensor([[1.0, 0.0]]))
        return float(outputs[0, 0].item())
```

### A full experiment

```python
from pathlib import Path
import polyneat as pn
from polyneat.evaluators.xor_evaluator import XORFitnessEvaluator

config = pn.NEATConfig.load_from_yaml_file(Path("examples/xor/neat.yaml"))
algorithm = pn.NEATAlgorithm.from_config(config)

runner = pn.EvolutionRunner(
    algorithm=algorithm,
    fitness_evaluator=XORFitnessEvaluator(),
    termination_criterion=pn.CompositeTermination([
        pn.TargetFitnessTermination(target_fitness=3.95),
        pn.MaxGenerationsTermination(max_generations=300),
    ]),
    callbacks=[
        pn.ConsoleStatisticsLogger(),
        pn.BestGenomePersister(output_directory=Path("runs")),
        pn.NetworkTopologyVisualizer(output_directory=Path("runs")),
        pn.TensorBoardLogger(log_directory=Path("runs"), run_name="xor"),
    ],
    random_seed=config.random_seed,
)

result = runner.run_evolution()
print(f"Best fitness: {result.best_fitness_ever_achieved:.4f}")
print(f"Termination reason: {result.termination_reason}")
```

TensorBoard:

```bash
tensorboard --logdir runs/
```

---

## Development

```bash
uv sync                        # installs the library + pytest
uv run pytest                  # run the test suite
uv pip install -e ".[dev]"     # ruff
uv run ruff check polyneat examples tests
uv run ruff format polyneat
```

---

## Literature

- Stanley, K. O. & Miikkulainen, R. (2002). **Evolving Neural Networks through Augmenting Topologies**. *Evolutionary Computation*, 10(2), 99–127.
- Whiteson, S., Stone, P., Stanley, K. O., Miikkulainen, R. & Kohl, N. (2005). **Automatic Feature Selection in Neuroevolution**. *GECCO 2005*.
- Stanley, K. O., D'Ambrosio, D. B. & Gauci, J. (2009). **A Hypercube-Based Encoding for Evolving Large-Scale Neural Networks**. *Artificial Life*, 15(2), 185–212.
- Stanovov, V., Akhmedova, S. & Semenkin, E. (2021). **Difference-based mutation operation for neuroevolution of augmented topologies**. DOI 10.1088/1757-899X/1047/1/012075.
- Chen, L. & Alahakoon, D. (2006). **NeuroEvolution of Augmenting Topologies with Learning for Data Classification**. *ICIA 2006*, pp. 367–371.
- Alfaham, Y., Van Raemdonck, W. & Mercelis, S. (2024). *ACAI 2024*, DOI 10.1109/ACAI63924.2024.10899662.
