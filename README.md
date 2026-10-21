# PolyNEAT

A Python library for neuroevolution algorithms, built around a full implementation of NEAT (Stanley & Miikkulainen, 2002). Variants subclass `NEATAlgorithm` and override only what they change, so the core is written once and shared.

Implemented algorithms:

| Algorithm | Class | Idea | Reference |
|---|---|---|---|
| NEAT | `NEATAlgorithm` | Evolves network topology and weights from a minimal start | Stanley & Miikkulainen, 2002 |
| FS-NEAT | `FSNEATAlgorithm` | Starts each genome with a single random input connection, so evolution selects the relevant input features | Whiteson et al., 2005 |
| FD-NEAT | `FDNEATAlgorithm` | Starts fully connected and deletes input connections, so evolution *deselects* the irrelevant features | Tan et al., 2012 |
| HyperNEAT | `HyperNEATAlgorithm` | Evolves a CPPN that paints the weights of a fixed substrate as a function of geometry | Stanley, D'Ambrosio & Gauci, 2009 |
| HyperNEAT-LEO | `HyperNEATLEOAlgorithm` | Gives the CPPN a second output deciding whether a connection is expressed at all, seeded toward local links so modular structure can emerge | Verbancsics & Stanley, 2011 |
| NEAT-DBM | `NEATDBMAlgorithm` | Recombines each child's weights from three donor genomes, differential-evolution style | Stanovov et al., 2021 |
| C-NEAT | `CNEATAlgorithm` | Scores each organism on one class only and keeps a container of the best recognizer per class | Alfaham et al., 2024 |
| L-NEAT | `LNEATAlgorithm` | Interleaves evolution with Lamarckian backpropagation sessions on a fixed learning subset | Chen & Alahakoon, 2006 |
| EXACT | `EXACTAlgorithm` | Evolves CNN filter topologies — nodes are filters, edges are convolutions — training every genome by backpropagation before it is scored, and co-evolves the training hyperparameters | Desell, 2017 |
| DeepNEAT | `DeepNEATAlgorithm` | Evolves deep architectures where each node is a whole *layer* with its own hyperparameters and each edge carries a tensor; genomes hold no weights, so every network is trained from scratch during evaluation | Miikkulainen et al., 2017 |

---

## Installation

Requires Python 3.11+. The recommended package manager is [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync                      # library + pytest
uv pip install -e ".[dev]"   # adds ruff
```

Alternatively with standard `pip`:

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS / Linux
pip install -e ".[dev]" pytest
```

`pip` does not read the `dev` dependency group from `pyproject.toml`, so `pytest` has to be named explicitly.

---

## Quick start

Examples live in `examples/<task>/<algorithm>.py`, each with a YAML config next to it. Run them as modules from the repository root:

```bash
uv run python -m examples.xor.neat            # NEAT on XOR
uv run python -m examples.iris.cneat          # C-NEAT on Iris
uv run python -m examples.mnist.hyperneat     # HyperNEAT on down-pooled MNIST
uv run python -m examples.mnist.exact         # EXACT on full-resolution MNIST
uv run python -m examples.fashion_mnist.deepneat  # DeepNEAT on Fashion-MNIST
```

All sixteen examples:

| Task | NEAT | FS-NEAT | FD-NEAT | HyperNEAT | HyperNEAT-LEO | NEAT-DBM | C-NEAT | L-NEAT | EXACT | DeepNEAT |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `xor` | ✓ | ✓ | ✓ | ✓ | | ✓ | | | | |
| `iris` | | | | | | ✓ | ✓ | ✓ | | |
| `mnist` | ✓ | | | ✓ | | | | | ✓ | ✓ |
| `fashion_mnist` | | | | | | | | | | ✓ |
| `retina` | | | | ✓ | ✓ | | | | | |
| `visual_discrimination` | | | | ✓ | | | | | | |

Artifacts — best genome as JSON and pickle, topology renders, TensorBoard event files — are written to `examples/<task>/artifacts/<algorithm>/`.

Every example exposes the same contract: a `CONFIG_FILE_PATH` constant and a `run_experiment(device, random_seed, artifacts_directory)` function returning an `ExperimentReport` with metrics, generation count and runtime. The registry of all runnable examples is `EXAMPLE_REGISTRY` in `examples/_experiment.py`.

---

## Datasets in the examples

Every example package has a `dataset.py` exposing a top-level `load_*` function. The two classification datasets return the same named bundle, `ClassificationDataset` (in `examples/_datasets.py`), with `train_features`, `train_labels`, `test_features`, `test_labels`, `number_of_classes` and a `number_of_features` property.

```python
from examples.iris.dataset import load_iris
from examples.mnist.dataset import load_mnist

iris = load_iris(random_seed=42)                       # 99/51 split of the 150 UCI samples
mnist = load_mnist(random_seed=42, grid_side=7)        # 28x28 images average-pooled to 7x7
mnist_full = load_mnist(random_seed=42, grid_side=28,  # full-resolution, no subset caps
                        max_train_samples=None, max_test_samples=None)
```

Knobs are plain parameters: `train_fraction`, `grid_side` (any divisor of 28), `max_train_samples` and `max_test_samples`. The building blocks are public too: `load_mnist_features_and_labels(grid_side)` returns the whole 70,000-sample set as one `(features, labels)` pair, and `pool_features_to_grid(images, grid_side)` re-pools raw uint8 images. Raw files download once into `examples/<task>/data/` and are cached there.

Fashion-MNIST (Xiao et al., 2017) is loaded the same way by `examples.fashion_mnist.dataset.load_fashion_mnist`, which takes the same parameters and returns the same bundle — it is a drop-in replacement for MNIST, being identically shaped (28x28 grayscale, 10 classes, 70,000 samples) but harder. It ships as four gzipped IDX archives rather than one `.npz`, so that package adds `read_idx_gz_file(path)`; pooling and standardization are imported from the MNIST loader rather than duplicated.

The two synthetic tasks follow the same convention: `examples.xor.dataset.load_xor()` returns the four patterns, and `examples.visual_discrimination.dataset.load_visual_discrimination_trials(...)` returns a `VisualDiscriminationTrials` bundle.

---

## Benchmarks

`benchmarks/run_benchmark.py` runs one example several times with consecutive seeds and records the spread:

```bash
uv run python -m benchmarks.run_benchmark iris/cneat --repeats 5 [--cpu | --gpu] [--base-seed 0]
```

Each invocation writes one JSON document to `benchmarks/results/` holding every run, a mean/std summary, and the full YAML config that produced it, so a result stays interpretable after the config has moved on.

---

## Monitoring with TensorBoard

Examples that attach a `TensorBoardLogger` write event files to `examples/<task>/artifacts/<algorithm>/tensorboard/`:

```bash
uv run tensorboard --logdir examples/xor/artifacts/neat/tensorboard
# or point at everything at once:
uv run tensorboard --logdir examples
```

Logged scalars: `fitness/best`, `fitness/mean`, `fitness/median`, `population/num_species`, plus algorithm-specific metrics under `extra/*`, for example `extra/innovation_id_high_water_mark`. To log your own runs, add the callback to the runner:

```python
pn.TensorBoardLogger(log_directory=Path("runs/tensorboard"), run_name="my-run")
```

---

## Choosing the evaluation device (`--cpu` / `--gpu`)

All example scripts accept two mutually exclusive flags:

- `--cpu` forces phenotype evaluation on the CPU.
- `--gpu` forces CUDA. If CUDA is unavailable the script prints an error and exits with code 1. There is no silent GPU→CPU fallback, so benchmark numbers cannot be produced on the wrong device without saying so.

With no flag, the `device_for_phenotype_evaluation` value from the example's YAML config is used.

In library code the same mechanism is `Algorithm.from_config(config, device_for_phenotype_computation=device)`.

---

## Project structure

The full NEAT implementation lives in `polyneat/core/neat/`; derived algorithms subclass `NEATAlgorithm` and override only what they change.

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
│   │   ├── lneat/               backpropagation trainer, trainable phenotype
│   │   ├── exact/               CNN genome, eight operators, conv phenotype, SHO
│   │   └── deepneat/            layer-level genome, shape propagation, layer-stack phenotype
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
│   ├── mnist/                   neat, hyperneat, exact, deepneat
│   ├── fashion_mnist/           deepneat
│   └── visual_discrimination/   hyperneat
├── benchmarks/                  repeat-runner + results
└── tests/
```

---

## What each module is responsible for

### `polyneat/configs/`

| File | Responsibility |
|---|---|
| `algorithm_config.py` | Base dataclass `AlgorithmConfig` - parameters shared by every algorithm (`population_size`, `number_of_input_nodes`, `random_seed`, etc.). Provides `load_from_yaml_file` and `from_dict`; both are strict, unknown keys raise `ConfigurationError`. |
| `configuration_errors.py` | `ConfigurationError` - raised when a configuration is invalid. The message always names the field, the value, and the reason. |
| `neat/neat_config.py` | `NEATConfig` - all NEAT hyperparameters: mutation probabilities, compatibility-distance coefficients, selection parameters. |
| `hyperneat/hyperneat_config.py` | `HyperNEATConfig(NEATConfig)` - substrate geometry, CPPN activation set, weight expression threshold. |
| `neatdbm/neatdbm_config.py` | `NEATDBMConfig(NEATConfig)` - difference-based mutation parameters. |
| `cneat/cneat_config.py` | `CNEATConfig(NEATConfig)` - number of class labels for the container. |
| `lneat/lneat_config.py` | `LNEATConfig(NEATConfig)` - learning interval, backpropagation session parameters, learning subset size. |
| `exact/exact_config.py` | `EXACTConfig(NEATConfig)` - input image geometry, the eight operator probabilities, filter-size change options, crossover inclusion rates, the full backpropagation block (learning rate / momentum / weight decay with their per-epoch schedules, velocity reset ω, dropout, batch normalization) and the simplex hyperparameter optimization settings. |
| `deepneat/deepneat_config.py` | `DeepNEATConfig(NEATConfig)` - input tensor geometry and class count, the four operator probabilities, the per-layer hyperparameter ranges evolution draws from (filter counts, kernel sizes, dense unit counts, dropout range), the total parameter budget above which a phenotype is rejected, and the fixed training block every genome is evaluated with (epochs, learning rate, batch size, determinism). The inherited connection-weight fields are unused: DeepNEAT genomes carry no weights. |

FS-NEAT has no config of its own; it runs on plain `NEATConfig`.

### `polyneat/core/`

Defines the protocols every algorithm implements, plus the shared data types.

| File | Responsibility |
|---|---|
| `component_protocols.py` | `@runtime_checkable` protocols: `Genome`, `Phenotype`, `PhenotypeDecoder`, `MutationOperator`, `CrossoverOperator`, `ParentSelection`, `Speciator`, `FitnessEvaluator`, `InitialPopulationStrategy`, `NeuroevolutionAlgorithm`, `InnovationTracker`. None of them is a base class - implementing the right methods is enough. |
| `population.py` | Frozen dataclass `Population(genomes, species_assignments, generation_number)`. Every generation creates a new object. |
| `generation_statistics.py` | Frozen dataclass `GenerationStatistics` - best and mean fitness, species count, timing for one generation. |
| `type_aliases.py` | `FitnessValue = float`, `InnovationId = int`, `SpeciesId = int`. |

### `polyneat/core/neat/`

The classic NEAT implementation, one aspect per file. `polyneat/algorithms/` below holds the derived algorithms.

| File | Responsibility |
|---|---|
| `neat_genome.py` | Frozen dataclasses `NodeGene` and `ConnectionGene`, plus `NEATGenome`. A genome cannot be modified in place - mutation always returns a new object. `__post_init__` validates that there are no duplicate nodes or innovations. |
| `global_innovation_tracker.py` | Assigns global `InnovationId`s to new connections. Within one generation the same `(source, target)` pair gets the same id, which lets crossover align structurally identical genes. After each generation the dedup table is cleared; the counter is never reset. |
| `initial_population.py` | Generation-0 strategies: `fully_connected` (vanilla NEAT minimal start) and `fs_neat` (single random input→output connection per genome), plus a name→strategy registry extensible via `register_initial_population_strategy`. |
| `mutations/add_node_mutation.py` | Picks one enabled connection, disables it, inserts a new hidden node, adds two new connections: input→node with weight 1.0 and node→output with the original weight. |
| `mutations/add_connection_mutation.py` | Tries to add a new connection between unconnected nodes. Checks for duplicates, self-loops, cycles (BFS from the target; if the source is reachable, a cycle exists) and topological constraints: input nodes cannot be targets, output nodes cannot be sources. |
| `mutations/weight_modification_mutation.py` | Per connection: with probability `p_perturb` adds Gaussian noise, with probability `p_replace` redraws the weight from the initialization range. |
| `mutations/toggle_connection_enabled_mutation.py` | Flips the `is_enabled` flag of a random connection. Before re-enabling a disabled connection it checks that no cycle is created, because `AddConnectionMutation` checks cycles only on the currently enabled connections. |
| `mutations/composite_neat_mutation.py` | Applies the mutations in order: WeightModification → AddConnection → AddNode → Toggle. Each mutation has its own probability. |
| `neat_crossover.py` | Crossover aligned by `innovation_id`. Matching genes are inherited randomly with the configured probability; disjoint and excess genes come from the fitter parent. After assembling the child, `_resolve_enabled_connection_cycles` removes any cycles, which can appear when a gene was disabled in one parent and the re-enable chance closes a loop. |
| `compatibility_distance_speciator.py` | Computes the compatibility distance δ = c₁·E/N + c₂·D/N + c₃·W̄ and assigns genomes to species by comparison with each species' representative. After every speciation pass the representative is resampled from the current members, as in the original paper; a frozen representative caused artificial species fragmentation. |
| `tournament_parent_selection.py` | Samples `tournament_size` individuals and returns the best. Repeats for every requested parent. |
| `torch_feedforward_phenotype.py` | The network as a PyTorch `nn.Module`. At construction it computes a topological order of the nodes with Kahn's algorithm; `forward_pass` then iterates the nodes in that order, sums weighted inputs and applies the activation function. Supports batched inputs `[batch, n_inputs]`. |
| `neat_phenotype_decoder.py` | Builds a `TorchFeedForwardPhenotype` from a `NEATGenome` and passes the device through. |
| `neat_algorithm.py` | `NEATAlgorithm`, the main class. `from_config()` builds every component from `NEATConfig` via overridable `_build_*` factory methods, so subclasses replace only what they change. `create_initial_population()` builds minimal genomes (inputs + bias → outputs). `advance_one_generation()` runs the full cycle described [below](#one-generation-cycle-advance_one_generation). |

### `polyneat/algorithms/`

| Package | Responsibility |
|---|---|
| `neat/` | Thin entry point: re-exports `NEATAlgorithm` from `polyneat/core/neat/`. |
| `fsneat/` | `FSNEATAlgorithm(NEATAlgorithm)` overrides only `create_initial_population`; every genome starts with a single random input→output connection, so evolution itself selects the relevant input features. Any `initial_population_strategy` from YAML is deliberately ignored. |
| `fdneat/` | `FDNEATAlgorithm(NEATAlgorithm)` overrides only `_build_mutation`, appending `DeleteInputConnectionMutation` to the standard four. The fully connected start is inherited unchanged — it *is* FD-NEAT's start, and evolution removes what does not earn its place. |
| `hyperneat/` | `HyperNEATAlgorithm` evolves CPPNs; `substrate.py` builds substrates (`build_grid_sandwich_substrate` for two-sheet sandwiches, `build_layered_substrate` for the general case, `build_substrate_from_explicit_layer_coordinates` when a task needs groups of nodes pushed apart); `HyperNEATPhenotypeDecoder` queries the CPPN for every substrate connection and thresholds the result, exposing the decoded substrate through `decode_substrate_genome`; `substrate_modularity.py` measures wiring locality and functional modularity of a decoded substrate; `AddNodeWithRandomActivationMutation` gives new CPPN nodes a random activation from the configured set. |
| `hyperneatleo/` | `HyperNEATLEOAlgorithm(HyperNEATAlgorithm)` overrides only `_build_phenotype_decoder`. The CPPN keeps HyperNEAT's four coordinate inputs but gains a second output, the *link expression output*, which decides on its own whether a connection exists — so a weak-but-present connection becomes expressible, which classic HyperNEAT cannot represent. `leo_seeded_initial_population.py` seeds generation 0 toward local connections by feeding an axis' two coordinates into a Gaussian node through equal and opposite weights, so their sum is the coordinate difference. |
| `neatdbm/` | `NEATDBMAlgorithm` - after standard reproduction, `DifferenceBasedWeightMutation` recombines each child's weights from three donors at shared innovation ids. |
| `cneat/` | `CNEATAlgorithm` - organisms are scored on one assigned class; `ClassGenomeContainer` keeps the best recognizer per class; `ContainerEnsemblePhenotype` classifies by argmax over the container networks; `ContainerUpdateCallback` and `ContainerProgressLogger` maintain and report the container during the run. |
| `lneat/` | `LNEATAlgorithm` - every `learning_interval_generations`, non-Type-1 offspring get a backpropagation session (`BackpropagationWeightTrainer`) on a fixed learning subset, and trained weights are inherited (Lamarckian). `TrainableTorchFeedForwardPhenotype` makes the phenotype's weights torch parameters; `RecognizerEnsemblePhenotype` assembles per-class recognizers into an argmax ensemble. |
| `exact/` | `EXACTAlgorithm` - a CNN genome (`EXACTGenome`) whose nodes are filters and whose edges are convolutions, evolved by eight operators in `mutations/`. `EXACTInnovationTracker` keeps the master innovation list for the whole search; `EXACTCrossover` is fitness-asymmetric and discards children with an unreachable output; `TorchConvolutionalPhenotype` executes the CNN in one depth-ordered sweep with optional batch normalization and dropout; `EXACTBackpropagationTrainer` trains every untrained genome before it is scored and writes the kernels back (Lamarckian, with epigenetic weight initialization); `SimplexHyperparameterOptimizer` co-evolves each genome's eleven training hyperparameters. |
| `deepneat/` | `DeepNEATAlgorithm` - a genome (`DeepNEATGenome`) whose nodes are whole *layers* carrying their own hyperparameters and whose edges carry tensors and no weight, evolved by four operators in `mutations/`. `layer_shape_propagation.py` prunes the genome to the nodes on an enabled input→output path and propagates tensor shapes along it, deciding how multiple incoming tensors are merged and coercing a flat tensor back to spatial form when a convolution needs one; `TorchLayerStackPhenotype` builds one `nn.Module` per surviving layer node and reports both `total_parameter_count` and `number_of_layer_modules`; `DeepNEATSpeciator` adds a layer-hyperparameter term to NEAT's compatibility distance. **`advance_one_generation` is not overridden** - because no weights are inherited, training belongs to the fitness evaluator (`TrainedNetworkAccuracyEvaluator`), and the inherited generational loop is enough. |

### `polyneat/nn/`

| File | Responsibility |
|---|---|
| `activation_functions.py` | `sigmoid`, `steepened_sigmoid`, `tanh`, `relu`, `leaky_relu`, `identity` as PyTorch callables, plus `sine`, `gaussian` and `absolute_value` for CPPNs. Dictionary `ACTIVATION_FUNCTION_NAME_TO_CALLABLE` plus `resolve_activation_function_by_name(name)`, which raises `ConfigurationError` for unknown names. |
| `topology_utilities.py` | `compute_topological_order_of_node_ids` - Kahn's algorithm, raises `ValueError` on a cycle. `would_directed_edge_create_cycle` - BFS from the candidate target; if the source is reachable, the edge would create a cycle. |

### `polyneat/evaluators/`

| File | Responsibility |
|---|---|
| `sequential_evaluator_base.py` | `SequentialFitnessEvaluator` - base class for evaluating one phenotype at a time. Overriding `evaluate_single_phenotype` is enough. |
| `parallel_evaluator_wrapper.py` | `ParallelFitnessEvaluatorWrapper` wraps any evaluator and evaluates in parallel via `joblib`. Defaults to `prefer="threads"`. |
| `class_indexed_evaluator_base.py` | Base for evaluators that score an organism against one assigned class label, used by C-NEAT. |
| `xor_evaluator.py` | `XORFitnessEvaluator` - evaluates a phenotype on the four XOR patterns. Fitness = Σ(1 - (expected - actual)²), max 4.0, solved threshold ≥ 3.95. |
| `xor_with_distractors_evaluator.py` | XOR plus noise distractor inputs - the FS-NEAT feature-selection benchmark. |
| `classification_accuracy_evaluator.py` | `ClassificationAccuracyEvaluator` - fraction of samples whose argmax output matches the label. |
| `softmax_likelihood_evaluator.py` | `SoftmaxLikelihoodFitnessEvaluator` - mean softmax probability of the correct class; a smooth training signal for many-class problems. |
| `binary_recognizer_evaluator.py` | Scores a single-output recognizer network for one target class, used by L-NEAT's per-class runs. |
| `multiclass_dataset_evaluator.py` | Scores each organism only on its assigned class over a labelled dataset, C-NEAT's training signal. |
| `visual_discrimination_evaluator.py` | Trial generator + evaluator for locating the larger of two squares in a 2-D field (Stanley et al. 2009, section 4). |
| `trained_network_accuracy_evaluator.py` | `TrainedNetworkAccuracyEvaluator` - trains each phenotype from a fresh random initialization by backpropagation, then scores it by validation accuracy. DeepNEAT's fitness signal: since its genomes inherit no weights, training lives here rather than in the generational loop. Trained weights are never written back anywhere. |

### `polyneat/runner/`

| File | Responsibility |
|---|---|
| `evolution_runner.py` | `EvolutionRunner` - the main loop: builds phenotypes → evaluates fitness → tracks the best genome → calls `advance_one_generation` → checks the termination criterion. Returns an `EvolutionResult`. |
| `run_context.py` | `RunContext` - the current run state (id, start time, generation number, statistics history, best genome so far). Passed to all callbacks. |
| `termination_criteria.py` | `MaxGenerationsTermination`, `TargetFitnessTermination`, `FitnessStagnationTermination`, `CompositeTermination` with OR logic. |
| `evolution_callback_protocol.py` | The `EvolutionCallback` protocol with six hooks: `on_run_started`, `on_generation_started`, `on_population_evaluated`, `on_generation_completed`, `on_new_best_genome_found`, `on_run_completed`. `BaseEvolutionCallback` provides empty default implementations. |
| `builtin_evolution_callbacks.py` | `ConsoleStatisticsLogger` (rich table), `TensorBoardLogger`, `BestGenomePersister` (JSON + pickle), `NetworkTopologyVisualizer`. |

### `polyneat/logging_utils/`

| File | Responsibility |
|---|---|
| `custom_logger.py` | `get_logger(__name__)` - the entry point for every logger in the library. Registers `CustomLogger` via `logging.setLoggerClass` at import time. Every logger attaches its own handler; propagation is off. |
| `colored_level_formatter.py` | `ColoredLevelFormatter` - colors the message body, not the whole line, with `colorama`. DEBUG blue, INFO green, WARNING yellow, ERROR red, CRITICAL dark red. |
| `logging_config.py` | `LoggingConfig` - log level, message format, optional directory for file logs. |

### `polyneat/viz/`

`network_topology_renderer.py` - `render_genome_topology(genome, output_path)`. Uses `matplotlib` and `networkx`; `matplotlib.use("Agg")` makes it work without a display, on a server or in CI. Input, bias, hidden and output nodes get different colors; disabled connections are dashed.

### `polyneat/utils/`

| File | Responsibility |
|---|---|
| `random_generator_factory.py` | `create_seeded_random_generator(seed)` returns a `numpy.random.Generator` - the single point of seed control. |
| `artifact_serialization.py` | `save_as_json`, `load_from_json`, `save_as_pickle`, `load_from_pickle`. |

---

## How NEAT works - the algorithm step by step

### Where the implementation comes from

NEAT is based on the original publication:

> Stanley, K. O. & Miikkulainen, R. (2002). **Evolving Neural Networks through Augmenting Topologies**. *Evolutionary Computation*, 10(2), 99–127.

The article is freely available on the author's website. Every implementation decision in the code — the compatibility distance formula, the crossover rules, offspring allocation — traces back to it.

---

### The problem NEAT solves

Classic neural network evolution has three fundamental problems.

**1. Competing conventions.**
If evolution independently discovers the same hidden node in two genomes, they are structurally identical but the nodes carry different numbers. Crossing over such genomes produces offspring with duplicated nodes.

*NEAT's solution:* every new structural edge gets a global `InnovationId`. If the same edge `(A→B)` appears in several genomes within the same generation, they all get the same id. Crossover aligns genes by `InnovationId`, not by index.

**2. Protecting structural innovation.**
A freshly added hidden node perturbs the network and fitness drops. Without protection, a lineage carrying a topological innovation dies before it can optimize it.

*NEAT's solution:* speciation. Structurally similar genomes form one species. Selection happens within species, and fitness is divided by the species size, so each species competes with itself.

**3. Minimal dimensionality.**
Randomly initialized networks have as much freedom as large ones but are harder to evolve, and large weight spaces converge slowly.

*NEAT's solution:* start from a minimal network — inputs + bias → outputs, no hidden nodes. Topological complexity grows gradually through the `AddNode` and `AddConnection` mutations.

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

`NodeGene`, `ConnectionGene` and `NEATGenome` are all frozen dataclasses. No operator modifies an existing genome; mutation always returns a new object.

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
    │
    ▼
[3] Stagnation
    If a species has not improved its best raw fitness
    for species_stagnation_generations_limit generations → removed.
    The species containing the globally best genome always survives.
    │
    ▼
[4] Offspring allocation
    Each species receives offspring slots proportional to the
    SUM of its members' adjusted fitness, which equals the species'
    mean raw fitness, as in the original paper. Largest-remainder
    rounding keeps the slot total equal to population_size.
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

If a gene was disabled in either parent, the offspring has a 75% chance of inheriting it disabled, which protects against topological chaos. After the whole child genome is assembled, `_resolve_enabled_connection_cycles` removes any cycles produced by the combination of enable states.

---

### Configuration

Every hyperparameter lives in the example's YAML file (`examples/xor/neat.yaml` for the XOR baseline) and maps 1:1 onto a `NEATConfig` field. The defaults follow the original paper where it specifies values — population 150, compatibility coefficients 1.0/1.0/0.4, threshold 3.0, interspecies mating 0.001 — and are commented where they deviate. Unknown keys raise `ConfigurationError`, so a typo cannot silently fall back to a default.

---

## The XOR benchmark

### Why XOR

XOR is the standard NEAT benchmark from the original paper. It is not linearly separable, so a network without hidden nodes cannot solve it and the algorithm must evolve the right topology. It is also trivially verifiable — four input patterns, no ambiguity — and Stanley and Miikkulainen used it to validate NEAT in 2002, which gives a reference point.

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

Maximum 4.0, all patterns perfect. Solved threshold: ≥ 3.95.

Squared error is used instead of absolute error because absolute error gives a flat gradient around the "3 patterns correct" local optimum: fitness stays at 3.0 no matter how badly the network misses the fourth pattern. Squared error penalizes large errors more and small errors less, so there is a gradient encouraging the network to reduce the error on the fourth pattern. A network that outputs 0.5 for pattern (1,1) gets fitness 3.75, not 3.5.

Configuration file: `examples/xor/neat.yaml`.

### Success criterion - a note on comparability with the original paper

The original paper counts a network as solving XOR when all four outputs land on the correct side of 0.5. The `fitness ≥ 3.95` threshold with squared error is much stricter: it requires each output to be on average within ~0.11 of its target. A network the paper would count as a solution, say outputs 0.3/0.7/0.7/0.3, has fitness of only 3.64 here. Generation counts measured against the two criteria are therefore not comparable, so both are reported. The helper is `XORFitnessEvaluator.classifies_all_patterns_correctly`.

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

10/10 seeds; on average 33.1 generations to fitness ≥ 3.95 and 24.9 generations to the paper's criterion. The paper reports an average of 32.

An earlier version of the implementation needed ~165 generations on average to reach fitness ≥ 3.95. The speedup comes from two changes, ablated on the same 10 seeds:

1. Survival threshold, interspecies mating, representative resampling and corrected offspring allocation: ~165 → 60.1 generations. The dominant contribution is the survival threshold, i.e. selection pressure. The resampling and allocation fixes alone do not change the pace on XOR, but they remove the species fragmentation that matters in longer runs.
2. Steepened sigmoid (4.9) instead of the standard one: 60.1 → 33.1 generations.

---

## Feature selection: FS-NEAT against FD-NEAT

The two feature-selection variants solve the same problem from opposite ends, so
they share a benchmark: XOR padded with six pure-noise inputs
(`XORWithDistractorsEvaluator`). Only inputs 0 and 1 carry signal, and both
examples report `number_of_connected_input_features` — inputs with an *enabled
path to the output*, computed by the same library function, so an input feeding a
dead-end hidden node does not count.

Ten seeds each (`--base-seed 0`, CPU). Feature counts are averaged over the
**solved** runs only, because in an unsolved run the count says nothing:

| Variant | Solved | Features used (solved) | Generations (solved) |
|---|:-:|:-:|:-:|
| FS-NEAT | **10/10** | 5.50 | 158.4 |
| FD-NEAT | 6/10 | **5.00** | **96.5** |

FD-NEAT deselects harder and, when it succeeds, converges faster — but it solves
XOR far less reliably. The mechanism is visible in the ablation: deletion has to
be balanced against `probability_of_add_connection_mutation`, which is the only
way a wrongly deleted input can return.

| `delete` | `add_connection` | Solved | Features | Generations |
|---:|---:|:-:|:-:|:-:|
| 0.05 | 0.03 | 3/10 | 3.67 | 270.0 |
| **0.05** | **0.10** | **6/10** | **5.00** | **96.5** |
| 0.02 | 0.10 | 6/10 | 6.00 | 245.5 |

Too little repair (row 1) halves the success rate: once the operator cuts input 0
or 1 the genome cannot get it back. Too little deletion (row 3) is dominated —
worse on every axis than row 2. The shipped `examples/xor/fdneat.yaml` uses row 2,
and the full data with the exact configs that produced it is written to
`benchmarks/results/xor_fdneat_*.json`.

---

## Modularity: HyperNEAT against HyperNEAT-LEO

The retina problem (Kashtan & Alon, 2005) is built from two independent
sub-problems: an eight-pixel retina whose left and right halves each may contain
an "object", where the left answer depends on no right-hand pixel and vice versa.
A network keeping its halves separate loses nothing; one wiring them together
pays for connections it cannot use. That makes it the natural test of LEO's
claim, and both `examples/retina/hyperneat.py` and `examples/retina/leo.py` run
it through the same code (`examples/retina/_shared.py`), so only the algorithm
differs.

Two metrics are reported, and the distinction matters:

- **`number_of_cross_hemisphere_connections`** — wiring locality: connections whose
  endpoints sit on opposite sides of `x = 0`. This is what the locality seed
  directly controls.
- **`number_of_cross_hemisphere_input_dependencies`** — functional modularity: pairs
  of (output, opposite-side input) joined by an enabled path, out of 8. **This is
  what the modularity claim is actually about.** A hidden node's side is
  arbitrary, so a network routing left inputs through right-side hidden nodes into
  the left output never mixes left and right information — perfectly modular, yet
  every one of its connections looks like a crossing to the first metric.

Single run, seed 42, 1500 generations, identical configuration apart from what
separates the two algorithms:

| Variant | Fitness (max 256) | Expressed links | Crossing links | Crossing dependencies |
|---|---:|---:|---:|---:|
| HyperNEAT-LEO | **211.95** | 25 | **0 (0%)** | **0 / 8** |
| HyperNEAT | 201.55 | 52 | 29 (56%) | 7 / 8 |

LEO comes out perfectly modular on both measures and ahead on fitness; the
baseline is almost entirely non-modular and barely above the 192.0 a
constant-output network scores.

**Read these numbers with three caveats.**

1. **One seed.** A multi-seed benchmark has not been run.
2. **The budget is far short of the literature.** Huizinga et al. (2014) run
   25,000–50,000 generations and report that the medians of all treatments reach
   perfect performance; differences become significant only after ~12,000. At
   1500 generations neither variant here comes close to 256, which is a property
   of the budget, not of the methods. A trajectory probe shows LEO flat at ~208
   through generation 300, then climbing to ~230 by 1500, while the baseline is
   flat throughout (192.6 → 199.3).
3. **The locality seed's advantage may be transient.** The same paper reports that
   seeded modularity "spikes during the first few generations, but then decreases
   over time", and notes that perfect-performing *non-modular* solutions to this
   task exist.

Provenance is recorded in
[`docs/superpowers/specs/2026-08-14-hyperneat-leo-design.md`](docs/superpowers/specs/2026-08-14-hyperneat-leo-design.md):
the task definition is verified against Kashtan & Alon, the seed constants come
from Huizinga et al., the Gaussian-to-LEO weight is stated by neither source, and
the substrate geometry plus the two-output form of the task are this
implementation's own choices.

---

## Using the library

### Logging

```python
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)
logger.info("Message")
logger.debug("Debug with %s", "an argument")
```

Never call `logging.getLogger` directly. Configure once at startup:

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

### A custom algorithm

Subclass `NEATAlgorithm` and override the `_build_*` factories you want to change; `from_config` picks up the override and the generational loop stays shared.

```python
import polyneat as pn

class MyNEAT(pn.NEATAlgorithm):
    @classmethod
    def _build_parent_selection(cls, config: pn.NEATConfig) -> pn.ParentSelection:
        return pn.TournamentParentSelection(tournament_size=7)
```

---

## Development

```bash
uv sync                        # library + pytest
uv run pytest                  # 356 tests
uv pip install -e ".[dev]"     # ruff
uv run ruff check polyneat examples tests
uv run ruff format polyneat
```

---

## Literature

- Stanley, K. O. & Miikkulainen, R. (2002). **Evolving Neural Networks through Augmenting Topologies**. *Evolutionary Computation*, 10(2), 99–127.
- Chen, L. & Alahakoon, D. (2006). **NeuroEvolution of Augmenting Topologies with Learning for Data Classification**. *ICIA 2006: 2nd International Conference on Information and Automation*, pp. 367–371.
- Whiteson, S., Stone, P., Stanley, K. O., Miikkulainen, R. & Kohl, N. (2005). **Automatic Feature Selection in Neuroevolution**. *GECCO 2005: Proceedings of the Genetic and Evolutionary Computation Conference*, pp. 1225–1232.
- Tan, M., Deklerck, R., Jansen, B. & Cornelis, J. (2012). **Analysis of a Feature-Deselective Neuroevolution Classifier (FD-NEAT) in a Computer-Aided Lung Nodule Detection System for CT Images**. *GECCO '12 Companion: Proceedings of the 14th Annual Conference Companion on Genetic and Evolutionary Computation*, pp. 539–546. DOI: 10.1145/2330784.2330869
- Stanley, K. O., D'Ambrosio, D. B. & Gauci, J. (2009). **A Hypercube-Based Encoding for Evolving Large-Scale Neural Networks**. *Artificial Life*, 15(2), 185–212. DOI: 10.1162/artl.2009.15.2.15202
- Verbancsics, P. & Stanley, K. O. (2011). **Constraining Connectivity to Encourage Modularity in HyperNEAT**. *GECCO '11: Proceedings of the 13th Annual Conference on Genetic and Evolutionary Computation*, pp. 1483–1490. DOI: 10.1145/2001576.2001776
- Kashtan, N. & Alon, U. (2005). **Spontaneous evolution of modularity and network motifs**. *Proceedings of the National Academy of Sciences*, 102(39), 13773–13778. DOI: 10.1073/pnas.0503610102
- Huizinga, J., Mouret, J.-B. & Clune, J. (2014). **Evolving Neural Networks That Are Both Modular and Regular: HyperNeat Plus the Connection Cost Technique**. *GECCO '14*, pp. 697–704. DOI: 10.1145/2576768.2598232 — not implemented here; it is the source of the locality-seed constants, since it reimplements HyperNEAT-LEO as one of its treatments and the 2011 paper was not obtainable.
- Stanovov, V., Akhmedova, Sh. & Semenkin, E. (2021). **Neuroevolution of augmented topologies with difference-based mutation**. *IOP Conference Series: Materials Science and Engineering*, 1047, 012075. DOI: 10.1088/1757-899X/1047/1/012075
- Alfaham, A., Van Raemdonck, S. & Mercelis, S. (2024). **Genetic NEAT-Based Method for Multi-Class Classification**. *ACAI 2024: 7th International Conference on Algorithms, Computing and Artificial Intelligence*. DOI: 10.1109/ACAI63924.2024.10899662
- Desell, T. (2017). **Developing a Volunteer Computing Project to Evolve Convolutional Neural Networks and Their Hyperparameters**. *2017 IEEE 13th International Conference on e-Science*, pp. 19–28. DOI: 10.1109/eScience.2017.14
- Miikkulainen, R., Liang, J., Meyerson, E., Rawal, A., Fink, D., Francon, O., Raju, B., Shahrzad, H., Navruzyan, A., Duffy, N. & Hodjat, B. (2017). **Evolving Deep Neural Networks**. *arXiv:1703.00548*. Published in *Artificial Intelligence in the Age of Neural Networks and Brain Computing* (2019), pp. 293–312. DOI: 10.1016/B978-0-12-815480-9.00015-3
- Xiao, H., Rasul, K. & Vollgraf, R. (2017). **Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms**. *arXiv:1708.07747*.

Every citation above is reproduced verbatim in the `References:` block of the corresponding module's docstring; the two sets are kept in sync deliberately.
