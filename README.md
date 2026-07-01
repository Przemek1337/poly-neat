# PolyNEAT

Python library for **neuroevolution algorithms**. Vanilla NEAT (Stanley &
Miikkulainen, 2002) is the first algorithm; the abstractions are designed to
accommodate NEAT variants (HyperNEAT, ES-HyperNEAT, novelty search) and deep
neuroevolution methods (Uber Deep GA, ES on fixed DNN topologies) without
redesigning the core.

## Status

| Phase | Deliverable | State |
|---|---|---|
| 1 | Backend skeleton — protocols, config, runner, evaluators, logging, viz | done |
| 2 | Vanilla NEAT — genome, mutations, crossover, speciation, phenotype | pending |
| 3 | XOR validation — `XORFitnessEvaluator`, baseline `examples/xor_baseline.py` | pending |

The public API in `polyneat/__init__.py` is stable inside Phase 1. Algorithm
protocols and runner will not change between phases.

## Installation

Requires Python 3.11 or newer. The recommended package manager is
[`uv`](https://github.com/astral-sh/uv).

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
```

If you prefer stdlib `venv` + `pip`:

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS / Linux
pip install -e ".[dev]"
```

## Project layout

```
polyneat/
├── config/               Config dataclasses (AlgorithmConfig, NEATConfig, LoggingConfig)
├── core/                 Component protocols + shared data types
├── algorithms/neat/      Vanilla NEAT implementation (Phase 2)
├── nn/                   Activation functions + topology helpers (Phase 2)
├── evaluators/           SequentialFitnessEvaluator base + ParallelFitnessEvaluatorWrapper
├── runner/               EvolutionRunner + RunContext + termination + callbacks
├── logging_utils/        Colored CustomLogger + get_logger factory
├── viz/                  Network topology renderer
└── utils/                Small helpers (RNG factory, JSON/pickle serialization)
```

## Core abstractions

Every algorithm composes seven `Protocol`s from
`polyneat.core.component_protocols`:

| Protocol | Responsibility |
|---|---|
| `Genome` | Immutable genotype; `clone_genome`, `to_serializable_dict` |
| `Phenotype` | Executable neural network; `forward_pass`, `reset_recurrent_state` |
| `PhenotypeBuilder` | `build_phenotype_from_genome(genome) -> Phenotype` |
| `MutationOperator` | `apply_to_genome(genome, rng, innovation_tracker) -> Genome` |
| `CrossoverOperator` | `apply_to_parents(fitter, less_fit, rng) -> Genome` |
| `ParentSelection` | `select_parents(genomes, fitnesses, n, rng) -> list[Genome]` |
| `Speciator` | `assign_genomes_to_species(genomes) -> list[SpeciesId]` |
| `FitnessEvaluator` | `evaluate_batch_of_phenotypes(phenotypes) -> list[FitnessValue]` |
| `NeuroevolutionAlgorithm` | Composes everything above; owns `create_initial_population` and `advance_one_generation` |

## Logging

Every module obtains its logger through the project's `CustomLogger`. It
supports coloured console output (via `colorama`) and optional per-logger file
handlers.

**Inside a module:**

```python
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)
logger.info("Hello, PolyNEAT")
```

**Configuring the logging subsystem** (call once at application startup, before
any `get_logger` call):

```python
import logging
from polyneat import LoggingConfig, set_logging_config

set_logging_config(
    LoggingConfig(
        log_level=logging.DEBUG,
        log_format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        file_log_directory="runs/logs",   # None disables file logging
    )
)
```

Do **not** call `logging.getLogger` directly anywhere in the codebase — always
go through `get_logger`.

## Running an experiment (Phase 2 preview)

Once NEAT lands in Phase 2, the entry point will look like this:

```python
from pathlib import Path

from polyneat import (
    NEATConfig,
    EvolutionRunner,
    MaxGenerationsTermination,
    TargetFitnessTermination,
    CompositeTermination,
    ConsoleStatisticsLogger,
    TensorBoardLogger,
    BestGenomePersister,
    NetworkTopologyVisualizer,
    ParallelFitnessEvaluatorWrapper,
)
# from polyneat.algorithms.neat.algorithm import NEATAlgorithm     # Phase 2
# from polyneat.evaluators.xor_evaluator import XORFitnessEvaluator  # Phase 3

config = NEATConfig.load_from_yaml_file(Path("examples/xor_baseline.yaml"))
# algorithm = NEATAlgorithm.from_config(config)
# xor_evaluator = XORFitnessEvaluator()
# parallel_evaluator = ParallelFitnessEvaluatorWrapper(xor_evaluator, number_of_parallel_workers=-1)

run_output_directory = Path("runs") / "xor_baseline"

runner = EvolutionRunner(
    algorithm=algorithm,
    fitness_evaluator=parallel_evaluator,
    termination_criterion=CompositeTermination([
        MaxGenerationsTermination(max_generations=300),
        TargetFitnessTermination(target_fitness=3.95),
    ]),
    callbacks=[
        ConsoleStatisticsLogger(),
        TensorBoardLogger(log_directory=run_output_directory),
        BestGenomePersister(output_directory=run_output_directory),
        NetworkTopologyVisualizer(output_directory=run_output_directory, render_every_n_generations=50),
    ],
    random_seed=config.random_seed,
)

result = runner.run_evolution()
print(f"Best fitness: {result.best_fitness_ever_achieved:.4f}")
print(f"Reason: {result.termination_reason}")
```

Each run writes artefacts to `runs/<run_id>/`:

```
runs/<run_id>/
├── events.out.tfevents.*        TensorBoard event file
├── best_genome.json             Human-readable best genome
├── best_genome.pkl              Fast-reload best genome
└── topology/                    PNG / SVG topology renders
    ├── gen_0050_best.png
    └── final_best.svg
```

## Live monitoring

```bash
tensorboard --logdir runs/
```

## Development

```bash
uv pip install -e ".[dev]"
ruff check polyneat
ruff format polyneat
```

## Design document

The full architecture rationale lives in
[`docs/superpowers/specs/2026-06-30-poly-neat-library-design.md`](docs/superpowers/specs/2026-06-30-poly-neat-library-design.md).
