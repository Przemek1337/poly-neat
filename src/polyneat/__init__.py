from polyneat.config.base import AlgorithmConfig
from polyneat.config.errors import ConfigurationError
from polyneat.config.neat_config import NEATConfig
from polyneat.core.population import Population
from polyneat.core.protocols import (
    CrossoverOperator,
    FitnessEvaluator,
    Genome,
    MutationOperator,
    NeuroevolutionAlgorithm,
    ParentSelection,
    Phenotype,
    PhenotypeBuilder,
    Speciator,
)
from polyneat.core.statistics import GenerationStatistics
from polyneat.core.type_aliases import FitnessValue, InnovationId, SpeciesId
from polyneat.evaluators.parallel import ParallelFitnessEvaluatorWrapper
from polyneat.runner.builtin_callbacks import (
    BestGenomePersister,
    ConsoleStatisticsLogger,
    NetworkTopologyVisualizer,
    TensorBoardLogger,
)
from polyneat.runner.callbacks import BaseEvolutionCallback, EvolutionCallback
from polyneat.runner.context import RunContext
from polyneat.runner.evolution_runner import EvolutionResult, EvolutionRunner
from polyneat.runner.termination import (
    CompositeTermination,
    FitnessStagnationTermination,
    MaxGenerationsTermination,
    TargetFitnessTermination,
)

__all__ = [
    # Config
    "AlgorithmConfig",
    "NEATConfig",
    "ConfigurationError",
    # Core protocols
    "Genome",
    "Phenotype",
    "PhenotypeBuilder",
    "MutationOperator",
    "CrossoverOperator",
    "ParentSelection",
    "Speciator",
    "FitnessEvaluator",
    "NeuroevolutionAlgorithm",
    # Data types
    "Population",
    "GenerationStatistics",
    "RunContext",
    "EvolutionResult",
    # Type aliases
    "FitnessValue",
    "InnovationId",
    "SpeciesId",
    # Runner
    "EvolutionRunner",
    "EvolutionCallback",
    "BaseEvolutionCallback",
    # Termination
    "MaxGenerationsTermination",
    "TargetFitnessTermination",
    "FitnessStagnationTermination",
    "CompositeTermination",
    # Built-in callbacks
    "ConsoleStatisticsLogger",
    "TensorBoardLogger",
    "BestGenomePersister",
    "NetworkTopologyVisualizer",
    # Evaluators
    "ParallelFitnessEvaluatorWrapper",
]
