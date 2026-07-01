from polyneat.config.algorithm_config import AlgorithmConfig
from polyneat.config.configuration_errors import ConfigurationError
from polyneat.config.logging_config import LoggingConfig
from polyneat.config.neat_config import NEATConfig
from polyneat.core.component_protocols import (
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
from polyneat.core.generation_statistics import GenerationStatistics
from polyneat.core.population import Population
from polyneat.core.type_aliases import FitnessValue, InnovationId, SpeciesId
from polyneat.evaluators.parallel_evaluator_wrapper import ParallelFitnessEvaluatorWrapper
from polyneat.logging_utils.custom_logger import (
    CustomLogger,
    get_active_logging_config,
    get_logger,
    set_logging_config,
)
from polyneat.runner.builtin_evolution_callbacks import (
    BestGenomePersister,
    ConsoleStatisticsLogger,
    NetworkTopologyVisualizer,
    TensorBoardLogger,
)
from polyneat.runner.evolution_callback_protocol import (
    BaseEvolutionCallback,
    EvolutionCallback,
)
from polyneat.runner.evolution_runner import EvolutionResult, EvolutionRunner
from polyneat.runner.run_context import RunContext
from polyneat.runner.termination_criteria import (
    CompositeTermination,
    FitnessStagnationTermination,
    MaxGenerationsTermination,
    TargetFitnessTermination,
)

__all__ = [
    # Config
    "AlgorithmConfig",
    "NEATConfig",
    "LoggingConfig",
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
    # Logging
    "CustomLogger",
    "get_logger",
    "set_logging_config",
    "get_active_logging_config",
]
