from polyneat.algorithms.neat.compatibility_distance_speciator import (
    CompatibilityDistanceSpeciator,
)
from polyneat.algorithms.neat.global_innovation_tracker import GlobalInnovationTracker
from polyneat.algorithms.neat.mutations.add_connection_mutation import AddConnectionMutation
from polyneat.algorithms.neat.mutations.add_node_mutation import AddNodeMutation
from polyneat.algorithms.neat.mutations.composite_neat_mutation import CompositeNEATMutation
from polyneat.algorithms.neat.mutations.toggle_connection_enabled_mutation import (
    ToggleConnectionEnabledMutation,
)
from polyneat.algorithms.neat.mutations.weight_modification_mutation import (
    WeightModificationMutation,
)
from polyneat.algorithms.neat.neat_algorithm import NEATAlgorithm
from polyneat.algorithms.neat.neat_crossover import NEATCrossover
from polyneat.algorithms.neat.neat_genome import (
    ConnectionGene,
    InvalidGenomeError,
    NEATGenome,
    NodeGene,
)
from polyneat.algorithms.neat.neat_phenotype_builder import NEATPhenotypeBuilder
from polyneat.algorithms.neat.torch_feedforward_phenotype import TorchFeedForwardPhenotype
from polyneat.algorithms.neat.tournament_parent_selection import TournamentParentSelection
from polyneat.config.algorithm_config import AlgorithmConfig
from polyneat.config.configuration_errors import ConfigurationError
from polyneat.config.neat_config import NEATConfig
from polyneat.core.component_protocols import (
    CrossoverOperator,
    FitnessEvaluator,
    Genome,
    InnovationTracker,
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
from polyneat.evaluators.xor_evaluator import XORFitnessEvaluator
from polyneat.logging_utils.custom_logger import (
    CustomLogger,
    get_active_logging_config,
    get_logger,
    set_logging_config,
)
from polyneat.logging_utils.logging_config import LoggingConfig
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
    "InnovationTracker",
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
    "XORFitnessEvaluator",
    # Logging
    "CustomLogger",
    "get_logger",
    "set_logging_config",
    "get_active_logging_config",
    # NEAT algorithm
    "NEATAlgorithm",
    "NEATGenome",
    "NodeGene",
    "ConnectionGene",
    "InvalidGenomeError",
    "GlobalInnovationTracker",
    "NEATPhenotypeBuilder",
    "TorchFeedForwardPhenotype",
    "NEATCrossover",
    "CompatibilityDistanceSpeciator",
    "TournamentParentSelection",
    "CompositeNEATMutation",
    "AddNodeMutation",
    "AddConnectionMutation",
    "WeightModificationMutation",
    "ToggleConnectionEnabledMutation",
]
