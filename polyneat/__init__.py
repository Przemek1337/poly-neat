"""PolyNEAT — a neuroevolution library whose core is NEAT.

The full NEAT implementation (Stanley & Miikkulainen, 2002) lives in
``polyneat.core.neat``; derived algorithms such as FS-NEAT subclass
``NEATAlgorithm`` and override only what they change. This module re-exports
the public API.
"""

from polyneat.algorithms.cneat.class_genome_container import ClassGenomeContainer
from polyneat.algorithms.cneat.cneat_algorithm import CNEATAlgorithm
from polyneat.algorithms.cneat.container_ensemble_phenotype import ContainerEnsemblePhenotype
from polyneat.algorithms.cneat.container_progress_logger import ContainerProgressLogger
from polyneat.algorithms.cneat.container_update_callback import ContainerUpdateCallback
from polyneat.algorithms.fsneat.fsneat_algorithm import FSNEATAlgorithm
from polyneat.algorithms.lneat.backpropagation_weight_trainer import (
    BackpropagationWeightTrainer,
)
from polyneat.algorithms.lneat.lneat_algorithm import LNEATAlgorithm
from polyneat.algorithms.lneat.recognizer_ensemble_phenotype import (
    RecognizerEnsemblePhenotype,
)
from polyneat.algorithms.lneat.trainable_torch_phenotype import (
    TrainableTorchFeedForwardPhenotype,
)
from polyneat.algorithms.hyperneat.add_node_random_activation_mutation import (
    AddNodeWithRandomActivationMutation,
)
from polyneat.algorithms.hyperneat.hyperneat_algorithm import HyperNEATAlgorithm
from polyneat.algorithms.hyperneat.hyperneat_phenotype_decoder import (
    HyperNEATPhenotypeDecoder,
)
from polyneat.algorithms.hyperneat.substrate import (
    Substrate,
    SubstrateLayer,
    SubstrateNode,
    build_grid_sandwich_substrate,
    build_layered_substrate,
)
from polyneat.algorithms.neatdbm.difference_based_weight_mutation import (
    DifferenceBasedWeightMutation,
)
from polyneat.algorithms.neatdbm.neatdbm_algorithm import NEATDBMAlgorithm
from polyneat.config.algorithm_config import AlgorithmConfig
from polyneat.config.cneat_config import CNEATConfig
from polyneat.config.configuration_errors import ConfigurationError
from polyneat.config.lneat_config import LNEATConfig
from polyneat.config.hyperneat_config import HyperNEATConfig
from polyneat.config.neat_config import NEATConfig
from polyneat.config.neatdbm_config import NEATDBMConfig
from polyneat.core.component_protocols import (
    CrossoverOperator,
    FitnessEvaluator,
    Genome,
    InitialPopulationStrategy,
    InnovationTracker,
    MutationOperator,
    NeuroevolutionAlgorithm,
    ParentSelection,
    Phenotype,
    PhenotypeDecoder,
    Speciator,
)
from polyneat.core.generation_statistics import GenerationStatistics
from polyneat.core.neat.compatibility_distance_speciator import (
    CompatibilityDistanceSpeciator,
)
from polyneat.core.neat.global_innovation_tracker import GlobalInnovationTracker
from polyneat.core.neat.initial_population import (
    build_fs_neat_initial_population,
    build_fully_connected_initial_population,
    register_initial_population_strategy,
    resolve_initial_population_strategy_by_name,
)
from polyneat.core.neat.mutations.add_connection_mutation import AddConnectionMutation
from polyneat.core.neat.mutations.add_node_mutation import AddNodeMutation
from polyneat.core.neat.mutations.composite_neat_mutation import CompositeNEATMutation
from polyneat.core.neat.mutations.toggle_connection_enabled_mutation import (
    ToggleConnectionEnabledMutation,
)
from polyneat.core.neat.mutations.weight_modification_mutation import (
    WeightModificationMutation,
)
from polyneat.core.neat.neat_algorithm import NEATAlgorithm
from polyneat.core.neat.neat_crossover import NEATCrossover
from polyneat.core.neat.neat_genome import (
    ConnectionGene,
    InvalidGenomeError,
    NEATGenome,
    NodeGene,
)
from polyneat.core.neat.neat_phenotype_decoder import NEATPhenotypeDecoder
from polyneat.core.neat.torch_feedforward_phenotype import TorchFeedForwardPhenotype
from polyneat.core.neat.tournament_parent_selection import TournamentParentSelection
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
    "PhenotypeDecoder",
    "MutationOperator",
    "CrossoverOperator",
    "ParentSelection",
    "Speciator",
    "FitnessEvaluator",
    "InitialPopulationStrategy",
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
    "FSNEATAlgorithm",
    # L-NEAT algorithm
    "LNEATConfig",
    "LNEATAlgorithm",
    "BackpropagationWeightTrainer",
    "TrainableTorchFeedForwardPhenotype",
    "RecognizerEnsemblePhenotype",
    "build_fully_connected_initial_population",
    "build_fs_neat_initial_population",
    "register_initial_population_strategy",
    "resolve_initial_population_strategy_by_name",
    "NEATGenome",
    "NodeGene",
    "ConnectionGene",
    "InvalidGenomeError",
    "GlobalInnovationTracker",
    "NEATPhenotypeDecoder",
    "TorchFeedForwardPhenotype",
    "NEATCrossover",
    "CompatibilityDistanceSpeciator",
    "TournamentParentSelection",
    "CompositeNEATMutation",
    "AddNodeMutation",
    "AddConnectionMutation",
    "WeightModificationMutation",
    "ToggleConnectionEnabledMutation",
    # HyperNEAT algorithm
    "HyperNEATConfig",
    "HyperNEATAlgorithm",
    "HyperNEATPhenotypeDecoder",
    "Substrate",
    "SubstrateLayer",
    "SubstrateNode",
    "build_layered_substrate",
    "build_grid_sandwich_substrate",
    "AddNodeWithRandomActivationMutation",
    # C-NEAT algorithm
    "CNEATConfig",
    "CNEATAlgorithm",
    "ClassGenomeContainer",
    "ContainerUpdateCallback",
    "ContainerEnsemblePhenotype",
    "ContainerProgressLogger",
    # NEAT-DBM algorithm
    "NEATDBMConfig",
    "NEATDBMAlgorithm",
    "DifferenceBasedWeightMutation",
]
