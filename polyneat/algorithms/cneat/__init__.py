"""C-NEAT: container-based NEAT for multi-class classification.

References:
    Alfaham, A., Van Raemdonck, S., & Mercelis, S. (2024). Genetic NEAT-Based
    Method for Multi-class Classification. *ACAI 2024*.
    DOI: 10.1109/ACAI63924.2024.10899662
"""

from polyneat.algorithms.cneat.class_genome_container import ClassGenomeContainer
from polyneat.algorithms.cneat.cneat_algorithm import CNEATAlgorithm
from polyneat.algorithms.cneat.container_ensemble_phenotype import ContainerEnsemblePhenotype
from polyneat.algorithms.cneat.container_progress_logger import ContainerProgressLogger
from polyneat.algorithms.cneat.container_update_callback import ContainerUpdateCallback

__all__ = [
    "CNEATAlgorithm",
    "ClassGenomeContainer",
    "ContainerEnsemblePhenotype",
    "ContainerProgressLogger",
    "ContainerUpdateCallback",
]
