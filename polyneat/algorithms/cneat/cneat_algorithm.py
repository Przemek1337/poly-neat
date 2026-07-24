from __future__ import annotations

from dataclasses import dataclass

from polyneat.configs.cneat.cneat_config import CNEATConfig
from polyneat.core.neat.neat_algorithm import NEATAlgorithm


@dataclass
class CNEATAlgorithm(NEATAlgorithm):
    """C-NEAT (Alfaham, Van Raemdonck & Mercelis, 2024).

    C-NEAT leaves NEAT's genetics untouched (section IV of the
    paper: "without changing its core implementation"), so this class inherits
    every operator, factory, and the generational loop from
    :class:`~polyneat.core.neat.neat_algorithm.NEATAlgorithm` unchanged.
    Build it with the inherited ``CNEATAlgorithm.from_config(cneat_config)``.
    What makes a run C-NEAT is the surrounding wiring:

    - a :class:`~polyneat.evaluators.class_indexed_evaluator_base.ClassIndexedFitnessEvaluator`
      scores each organism only on its assigned class label
      (``organism_index mod number_of_class_labels``), and
    - a :class:`~polyneat.algorithms.cneat.container_update_callback.ContainerUpdateCallback`
      preserves the best genome per class in a
      :class:`~polyneat.algorithms.cneat.class_genome_container.ClassGenomeContainer`,
      from which
      :class:`~polyneat.algorithms.cneat.container_ensemble_phenotype.ContainerEnsemblePhenotype`
      builds the final argmax classifier.

    The subclass carries the
    :class:`~polyneat.configs.cneat.cneat_config.CNEATConfig` whose
    ``number_of_class_labels`` the evaluator and container share.

    References:
        Alfaham, A., Van Raemdonck, S., & Mercelis, S. (2024). Genetic
        NEAT-Based Method for Multi-class Classification. *ACAI 2024*.
        DOI: 10.1109/ACAI63924.2024.10899662
    """

    config: CNEATConfig
