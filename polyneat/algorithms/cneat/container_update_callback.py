from __future__ import annotations

from typing import TYPE_CHECKING

from polyneat.algorithms.cneat.class_genome_container import ClassGenomeContainer
from polyneat.core.type_aliases import FitnessValue
from polyneat.evaluators.class_indexed_evaluator_base import ClassIndexedFitnessEvaluator
from polyneat.runner.evolution_callback_protocol import BaseEvolutionCallback

if TYPE_CHECKING:
    from polyneat.core.population import Population
    from polyneat.runner.run_context import RunContext


class ContainerUpdateCallback(BaseEvolutionCallback):
    """Keeps a :class:`ClassGenomeContainer` up to date during a C-NEAT run.

    After every fitness evaluation, each organism is offered to the container
    cell of its assigned class (organism index modulo container size,
    equation 1 of Alfaham et al., 2024). The container accepts only strict
    improvements, so it always holds the best genome found so far for every
    class label.

    The container's correctness depends on the evaluator and the container
    using the same class count — a silent mismatch would store genomes
    evaluated for class ``i`` into cell ``j`` — so the constructor takes the
    evaluator and fails fast on disagreement. Register the callback with
    :class:`~polyneat.runner.evolution_runner.EvolutionRunner`.
    """

    def __init__(
        self,
        container: ClassGenomeContainer,
        fitness_evaluator: ClassIndexedFitnessEvaluator,
    ) -> None:
        """Bind the container to the evaluator whose assignments it mirrors.

        Args:
            container: Best-genome-per-class store to keep updated.
            fitness_evaluator: The class-indexed evaluator used in the run;
                only its ``number_of_class_labels`` is consulted.

        Raises:
            ValueError: If container and evaluator disagree on the class count.
        """
        if container.number_of_class_labels != fitness_evaluator.number_of_class_labels:
            raise ValueError(
                f"container has {container.number_of_class_labels} class labels but "
                f"the fitness evaluator has {fitness_evaluator.number_of_class_labels}; "
                f"construct both from the same CNEATConfig.number_of_class_labels"
            )
        self._container = container

    def on_population_evaluated(
        self,
        context: RunContext,
        population: Population,
        fitnesses: list[FitnessValue],
    ) -> None:
        """Offer every evaluated organism to its class cell in the container."""
        for organism_index, (genome, fitness) in enumerate(
            zip(population.genomes, fitnesses, strict=True)
        ):
            class_label_index = self._container.assigned_class_for_organism_index(organism_index)
            self._container.offer_genome_for_class(class_label_index, genome, fitness)
