from __future__ import annotations

from typing import Generic

from numpy.random import Generator

from polyneat.core.component_protocols import GenomeType
from polyneat.core.type_aliases import FitnessValue


class TournamentParentSelection(Generic[GenomeType]):
    """K-way tournament: sample ``tournament_size`` candidates with replacement,
    return the one with the highest fitness.

    Larger ``tournament_size`` values push selection pressure up (favours the
    fittest more aggressively); ``tournament_size=1`` is uniform random.
    """

    def __init__(self, tournament_size: int) -> None:
        if tournament_size < 1:
            raise ValueError(
                f"TournamentParentSelection.tournament_size must be >= 1, got {tournament_size}"
            )
        self._tournament_size = tournament_size

    def select_parents(
        self,
        candidate_genomes: list[GenomeType],
        candidate_fitnesses: list[FitnessValue],
        number_of_parents_to_select: int,
        rng: Generator,
    ) -> list[GenomeType]:
        """Run one tournament per requested parent.

        Args:
            candidate_genomes: Pool to select from.
            candidate_fitnesses: Fitness per candidate, aligned by index.
            number_of_parents_to_select: How many parents to return.
            rng: Source of randomness for tournament sampling.

        Returns:
            The selected parents (possibly with repeats).

        Raises:
            ValueError: If the pool is empty or the lists are misaligned.
        """
        if len(candidate_genomes) != len(candidate_fitnesses):
            raise ValueError(
                f"TournamentParentSelection: candidate_genomes has length "
                f"{len(candidate_genomes)} but candidate_fitnesses has length "
                f"{len(candidate_fitnesses)}"
            )
        if not candidate_genomes:
            raise ValueError("TournamentParentSelection: no candidate genomes given")

        selected_parent_genomes: list[GenomeType] = []
        for _selection_step_index in range(number_of_parents_to_select):
            tournament_participant_indices = rng.integers(
                low=0,
                high=len(candidate_genomes),
                size=self._tournament_size,
            )
            winner_index_in_population = max(
                tournament_participant_indices,
                key=lambda participant_index: candidate_fitnesses[int(participant_index)],
            )
            selected_parent_genomes.append(candidate_genomes[int(winner_index_in_population)])
        return selected_parent_genomes
