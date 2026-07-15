from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from polyneat.core.neat.tournament_parent_selection import TournamentParentSelection


@dataclass(frozen=True)
class _StubGenome:
    """Minimal structural Genome stand-in that is NOT a NEATGenome."""

    label: str

    def to_serializable_dict(self) -> dict:
        return {"label": self.label}

    @classmethod
    def from_serializable_dict(cls, payload: dict) -> _StubGenome:
        return cls(label=payload["label"])


def test_selection_works_with_non_neat_genome_type() -> None:
    genomes = [_StubGenome("weak"), _StubGenome("strong")]
    fitnesses = [0.0, 10.0]
    selection: TournamentParentSelection[_StubGenome] = TournamentParentSelection(tournament_size=5)
    selected = selection.select_parents(
        candidate_genomes=genomes,
        candidate_fitnesses=fitnesses,
        number_of_parents_to_select=4,
        rng=np.random.default_rng(0),
    )
    assert len(selected) == 4
    assert all(isinstance(selected_genome, _StubGenome) for selected_genome in selected)


def test_large_tournament_prefers_fitter_genome() -> None:
    genomes = [_StubGenome("weak"), _StubGenome("strong")]
    fitnesses = [0.0, 10.0]
    selection: TournamentParentSelection[_StubGenome] = TournamentParentSelection(tournament_size=5)
    selected = selection.select_parents(
        candidate_genomes=genomes,
        candidate_fitnesses=fitnesses,
        number_of_parents_to_select=20,
        rng=np.random.default_rng(1),
    )
    strong_count = sum(1 for genome in selected if genome.label == "strong")
    # P(weak wins one 5-way tournament) = 0.5**5 ≈ 3%; 15/20 is a very safe bound.
    assert strong_count >= 15


def test_tournament_size_below_one_rejected() -> None:
    with pytest.raises(ValueError, match="tournament_size"):
        TournamentParentSelection(tournament_size=0)
