"""Sequential evaluation with explicit boundaries for stop and resume.

This opt-in loop leaves the historical EvolutionRunner API unchanged. Algorithm
reproduction remains an atomic boundary: a interrupted generation transition is
repeated from its input state, including any algorithm-owned training.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol

import numpy as np

from polyneat.core.component_protocols import Genome, NeuroevolutionAlgorithm, Phenotype
from polyneat.core.population import Population
from polyneat.logging_utils.custom_logger import get_logger
from polyneat.runner.evaluation_record import EvaluationRecord
from polyneat.runner.evolution_runner import NoSuccessfulEvaluationError
from polyneat.runner.run_checkpoint import SupportsEvolutionStateExport
from polyneat.runner.search_session import SearchSession

logger = get_logger(__name__)


def _budget_note(session: SearchSession) -> str:
    """A short ', <n>s used' when a wall-clock budget is bounding the search.

    The search evaluators are otherwise silent per candidate, so a run against a
    long budget looks dead for minutes. This makes each candidate and each
    generation visible, including how much of the budget has been consumed.
    """
    if session.budget is None:
        return ""
    return f", {session.budget.consumed_seconds:.0f}s used"


class StatefulAlgorithm(NeuroevolutionAlgorithm, SupportsEvolutionStateExport, Protocol):
    pass


class StatefulEvaluator(Protocol):
    def evaluate_candidate(self, phenotype: Phenotype, evaluation_id: str) -> EvaluationRecord: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...


def run_resumable_evolution(
    algorithm: StatefulAlgorithm,
    evaluator: StatefulEvaluator,
    *,
    session: SearchSession,
    genome_from_dict: Callable[[dict], Genome],
    seed: int,
    number_of_generations: int,
    export_training_state: Callable[[], dict] = dict,
    import_training_state: Callable[[dict], None] = lambda state: None,
    prepare_genome: Callable[[Genome], Genome] = lambda genome: genome,
) -> tuple[Genome, float, int]:
    """Return the best completed candidate, including one from a partial generation."""
    if number_of_generations < 1:
        raise ValueError("number_of_generations must be positive")
    rng = np.random.default_rng(seed)
    state = session.state
    population = None
    fitnesses: list[float] = []
    best_genome = None
    best_fitness = None
    best_size = None
    completed = 0
    done = False
    if state is not None:
        if state["kind"] != "evolution":
            raise ValueError("snapshot belongs to a different search kind")
        algorithm.import_evolution_state(state["algorithm"])
        evaluator.load_state_dict(state["evaluator"])
        import_training_state(state["training"])
        rng.bit_generator.state = state["rng"]
        if state["population"] is not None:
            population = Population(
                genomes=[genome_from_dict(row) for row in state["population"]],
                species_assignments=state["species"],
                generation_number=state["generation"],
            )
        fitnesses = state["fitnesses"]
        best_genome = None if state["best"] is None else genome_from_dict(state["best"])
        best_fitness, best_size = state["best_fitness"], state["best_size"]
        completed, done = state["completed"], state["done"]

    def commit() -> None:
        session.commit(
            {
                "kind": "evolution",
                "algorithm": algorithm.export_evolution_state(),
                "evaluator": evaluator.state_dict(),
                "training": export_training_state(),
                "rng": rng.bit_generator.state,
                "population": None
                if population is None
                else [genome.to_serializable_dict() for genome in population.genomes],
                "species": [] if population is None else population.species_assignments,
                "generation": 0 if population is None else population.generation_number,
                "fitnesses": list(fitnesses),
                "best": None if best_genome is None else best_genome.to_serializable_dict(),
                "best_fitness": best_fitness,
                "best_size": best_size,
                "completed": completed,
                "done": done,
            }
        )

    def expired() -> bool:
        return session.budget is not None and session.budget.should_stop()

    if state is None:
        commit()  # Also makes interruptions during initial EXACT training resumable.
    if population is None and not expired():
        population = algorithm.create_initial_population(rng)
        commit()
    while population is not None and not done and not expired():
        for position in range(len(fitnesses), len(population.genomes)):
            if expired():
                break
            prepared = prepare_genome(population.genomes[position])
            population = Population(
                genomes=[
                    prepared if index == position else genome
                    for index, genome in enumerate(population.genomes)
                ],
                species_assignments=population.species_assignments,
                generation_number=population.generation_number,
            )
            phenotype = algorithm.phenotype_decoder.build_phenotype_from_genome(
                population.genomes[position]
            )
            record = evaluator.evaluate_candidate(
                phenotype, f"gen{population.generation_number}/cand{position}"
            )
            del phenotype
            selectable = record.is_selectable and not expired()
            value = record.fitness if selectable else None
            fitnesses.append(float("-inf") if value is None else float(value))
            if value is not None and (
                best_fitness is None
                or value > best_fitness
                or (value == best_fitness and best_size is not None
                    and record.parameter_count < best_size)
            ):
                best_genome = population.genomes[position]
                best_fitness, best_size = value, record.parameter_count
            logger.info(
                "gen %d cand %d/%d: %s%s, %d params%s",
                population.generation_number,
                position,
                len(population.genomes),
                record.status.value,
                "" if value is None else f", fitness {value:.4f}",
                record.parameter_count or 0,
                _budget_note(session),
            )
            commit()
        if expired():
            break
        completed = population.generation_number + 1
        logger.info(
            "generation %d complete: %d candidates, best fitness %s%s",
            population.generation_number,
            len(fitnesses),
            "n/a" if best_fitness is None else f"{best_fitness:.4f}",
            _budget_note(session),
        )
        if completed >= number_of_generations or not any(map(math.isfinite, fitnesses)):
            done = True
            commit()
            break
        # A transition is committed only when fully constructed. On interrupt
        # the previous population, RNG, scorer and training streams are restored.
        population, _statistics = algorithm.advance_one_generation(population, fitnesses, rng)
        fitnesses = []
        commit()
    if best_genome is None or best_fitness is None:
        raise NoSuccessfulEvaluationError(
            "search finished without a successful completed candidate"
        )
    done = True
    commit()
    return best_genome, float(best_fitness), completed


def run_resumable_candidates(
    algorithm: StatefulAlgorithm,
    evaluator: StatefulEvaluator,
    *,
    session: SearchSession,
    genome_from_dict: Callable[[dict], Genome],
    rng: np.random.Generator,
    draw: Callable[[], Genome],
    count: int,
) -> tuple[Genome, float]:
    """Independent search candidates, streamed rather than all resident on GPU."""
    position = 0
    best = None
    fitness = None
    size = None
    if session.state is not None:
        state = session.state
        if state["kind"] != "candidates":
            raise ValueError("snapshot belongs to a different search kind")
        algorithm.import_evolution_state(state["algorithm"])
        evaluator.load_state_dict(state["evaluator"])
        rng.bit_generator.state = state["rng"]
        position, fitness, size = state["position"], state["fitness"], state["size"]
        best = None if state["best"] is None else genome_from_dict(state["best"])

    def commit() -> None:
        session.commit(
            {
                "kind": "candidates",
                "position": position,
                "algorithm": algorithm.export_evolution_state(),
                "evaluator": evaluator.state_dict(),
                "rng": rng.bit_generator.state,
                "best": None if best is None else best.to_serializable_dict(),
                "fitness": fitness,
                "size": size,
            }
        )

    commit()
    while position < count:
        if session.budget is not None and session.budget.should_stop():
            break
        genome = draw()
        phenotype = algorithm.phenotype_decoder.build_phenotype_from_genome(genome)
        record = evaluator.evaluate_candidate(phenotype, f"gen0/cand{position}")
        del phenotype
        if (
            record.is_selectable
            and record.fitness is not None
            and (
                fitness is None
                or record.fitness > fitness
                or (record.fitness == fitness and size is not None
                    and record.parameter_count < size)
            )
        ):
            best, fitness, size = genome, record.fitness, record.parameter_count
        logger.info(
            "cand %d/%d: %s%s, %d params%s",
            position,
            count,
            record.status.value,
            "" if record.fitness is None else f", fitness {record.fitness:.4f}",
            record.parameter_count or 0,
            _budget_note(session),
        )
        position += 1
        commit()
    if best is None or fitness is None:
        raise NoSuccessfulEvaluationError("no successful independent candidate")
    return best, float(fitness)
