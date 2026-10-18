"""Resume must reproduce the search, not merely reload the incumbent."""

import json

import pytest
import torch

from polyneat.algorithms.deepneat.deepneat_algorithm import DeepNEATAlgorithm
from polyneat.algorithms.deepneat.deepneat_genome import DeepNEATGenome
from polyneat.algorithms.exact.exact_algorithm import EXACTAlgorithm
from polyneat.algorithms.exact.exact_genome import EXACTGenome
from polyneat.configs.deepneat.deepneat_config import DeepNEATConfig
from polyneat.configs.exact.exact_config import EXACTConfig
from polyneat.runner.evaluation_record import EvaluationRecord, EvaluationStatus
from polyneat.runner.resumable_search import run_resumable_evolution
from polyneat.runner.run_checkpoint import CheckpointResumeError
from polyneat.runner.search_session import SearchSession
from polyneat.runner.wall_clock_budget import WallClockBudget


class Evaluator:
    def __init__(self, interrupt_after=None):
        self.records = []
        self.interrupt_after = interrupt_after

    def evaluate_candidate(self, phenotype, evaluation_id):
        if len(self.records) == self.interrupt_after:
            raise KeyboardInterrupt
        score = float(torch.rand(()))
        self.records.append((evaluation_id, score))
        return EvaluationRecord(
            evaluation_id=evaluation_id,
            status=EvaluationStatus.SUCCEEDED,
            fitness=score,
            parameter_count=10,
        )

    def state_dict(self):
        return {"records": list(self.records)}

    def load_state_dict(self, state):
        self.records = state["records"]


def algorithm_for(kind):
    if kind == "deepneat":
        return DeepNEATAlgorithm.from_config(
            DeepNEATConfig(
                population_size=3,
                number_of_input_nodes=1,
                number_of_output_nodes=2,
                number_of_classes=2,
                input_image_channels=1,
                input_image_height=8,
                input_image_width=8,
                available_filter_counts=(2,),
                available_dense_unit_counts=(4,),
            )
        ), DeepNEATGenome.from_serializable_dict
    return EXACTAlgorithm.from_config(
        EXACTConfig(
            population_size=3,
            number_of_input_nodes=1,
            number_of_output_nodes=2,
            input_image_height=8,
            input_image_width=8,
            use_simplex_hyperparameter_optimization=False,
        )
    ), EXACTGenome.from_serializable_dict


@pytest.mark.parametrize("kind", ["deepneat", "exact"])
@pytest.mark.parametrize("interrupt_after", [1, 4, 7])
def test_resume_reproduces_population_innovations_rng_and_scores(tmp_path, kind, interrupt_after):
    torch.manual_seed(99)
    continuous, factory = algorithm_for(kind)
    scorer = Evaluator()
    with SearchSession(None, binding={}, budget=None) as session:
        expected = run_resumable_evolution(
            continuous,
            scorer,
            session=session,
            genome_from_dict=factory,
            seed=33,
            number_of_generations=3,
        )
    expected_algorithm_state = continuous.export_evolution_state()

    torch.manual_seed(99)
    interrupted, factory = algorithm_for(kind)
    with pytest.raises(KeyboardInterrupt):
        with SearchSession(tmp_path, binding={"config": "same"}, budget=None) as session:
            run_resumable_evolution(
                interrupted,
                Evaluator(interrupt_after),
                session=session,
                genome_from_dict=factory,
                seed=33,
                number_of_generations=3,
            )
    # Perturb the process RNG and construct new components, as in a new process.
    torch.manual_seed(123456)
    resumed, factory = algorithm_for(kind)
    resumed_scorer = Evaluator()
    with SearchSession(tmp_path, binding={"config": "same"}, budget=None, resume=True) as session:
        actual = run_resumable_evolution(
            resumed,
            resumed_scorer,
            session=session,
            genome_from_dict=factory,
            seed=33,
            number_of_generations=3,
        )
    assert actual[0].to_serializable_dict() == expected[0].to_serializable_dict()
    assert actual[1:] == expected[1:]
    assert resumed_scorer.records == scorer.records
    # Timing fields in stagnation state do not exist; all genetic state is exact.
    assert resumed.export_evolution_state() == expected_algorithm_state


def test_interrupted_work_is_charged_and_downtime_is_not(tmp_path):
    now = [0.0]
    budget = WallClockBudget(100, clock=lambda: now[0])
    with pytest.raises(KeyboardInterrupt):
        with SearchSession(tmp_path, binding={}, budget=budget) as session:
            session.commit({"checkpoint": 1})
            now[0] = 7
            raise KeyboardInterrupt
    now[0] = 1000  # A long infrastructure outage is not training time.
    resumed_budget = WallClockBudget(100, clock=lambda: now[0])
    with SearchSession(tmp_path, binding={}, budget=resumed_budget, resume=True):
        assert resumed_budget.consumed_seconds == 7
        now[0] += 3
        assert resumed_budget.consumed_seconds == 10


def test_unclean_shutdown_requires_lost_time_and_binding_is_enforced(tmp_path):
    with SearchSession(tmp_path, binding={"config": 1}, budget=WallClockBudget(100)) as session:
        session.commit({"checkpoint": 1})
    journal_path = tmp_path / "budget.json"
    journal = json.loads(journal_path.read_text())
    journal["running"] = True
    journal_path.write_text(json.dumps(journal))
    with pytest.raises(CheckpointResumeError, match="unclean"):
        with SearchSession(
            tmp_path, binding={"config": 1}, budget=WallClockBudget(100), resume=True
        ):
            pass
    with pytest.raises(CheckpointResumeError, match="changed"):
        with SearchSession(tmp_path, binding={"config": 2}, budget=None, resume=True):
            pass


def test_corrupt_snapshot_is_rejected(tmp_path):
    with SearchSession(tmp_path, binding={}, budget=None) as session:
        session.commit({"checkpoint": 1})
    pointer = json.loads((tmp_path / "latest.json").read_text())
    (tmp_path / pointer["snapshot"]).write_bytes(b"corrupt")
    with pytest.raises(CheckpointResumeError, match="checksum"):
        with SearchSession(tmp_path, binding={}, budget=None, resume=True):
            pass
