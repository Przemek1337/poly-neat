"""Layer assignment and the two topology artifacts, on hand-built genomes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from gpu_sweep.topology_report import (
    compute_layer_index_by_node_id,
    describe_topology,
    render_layered_topology,
    render_topology_records,
    topology_record_from_dict,
    write_topology_record,
    write_topology_report,
)


@dataclass(frozen=True)
class FakeNodeGene:
    node_id: int
    node_type: str
    activation_function_name: str = "tanh"


@dataclass(frozen=True)
class FakeConnectionGene:
    source_node_id: int
    target_node_id: int
    weight: float = 1.0
    is_enabled: bool = True


@dataclass(frozen=True)
class FakeGenome:
    node_genes: list[FakeNodeGene]
    connection_genes: list[FakeConnectionGene]


def build_direct_genome() -> FakeGenome:
    """Two inputs and a bias wired straight to one output - no hidden nodes."""
    return FakeGenome(
        node_genes=[
            FakeNodeGene(0, "input"),
            FakeNodeGene(1, "input"),
            FakeNodeGene(2, "bias"),
            FakeNodeGene(3, "output"),
        ],
        connection_genes=[
            FakeConnectionGene(0, 3),
            FakeConnectionGene(1, 3),
            FakeConnectionGene(2, 3),
        ],
    )


def build_two_hidden_layer_genome() -> FakeGenome:
    """Two inputs -> hidden 4 -> hidden 5 -> output 6, plus one disabled edge."""
    return FakeGenome(
        node_genes=[
            FakeNodeGene(0, "input"),
            FakeNodeGene(1, "input"),
            FakeNodeGene(4, "hidden"),
            FakeNodeGene(5, "hidden"),
            FakeNodeGene(6, "output"),
        ],
        connection_genes=[
            FakeConnectionGene(0, 4),
            FakeConnectionGene(1, 4),
            FakeConnectionGene(4, 5),
            FakeConnectionGene(5, 6),
            FakeConnectionGene(1, 6, is_enabled=False),
        ],
    )


def test_inputs_and_bias_sit_in_layer_zero() -> None:
    layers = compute_layer_index_by_node_id(build_direct_genome())

    assert layers[0] == 0
    assert layers[1] == 0
    assert layers[2] == 0


def test_output_sits_one_layer_past_the_deepest_non_output_node() -> None:
    layers = compute_layer_index_by_node_id(build_direct_genome())

    assert layers[3] == 1


def test_chained_hidden_nodes_get_increasing_layers() -> None:
    layers = compute_layer_index_by_node_id(build_two_hidden_layer_genome())

    assert layers[4] == 1
    assert layers[5] == 2
    assert layers[6] == 3


def test_disabled_connections_do_not_deepen_a_node() -> None:
    genome = build_two_hidden_layer_genome()

    layers = compute_layer_index_by_node_id(genome)

    # node 6 is reached through 5, not through the disabled 1 -> 6 edge,
    # so it sits past hidden layer 2 rather than directly behind the inputs.
    assert layers[6] == 3


def test_an_orphaned_hidden_node_is_not_drawn_in_the_input_column() -> None:
    """AddNodeMutation disables the edge it splits, so this shape is routine.

    Node 4 is hidden but has no enabled incoming connection. Placing it in
    layer 0 would draw it alongside the inputs, where it reads as one of them.
    """
    genome = FakeGenome(
        node_genes=[
            FakeNodeGene(0, "input"),
            FakeNodeGene(1, "input"),
            FakeNodeGene(4, "hidden"),
            FakeNodeGene(3, "output"),
        ],
        connection_genes=[
            FakeConnectionGene(0, 3, is_enabled=False),
            FakeConnectionGene(0, 4, is_enabled=False),
            FakeConnectionGene(4, 3),
            FakeConnectionGene(1, 3),
        ],
    )

    layers = compute_layer_index_by_node_id(genome)

    assert layers[0] == 0
    assert layers[1] == 0
    assert layers[4] > 0, "an orphaned hidden node must not sit in the input column"


def test_an_unreachable_output_still_gets_the_final_layer() -> None:
    genome = FakeGenome(
        node_genes=[FakeNodeGene(0, "input"), FakeNodeGene(1, "output")],
        connection_genes=[],
    )

    layers = compute_layer_index_by_node_id(genome)

    assert layers[1] == 1


def test_describe_topology_counts_nodes_connections_and_layers() -> None:
    description = describe_topology(
        build_two_hidden_layer_genome(),
        title="tiny/neat",
        structure_notes={"genome_kind": "single network"},
    )

    assert "tiny/neat" in description
    assert "input 2" in description
    assert "hidden 2" in description
    assert "output 1" in description
    assert "4 enabled of 5 total" in description
    assert "layer 0" in description
    assert "single network" in description


def test_render_layered_topology_writes_a_png(tmp_path: Path) -> None:
    output_path = tmp_path / "topology.png"

    render_layered_topology(build_two_hidden_layer_genome(), output_path, title="tiny")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_write_topology_report_writes_both_artifacts(tmp_path: Path) -> None:
    write_topology_report(
        build_direct_genome(),
        tmp_path,
        "tiny__neat__best",
        title="tiny/neat best",
        structure_notes={},
    )

    assert (tmp_path / "tiny__neat__best.png").exists()
    assert (tmp_path / "tiny__neat__best.txt").exists()


def test_write_topology_record_stores_json_and_draws_nothing(tmp_path: Path) -> None:
    write_topology_record(
        build_two_hidden_layer_genome(),
        tmp_path,
        "tiny__neat__best",
        title="tiny/neat best",
        structure_notes={"genome_kind": "single network"},
    )

    assert (tmp_path / "tiny__neat__best.json").exists()
    assert not (tmp_path / "tiny__neat__best.png").exists()


def test_a_stored_record_round_trips_to_the_same_layers(tmp_path: Path) -> None:
    genome = build_two_hidden_layer_genome()
    write_topology_record(
        genome, tmp_path, "roundtrip", title="t", structure_notes={}
    )

    payload = json.loads((tmp_path / "roundtrip.json").read_text(encoding="utf-8"))
    record, title, notes = topology_record_from_dict(payload)

    assert compute_layer_index_by_node_id(record) == compute_layer_index_by_node_id(genome)
    assert title == "t"
    assert notes == {}


def test_render_topology_records_draws_every_stored_record(tmp_path: Path) -> None:
    write_topology_record(
        build_direct_genome(), tmp_path, "a__neat__best", title="a", structure_notes={}
    )
    write_topology_record(
        build_two_hidden_layer_genome(),
        tmp_path,
        "b__cneat__class_0",
        title="b",
        structure_notes={},
    )

    number_drawn = render_topology_records(tmp_path)

    assert number_drawn == 2
    assert (tmp_path / "a__neat__best.png").exists()
    assert (tmp_path / "b__cneat__class_0.txt").exists()


def test_render_topology_records_skips_a_corrupt_record(tmp_path: Path) -> None:
    write_topology_record(
        build_direct_genome(), tmp_path, "good", title="g", structure_notes={}
    )
    (tmp_path / "bad.json").write_text("{ truncated", encoding="utf-8")

    number_drawn = render_topology_records(tmp_path)

    assert number_drawn == 1
    assert (tmp_path / "good.png").exists()
