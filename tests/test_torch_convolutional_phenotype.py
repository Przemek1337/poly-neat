from __future__ import annotations

import torch

from polyneat.algorithms.exact.exact_genome import (
    ConvolutionEdgeGene,
    EXACTGenome,
    FilterNodeGene,
)
from polyneat.algorithms.exact.exact_phenotype_decoder import EXACTPhenotypeDecoder
from polyneat.algorithms.exact.torch_convolutional_phenotype import (
    TorchConvolutionalPhenotype,
)

_CPU = torch.device("cpu")


def _tiny_genome(kernel: tuple[tuple[float, ...], ...] | None) -> EXACTGenome:
    """2x2 input directly convolved into a single 1x1 output (kernel 2x2)."""
    return EXACTGenome(
        node_genes=(
            FilterNodeGene(
                node_id=0, node_type="input", filter_height=2, filter_width=2, depth=0.0
            ),
            FilterNodeGene(
                node_id=1, node_type="output", filter_height=1, filter_width=1, depth=1.0
            ),
        ),
        edge_genes=(
            ConvolutionEdgeGene(
                innovation_id=0,
                source_node_id=0,
                target_node_id=1,
                is_enabled=True,
                kernel_weights=kernel,
            ),
        ),
    )


def _phenotype(genome: EXACTGenome) -> TorchConvolutionalPhenotype:
    return TorchConvolutionalPhenotype(
        exact_genome=genome,
        input_image_height=2,
        input_image_width=2,
        leaky_relu_negative_slope=0.1,
        activation_clamp_maximum=5.5,
        device_for_computation=_CPU,
    )


def test_forward_pass_computes_the_exact_convolution() -> None:
    kernel = ((1.0, 2.0), (3.0, 4.0))
    phenotype = _phenotype(_tiny_genome(kernel))
    # Input all ones -> valid 2x2 conv over a 2x2 map = sum of the kernel = 10.
    logits = phenotype.forward_pass(torch.ones(3, 4))
    assert logits.shape == (3, 1)
    assert torch.allclose(logits, torch.full((3, 1), 10.0))


def test_missing_kernels_are_he_initialized() -> None:
    phenotype = _phenotype(_tiny_genome(kernel=None))
    kernel_parameter = next(iter(phenotype.parameters()))
    assert kernel_parameter.shape == (1, 1, 2, 2)
    assert not torch.allclose(kernel_parameter, torch.zeros_like(kernel_parameter))


def test_upscaling_edge_pads_to_the_larger_target() -> None:
    # 2x2 input -> 3x3 hidden: kernel |3-2|+1 = 2, padding 1 -> output 3x3.
    genome = EXACTGenome(
        node_genes=(
            FilterNodeGene(
                node_id=0, node_type="input", filter_height=2, filter_width=2, depth=0.0
            ),
            FilterNodeGene(
                node_id=1, node_type="output", filter_height=1, filter_width=1, depth=1.0
            ),
            FilterNodeGene(
                node_id=2, node_type="hidden", filter_height=3, filter_width=3, depth=0.5
            ),
        ),
        edge_genes=(
            ConvolutionEdgeGene(
                innovation_id=0, source_node_id=0, target_node_id=2, is_enabled=True
            ),
            ConvolutionEdgeGene(
                innovation_id=1, source_node_id=2, target_node_id=1, is_enabled=True
            ),
        ),
    )
    logits = _phenotype(genome).forward_pass(torch.ones(1, 4))
    assert logits.shape == (1, 1)


def test_hidden_activation_is_clamped_leaky_relu() -> None:
    # Force a huge positive hidden pre-activation; the clamp caps it at 5.5,
    # so the 1x1 identity-weighted output edge reports exactly 5.5.
    genome = EXACTGenome(
        node_genes=(
            FilterNodeGene(
                node_id=0, node_type="input", filter_height=2, filter_width=2, depth=0.0
            ),
            FilterNodeGene(
                node_id=1, node_type="output", filter_height=1, filter_width=1, depth=1.0
            ),
            FilterNodeGene(
                node_id=2, node_type="hidden", filter_height=1, filter_width=1, depth=0.5
            ),
        ),
        edge_genes=(
            ConvolutionEdgeGene(
                innovation_id=0,
                source_node_id=0,
                target_node_id=2,
                is_enabled=True,
                kernel_weights=((100.0, 100.0), (100.0, 100.0)),
            ),
            ConvolutionEdgeGene(
                innovation_id=1,
                source_node_id=2,
                target_node_id=1,
                is_enabled=True,
                kernel_weights=((1.0,),),
            ),
        ),
    )
    logits = _phenotype(genome).forward_pass(torch.ones(1, 4))
    assert torch.allclose(logits, torch.tensor([[5.5]]))


def test_disabled_edges_carry_no_parameters_or_signal() -> None:
    genome = _tiny_genome(kernel=((1.0, 2.0), (3.0, 4.0)))
    disabled = EXACTGenome(
        node_genes=genome.node_genes,
        edge_genes=(
            ConvolutionEdgeGene(
                innovation_id=0,
                source_node_id=0,
                target_node_id=1,
                is_enabled=False,
                kernel_weights=((1.0, 2.0), (3.0, 4.0)),
            ),
        ),
    )
    phenotype = _phenotype(disabled)
    assert len(list(phenotype.parameters())) == 0
    logits = phenotype.forward_pass(torch.ones(1, 4))
    assert torch.allclose(logits, torch.zeros(1, 1))


def test_extract_genome_round_trips_kernels_and_marks_trained() -> None:
    kernel = ((1.0, 2.0), (3.0, 4.0))
    phenotype = _phenotype(_tiny_genome(kernel))
    extracted = phenotype.extract_genome_with_trained_weights()
    assert extracted.is_trained is True
    assert extracted.get_edge_gene_by_innovation_id(0).kernel_weights == kernel


def test_gradients_reach_the_kernels() -> None:
    phenotype = _phenotype(_tiny_genome(kernel=((1.0, 2.0), (3.0, 4.0))))
    loss = phenotype.forward_pass(torch.ones(1, 4)).sum()
    loss.backward()
    kernel_parameter = next(iter(phenotype.parameters()))
    assert kernel_parameter.grad is not None
    assert torch.allclose(kernel_parameter.grad, torch.ones(1, 1, 2, 2))


def test_decoder_builds_the_phenotype() -> None:
    decoder = EXACTPhenotypeDecoder(
        input_image_height=2,
        input_image_width=2,
        leaky_relu_negative_slope=0.1,
        activation_clamp_maximum=5.5,
        device_for_computation=_CPU,
    )
    phenotype = decoder.build_phenotype_from_genome(_tiny_genome(kernel=None))
    assert isinstance(phenotype, TorchConvolutionalPhenotype)
    assert phenotype.forward_pass(torch.ones(2, 4)).shape == (2, 1)


def _genome_with_one_hidden_node(
    batch_normalization_state: tuple[float, float, float, float] | None = None,
) -> EXACTGenome:
    """2x2 input -> 2x2 hidden -> 1x1 output; the hidden node is the only path."""
    return EXACTGenome(
        node_genes=(
            FilterNodeGene(
                node_id=0, node_type="input", filter_height=2, filter_width=2, depth=0.0
            ),
            FilterNodeGene(
                node_id=1, node_type="output", filter_height=1, filter_width=1, depth=1.0
            ),
            FilterNodeGene(
                node_id=2,
                node_type="hidden",
                filter_height=2,
                filter_width=2,
                depth=0.5,
                batch_normalization_state=batch_normalization_state,
            ),
        ),
        edge_genes=(
            ConvolutionEdgeGene(
                innovation_id=0,
                source_node_id=0,
                target_node_id=2,
                is_enabled=True,
                kernel_weights=((1.0,),),
            ),
            ConvolutionEdgeGene(
                innovation_id=1,
                source_node_id=2,
                target_node_id=1,
                is_enabled=True,
                kernel_weights=((1.0, 1.0), (1.0, 1.0)),
            ),
        ),
    )


def _build_phenotype(genome: EXACTGenome, **overrides) -> TorchConvolutionalPhenotype:
    return TorchConvolutionalPhenotype(
        exact_genome=genome,
        input_image_height=2,
        input_image_width=2,
        leaky_relu_negative_slope=0.1,
        activation_clamp_maximum=5.5,
        device_for_computation=_CPU,
        **overrides,
    )


def test_phenotype_starts_in_eval_mode() -> None:
    phenotype = _build_phenotype(_tiny_genome(kernel=((1.0, 2.0), (3.0, 4.0))))
    assert phenotype.training is False


def test_batch_normalization_uses_stored_state_in_eval_mode() -> None:
    """Hidden node with state (gamma=2, beta=1, mean=0.5, var=4): eval-mode BN
    maps a pre-activation value v to ((v - 0.5) / sqrt(4 + eps)) * 2 + 1."""
    genome = _genome_with_one_hidden_node(
        batch_normalization_state=(2.0, 1.0, 0.5, 4.0)
    )
    phenotype = _build_phenotype(genome, use_batch_normalization=True)
    input_tensor = torch.ones(1, 4)
    logits_with_batch_normalization = phenotype.forward_pass(input_tensor)
    logits_without_batch_normalization = _build_phenotype(
        genome, use_batch_normalization=False
    ).forward_pass(input_tensor)
    assert not torch.allclose(
        logits_with_batch_normalization, logits_without_batch_normalization
    )


def test_extraction_round_trips_batch_norm_state() -> None:
    genome = _genome_with_one_hidden_node()
    phenotype = _build_phenotype(
        genome, use_batch_normalization=True, batch_normalization_alpha=0.1
    )
    phenotype.train()
    phenotype.forward_pass(torch.randn(8, 4))
    extracted = phenotype.extract_genome_with_trained_weights()
    hidden_node = next(
        node_gene
        for node_gene in extracted.node_genes
        if node_gene.node_type == "hidden"
    )
    assert hidden_node.batch_normalization_state is not None
    assert len(hidden_node.batch_normalization_state) == 4


def test_hidden_dropout_probability_one_zeroes_logits_in_train_mode() -> None:
    genome = _genome_with_one_hidden_node()
    phenotype = _build_phenotype(genome, hidden_dropout_probability=1.0)
    phenotype.train()
    logits = phenotype.forward_pass(torch.ones(1, 4))
    assert torch.allclose(logits, torch.zeros_like(logits))
    phenotype.eval()
    assert not torch.allclose(
        phenotype.forward_pass(torch.ones(1, 4)), torch.zeros_like(logits)
    )
