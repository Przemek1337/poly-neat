from __future__ import annotations

import torch

from polyneat.core.component_protocols import Genome, Phenotype, PhenotypeDecoder


class RecognizerEnsemblePhenotype:
    """Argmax classifier assembled from single-output recognizer phenotypes.

    Implements the assembly step of L-NEAT's divide and conquer (Chen &
    Alahakoon, 2006, section IV.A): after one recognizer network has been
    evolved per class label, "when the best networks are identified and
    assembled together, the final solution is found". Column ``k`` of the
    forward-pass output is recognizer ``k``'s output, so the predicted class
    of a sample is the argmax over columns — the same interface a native
    multi-output classifier exposes, which makes the ensemble directly
    scorable by
    :class:`~polyneat.evaluators.classification_accuracy_evaluator.ClassificationAccuracyEvaluator`.

    References:
        Chen, L., & Alahakoon, D. (2006). NeuroEvolution of Augmenting Topologies with Learning
            for Data Classification. *ICIA 2006: 2nd International Conference on Information and
            Automation*, pp. 367-371.
    """

    def __init__(self, class_recognizer_phenotypes: list[Phenotype]) -> None:
        """Assemble the ensemble from already-built recognizer phenotypes.

        Args:
            class_recognizer_phenotypes: One single-output phenotype per
                class label, ordered by class label index.

        Raises:
            ValueError: If fewer than two recognizers are supplied.
        """
        if len(class_recognizer_phenotypes) < 2:
            raise ValueError(
                "RecognizerEnsemblePhenotype: at least two class recognizers are "
                f"required, got {len(class_recognizer_phenotypes)}"
            )
        self._class_recognizer_phenotypes = class_recognizer_phenotypes

    @classmethod
    def from_genomes(
        cls,
        class_recognizer_genomes: list[Genome],
        phenotype_decoder: PhenotypeDecoder,
    ) -> RecognizerEnsemblePhenotype:
        """Decode one genome per class label and assemble the ensemble.

        Args:
            class_recognizer_genomes: The best recognizer genome of each
                subtask evolution, ordered by class label index.
            phenotype_decoder: Decoder shared by all subtask runs.

        Returns:
            An ensemble ready for ``forward_pass``.

        Raises:
            ValueError: If fewer than two genomes are supplied.
        """
        return cls(
            class_recognizer_phenotypes=[
                phenotype_decoder.build_phenotype_from_genome(genome)
                for genome in class_recognizer_genomes
            ]
        )

    def forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Evaluate every recognizer and stack their outputs into columns.

        Args:
            input_tensor: Shape ``(batch, num_inputs)`` — or ``(num_inputs,)``,
                which is treated as a batch of one.

        Returns:
            Tensor of shape ``(batch, number_of_classes)`` where column ``k``
            holds recognizer ``k``'s output; the predicted class of a sample
            is the argmax over its row.

        Raises:
            ValueError: If a recognizer produces anything but a single
                output column.
        """
        recognizer_output_columns: list[torch.Tensor] = []
        for class_label_index, recognizer_phenotype in enumerate(
            self._class_recognizer_phenotypes
        ):
            recognizer_outputs = recognizer_phenotype.forward_pass(input_tensor)
            if recognizer_outputs.dim() != 2 or recognizer_outputs.shape[1] != 1:
                raise ValueError(
                    f"RecognizerEnsemblePhenotype: recognizer for class "
                    f"{class_label_index} must produce a single output column, got "
                    f"shape {tuple(recognizer_outputs.shape)}"
                )
            recognizer_output_columns.append(recognizer_outputs[:, 0])
        return torch.stack(recognizer_output_columns, dim=1)

    def reset_recurrent_state(self) -> None:
        """Reset every member recognizer."""
        for recognizer_phenotype in self._class_recognizer_phenotypes:
            recognizer_phenotype.reset_recurrent_state()
