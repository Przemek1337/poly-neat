"""Configuration for the NEAT-DBM algorithm.

References:
    Stanovov, V., Akhmedova, Sh., & Semenkin, E. (2021). Neuroevolution of
    augmented topologies with difference-based mutation. *IOP Conference
    Series: Materials Science and Engineering*, 1047, 012075.
    DOI: 10.1088/1757-899X/1047/1/012075
"""
from __future__ import annotations

from dataclasses import dataclass

from polyneat.configs.configuration_errors import ConfigurationError
from polyneat.configs.neat.neat_config import NEATConfig


@dataclass
class NEATDBMConfig(NEATConfig):
    """Hyperparameters for NEAT-DBM (Stanovov et al., 2021).

    NEAT-DBM keeps every NEAT hyperparameter unchanged and adds the
    difference-based mutation (paper, section 3): with a configured
    probability, a freshly produced child has its weights recombined from
    three distinct donor genomes at the connection genes whose innovation
    numbers all four genomes share.

    Attributes:
        probability_of_difference_based_mutation: Probability that a freshly
            produced (non-elite) child undergoes the difference-based
            mutation. The paper uses the operator alongside its other
            mutations with probability 0.1 (section 3).
        difference_mutation_scaling_factor: Scaling factor ``F`` in the
            difference formula ``w_r1 + F * (w_r2 - w_r3)`` (paper,
            section 3). The paper does not report the value it used; the
            default 0.5 follows differential-evolution convention
            (Storn & Price, 1997).
    """

    probability_of_difference_based_mutation: float = 0.1
    difference_mutation_scaling_factor: float = 0.5

    def validate(self) -> None:
        """Validate NEAT-DBM fields on top of the NEAT and shared validation.

        Raises:
            ConfigurationError: If the difference-based mutation probability
                lies outside ``[0, 1]`` or the scaling factor is not
                strictly positive.
        """
        super().validate()
        if not (0.0 <= self.probability_of_difference_based_mutation <= 1.0):
            raise ConfigurationError(
                f"probability_of_difference_based_mutation must be in [0.0, 1.0], "
                f"got {self.probability_of_difference_based_mutation}"
            )
        if self.difference_mutation_scaling_factor <= 0.0:
            raise ConfigurationError(
                f"difference_mutation_scaling_factor must be > 0.0, "
                f"got {self.difference_mutation_scaling_factor}"
            )
