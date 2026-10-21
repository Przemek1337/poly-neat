"""HyperNEAT-LEO: link expression output plus a locality-seeded generation 0.

References:
    Verbancsics, P., & Stanley, K. O. (2011). Constraining Connectivity to Encourage
        Modularity in HyperNEAT. *GECCO '11*, pp. 1483-1490. DOI: 10.1145/2001576.2001776
"""

from polyneat.algorithms.hyperneatleo.hyperneatleo_algorithm import HyperNEATLEOAlgorithm
from polyneat.algorithms.hyperneatleo.leo_phenotype_decoder import (
    HyperNEATLEOPhenotypeDecoder,
    scale_leo_output_to_substrate_weight,
)

# Importing this module registers the "leo_seeded" generation-0 strategy, which
# HyperNEATLEOConfig names by default. The import lives here rather than in
# initial_population.py to keep the core free of any dependency on a variant.
from polyneat.algorithms.hyperneatleo.leo_seeded_initial_population import (
    build_leo_seeded_initial_population,
)

__all__ = [
    "HyperNEATLEOAlgorithm",
    "HyperNEATLEOPhenotypeDecoder",
    "build_leo_seeded_initial_population",
    "scale_leo_output_to_substrate_weight",
]
