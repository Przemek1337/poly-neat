from __future__ import annotations

import polyneat as pn


def test_hyperneat_symbols_are_exported():
    expected_names = [
        "HyperNEATConfig",
        "HyperNEATAlgorithm",
        "HyperNEATPhenotypeDecoder",
        "Substrate",
        "SubstrateLayer",
        "SubstrateNode",
        "build_layered_substrate",
        "AddNodeWithRandomActivationMutation",
    ]
    for name in expected_names:
        assert name in pn.__all__, f"{name} missing from polyneat.__all__"
        assert hasattr(pn, name), f"{name} not importable from polyneat"


def test_hyperneat_factory_is_wired():
    config = pn.HyperNEATConfig(population_size=8)
    algorithm = pn.HyperNEATAlgorithm.from_config(config)
    assert isinstance(algorithm.phenotype_decoder, pn.HyperNEATPhenotypeDecoder)
