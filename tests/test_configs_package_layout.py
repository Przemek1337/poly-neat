"""Layout tests for the restructured ``polyneat.configs`` package."""

import importlib

import pytest


def test_shared_config_infrastructure_lives_at_configs_root():
    from polyneat.configs.algorithm_config import AlgorithmConfig
    from polyneat.configs.configuration_errors import ConfigurationError

    assert AlgorithmConfig.__name__ == "AlgorithmConfig"
    assert issubclass(ConfigurationError, Exception)


def test_each_algorithm_config_lives_in_its_own_subpackage():
    from polyneat.configs.algorithm_config import AlgorithmConfig
    from polyneat.configs.cneat.cneat_config import CNEATConfig
    from polyneat.configs.hyperneat.hyperneat_config import HyperNEATConfig
    from polyneat.configs.lneat.lneat_config import LNEATConfig
    from polyneat.configs.neat.neat_config import NEATConfig
    from polyneat.configs.neatdbm.neatdbm_config import NEATDBMConfig

    assert issubclass(NEATConfig, AlgorithmConfig)
    assert issubclass(CNEATConfig, NEATConfig)
    assert issubclass(HyperNEATConfig, NEATConfig)
    assert issubclass(LNEATConfig, NEATConfig)
    assert issubclass(NEATDBMConfig, NEATConfig)


def test_config_subpackages_reexport_their_config_class():
    from polyneat.configs.cneat import CNEATConfig
    from polyneat.configs.hyperneat import HyperNEATConfig
    from polyneat.configs.lneat import LNEATConfig
    from polyneat.configs.neat import NEATConfig
    from polyneat.configs.neatdbm import NEATDBMConfig

    for config_class in (NEATConfig, CNEATConfig, HyperNEATConfig, LNEATConfig, NEATDBMConfig):
        assert config_class.__name__.endswith("Config")


def test_old_config_package_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("polyneat.config")


def test_top_level_exports_still_expose_all_config_classes():
    import polyneat as pn

    for class_name in (
        "AlgorithmConfig",
        "ConfigurationError",
        "NEATConfig",
        "CNEATConfig",
        "HyperNEATConfig",
        "LNEATConfig",
        "NEATDBMConfig",
    ):
        assert hasattr(pn, class_name)
