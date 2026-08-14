from __future__ import annotations

import polyneat as pn

_FDNEAT_PUBLIC_NAMES = [
    "FDNEATConfig",
    "FDNEATAlgorithm",
    "DeleteInputConnectionMutation",
]


def test_fdneat_classes_are_exported_at_package_level() -> None:
    for public_name in _FDNEAT_PUBLIC_NAMES:
        assert hasattr(pn, public_name), f"polyneat.{public_name} missing"
        assert public_name in pn.__all__, f"{public_name} not in polyneat.__all__"


def test_exported_classes_are_the_same_objects_as_the_module_level_ones() -> None:
    from polyneat.algorithms.fdneat.fdneat_algorithm import FDNEATAlgorithm
    from polyneat.algorithms.fdneat.mutations.delete_input_connection_mutation import (
        DeleteInputConnectionMutation,
    )
    from polyneat.configs.fdneat.fdneat_config import FDNEATConfig

    assert pn.FDNEATAlgorithm is FDNEATAlgorithm
    assert pn.FDNEATConfig is FDNEATConfig
    assert pn.DeleteInputConnectionMutation is DeleteInputConnectionMutation


def test_every_name_in_dunder_all_actually_resolves() -> None:
    unresolvable_names = [name for name in pn.__all__ if not hasattr(pn, name)]
    assert unresolvable_names == []


def test_dunder_all_has_no_duplicate_names() -> None:
    duplicated_names = sorted({name for name in pn.__all__ if pn.__all__.count(name) > 1})
    assert duplicated_names == []
