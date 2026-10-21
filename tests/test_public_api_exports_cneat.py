from __future__ import annotations

import polyneat as pn

_CNEAT_PUBLIC_NAMES = [
    "CNEATConfig",
    "CNEATAlgorithm",
    "ClassGenomeContainer",
    "ContainerUpdateCallback",
    "ContainerEnsemblePhenotype",
    "ContainerProgressLogger",
]


def test_cneat_classes_are_exported_at_package_level() -> None:
    for public_name in _CNEAT_PUBLIC_NAMES:
        assert hasattr(pn, public_name), f"polyneat.{public_name} missing"
        assert public_name in pn.__all__, f"{public_name} not in polyneat.__all__"


def test_every_name_in_dunder_all_actually_resolves() -> None:
    unresolvable_names = [name for name in pn.__all__ if not hasattr(pn, name)]
    assert unresolvable_names == []


def test_dunder_all_has_no_duplicate_names() -> None:
    duplicated_names = sorted({name for name in pn.__all__ if pn.__all__.count(name) > 1})
    assert duplicated_names == []
