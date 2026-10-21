from __future__ import annotations

import polyneat as pn

_LEO_PUBLIC_NAMES = [
    "HyperNEATLEOConfig",
    "HyperNEATLEOAlgorithm",
    "HyperNEATLEOPhenotypeDecoder",
    "build_leo_seeded_initial_population",
    "build_substrate_from_explicit_layer_coordinates",
    "RetinaProblemEvaluator",
    "count_cross_hemisphere_connections",
    "count_expressed_connections",
]


def test_leo_names_are_exported_at_package_level() -> None:
    for public_name in _LEO_PUBLIC_NAMES:
        assert hasattr(pn, public_name), f"polyneat.{public_name} missing"
        assert public_name in pn.__all__, f"{public_name} not in polyneat.__all__"


def test_leo_seeded_strategy_is_registered_on_import() -> None:
    from polyneat.core.neat.initial_population import (
        INITIAL_POPULATION_STRATEGY_NAME_TO_CALLABLE,
    )

    assert "leo_seeded" in INITIAL_POPULATION_STRATEGY_NAME_TO_CALLABLE


def test_every_name_in_dunder_all_actually_resolves() -> None:
    unresolvable_names = [name for name in pn.__all__ if not hasattr(pn, name)]
    assert unresolvable_names == []


def test_dunder_all_has_no_duplicate_names() -> None:
    duplicated_names = sorted({name for name in pn.__all__ if pn.__all__.count(name) > 1})
    assert duplicated_names == []
