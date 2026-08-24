"""The innovation tracker must stay on the single-threaded reproduction path."""

from __future__ import annotations

from pathlib import Path

import pytest

_LIBRARY_ROOT = Path(__file__).resolve().parent.parent / "polyneat"
_PACKAGES_THAT_MUST_NOT_TOUCH_THE_TRACKER = ("evaluators", "runner")


@pytest.mark.parametrize("package_name", _PACKAGES_THAT_MUST_NOT_TOUCH_THE_TRACKER)
def test_package_never_references_the_innovation_tracker(package_name: str) -> None:
    # GlobalInnovationTracker is not thread-safe: get_or_assign_* are
    # check-then-act sequences on a dict and a counter. That is harmless only
    # while every caller sits on the single-threaded reproduction path.
    # Evaluators and the runner are where parallelism lives, so they must never
    # see it.
    offending_files = []
    for source_file in (_LIBRARY_ROOT / package_name).rglob("*.py"):
        if "innovation" in source_file.read_text(encoding="utf-8").lower():
            offending_files.append(source_file.name)
    assert offending_files == [], (
        f"polyneat/{package_name}/ references the innovation tracker in "
        f"{offending_files}; parallel evaluation must never touch it"
    )


def test_tracker_docstring_states_the_threading_contract() -> None:
    from polyneat.core.neat.global_innovation_tracker import GlobalInnovationTracker

    docstring = (GlobalInnovationTracker.__doc__ or "").lower()
    assert "thread" in docstring, "the threading contract must be documented"
