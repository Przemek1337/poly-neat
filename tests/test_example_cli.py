from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_EXAMPLES_DIR = str(Path(__file__).parent.parent / "examples")
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

from _example_cli import parse_device_from_cli  # noqa: E402


def test_no_flag_returns_none_so_yaml_value_wins() -> None:
    assert parse_device_from_cli([]) is None


def test_cpu_flag_returns_cpu_device() -> None:
    assert parse_device_from_cli(["--cpu"]) == torch.device("cpu")


def test_gpu_flag_without_cuda_exits_with_code_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as raised:
        parse_device_from_cli(["--gpu"])
    assert raised.value.code == 1


def test_gpu_flag_with_cuda_returns_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert parse_device_from_cli(["--gpu"]) == torch.device("cuda")


def test_cpu_and_gpu_together_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit) as raised:
        parse_device_from_cli(["--cpu", "--gpu"])
    assert raised.value.code == 2
