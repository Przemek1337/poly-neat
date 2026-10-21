from __future__ import annotations

import argparse

import pytest
import torch

from examples._example_cli import (
    add_device_arguments,
    parse_device_from_cli,
    resolve_device,
)


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


def _parser_with_device_flags() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_device_arguments(parser)
    return parser


def test_resolve_device_returns_none_when_no_flag_is_set() -> None:
    namespace = _parser_with_device_flags().parse_args([])
    assert resolve_device(namespace) is None


def test_resolve_device_returns_cpu_for_cpu_flag() -> None:
    namespace = _parser_with_device_flags().parse_args(["--cpu"])
    assert resolve_device(namespace) == torch.device("cpu")


def test_resolve_device_exits_for_gpu_flag_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    namespace = _parser_with_device_flags().parse_args(["--gpu"])
    with pytest.raises(SystemExit) as raised:
        resolve_device(namespace)
    assert raised.value.code == 1


def test_add_device_arguments_composes_with_foreign_arguments() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--repeats", type=int, default=5)
    add_device_arguments(parser)
    namespace = parser.parse_args(["--cpu", "--repeats", "3"])
    assert namespace.repeats == 3
    assert resolve_device(namespace) == torch.device("cpu")
