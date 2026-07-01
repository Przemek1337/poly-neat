from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from polyneat.config.errors import ConfigurationError


@dataclass
class AlgorithmConfig:
    population_size: int = 150
    number_of_input_nodes: int = 2
    number_of_output_nodes: int = 1
    random_seed: int | None = None
    device_for_phenotype_evaluation: str = "cpu"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.population_size < 1:
            raise ConfigurationError(
                f"population_size must be >= 1, got {self.population_size}"
            )
        if self.number_of_input_nodes < 1:
            raise ConfigurationError(
                f"number_of_input_nodes must be >= 1, got {self.number_of_input_nodes}"
            )
        if self.number_of_output_nodes < 1:
            raise ConfigurationError(
                f"number_of_output_nodes must be >= 1, got {self.number_of_output_nodes}"
            )

    @classmethod
    def load_from_yaml_file(cls, yaml_file_path: Path) -> "AlgorithmConfig":
        data = yaml.safe_load(yaml_file_path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlgorithmConfig":
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(data.keys()) - known_fields
        if unknown:
            raise ConfigurationError(
                f"Unknown configuration keys: {sorted(unknown)}. "
                f"Check for typos. Known keys: {sorted(known_fields)}"
            )
        return cls(**{k: v for k, v in data.items() if k in known_fields})

    def save_to_yaml_file(self, yaml_file_path: Path) -> None:
        yaml_file_path.write_text(
            yaml.dump(self.to_dict(), default_flow_style=False), encoding="utf-8"
        )

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)
