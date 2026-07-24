from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from polyneat.configs.configuration_errors import ConfigurationError


@dataclass
class AlgorithmConfig:
    """Configuration shared by every neuroevolution algorithm.

    Attributes:
        population_size: Number of genomes per generation.
        number_of_input_nodes: Input nodes of every network (bias excluded).
        number_of_output_nodes: Output nodes of every network.
        random_seed: Seed for the run's RNG; ``None`` means non-deterministic.
        device_for_phenotype_evaluation: Torch device name used for phenotype
            evaluation unless overridden from the CLI/from_config.
    """

    population_size: int = 150
    number_of_input_nodes: int = 2
    number_of_output_nodes: int = 1
    random_seed: int | None = None
    device_for_phenotype_evaluation: str = "cpu"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check the base fields; subclasses extend and call ``super().validate()``.

        Raises:
            ConfigurationError: Naming the field, the value, and the reason.
        """
        if self.population_size < 1:
            raise ConfigurationError(f"population_size must be >= 1, got {self.population_size}")
        if self.number_of_input_nodes < 1:
            raise ConfigurationError(
                f"number_of_input_nodes must be >= 1, got {self.number_of_input_nodes}"
            )
        if self.number_of_output_nodes < 1:
            raise ConfigurationError(
                f"number_of_output_nodes must be >= 1, got {self.number_of_output_nodes}"
            )

    @classmethod
    def load_from_yaml_file(cls, yaml_file_path: Path) -> AlgorithmConfig:
        """Load and validate a config from a YAML file (strict keys).

        Args:
            yaml_file_path: Path of the YAML file to read.

        Returns:
            A validated config instance of ``cls``.

        Raises:
            ConfigurationError: On unknown keys or invalid values.
        """
        yaml_payload = yaml.safe_load(yaml_file_path.read_text(encoding="utf-8"))
        return cls.from_dict(yaml_payload)

    @classmethod
    def from_dict(cls, raw_config_data: dict[str, Any]) -> AlgorithmConfig:
        """Strict loader: unknown keys raise ``ConfigurationError`` (catches typos).

        Values loaded from YAML for tuple-typed fields arrive as lists; they are
        coerced back to tuples so a saved config reloads to an equal instance.
        """
        dataclass_fields = fields(cls)
        known_field_names = {field.name for field in dataclass_fields}
        unknown_keys = set(raw_config_data.keys()) - known_field_names
        if unknown_keys:
            raise ConfigurationError(
                f"Unknown configuration keys: {sorted(unknown_keys)}. "
                f"Check for typos. Known keys: {sorted(known_field_names)}"
            )
        coerced_config_data = dict(raw_config_data)
        for dataclass_field in dataclass_fields:
            if isinstance(dataclass_field.default, tuple) and isinstance(
                coerced_config_data.get(dataclass_field.name), list
            ):
                coerced_config_data[dataclass_field.name] = tuple(
                    coerced_config_data[dataclass_field.name]
                )
        return cls(**coerced_config_data)

    def save_to_yaml_file(self, yaml_file_path: Path) -> None:
        """Dump the config as YAML, loadable back via ``load_from_yaml_file``."""
        yaml_file_path.write_text(
            yaml.dump(self.to_dict(), default_flow_style=False), encoding="utf-8"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain dict, tuple fields rendered as lists.

        Lists keep the saved YAML free of ``!!python/tuple`` tags so it stays
        ``safe_load``-able; :meth:`from_dict` restores the tuples on load.
        """
        return {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(self).items()
        }
