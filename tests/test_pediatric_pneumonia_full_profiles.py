"""The full profiles must refuse fake data and must stay comparable to each other.

A smoke profile may generate a synthetic archive when none is given, because
its whole purpose is to exercise the plumbing without a download. A full
profile must not: fake radiographs would flow through the same code and land in
a run report that looks exactly like a real one.

The other property checked here is comparability. Track B exists so that every
selected topology is retrained under one recipe rather than each method's own,
which only holds if the four profiles that have a track B declare the same
recipe. That is a property of the configuration files, so it is tested on them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from examples._experiment import ExperimentReport
from examples.pediatric_pneumonia._profiles import (
    CONFIGS_DIRECTORY,
    ProfileDataError,
    load_profile_settings,
    run_profile_main,
)
from polyneat.algorithms.deepneat.deepneat_algorithm import DeepNEATAlgorithm
from polyneat.configs.deepneat.deepneat_config import DeepNEATConfig

_FULL_PROFILE_NAMES = (
    "deepneat_full",
    "exact_full",
    "fixed_cnn_full",
    "random_search_full",
    "transfer_learning_full",
)
_SMOKE_PROFILE_NAMES = (
    "deepneat_smoke",
    "exact_smoke",
    "fixed_cnn_smoke",
    "random_search_smoke",
    "transfer_learning_smoke",
)
# transfer_learning has no track B: retraining the backbone from scratch would
# delete the pretrained features that are the reason it is in the comparison.
_TRACK_B_PROFILE_NAMES = (
    "deepneat_full",
    "exact_full",
    "fixed_cnn_full",
    "random_search_full",
)
_EXPECTED_IMAGE_SIDE = 128


def _payload(profile_name: str) -> dict:
    return yaml.safe_load((CONFIGS_DIRECTORY / f"{profile_name}.yaml").read_text("utf-8"))


@pytest.mark.parametrize("profile_name", _FULL_PROFILE_NAMES)
class TestFullProfilesRefuseSyntheticData:
    def test_the_profile_does_not_allow_a_synthetic_archive(
        self, profile_name: str
    ) -> None:
        protocol = _payload(profile_name)["protocol"]

        assert protocol.get("allows_synthetic_archive", False) is False

    def test_loading_without_a_data_directory_is_refused(self, profile_name: str) -> None:
        with pytest.raises(ProfileDataError, match="needs a real archive"):
            load_profile_settings(CONFIGS_DIRECTORY / f"{profile_name}.yaml")

    def test_loading_with_a_data_directory_succeeds(
        self, profile_name: str, tmp_path: Path
    ) -> None:
        settings = load_profile_settings(
            CONFIGS_DIRECTORY / f"{profile_name}.yaml", data_directory=tmp_path
        )

        assert settings.data_directory == tmp_path
        assert settings.image_side == _EXPECTED_IMAGE_SIDE

    def test_the_profile_does_not_claim_a_verified_license(self, profile_name: str) -> None:
        # The license must be replaced with what the download states; until then
        # the run report has to carry the fact that nobody checked.
        protocol = _payload(profile_name)["protocol"]

        assert protocol["dataset_license"] == "unverified-confirm-against-the-download"


@pytest.mark.parametrize("profile_name", _SMOKE_PROFILE_NAMES)
def test_smoke_profiles_still_allow_their_synthetic_archive(profile_name: str) -> None:
    protocol = _payload(profile_name)["protocol"]

    assert protocol["allows_synthetic_archive"] is True


def test_every_track_b_profile_declares_the_same_recipe() -> None:
    recipes = {name: _payload(name)["track_b_recipe"] for name in _TRACK_B_PROFILE_NAMES}
    distinct = {repr(sorted(recipe.items())) for recipe in recipes.values()}

    assert len(distinct) == 1, f"track B recipes disagree between methods: {recipes}"


def test_the_transfer_learning_profile_has_no_track_b_seeds() -> None:
    protocol = _payload("transfer_learning_full")["protocol"]

    assert protocol["retraining_seeds"] == []


@pytest.mark.parametrize("profile_name", ("deepneat_full", "random_search_full"))
def test_full_search_profiles_pin_the_extra_preprocessing_genes(profile_name: str) -> None:
    config = DeepNEATConfig.from_dict(_payload(profile_name)["algorithm"])
    assert isinstance(config, DeepNEATConfig)

    assert config.global_cropped_image_size_max == 0
    assert config.global_spatial_scaling_max == 0.0
    assert config.available_horizontal_flip_options == (False,)
    assert config.available_variance_normalization_options == (False,)


@pytest.mark.parametrize("profile_name", ("deepneat_full", "random_search_full"))
def test_no_initial_full_genome_carries_extra_preprocessing(profile_name: str) -> None:
    config = DeepNEATConfig.from_dict(_payload(profile_name)["algorithm"])
    assert isinstance(config, DeepNEATConfig)
    population = DeepNEATAlgorithm.from_config(config).create_initial_population(
        np.random.default_rng(7)
    )

    for genome in population.genomes:
        hyperparameters = genome.global_hyperparameters
        assert hyperparameters.cropped_image_size == 0
        assert hyperparameters.spatial_scaling == 0.0
        assert hyperparameters.uses_horizontal_flips is False
        assert hyperparameters.uses_variance_normalization is False


def test_a_missing_archive_becomes_an_exit_code_not_a_traceback() -> None:
    """The entry point turns the refusal into a command-line error."""

    def run_experiment_that_finds_no_archive(**_: object) -> None:
        raise ProfileDataError("deepneat_full.yaml needs a real archive")

    with pytest.raises(SystemExit) as exit_info:
        run_profile_main(
            run_experiment_that_finds_no_archive,
            default_config_file_path=CONFIGS_DIRECTORY / "deepneat_full.yaml",
            artifacts_directory=Path("unused"),
            argument_list=["--cpu"],
        )

    assert exit_info.value.code == 1


def test_the_entry_point_forwards_the_config_and_archive_it_parsed(tmp_path: Path) -> None:
    received: dict = {}

    def recording_run_experiment(**keyword_arguments: object) -> ExperimentReport:
        received.update(keyword_arguments)
        return ExperimentReport(
            metric_values={"placeholder": 0.0}, number_of_generations=1, runtime_seconds=0.0
        )

    run_profile_main(
        recording_run_experiment,
        default_config_file_path=CONFIGS_DIRECTORY / "deepneat_smoke.yaml",
        artifacts_directory=tmp_path / "artifacts",
        argument_list=[
            "--cpu",
            "--data-directory",
            str(tmp_path / "archive"),
            "--config",
            str(CONFIGS_DIRECTORY / "deepneat_full.yaml"),
        ],
    )

    assert received["data_directory"] == tmp_path / "archive"
    assert received["config_file_path"] == CONFIGS_DIRECTORY / "deepneat_full.yaml"
