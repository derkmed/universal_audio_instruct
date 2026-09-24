"""Offline tests for building `complete.json` from the Hub's `complete-1..5`.

`build_complete_config.combine_configs` is the pure core; `main` reads the five
configs through `hub.download_file`, replaced here with a fake. The combined
config must load like any other run config.

Runnable directly (`python tests/test_complete_config.py`) or under pytest.
Only requires `datasets`, `jinja2`, `huggingface_hub`.
"""
import json
import os
import tempfile

from uad_data import build_complete_config, hub
from uad_data.json_config_loader import UniversalJsonConfig

SOURCES = {
    "complete-1.json": {"name": "Complete 1", "randomize_prompt_format": False,
                        "datasets": [{"name": "Clotho", "tasks": ["caption"], "splits": ["test"]},
                                     {"name": "AESDD", "tasks": ["classification"]}]},
    "complete-2.json": {"name": "Complete 2",
                        "datasets": [{"name": "EMNS", "tasks": ["classification", "asr"],
                                      "splits": ["train"]}]},
    "complete-3.json": {"name": "Complete 3",
                        "datasets": [{"name": "AudioMNIST", "tasks": ["asr"]}]},
    "complete-4.json": {"name": "Complete 4", "row_filter": "all_pass",
                        "datasets": [{"name": "AudioMNISTCommonSense", "tasks": ["qa"]}]},
    "complete-5.json": {"name": "Complete 5",
                        "datasets": [{"name": "colombian_spanish", "tasks": ["asr"]}]},
}


def test_datasets_keep_the_order_of_complete_1_to_5() -> None:
    combined = build_complete_config.combine_configs(SOURCES)

    assert [d["name"] for d in combined["datasets"]] == [
        "Clotho", "AESDD", "EMNS", "AudioMNIST", "AudioMNISTCommonSense",
        "colombian_spanish"], combined
    assert combined["datasets"][0] == SOURCES["complete-1.json"]["datasets"][0], combined

    print("PASS: the combined datasets keep the order of complete-1..5.")


def test_combined_config_has_the_fixed_fields_and_its_sources() -> None:
    combined = build_complete_config.combine_configs(SOURCES)

    assert combined["name"] == "Complete UAD", combined
    assert combined["randomize_prompt_format"] is True, combined
    assert combined["built_from"] == [f"complete-{i}.json" for i in range(1, 6)], combined
    assert set(combined) == {"name", "randomize_prompt_format", "built_from", "datasets"}, combined

    print("PASS: the combined config has the fixed name, random templates and built_from.")


def test_an_internal_dataset_in_two_configs_is_an_error() -> None:
    sources = {**SOURCES, "complete-5.json": {
        "name": "Complete 5", "datasets": [{"name": "EMNS", "tasks": ["asr"]}]}}

    try:
        build_complete_config.combine_configs(sources)
    except ValueError as e:
        assert "EMNS" in str(e), e
        assert "complete-2.json" in str(e) and "complete-5.json" in str(e), e
    else:
        raise AssertionError("expected a ValueError for EMNS listed twice")

    print("PASS: an internal dataset in two of the configs is an error.")


def _fake_download(root: str, requests: list[str]):
    def download_file(path_or_url, *, repo_id=None, revision=None, token=None):
        path = hub.to_repo_path(path_or_url)
        requests.append(path)
        local = os.path.join(root, os.path.basename(path))
        with open(local, "w", encoding="utf-8") as f:
            json.dump(SOURCES[os.path.basename(path)], f)
        return local
    return download_file


def test_main_reads_the_five_hub_configs_and_writes_a_loadable_config() -> None:
    requests = []
    with tempfile.TemporaryDirectory() as root:
        output = os.path.join(root, "outputs", "complete.json")
        original = hub.download_file
        hub.download_file = _fake_download(root, requests)
        try:
            build_complete_config.main(["--output", output])
        finally:
            hub.download_file = original

        collection = UniversalJsonConfig(filepath=output).toCollection()

    assert requests == [
        f"universal_audio_dataset_configs/complete-{i}.json" for i in range(1, 6)], requests
    assert collection.name == "Complete UAD", collection.name
    assert collection.randomize_prompt_format is True
    assert [d.name for d in collection.internal_datasets] == [
        "Clotho", "AESDD", "EMNS", "AudioMNIST", "AudioMNISTCommonSense",
        "colombian_spanish"], collection.internal_datasets

    print("PASS: main reads complete-1..5 from the Hub and writes a config that loads.")


def test_output_defaults_to_outputs_not_the_working_directory() -> None:
    args = build_complete_config.build_parser().parse_args([])

    assert os.path.normpath(args.output) == os.path.join("outputs", "complete.json"), args

    print("PASS: --output defaults to outputs/complete.json.")


if __name__ == "__main__":
    for _name, _test in list(globals().items()):
        if _name.startswith("test_"):
            _test()
