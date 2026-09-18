"""Build the `complete.json` run config from the Hub's `complete-1..5`.

Usage:
```
python -m uad_data.build_complete_config [--output outputs/complete.json]
```

`complete-1.json` through `complete-5.json` stay the source of truth: edit them,
rerun this command, and upload the result to the Hub as
`universal_audio_dataset_configs/complete.json`. The output lists the five
configs' internal datasets in order, under the fixed name "Complete UAD" with
random prompt templates, and records its sources in a top-level `built_from`.

The output defaults to `outputs/complete.json`, never the working directory: a
local `complete.json` there would shadow the Hub copy when a run names it.
"""
import argparse
import json
import os

from . import hub

SOURCES = [f"complete-{i}.json" for i in range(1, 6)]
CONFIGS_DIR = "universal_audio_dataset_configs"


def combine_configs(configs: dict[str, dict]) -> dict:
    """Combine run configs, keyed by file name in order, into `complete.json`'s content.

    Each internal dataset's entry is kept as it is. Raises ValueError when an
    internal dataset appears in more than one config.
    """
    datasets, source_of = [], {}
    for filename, config in configs.items():
        for entry in config["datasets"]:
            if entry["name"] in source_of:
                raise ValueError(
                    f"{entry['name']} is in both {source_of[entry['name']]} and {filename}; "
                    "an internal dataset may appear in only one of them.")
            source_of[entry["name"]] = filename
            datasets.append(entry)
    return {
        "name": "Complete UAD",
        "randomize_prompt_format": True,
        "built_from": list(configs),
        "datasets": datasets,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m uad_data.build_complete_config",
        description="Build complete.json from the Hub's complete-1..5 run configs.")
    parser.add_argument("--output", default=os.path.join("outputs", "complete.json"),
                        help="Where to write it (default: outputs/complete.json).")
    parser.add_argument("--repo-id", default=hub.DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=None, help="Hub revision to read the configs at.")
    parser.add_argument("--token", default=None,
                        help="HF token (default: the saved Hugging Face login).")
    return parser


def _read_config(filename: str, args: argparse.Namespace) -> dict:
    path = hub.download_file(
        f"{CONFIGS_DIR}/{filename}", repo_id=args.repo_id,
        revision=args.revision, token=args.token)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    combined = combine_configs({filename: _read_config(filename, args) for filename in SOURCES})
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
        f.write("\n")
    print(f"Wrote {args.output}: {len(combined['datasets'])} internal datasets "
          f"from {', '.join(SOURCES)}.")


if __name__ == "__main__":
    main()
