"""Rows, clips and download sizes per internal dataset in complete-1..5 (issue #44).

For every internal dataset in the Hub run configs complete-1..5 this reports:
  * the clips and rendered rows per task that `uad_data.loader.iter_samples`
    yields for each configured split (train, validation, test);
  * whether a validation split exists (in the run config, and on the Hub);
  * the archive size on the Hub, and the audio bytes inside it.

What comes from where:
  * Configs, prompt files and split metadata JSONs are downloaded from the Hub
    at its current `main` commit (small files only) into --cache-dir.
  * File sizes and LFS sha256s come from `HfApi.list_repo_tree` at that commit.
  * Audio archives are NOT downloaded. Which metadata records have an archive
    entry (the loader keeps only those) comes from listing the LOCAL copy of
    each archive (--uad-root), and only when its sha256 recorded in the local
    `huggingface_hub` snapshot equals the Hub's current LFS sha256. So the
    listing is exactly what the Hub serves today.
  * Rows are rendered with the real `uad_data` code (config validation,
    `_load_split_metadata`, `_get_prompt_templates`, `Task.utterance_indices`,
    `Sample.to_output`) with empty audio bytes. Rendering needs no audio.

Usage (repo venv; needs datasets, jinja2, huggingface_hub and an HF token):
    python docs/research/train-rows-and-sizes.py --cache-dir <scratch> --json-out <scratch>/rows.json

Findings: docs/research/train-rows-and-sizes.md
"""
from __future__ import annotations

import argparse
import collections
import importlib
import json
import os
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_ID = "AudioInstruct/Universal-Audio-Understanding"
DEFAULT_UAD_ROOT = r"C:\Users\derek\Desktop\UAD-DEV\Universal-Audio-Understanding"
CONFIGS = [f"complete-{i}.json" for i in range(1, 6)]
SPLITS = ["train", "validation", "test"]


def list_archive(args) -> dict:
    """Stream a local archive ("r|gz", as the loader does): name -> size of every
    regular file, plus the error if the stream breaks. Cached by path+size+mtime."""
    path, cache_dir = args
    st = os.stat(path)
    cf = os.path.join(cache_dir, "listing__" + Path(path).name + ".json")
    if os.path.exists(cf):
        with open(cf, encoding="utf-8") as f:
            cached = json.load(f)
        if cached["size"] == st.st_size and cached["mtime"] == st.st_mtime:
            return cached
    files, error = {}, None
    try:
        with tarfile.open(path, mode="r|gz") as tar:
            for m in tar:
                if m.isfile():
                    files[m.name] = m.size
    except Exception as e:  # truncated archive (MLEnd_Intonation)
        error = f"{type(e).__name__}: {e}"
    out = {"path": path, "size": st.st_size, "mtime": st.st_mtime,
           "files": files, "error": error}
    with open(cf, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uad-root", default=DEFAULT_UAD_ROOT)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json-out")
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    revision = api.dataset_info(REPO_ID).sha
    tree = {}
    for item in api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True,
                                   revision=revision):
        if hasattr(item, "size"):
            lfs = getattr(item, "lfs", None)
            tree[item.path] = {"size": item.size,
                               "sha256": lfs.sha256 if lfs else None}

    def fetch(rel: str) -> str:
        return hf_hub_download(REPO_ID, rel, repo_type="dataset", revision=revision,
                               local_dir=os.path.join(args.cache_dir, "hub"))

    sys.path.insert(0, str(REPO_ROOT))
    from uad_data import hub, internal_datasets, json_config_loader, loader, prompts, sample as sample_lib
    prompt_files = [p for p in tree if p.startswith("prompts/") and p.endswith(".json")]
    prompts.PROMPTS_DIR = os.path.dirname(fetch(prompt_files[0]))
    for p in prompt_files:
        fetch(p)

    # Local snapshot sha256s, to decide whether a local archive equals the Hub's.
    snap = {}
    for t in Path(args.uad_root, ".cache", "huggingface", "trees").glob("*.json"):
        snap.update(json.loads(t.read_text(encoding="utf-8"))["files"])

    plan = []  # (config, dataset, tasks, configured splits, registered splits)
    for cfg in CONFIGS:
        importlib.reload(internal_datasets)  # toCollection mutates the registry
        registered = {n: [str(s) for s in d.get_splits()]
                      for n, d in internal_datasets.DATASETS_DIRECTORY.items()}
        coll = json_config_loader.UniversalJsonConfig(
            filepath=fetch(f"universal_audio_dataset_configs/{cfg}")).toCollection()
        assert coll.is_random_prompt_format_selection(), cfg
        for d in coll.internal_datasets:
            plan.append((cfg, d, [str(s) for s in d.get_splits()], registered[d.name]))

    archives = {}
    for _, d, _, _ in plan:
        rel = hub.to_repo_path(d.data_url)
        hub_sha = tree[rel]["sha256"]
        local = os.path.join(args.uad_root, *rel.split("/"))
        same = (os.path.exists(local) and snap.get(rel, {}).get("lfs_sha256") == hub_sha
                and os.path.getsize(local) == tree[rel]["size"])
        archives[rel] = {"hub_size": tree[rel]["size"], "local_matches_hub": same,
                         "local": local}
    with ProcessPoolExecutor(args.workers) as pool:
        todo = [r for r, a in archives.items() if a["local_matches_hub"]]
        for rel, res in zip(todo, pool.map(
                list_archive, [(archives[r]["local"], args.cache_dir) for r in todo])):
            archives[rel].update(files=res["files"], error=res["error"])
            print(f"listed {rel}: {len(res['files'])} files, error={res['error']}", flush=True)

    rows_out = []
    for cfg, d, cfg_splits, reg_splits in plan:
        rel = hub.to_repo_path(d.data_url)
        arc = archives[rel]
        files = arc.get("files")
        entry = {"config": cfg, "dataset": d.name, "tasks": [t.value for t in d.tasks],
                 "configured_splits": cfg_splits, "registered_splits": reg_splits,
                 "archive_bytes": arc["hub_size"],
                 "archive_audio_bytes": sum(files.values()) if files else None,
                 "archive_error": arc.get("error"),
                 "archive_listed": files is not None, "splits": {}}
        for split in SPLITS:
            meta_rel = f"data/{d.name}/{d.name}_{split}.json"
            s = {"configured": split in cfg_splits, "registered": split in reg_splits,
                 "metadata_on_hub": meta_rel in tree}
            if s["metadata_on_hub"]:
                md = loader._load_split_metadata(fetch(meta_rel), d.tasks, split)
                paths = [k for k in md if k != "split"]
                with open(fetch(meta_rel), encoding="utf-8") as f:
                    s["records"] = len(json.load(f))
                s["unique_clips"] = len(paths)
                if files is not None:
                    matched = [p for p in paths if p in files]
                    s["clips"] = len(matched)
                    s["audio_bytes"] = sum(files[p] for p in matched)
                else:
                    matched = paths
                    s["clips"] = None
                rows = collections.Counter()
                errors = collections.Counter()
                for task in d.tasks:
                    templates = loader._get_prompt_templates(task, True)
                    for p in matched:
                        rec = md[p]
                        for ui in task.utterance_indices(rec):
                            for si_t, p_t, o_t in templates:
                                try:
                                    sample_lib.Sample(
                                        audio_path=p, dataset_name=d.name, split=split,
                                        task=task, audio_data=b"", metadata=rec,
                                        system_instruction_template=si_t,
                                        prompt_template=p_t, output_template=o_t,
                                        utterance_index=ui).to_output()
                                    rows[task.value] += 1
                                except Exception:
                                    errors[task.value] += 1
                s["rows"] = dict(rows)
                s["render_errors"] = dict(errors)
            entry["splits"][split] = s
        rows_out.append(entry)
        print(cfg, d.name, {k: (v.get("clips"), v.get("rows")) for k, v in entry["splits"].items()
                            if v.get("configured")}, flush=True)

    result = {"revision": revision, "datasets": rows_out,
              "archives": {r: {k: v for k, v in a.items() if k != "files"}
                           for r, a in archives.items()}}
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
