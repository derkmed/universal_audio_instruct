"""Offline load check for the run configs complete-1..5 (issue #3).

Answers "would a smoke run over complete-1..5 load cleanly today?" by running the
real `uad_data` code (config validation, metadata loading, prompt matching,
`Sample.to_output`) against a LOCAL copy of the Hub dataset repo, plus a full
pass over every audio archive's entry list.

Known limitation (issue #20): this script predates per-utterance rendering (#14)
and does not handle asr_timestamp_search, so don't trust its output for
libricss, libricss_subseg or SparseLibriMix. It imports `uad_data` from the
checkout it sits in, and from #14 on it misreports those datasets in two ways:
  * it builds `Sample` without an `utterance_index`, so every
    asr_timestamp_search row counts as a render error;
  * `vars_not_in_features` compares template variables with `Task.features`
    keys, so it flags `start_time`, `end_time` and `transcription`, which are
    utterance fields.
complete-1..5 don't use these datasets, so the default run is unaffected.

Standard library only. `uad_data` imports three third-party packages that are
not installed here, so this script registers minimal stand-ins in `sys.modules`
BEFORE importing `uad_data`:

  * `datasets`        -- `Split`, `NamedSplit`, `Value`, `Version`, `Audio`.
                         `Split(name)` mimics `datasets.splits.NamedSplit`:
                         compares/hashes by name, `str()` gives the name, and the
                         name must match `^\\w+(\\.\\w+)*$`. That behaviour is
                         reproduced from memory of `datasets/splits.py`, not
                         verified against an installed copy.
  * `jinja2`          -- `Template(src).render(ctx)` supporting only
                         `{{ name }}` substitution (all that prompts/*.json use).
                         Undefined names render as "" (jinja2's default
                         `Undefined`); other values render via `str()`. Any other
                         jinja syntax raises, so the fake cannot silently
                         under-render.
  * `huggingface_hub` -- every entry point raises; nothing touches the network.

Usage:
    python docs/research/complete-configs-load-check.py \
        --uad-root C:/path/to/Universal-Audio-Understanding \
        --cache-dir C:/tmp/uad-listings --workers 4 --e2e-rows 3 --json-out results.json

`--list-only` just builds the archive-entry cache (the slow part: every archive is
decompressed end to end with tarfile mode "r|gz", exactly as the loader reads it).
`--skip-archives` runs only the fast checks (config, metadata, prompts, fields,
rendering). `--e2e-rows N` additionally drives the real `loader.iter_samples` per
(dataset, split) against local files, capped at N rows, plus uncapped passes for
any dataset whose archive fails to list.

Findings: docs/research/complete-configs-load-check.md (issue #3).
"""
from __future__ import annotations

import argparse
import collections
import importlib
import json
import os
import re
import sys
import tarfile
import time
import traceback
import types
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UAD_ROOT = r"C:\Users\derek\Desktop\UAD-DEV\Universal-Audio-Understanding"
DEFAULT_CONFIGS = [f"complete-{i}.json" for i in range(1, 6)]


# --------------------------------------------------------------------------- #
# Stand-ins for packages that are not installed (see module docstring).
# --------------------------------------------------------------------------- #
FAKE_NOTES: list[str] = []
UNDEFINED_TEMPLATE_VARS: collections.Counter = collections.Counter()


def install_fakes() -> None:
    # ---- datasets ----------------------------------------------------------
    ds = types.ModuleType("datasets")
    ds.__fake__ = True
    split_re = re.compile(r"^\w+(\.\w+)*$")

    class NamedSplit(str):
        """str subclass: eq/hash/str/format by name, like datasets.NamedSplit."""

        def __new__(cls, name):
            for part in [p.split("[")[0] for p in str(name).split("+")]:
                if not split_re.match(part):
                    raise ValueError(
                        f"Split name should match '{split_re.pattern}' but got '{part}'.")
            return super().__new__(cls, name)

        def __repr__(self):
            return f"NamedSplit({str.__repr__(self)})"

    class _SplitMeta(type):
        pass

    class Split(metaclass=_SplitMeta):
        TRAIN = NamedSplit("train")
        TEST = NamedSplit("test")
        VALIDATION = NamedSplit("validation")

        def __new__(cls, name):
            if name == "all":
                raise NotImplementedError("fake datasets.Split does not model Split('all')")
            return NamedSplit(name)

    class Value:
        def __init__(self, dtype, id=None):
            self.dtype = dtype

        def __repr__(self):
            return f"Value({self.dtype!r})"

    class Version:
        def __init__(self, version_str, description=None, **kw):
            self.version_str = version_str

        def __str__(self):
            return self.version_str

    class Audio:
        def __init__(self, sampling_rate=None, mono=True, decode=True, id=None):
            self.sampling_rate = sampling_rate
            self.decode = decode

    ds.NamedSplit = NamedSplit
    ds.Split = Split
    ds.Value = Value
    ds.Version = Version
    ds.Audio = Audio
    sys.modules["datasets"] = ds
    FAKE_NOTES.append(
        "datasets: fake Split/NamedSplit (str subclass, name equality), Value, Version, Audio")

    # ---- jinja2 ------------------------------------------------------------
    j2 = types.ModuleType("jinja2")
    j2.__fake__ = True
    var_re = re.compile(r"\{\{\s*([A-Za-z_]\w*)\s*\}\}")

    class Template:
        def __init__(self, source):
            leftover = var_re.sub("", source)
            if "{{" in leftover or "{%" in leftover or "{#" in leftover:
                raise NotImplementedError(
                    f"fake jinja2 only supports '{{{{ name }}}}'; got: {source!r}")
            self.source = source

        def render(self, *args, **kwargs):
            ctx = dict(*args, **kwargs)

            def sub(m):
                name = m.group(1)
                if name not in ctx:
                    UNDEFINED_TEMPLATE_VARS[name] += 1
                    return ""  # jinja2 default Undefined renders as empty string
                return str(ctx[name])

            return var_re.sub(sub, self.source)

    j2.Template = Template
    sys.modules["jinja2"] = j2
    FAKE_NOTES.append("jinja2: fake Template supporting only {{ name }} substitution")

    # ---- huggingface_hub ---------------------------------------------------
    hh = types.ModuleType("huggingface_hub")
    hh.__fake__ = True

    def _no_network(*a, **k):
        raise RuntimeError("network access disabled in offline load check")

    class HfFileSystem:
        def __init__(self, *a, **k):
            _no_network()

    hh.hf_hub_download = _no_network
    hh.snapshot_download = _no_network
    hh.HfFileSystem = HfFileSystem
    sys.modules["huggingface_hub"] = hh
    FAKE_NOTES.append("huggingface_hub: every call raises (no network)")


# --------------------------------------------------------------------------- #
# Archive listing (the slow, I/O-bound part). Runs in worker processes, so it
# must not depend on uad_data or the fakes.
# --------------------------------------------------------------------------- #
def _cache_file(cache_dir: str, rel_archive: str) -> str:
    return os.path.join(cache_dir, rel_archive.replace("/", "__") + ".listing.json")


def list_archive(uad_root: str, rel_archive: str, cache_dir: str | None) -> dict:
    """Stream the whole archive ("r|gz", as loader._open_archive does) and record
    every entry name, its type, and the archive's size/mtime for cache validity."""
    path = os.path.join(uad_root, rel_archive)
    st = os.stat(path)
    if cache_dir:
        cf = _cache_file(cache_dir, rel_archive)
        if os.path.exists(cf):
            with open(cf, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("size") == st.st_size and cached.get("mtime") == st.st_mtime:
                cached["from_cache"] = True
                return cached
    t0 = time.time()
    files, nonfiles = [], collections.Counter()
    nonfile_names = []
    error = None
    try:
        with tarfile.open(path, mode="r|gz") as tar:
            for m in tar:
                if m.isfile():
                    files.append(m.name)
                else:
                    kind = ("dir" if m.isdir() else "symlink" if m.issym() else
                            "hardlink" if m.islnk() else "other")
                    nonfiles[kind] += 1
                    if kind != "dir":
                        nonfile_names.append([kind, m.name])
    except Exception as e:  # truncated / corrupt archive
        error = f"{type(e).__name__}: {e}"
    result = {
        "archive": rel_archive,
        "size": st.st_size,
        "mtime": st.st_mtime,
        "files": files,
        "nonfiles": dict(nonfiles),
        "nonfile_names": nonfile_names[:2000],
        "error": error,
        "seconds": round(time.time() - t0, 1),
        "from_cache": False,
    }
    if cache_dir:  # errors (e.g. truncation) are deterministic too, so cache them
        os.makedirs(cache_dir, exist_ok=True)
        tmp = _cache_file(cache_dir, rel_archive) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f)
        os.replace(tmp, _cache_file(cache_dir, rel_archive))
    return result


def list_archives(uad_root, rel_archives, cache_dir, workers) -> dict[str, dict]:
    out = {}
    # Biggest first so the long ones start early.
    order = sorted(rel_archives, key=lambda r: -os.path.getsize(os.path.join(uad_root, r)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(list_archive, uad_root, r, cache_dir): r for r in order}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"archive": r, "files": [], "nonfiles": {}, "nonfile_names": [],
                       "error": f"{type(e).__name__}: {e}", "seconds": None, "from_cache": False}
            out[r] = res
            print(f"  listed {r}: {len(res['files'])} files, nonfiles={res['nonfiles']}, "
                  f"error={res['error']}, {res['seconds']}s, cache={res['from_cache']}",
                  flush=True)
    return out


# --------------------------------------------------------------------------- #
# Checks that use the real uad_data code.
# --------------------------------------------------------------------------- #
def fmt_exc(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def reload_registry(uad):
    """InternalDataset objects in DATASETS_DIRECTORY are module-level singletons
    that toInternalDataset() mutates; reload to give each config a fresh registry."""
    importlib.reload(uad["internal_datasets"])


def check_prompt_dir(uad, prompts_dir: str) -> dict:
    """Every *.json in PROMPTS_DIR is parsed on every _get_prompt_templates call,
    so one bad file breaks every task. Validate them all, and compare each file's
    template variables with its Task.features keys."""
    prompts = uad["prompts"]
    tasks = uad["tasks"]
    res = {"files": {}, "errors": []}
    var_re = re.compile(r"\{\{\s*([A-Za-z_]\w*)\s*\}\}")
    for fp in sorted(Path(prompts_dir).glob("*.json")):
        info = {}
        try:
            pf = prompts.PromptFilepath(filepath=str(fp))
            info["task"] = pf.task.value
            info["n_templates"] = len(pf.all_templates)
            srcs = [t.template for t in pf.system_instruction_templates + pf.prompt_templates
                    + pf.output_templates]
            used = sorted({v for s in srcs for v in var_re.findall(s)})
            try:
                feats = sorted(pf.task.features.keys())
            except NotImplementedError as e:
                feats = None
                info["features_error"] = fmt_exc(e)
            info["template_vars"] = used
            info["features"] = feats
            info["vars_not_in_features"] = sorted(set(used) - set(feats or []))
            raw = fp.read_bytes()
            info["non_ascii"] = any(b > 127 for b in raw)
        except Exception as e:
            info["error"] = fmt_exc(e)
            res["errors"].append(f"{fp.name}: {fmt_exc(e)}")
        res["files"][fp.name] = info
    registered = {t.value for t in tasks.Task}
    covered = {i.get("task") for i in res["files"].values()}
    res["tasks_without_prompt_file"] = sorted(registered - covered)
    return res


def near_misses(missing: list[str], files: list[str], limit: int = 5) -> dict:
    """Classify metadata paths that have no exact archive entry."""
    by_lower = collections.defaultdict(list)
    by_base = collections.defaultdict(list)
    by_nfc = {}
    stripped = {}
    for f in files:
        by_lower[f.lower()].append(f)
        by_base[f.rsplit("/", 1)[-1]].append(f)
        by_nfc[unicodedata.normalize("NFC", f)] = f
        s = f[2:] if f.startswith("./") else f
        stripped[s] = f
    kinds = collections.Counter()
    examples = collections.defaultdict(list)
    for ap in missing:
        norm = ap.replace("\\", "/")
        if norm.startswith("./"):
            norm = norm[2:]
        if norm in stripped:
            kind, hit = "leading ./ or backslash difference", stripped[norm]
        elif unicodedata.normalize("NFC", ap) in by_nfc:
            kind, hit = "unicode normalisation difference", by_nfc[unicodedata.normalize("NFC", ap)]
        elif ap.lower() in by_lower:
            kind, hit = "case difference", by_lower[ap.lower()][0]
        elif ap.strip() != ap and ap.strip() in stripped:
            kind, hit = "surrounding whitespace", stripped[ap.strip()]
        else:
            base = ap.replace("\\", "/").rsplit("/", 1)[-1]
            cands = by_base.get(base, [])
            if cands:
                kind, hit = "same filename, different directory", cands[0]
            else:
                kind, hit = "no entry with this filename", None
        kinds[kind] += 1
        if len(examples[kind]) < limit:
            examples[kind].append({"audio_path": ap, "archive_entry": hit})
    return {"kinds": dict(kinds), "examples": dict(examples)}


def check_split(uad, internal_dataset, split, randomize, uad_root, listing,
                render_limit) -> dict:
    loader, sample_mod, hub = uad["loader"], uad["sample"], uad["hub"]
    r = {"dataset": internal_dataset.name, "split": str(split),
         "tasks": [t.value for t in internal_dataset.tasks]}

    # ---- 2. metadata file ---------------------------------------------------
    rel = hub.to_repo_path(internal_dataset.split_metadata_path(split))
    r["metadata_file"] = rel
    local = os.path.join(uad_root, *rel.split("/"))
    r["metadata_exists"] = os.path.exists(local)
    if not r["metadata_exists"]:
        return r
    try:
        with open(local, encoding="utf-8") as f:
            raw = json.load(f)
        metadata = loader._load_split_metadata(local, internal_dataset.tasks, split)
        r["metadata_parses"] = True
    except Exception as e:
        r["metadata_parses"] = False
        r["metadata_error"] = fmt_exc(e)
        return r
    r["n_records"] = len(raw)
    aps = [rec.get("audio_path") for rec in raw]
    uniq = set(aps)
    r["n_unique_audio_paths"] = len(uniq)
    r["n_records_lost_to_duplicate_audio_path"] = len(raw) - len(uniq)
    dup_counts = collections.Counter(aps)
    dups = [ap for ap, c in dup_counts.items() if c > 1]
    r["duplicate_audio_path_examples"] = []
    for ap in dups[:3]:
        recs = [x for x in raw if x.get("audio_path") == ap]
        diff_keys = sorted({k for x in recs for k in x
                            if any(y.get(k) != x.get(k) for y in recs)})
        r["duplicate_audio_path_examples"].append(
            {"audio_path": ap, "copies": len(recs), "fields_that_differ": diff_keys,
             "kept_by_loader": {k: recs[-1].get(k) for k in diff_keys}})
    if dups:
        # Do the copies differ only in task-feature fields? (i.e. real label loss)
        feature_keys = {k for t in internal_dataset.tasks for k in t.features}
        differing_feature = 0
        for ap in dups:
            recs = [x for x in raw if x.get("audio_path") == ap] if len(dups) < 50 else None
            if recs is None:
                break
            if any(any(y.get(k) != recs[0].get(k) for y in recs) for k in feature_keys):
                differing_feature += 1
        if len(dups) >= 50:
            groups = collections.defaultdict(list)
            for x in raw:
                if dup_counts[x.get("audio_path")] > 1:
                    groups[x.get("audio_path")].append(x)
            differing_feature = sum(
                1 for recs in groups.values()
                if any(any(y.get(k) != recs[0].get(k) for y in recs) for k in feature_keys))
        r["n_duplicated_paths_with_differing_task_fields"] = differing_feature
    r["n_records_without_audio_path"] = sum(1 for ap in aps if ap is None)

    # ---- 3/4/5. per task ------------------------------------------------------
    r["task_checks"] = {}
    for task in internal_dataset.tasks:
        tc = {}
        try:
            templates = loader._get_prompt_templates(task, False)
            tc["prompt_file_ok"] = True
            tc["n_templates"] = len(templates)
            if randomize:
                loader._get_prompt_templates(task, True)
        except Exception as e:
            tc["prompt_file_ok"] = False
            tc["prompt_error"] = fmt_exc(e)
            templates = []
        feats = list(task.features.keys())
        tc["features"] = feats
        missing = collections.Counter()
        missing_examples = []
        empty = collections.Counter()
        empty_examples = []
        nonstr = collections.Counter()
        for rec in raw:
            for k in feats:
                if k not in rec:
                    missing[k] += 1
                    if len(missing_examples) < 3:
                        missing_examples.append({"audio_path": rec.get("audio_path"), "key": k})
                else:
                    v = rec[k]
                    if not isinstance(v, str):
                        nonstr[f"{k}:{type(v).__name__}"] += 1
                    if v is None or (isinstance(v, str) and not v.strip()):
                        empty[k] += 1
                        if len(empty_examples) < 3:
                            empty_examples.append({"audio_path": rec.get("audio_path"), "key": k,
                                                   "value": v})
        tc["n_records_missing_key"] = dict(missing)
        tc["missing_key_examples"] = missing_examples
        tc["n_records_empty_value"] = dict(empty)
        tc["empty_value_examples"] = empty_examples
        tc["non_string_values"] = dict(nonstr)

        # Rendering through the real Sample.to_output, over the records the loader
        # would actually keep (the audio_path-keyed dict), all templates.
        n_ok, errors, example_rows = 0, collections.Counter(), []
        undefined_before = sum(UNDEFINED_TEMPLATE_VARS.values())
        recs = [(k, v) for k, v in metadata.items() if k != "split"]
        if render_limit is not None:
            recs = recs[:render_limit]
        for ap, rec in recs:
            for si_t, p_t, o_t in templates:
                try:
                    row = sample_mod.Sample(
                        audio_path=ap, dataset_name=internal_dataset.name,
                        split=metadata["split"], task=task, audio_data=b"",
                        metadata=rec, system_instruction_template=si_t,
                        prompt_template=p_t, output_template=o_t,
                    ).to_output()
                    n_ok += 1
                    if len(example_rows) < 1:
                        example_rows.append({k: row[k] for k in
                                             ("system_instruction", "prompt", "output")
                                             if k in row})
                except Exception as e:
                    errors[fmt_exc(e)] += 1
        tc["n_records_rendered"] = len(recs)
        tc["n_rows_rendered_ok"] = n_ok
        tc["render_errors"] = dict(errors.most_common(5))
        tc["n_render_errors"] = sum(errors.values())
        tc["undefined_template_vars"] = sum(UNDEFINED_TEMPLATE_VARS.values()) - undefined_before
        tc["example_row"] = example_rows[0] if example_rows else None
        r["task_checks"][task.value] = tc

    # ---- 6. audio paths vs archive entries -----------------------------------
    if listing is None:
        r["archive_checked"] = False
        return r
    r["archive_checked"] = True
    r["archive"] = listing["archive"]
    r["archive_error"] = listing.get("error")
    files = listing["files"]
    file_set = set(files)
    keys = [k for k in metadata if k != "split"]
    matched = [k for k in keys if k in file_set]
    missing = [k for k in keys if k not in file_set]
    r["n_paths_matched"] = len(matched)
    r["n_paths_missing"] = len(missing)
    r["missing_examples"] = missing[:5]
    r["near_misses"] = near_misses(missing, files) if missing else {"kinds": {}, "examples": {}}
    counts = collections.Counter(files)
    r["n_matched_paths_duplicated_in_archive"] = sum(1 for k in matched if counts[k] > 1)
    key_set = set(keys)
    positions = [i for i, f in enumerate(files) if f in key_set]
    r["first_match_entry_index"] = positions[0] if positions else None
    r["last_match_entry_index"] = positions[-1] if positions else None
    r["n_archive_file_entries_read"] = len(files)
    if listing.get("error"):
        # The loader iterates the same "r|gz" stream, so it raises this error when
        # it reaches entry len(files), unless max_samples stops it earlier.
        r["archive_error_after_n_matched_clips"] = len(positions)
    nonfile = {n for _, n in listing.get("nonfile_names", [])}
    r["n_missing_paths_present_as_nonfile_entry"] = sum(1 for k in missing if k in nonfile)
    r["split_key_collides_with_archive_entry"] = "split" in file_set
    return r


def e2e_one(uad_root: str, cfg: str, dataset_name: str, split: str,
            max_rows: int | None) -> dict:
    """Run the REAL loader.iter_samples for one (config, dataset, split), with the
    hub functions pointed at local files (as tests/test_loader.py does). Rows are
    counted and checked, not kept, so a full pass does not hold audio in memory.
    Runs in a worker process."""
    if "datasets" not in sys.modules:
        install_fakes()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    hub = importlib.import_module("uad_data.hub")
    loader = importlib.import_module("uad_data.loader")
    prompts = importlib.import_module("uad_data.prompts")
    jcl = importlib.import_module("uad_data.json_config_loader")
    coll_mod = importlib.import_module("uad_data.collection")
    importlib.reload(importlib.import_module("uad_data.internal_datasets"))
    prompts.PROMPTS_DIR = os.path.join(uad_root, "prompts")

    opened = {"stream": 0, "download": []}

    def local_path(path_or_url):
        return os.path.join(uad_root, *hub.to_repo_path(path_or_url).split("/"))

    def fake_download_file(path_or_url, *, repo_id=None, revision=None, token=None):
        opened["download"].append(hub.to_repo_path(path_or_url))
        return local_path(path_or_url)

    def fake_open_archive_stream(path_or_url, *, repo_id=None, revision=None, token=None):
        opened["stream"] += 1
        return open(local_path(path_or_url), "rb")

    hub.download_file = fake_download_file
    hub.open_archive_stream = fake_open_archive_stream

    out = {"config": cfg, "dataset": dataset_name, "split": split, "max_rows": max_rows}
    coll = jcl.UniversalJsonConfig(filepath=os.path.join(
        uad_root, "universal_audio_dataset_configs", cfg)).toCollection()
    d = next(x for x in coll.internal_datasets if x.name == dataset_name)
    one = coll_mod.UadCollection(
        name=coll.name, internal_datasets=[d],
        randomize_prompt_format=coll.randomize_prompt_format,
        sample_filter=coll.sample_filter)
    split_key = sys.modules["datasets"].Split(split)
    configured = {t.value for t in d.tasks}
    n = 0
    problems = collections.Counter()
    tasks_seen = collections.Counter()
    clips = set()
    t0 = time.time()
    try:
        for row in loader.iter_samples(
                one, split_key, repo_id=hub.DEFAULT_REPO_ID, revision=None, token=None,
                max_samples=max_rows, stream=max_rows is not None):
            n += 1
            tasks_seen[row["task"]] += 1
            clips.add(row["audio_path"])
            if row["task"] not in configured:
                problems["task not configured"] += 1
            if row["split"] != split:
                problems["split field mismatch"] += 1
            if row["originating_dataset"] != dataset_name:
                problems["originating_dataset mismatch"] += 1
            if not row["audio"]["bytes"]:
                problems["empty audio bytes"] += 1
            if not (row.get("output") or "").strip():
                problems["empty output"] += 1
            if not ((row.get("system_instruction") or "").strip()
                    or (row.get("prompt") or "").strip()):
                problems["empty instruction and prompt"] += 1
        out["exception"] = None
    except Exception as e:
        out["exception"] = fmt_exc(e)
    out["seconds"] = round(time.time() - t0, 1)
    out["rows"] = n
    out["clips"] = len(clips)
    out["rows_per_task"] = dict(tasks_seen)
    out["row_problems"] = dict(problems)
    out["archive_opened_via"] = "stream" if opened["stream"] else "download"
    return out


def run_e2e(uad_root, plan, workers) -> list[dict]:
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(e2e_one, uad_root, *p): p for p in plan}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"config": p[0], "dataset": p[1], "split": p[2], "max_rows": p[3],
                       "exception": f"harness error: {fmt_exc(e)}", "rows": None}
            print(f"  e2e {res['dataset']}/{res['split']} max_rows={res['max_rows']}: "
                  f"rows={res['rows']} exc={res['exception']} {res.get('seconds')}s",
                  flush=True)
            results.append(res)
    order = {p: i for i, p in enumerate(plan)}
    results.sort(key=lambda r: order.get((r["config"], r["dataset"], r["split"], r["max_rows"]), 0))
    return results


def compare_to_snapshot(uad_root: str, rel_paths: list[str], hash_paths: list[str]) -> dict:
    """Compare local files with the Hub snapshot that `huggingface_hub` recorded
    when the local copy was downloaded (`.cache/huggingface/trees/<commit>.json`
    and `.cache/huggingface/download/<path>.metadata`). This is local data only;
    it says what the Hub served at that commit, not what the Hub holds today."""
    import hashlib
    cache = os.path.join(uad_root, ".cache", "huggingface")
    trees = sorted(Path(cache, "trees").glob("*.json")) if os.path.isdir(cache) else []
    if not trees:
        return {"available": False}
    out = {"available": True, "snapshots": [t.stem for t in trees], "files": {},
           "mismatches": []}
    tree = {}
    for t in trees:
        with open(t, encoding="utf-8") as f:
            tree.update(json.load(f).get("files", {}))
    for rel in rel_paths:
        info = tree.get(rel)
        local = os.path.join(uad_root, *rel.split("/"))
        rec = {"in_snapshot": info is not None, "local_exists": os.path.exists(local)}
        if info is not None and rec["local_exists"]:
            rec["local_size"] = os.path.getsize(local)
            rec["snapshot_size"] = info.get("lfs_size", info.get("size"))
            rec["size_matches"] = rec["local_size"] == rec["snapshot_size"]
            md = os.path.join(cache, "download", *rel.split("/")) + ".metadata"
            if os.path.exists(md):
                with open(md, encoding="utf-8") as f:
                    lines = f.read().split("\n")
                rec["downloaded_commit"] = lines[0]
                rec["download_etag_matches_snapshot_sha256"] = (
                    lines[1] == info.get("lfs_sha256") if info.get("lfs_sha256") else None)
            if rel in hash_paths:
                h = hashlib.sha256()
                with open(local, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 24), b""):
                        h.update(chunk)
                rec["local_sha256"] = h.hexdigest()
                rec["local_sha256_matches_snapshot"] = h.hexdigest() == info.get("lfs_sha256")
        if (not rec["in_snapshot"] or not rec["local_exists"]
                or rec.get("size_matches") is False
                or rec.get("download_etag_matches_snapshot_sha256") is False
                or rec.get("local_sha256_matches_snapshot") is False):
            out["mismatches"].append(rel)
        out["files"][rel] = rec
    return out


def run(args) -> dict:
    install_fakes()
    sys.path.insert(0, str(REPO_ROOT))
    uad = {name: importlib.import_module(f"uad_data.{name}") for name in
           ("tasks", "prompts", "io_templates", "sample", "internal_dataset",
            "internal_datasets", "collection", "filters", "json_config_loader",
            "hub", "loader")}
    import uad_data  # noqa: F401  (package __init__ must import cleanly too)

    uad_root = args.uad_root
    prompts_dir = os.path.join(uad_root, "prompts")
    uad["prompts"].PROMPTS_DIR = prompts_dir
    config_dir = os.path.join(uad_root, "universal_audio_dataset_configs")

    results = {"fakes": FAKE_NOTES, "uad_root": uad_root, "repo_root": str(REPO_ROOT),
               "configs": {}, "prompt_dir": check_prompt_dir(uad, prompts_dir)}

    # ---- 1. config validation (fresh registry per config) --------------------
    collections_by_cfg = {}
    for cfg in args.configs:
        reload_registry(uad)
        path = os.path.join(config_dir, cfg)
        c = {"path": path}
        try:
            ujc = uad["json_config_loader"].UniversalJsonConfig(filepath=path)
            coll = ujc.toCollection()
            c["loads"] = True
            c["randomize_prompt_format"] = coll.is_random_prompt_format_selection()
            c["sample_filter"] = type(coll.sample_filter).__name__
            c["datasets"] = [
                {"name": d.name, "tasks": [t.value for t in d.tasks],
                 "splits": [str(s) for s in d.get_splits()],
                 "archive": uad["hub"].to_repo_path(d.data_url)}
                for d in coll.internal_datasets]
            # What iter_samples would visit for each split name a run can pass.
            c["datasets_per_split"] = {
                s: [d.name for d in coll.get_datasets_with_splits(
                    sys.modules["datasets"].Split(s))]
                for s in ("train", "validation", "test")}
            collections_by_cfg[cfg] = coll
        except Exception as e:
            c["loads"] = False
            c["error"] = fmt_exc(e)
            c["traceback"] = traceback.format_exc()
        results["configs"][cfg] = c

    # Same five configs, one process, no registry reload between them.
    reload_registry(uad)
    seq = {}
    for cfg in args.configs:
        try:
            uad["json_config_loader"].UniversalJsonConfig(
                filepath=os.path.join(config_dir, cfg)).toCollection()
            seq[cfg] = "ok"
        except Exception as e:
            seq[cfg] = fmt_exc(e)
    results["sequential_same_process"] = seq

    # Registry-mutation probe: after complete-1 narrows Clotho to caption, can a
    # later config in the same process still ask for another registered task?
    reload_registry(uad)
    probe = {}
    try:
        uad["json_config_loader"].UniversalJsonConfig(
            filepath=os.path.join(config_dir, "complete-1.json")).toCollection()
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump({"name": "probe", "datasets": [
                {"name": "Clotho", "splits": ["test"], "tasks": ["commonsense"]}]}, tf)
        try:
            uad["json_config_loader"].UniversalJsonConfig(filepath=tf.name).toCollection()
            probe["clotho_commonsense_after_complete_1"] = "ok"
        except Exception as e:
            probe["clotho_commonsense_after_complete_1"] = fmt_exc(e)
        finally:
            os.unlink(tf.name)
    except Exception as e:
        probe["error"] = fmt_exc(e)
    results["registry_mutation_probe"] = probe

    # ---- 6 (prep). archive listings ------------------------------------------
    archives = sorted({d["archive"] for c in results["configs"].values() if c.get("loads")
                       for d in c["datasets"]})
    listings = {}
    if not args.skip_archives:
        print(f"Listing {len(archives)} archives with {args.workers} workers ...", flush=True)
        listings = list_archives(uad_root, archives, args.cache_dir, args.workers)
    results["archives"] = {
        a: {k: v for k, v in listings[a].items() if k not in ("files", "nonfile_names")}
        | {"n_files": len(listings[a]["files"])}
        for a in listings}

    # ---- 2-6. per (dataset, split, task) ---------------------------------------
    for cfg, coll in collections_by_cfg.items():
        rows = []
        randomize = coll.is_random_prompt_format_selection()
        for d in coll.internal_datasets:
            archive = uad["hub"].to_repo_path(d.data_url)
            for split in d.get_splits():
                print(f"checking {cfg} {d.name}/{split}", flush=True)
                rows.append(check_split(uad, d, split, randomize, uad_root,
                                        listings.get(archive), args.render_limit))
        results["configs"][cfg]["splits"] = rows
    results["undefined_template_vars_total"] = dict(UNDEFINED_TEMPLATE_VARS)

    # ---- local copy vs the Hub snapshot it was downloaded from ---------------
    rel_paths = sorted(
        set(archives)
        | {s["metadata_file"] for c in results["configs"].values()
           for s in c.get("splits", [])}
        | {f"prompts/{p.name}" for p in Path(prompts_dir).glob("*.json")}
        | {f"universal_audio_dataset_configs/{cfg}" for cfg in args.configs})
    errored = [a for a, l in listings.items() if l.get("error")]
    results["local_copy_vs_snapshot"] = compare_to_snapshot(uad_root, rel_paths, errored)

    # ---- 7. end-to-end through the real iter_samples --------------------------
    if args.e2e_rows is not None:
        plan = [(cfg, d["name"], s, args.e2e_rows)
                for cfg, c in results["configs"].items() if c.get("loads")
                for d in c["datasets"] for s in d["splits"]]
        # Full (uncapped) passes for datasets whose archive failed to list, plus any
        # requested explicitly, to see what an uncapped run does.
        full = set(args.e2e_full)
        for cfg, c in results["configs"].items():
            for d in c.get("datasets", []):
                if listings.get(d["archive"], {}).get("error"):
                    full.add(d["name"])
        plan += [(cfg, d["name"], s, None)
                 for cfg, c in results["configs"].items() if c.get("loads")
                 for d in c["datasets"] if d["name"] in full for s in d["splits"]]
        print(f"End-to-end: {len(plan)} runs with {args.workers} workers ...", flush=True)
        results["e2e"] = run_e2e(uad_root, plan, args.workers)
    return results


def summarise(results: dict) -> None:
    print("\n==== SUMMARY ====")
    pd = results["prompt_dir"]
    print("prompt files:", {k: (v.get("task"), v.get("n_templates"), v.get("vars_not_in_features"),
                                v.get("error")) for k, v in pd["files"].items()})
    print("tasks with no prompt file:", pd["tasks_without_prompt_file"])
    print("sequential same-process:", results["sequential_same_process"])
    print("registry mutation probe:", results["registry_mutation_probe"])
    snap = results.get("local_copy_vs_snapshot", {})
    print("local copy vs Hub snapshot:", snap.get("snapshots"), "mismatches:",
          snap.get("mismatches"))
    for a, info in results.get("archives", {}).items():
        if info.get("error"):
            print(f"ARCHIVE ERROR {a}: {info['error']} after {info['n_files']} file entries")
    for cfg, c in results["configs"].items():
        print(f"\n{cfg}: loads={c['loads']} {c.get('error', '')}")
        for s in c.get("splits", []):
            head = (f"  {s['dataset']}/{s['split']}: meta={s.get('metadata_exists')}/"
                    f"{s.get('metadata_parses')} n={s.get('n_records')} "
                    f"dup_lost={s.get('n_records_lost_to_duplicate_audio_path')} "
                    f"matched={s.get('n_paths_matched')} missing={s.get('n_paths_missing')} "
                    f"near={s.get('near_misses', {}).get('kinds')} "
                    f"first_match@{s.get('first_match_entry_index')}")
            print(head)
            for t, tc in s.get("task_checks", {}).items():
                print(f"    {t}: prompt_ok={tc['prompt_file_ok']} templates={tc.get('n_templates')} "
                      f"missing={tc['n_records_missing_key']} empty={tc['n_records_empty_value']} "
                      f"rows_ok={tc['n_rows_rendered_ok']} render_err={tc['n_render_errors']} "
                      f"undef={tc['undefined_template_vars']}")
    if results.get("e2e"):
        print("\nend-to-end (real iter_samples):")
        for e in results["e2e"]:
            print(f"  {e['config']} {e['dataset']}/{e['split']} max_rows={e['max_rows']}: "
                  f"rows={e['rows']} clips={e.get('clips')} per_task={e.get('rows_per_task')} "
                  f"problems={e.get('row_problems')} exc={e['exception']} "
                  f"via={e.get('archive_opened_via')} {e.get('seconds')}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--uad-root", default=DEFAULT_UAD_ROOT)
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    p.add_argument("--cache-dir", default=None,
                   help="Directory to cache archive entry listings (keyed by size+mtime).")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--render-limit", type=int, default=None,
                   help="Render only the first N records per (dataset, split, task). "
                        "Default: all records.")
    p.add_argument("--skip-archives", action="store_true")
    p.add_argument("--e2e-rows", type=int, default=None,
                   help="Also run the real loader.iter_samples per (dataset, split), "
                        "capped at this many rows (stream mode, as a smoke run would).")
    p.add_argument("--e2e-full", nargs="*", default=[],
                   help="Dataset names to also run uncapped end-to-end (datasets whose "
                        "archive failed to list are always added).")
    p.add_argument("--list-only", action="store_true",
                   help="Only build the archive listing cache for the configs' archives.")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    except AttributeError:
        pass

    if args.list_only:
        config_dir = os.path.join(args.uad_root, "universal_audio_dataset_configs")
        # Archive paths come from the registry, so import it (with fakes) first.
        install_fakes()
        sys.path.insert(0, str(REPO_ROOT))
        reg = importlib.import_module("uad_data.internal_datasets").DATASETS_DIRECTORY
        hub = importlib.import_module("uad_data.hub")
        names = {d["name"] for cfg in args.configs
                 for d in json.load(open(os.path.join(config_dir, cfg)))["datasets"]}
        archives = sorted({hub.to_repo_path(reg[n].data_url) for n in names if n in reg})
        list_archives(args.uad_root, archives, args.cache_dir, args.workers)
        return

    results = run(args)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1, default=str)
    summarise(results)


if __name__ == "__main__":
    main()
