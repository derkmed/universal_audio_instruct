"""How deep into each UAD audio archive do the first clips of each split sit?

Research script for issue #2 (findings: docs/research/archive-split-depth.md).

For every (internal dataset, split) listed in the run configs complete-1..5, this
reads the dataset's single `.tar.gz` once, front to back, the same way
`uad_data.loader.iter_samples` does (`tarfile` mode "r|gz", keep a regular file
only if `member.name` exactly equals a metadata `audio_path`), and records:

* the archive size on disk (compressed bytes) and its sha256,
* for the 1st, 5th and 20th clip of each split in archive order: tar members
  read so far and compressed bytes consumed so far (after reading the clip's
  data, as the loader does),
* where each split starts and ends in the archive,
* whether the split's clips come out in metadata-JSON order,
* metadata records with no exactly matching archive entry (with near-misses),
* archive files that no metadata JSON in the dataset folder refers to.

Compressed byte counts come from a wrapper that counts bytes handed to tarfile.
tarfile reads its file object in `bufsize` chunks (default tarfile.RECORDSIZE,
10 240 bytes), so each count can overshoot the true position by about that much.

Standard library only. Needs a local copy of the Hub dataset repo
(`AudioInstruct/Universal-Audio-Understanding`); it never touches the network.

Usage:
    python docs/research/archive-split-depth.py \
        --data-root  <local Hub copy> \
        --out        results.json \
        [--workers N] [--only NAME ...]
    python docs/research/archive-split-depth.py --table results.json
"""
import argparse
import ast
import collections
import glob
import hashlib
import json
import os
import posixpath
import sys
import tarfile
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
REGISTRY = os.path.join(REPO_ROOT, "uad_data", "internal_datasets.py")
CONFIG_NAMES = [f"complete-{i}.json" for i in range(1, 6)]
MILESTONES = (1, 5, 20)
DEFAULT_TAR = "data/{name}/{name}.tar.gz"  # uad_data/internal_dataset.py TAR_GZ_FILEPATH
EXAMPLES = 8


# --------------------------------------------------------------------------- inputs

def registry_data_urls(path=REGISTRY):
    """name -> data_url, read from uad_data/internal_datasets.py without importing it."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    urls = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "InternalDataset":
            kw = {k.arg: k.value for k in node.keywords}
            name = kw["name"].value
            url = kw["data_url"].value if "data_url" in kw else DEFAULT_TAR.format(name=name)
            urls[name] = url
    return urls


def configured_splits(config_dir):
    """name -> {"splits": [...], "configs": [...]} over complete-1..5, in config order."""
    out = collections.OrderedDict()
    for cfg in CONFIG_NAMES:
        with open(os.path.join(config_dir, cfg), encoding="utf-8") as f:
            data = json.load(f)
        for d in data["datasets"]:
            entry = out.setdefault(d["name"], {"splits": [], "configs": []})
            for s in d.get("splits", []):
                if s not in entry["splits"]:
                    entry["splits"].append(s)
            entry["configs"].append(cfg)
    return out


# --------------------------------------------------------------------------- reading

class CountingReader:
    """File wrapper that counts (and hashes) every compressed byte handed to tarfile."""

    def __init__(self, raw):
        self.raw = raw
        self.count = 0
        self.sha = hashlib.sha256()

    def read(self, size=-1):
        buf = self.raw.read(size)
        self.count += len(buf)
        self.sha.update(buf)
        return buf

    def drain(self):
        """Read whatever tarfile left unread, so the sha256 covers the whole file."""
        while True:
            buf = self.read(1 << 20)
            if not buf:
                return

    def close(self):
        self.raw.close()


def _dotslash(p):
    while p.startswith("./"):
        p = p[2:]
    return p


NEAR_MISS_KEYS = [
    ("leading './' or '/' differs", lambda p: _dotslash(p).lstrip("/")),
    ("backslash vs slash", lambda p: p.replace("\\", "/")),
    ("surrounding whitespace", lambda p: p.strip()),
    ("unicode normalisation", lambda p: unicodedata.normalize("NFC", p)),
    ("letter case", lambda p: _dotslash(p).lower()),
    ("same path, different extension", lambda p: posixpath.splitext(_dotslash(p).lower())[0]),
    ("same file name, different directory", lambda p: posixpath.basename(p).lower()),
]


def classify_misses(unmatched, file_names, nonfile_names, owner):
    """For each unmatched metadata path, name the closest kind of near-miss (if any).

    `owner` maps an archive path to the metadata JSON files that list it, so a
    near-miss that is really another split's clip is visible as such.
    """
    indexes = []
    for label, fn in NEAR_MISS_KEYS:
        idx = collections.defaultdict(list)
        for n in file_names:
            idx[fn(n)].append(n)
        indexes.append((label, fn, idx))
    kinds = collections.Counter()
    examples = collections.defaultdict(list)
    for p in unmatched:
        if p in nonfile_names:
            kind, hit = "exists but is not a regular file", p
        else:
            kind, hit = "no near-miss", None
            for label, fn, idx in indexes:
                cands = idx.get(fn(p))
                if cands:
                    kind, hit = label, cands[0]
                    break
        kinds[kind] += 1
        if hit is not None and owner.get(hit):
            kinds[f"(of which the archive file is listed by {', '.join(sorted(owner[hit]))})"] += 1
        if len(examples[kind]) < EXAMPLES:
            examples[kind].append({"metadata": p, "archive": hit,
                                   "archive_listed_by": sorted(owner.get(hit, ()))})
    return dict(kinds), dict(examples)


def download_record(data_root, rel, sha256):
    """What `huggingface_hub`'s local-dir download recorded for this file, if anything.

    `<local copy>/.cache/huggingface/download/<path>.metadata` holds three lines:
    the commit the file was fetched at, its etag (the LFS sha256 for LFS files),
    and a timestamp. A matching sha256 shows the local file is intact for that
    commit; it says nothing about later commits on the Hub.
    """
    p = os.path.join(data_root, ".cache", "huggingface", "download", *rel.split("/")) + ".metadata"
    if not os.path.exists(p):
        return None
    lines = open(p, encoding="utf-8").read().split()
    rec = {"commit": lines[0], "etag": lines[1]} if len(lines) >= 2 else {"raw": lines}
    if len(lines) >= 3:
        rec["fetched_at_unix"] = float(lines[2])
    rec["sha256_matches_etag"] = rec.get("etag") == sha256
    return rec


def spearman(xs):
    """Rank correlation between position and value for a permutation-like list."""
    n = len(xs)
    if n < 2:
        return None
    order = sorted(range(n), key=lambda i: xs[i])
    rank = [0] * n
    for r, i in enumerate(order):
        rank[i] = r
    d2 = sum((i - rank[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def analyse(job):
    name, data_root, archive_rel, splits = job
    folder = os.path.join(data_root, "data", name)
    archive_path = os.path.join(data_root, *archive_rel.split("/"))
    result = {"dataset": name, "archive": archive_rel, "configured_splits": splits}

    # Every metadata JSON in the folder, configured or not.
    json_files = sorted(glob.glob(os.path.join(folder, "*.json")))
    metas = collections.OrderedDict()
    for f in json_files:
        with open(f, encoding="utf-8") as fh:
            metas[os.path.basename(f)] = [r["audio_path"] for r in json.load(fh)]
    result["json_files"] = {k: len(v) for k, v in metas.items()}
    result["other_files"] = sorted(
        x for x in os.listdir(folder)
        if not x.endswith(".json") and x != posixpath.basename(archive_rel))

    # Identical JSON contents (e.g. a _dev.json next to a _validation.json).
    by_content = collections.defaultdict(list)
    for k, v in metas.items():
        by_content[hashlib.sha256("\n".join(v).encode()).hexdigest()].append(k)
    result["same_audio_paths_as"] = [g for g in by_content.values() if len(g) > 1]

    state = collections.OrderedDict()
    for s in splits:
        fname = f"{name}_{s}.json"
        if fname not in metas:
            state[s] = None
            continue
        paths = metas[fname]
        index = {}
        for i, p in enumerate(paths):
            index.setdefault(p, i)
        state[s] = {
            "file": fname, "records": len(paths), "unique": len(index),
            "index": index, "matched": [], "matched_hits": 0,
            "milestones": {}, "first": None, "last": None,
        }

    size = os.path.getsize(archive_path)
    result["archive_bytes"] = size
    raw = open(archive_path, "rb")
    reader = CountingReader(raw)
    entries = files = 0
    types = collections.Counter()
    file_names = []
    nonfile_names = set()
    t0 = time.perf_counter()
    error = None
    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as tar:
            for m in tar:
                entries += 1
                if not m.isfile():
                    types["dir" if m.isdir() else "other"] += 1
                    nonfile_names.add(m.name)
                    continue
                files += 1
                types["file"] += 1
                file_names.append(m.name)
                hits = [st for st in state.values() if st and m.name in st["index"]]
                if not hits:
                    continue
                header_at = reader.count
                if any(len(st["matched"]) < MILESTONES[-1] for st in hits):
                    tar.extractfile(m).read()  # what the loader does with a match
                for st in hits:
                    st["matched_hits"] += 1
                    seen = st.setdefault("_seen", set())
                    if m.name in seen:
                        continue
                    seen.add(m.name)
                    st["matched"].append(m.name)
                    k = len(st["matched"])
                    pos = {"entries": entries, "compressed": header_at}
                    if st["first"] is None:
                        st["first"] = pos
                    st["last"] = pos
                    if k in MILESTONES:
                        st["milestones"][k] = {
                            "entries": entries,
                            "compressed": reader.count,
                            "uncompressed_end": m.offset_data + m.size,
                            "audio_path": m.name,
                            "metadata_index": st["index"][m.name],
                        }
            result["tar_uncompressed_bytes"] = tar.offset
    except Exception as exc:  # record, don't die: the table should say what broke
        error = f"{type(exc).__name__}: {exc}"
    consumed_by_tar = reader.count
    reader.drain()
    reader.close()
    result["seconds"] = round(time.perf_counter() - t0, 1)
    result["error"] = error
    result["sha256"] = reader.sha.hexdigest()
    result["local_download_record"] = download_record(data_root, archive_rel, result["sha256"])
    result["bytes_read_by_tar"] = consumed_by_tar
    result["entries"] = entries
    result["member_types"] = dict(types)

    name_counts = collections.Counter(file_names)
    dups = [n for n, c in name_counts.items() if c > 1]
    result["duplicate_file_names"] = {"count": len(dups), "examples": dups[:EXAMPLES]}
    result["file_names_sha256"] = hashlib.sha256("\n".join(file_names).encode()).hexdigest()
    result["archive_is_sorted"] = file_names == sorted(file_names)
    top = collections.Counter(n.split("/", 1)[0] if "/" in n else "(root)" for n in file_names)
    result["top_level_dirs"] = dict(top.most_common(20))

    # Archive files that no JSON (any split, configured or not) refers to.
    all_meta = set().union(*[set(v) for v in metas.values()]) if metas else set()
    configured_meta = set().union(*[set(st["index"]) for st in state.values() if st]) \
        if any(state.values()) else set()
    names = set(file_names)
    orphans = sorted(names - all_meta)
    unconfigured = sorted((names & all_meta) - configured_meta)
    result["archive_files_in_no_json"] = {
        "count": len(orphans), "examples": orphans[:EXAMPLES],
        "by_top_dir": dict(collections.Counter(
            n.split("/", 1)[0] for n in orphans).most_common(10)),
    }
    owner = collections.defaultdict(set)
    for k, v in metas.items():
        for p in v:
            owner[p].add(k)
    who = collections.Counter()
    for n in unconfigured:
        for k in owner[n]:
            who[k] += 1
    result["archive_files_only_in_unconfigured_json"] = {
        "count": len(unconfigured), "by_json": dict(who)}

    # Pairwise overlap between configured splits.
    overlap = {}
    live = [(s, st) for s, st in state.items() if st]
    for i, (a, sa) in enumerate(live):
        for b, sb in live[i + 1:]:
            both = set(sa["index"]) & set(sb["index"])
            if both:
                overlap[f"{a}&{b}"] = {"count": len(both), "examples": sorted(both)[:3]}
    result["split_overlap"] = overlap

    out = collections.OrderedDict()
    for s, st in state.items():
        if st is None:
            out[s] = {"error": f"missing {name}_{s}.json"}
            continue
        meta_order = list(st["index"])  # unique paths, first-occurrence order
        matched = st["matched"]
        matched_set = set(matched)
        unmatched = [p for p in meta_order if p not in names]
        meta_matched = [p for p in meta_order if p in matched_set]
        positions = [st["index"][p] for p in matched]
        kinds, examples = classify_misses(unmatched, file_names, nonfile_names, owner)
        out[s] = {
            "file": st["file"],
            "records": st["records"],
            "unique_paths": st["unique"],
            "duplicate_records": st["records"] - st["unique"],
            "matched_clips": len(matched),
            "matched_entries": st["matched_hits"],
            "unmatched": len(unmatched),
            "unmatched_kinds": kinds,
            "unmatched_examples": examples,
            "milestones": {str(k): v for k, v in sorted(st["milestones"].items())},
            "first": st["first"],
            "last": st["last"],
            "order": {
                "first20_equal_raw": meta_order[:20] == matched[:20],
                "first20_equal_matched_only": meta_matched[:20] == matched[:20],
                "full_equal_matched_only": meta_matched == matched,
                "spearman": None if spearman(positions) is None else round(spearman(positions), 4),
                "metadata_sorted": meta_order == sorted(meta_order),
                "first_clip_metadata_index": positions[0] if positions else None,
                "first20_metadata_indexes": positions[:20],
            },
        }
    result["splits"] = out
    return result


# --------------------------------------------------------------------------- report

def mib(n):
    return f"{n / 2**20:,.1f}" if n is not None else "-"


def pct(n, size):
    return f"{100 * n / size:.1f}%" if n is not None else "-"


def table(results):
    lines = [
        "| Dataset | Split | Archive MiB | Clips (matched / records) | 1st clip: entries, MiB (%) "
        "| 5th clip: entries, MiB | 20th clip: entries, MiB (%) | Split spans (% of archive) "
        "| Metadata order kept? | Unmatched paths | Duplicate records |",
        "|---|---|---:|---:|---|---|---|---|---|---:|---:|",
    ]
    for r in results:
        size = r["archive_bytes"]
        for s, d in r["splits"].items():
            if "error" in d:
                lines.append(f"| {r['dataset']} | {s} | {mib(size)} | {d['error']} | | | | | | | |")
                continue
            ms = d["milestones"]

            def cell(k, with_pct):
                m = ms.get(k)
                if not m:
                    return "not reached"
                txt = f"{m['entries']:,}, {mib(m['compressed'])}"
                return txt + (f" ({pct(m['compressed'], size)})" if with_pct else "")

            o = d["order"]
            if not d["matched_clips"]:
                order = "n/a (no clips)"
            elif o["full_equal_matched_only"]:
                order = "yes, all"
            elif o["first20_equal_matched_only"]:
                order = f"first 20 yes; rho {o['spearman']}"
            else:
                order = f"no; rho {o['spearman']}"
            span = "-"
            if d["first"] and d["last"]:
                span = f"{pct(d['first']['compressed'], size)} to {pct(d['last']['compressed'], size)}"
            lines.append(
                f"| {r['dataset']} | {s} | {mib(size)} | {d['matched_clips']:,} / {d['records']:,} "
                f"| {cell('1', True)} | {cell('5', False)} | {cell('20', True)} | {span} "
                f"| {order} | {d['unmatched']:,} | {d['duplicate_records']:,} |")
    return "\n".join(lines)


def archive_table(results):
    lines = [
        "| Dataset | Archive bytes | Tar bytes (uncompressed) | Tar members (files / dirs) "
        "| Files no JSON lists | Pass seconds | sha256 = Hub etag? | Read error |",
        "|---|---:|---:|---|---:|---:|---|---|",
    ]
    for r in results:
        t = r["member_types"]
        rec = r.get("local_download_record") or {}
        ok = {True: "yes", False: "NO"}.get(rec.get("sha256_matches_etag"), "no record")
        tar_bytes = r.get("tar_uncompressed_bytes")
        lines.append(
            f"| {r['dataset']} | {r['archive_bytes']:,} | "
            f"{f'{tar_bytes:,}' if tar_bytes else '-'} | "
            f"{t.get('file', 0):,} / {t.get('dir', 0):,} | "
            f"{r['archive_files_in_no_json']['count']:,} | {r['seconds']} | {ok} | "
            f"{r['error'] or ''} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", help="local copy of the Hub dataset repo")
    ap.add_argument("--config-dir", help="defaults to <data-root>/universal_audio_dataset_configs")
    ap.add_argument("--out", help="write results JSON here")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--only", nargs="*", help="restrict to these dataset names")
    ap.add_argument("--table", help="print the markdown table for an existing results JSON")
    args = ap.parse_args()

    if args.table:
        with open(args.table, encoding="utf-8") as f:
            archives = json.load(f)["archives"]
        print(table(archives))
        print()
        print(archive_table(archives))
        return

    config_dir = args.config_dir or os.path.join(args.data_root, "universal_audio_dataset_configs")
    urls = registry_data_urls()
    wanted = configured_splits(config_dir)
    jobs = [(n, args.data_root, urls[n], w["splits"]) for n, w in wanted.items()
            if not args.only or n in args.only]
    results = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(analyse, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            results[r["dataset"]] = r
            print(f"{r['dataset']}: {r['seconds']}s, {r['archive_bytes']:,} bytes, "
                  f"error={r['error']}", flush=True)
    ordered = [results[j[0]] for j in jobs]
    payload = {
        "data_root": args.data_root,
        "configs": {n: w for n, w in wanted.items()},
        "workers": args.workers,
        "total_seconds": round(time.perf_counter() - t0, 1),
        "python": sys.version,
        "archives": ordered,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
    print(table(ordered))
    print()
    print(archive_table(ordered))


if __name__ == "__main__":
    main()
