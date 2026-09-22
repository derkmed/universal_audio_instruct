# ADR-0006: A finetune run is identified by an operator-supplied `--run-id`

- **Status:** accepted (design; not yet built — feeds the #42 spec)
- **Date:** 2026-09-21

## Context

Each run's checkpoints live under `gs://<bucket>/finetunes/<model>/<run-id>/`,
where `<run-id>` is a UTC timestamp plus a short git SHA (#42). A fresh VM must
resume into the **same** run-id, or a new timestamp is minted on every launch and
resume never triggers (the problem #51 was opened to settle). Two rules were
weighed: a required `--run-id` flag, versus auto-discovering the newest run under
the model prefix that has no `final/`.

## Decision

`--run-id` is the operator's contract.

- On the **first** launch, if `--run-id` is omitted, the script mints one
  (timestamp + short git SHA) and creates the run prefix.
- To **resume**, the operator passes that run-id back with `--run-id`. The script
  resumes strictly within that prefix (per ADR-0005's marker-based discovery of
  the newest complete checkpoint *inside the prefix*).
- There is **no auto-discovery** across run-ids, and **no completion sentinel**
  for resume. `final/` is written for `eval.main` to find the artifact, not to
  tell a launch whether to skip a run.

Concurrent runs for the same `<model>` are allowed (e.g. hyperparameter sweeps),
which is the deciding reason auto-discovery is rejected: "newest run without
`final/`" is ambiguous the moment two runs are in flight.

## Consequences

- The runbook must record the run-id that the first launch prints, and pass it on
  every resume. A resume that forgets the run-id mints a new empty run instead of
  continuing — loud (a new prefix, no checkpoints) rather than silently wrong.
- No abandoned-run hazard: a crashed run is never auto-resumed by an unrelated
  launch; it simply sits until the operator resumes it by id or deletes it.
- `run_config.json` is written once at first launch and is immutable; a resumed
  run leaves it untouched, so a run's recorded git SHA is always its *original*
  launch SHA even if the operator re-clones at a newer commit. Resuming from a
  different SHA is an operator-discipline concern the runbook must call out.
