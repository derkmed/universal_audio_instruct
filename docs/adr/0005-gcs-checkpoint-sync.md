# ADR-0005: Finetune checkpoints sync to GCS via an on_save callback, fail-loud

- **Status:** accepted (design; not yet built — feeds the #42 spec)
- **Date:** 2026-09-21

## Context

The #42 finetune scripts run on a disposable GCP GPU VM (Spot allowed), whose
local disk must be assumed gone after preemption, so GCS is the only source of
truth for checkpoints. HF `Trainer` writes each `checkpoint-<step>/` in place,
file by file, with no temp-dir-and-rename step, and `get_last_checkpoint` picks
the highest-step directory without checking its contents. A missing
`optimizer.pt`/`scheduler.pt`/`rng_state.pth` does **not** raise on resume — it
silently gives a fresh optimizer and restarts the scheduler. So a partially
synced checkpoint can resume silently wrong. The research for #45
(`docs/research/gcs-checkpoint-resume.md`) compared an `on_save` upload callback,
a gcsfuse mount as `output_dir`, and a background rsync loop.

## Decision

Keep `output_dir` on the VM's local disk (`save_total_limit=2`,
`save_only_model=False`). Add a `TrainerCallback` whose `on_save`, on the world-
process-zero rank only:

1. Uploads every file under the just-written `checkpoint-<step>/` to
   `<run prefix>/checkpoints/checkpoint-<step>/` with the `google-cloud-storage`
   client.
2. Writes an empty `_COMPLETE` **checkpoint marker** object last.
3. Prunes remote checkpoints to the newest 2 marked ones, deleting each victim's
   marker before its other objects.

On startup, the script lists `<run prefix>/checkpoints/`, keeps the steps that
have a marker, downloads the highest into a fresh local dir, and calls
`trainer.train(resume_from_checkpoint=<explicit local path>)` — never `True`, so
a stale local dir can never be picked. If no marked checkpoint exists it calls
`trainer.train()`.

Reject a gcsfuse mount as `output_dir` (non-atomic in-place writes and directory
renames on a flat-namespace bucket; unreliable mtime-based rotation) and a
background rsync loop (same partial-checkpoint problem, no natural point to write
a marker, races with local rotation).

**Failure policy: fail-loud.** Any GCS operation failure in the callback — an
object upload, the marker write, or the prune — is raised, which stops the run.
The client's built-in idempotent retries still apply; there is no additional
app-level retry loop. The operator restarts the VM and resumes from the last good
marker. This keeps "what is in GCS" always equal to what training actually saw,
at the cost of a transient GCS blip ending a run.

## Consequences

- A preemption loses at most `save_steps` optimizer steps plus the upload in
  flight; nothing depends on the (0 s default) Spot notice.
- The upload blocks the training step. For an attention-only r=16 LoRA the adapter
  plus 8-bit optimizer state is small, so this should cost seconds per save — to
  be confirmed on the GPU.
- Fail-loud means a persistent GCS outage wastes no further GPU time, but a
  transient 5xx can end a multi-hour run. Accepted deliberately over silent-wrong
  resumes.
- Resume reproduces the original data order only if the shuffled concatenation of
  train splits is deterministic, seeded, and independent of the VM (guaranteed by
  the #42 frozen run config + seed). This is a hard constraint on the loader.
