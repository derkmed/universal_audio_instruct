# ADR-0007: `--merge` is per-model and Gemma-only in v1

- **Status:** accepted (design; not yet built — feeds the #42 spec)
- **Date:** 2026-09-21

## Context

Runs always upload the trained LoRA adapter to `final/adapter/`, and `eval.main`
can load a model from its adapter directly (#42). `--merge` additionally folds the
adapter into the base and uploads full weights to `final/merged/`. Because QLoRA
trains a 4-bit base, a merge must first reload the base in **bf16** — you cannot
merge into a quantized base — on either device. The two v1 models differ sharply
in size: Gemma 4 12B is 23.9 GB in bf16 (fits the 80 GB card with room to merge),
while Qwen3-Omni-30B-A3B is 70.5 GB in bf16 (does not fit on an 80 GB card with
headroom, and only its thinker is trained), so merging it would need a big-RAM CPU
merge (~140 GB+) and a ~70 GB upload plus thinker/talker reassembly.

## Decision

- Each backend **declares its merge device**: GPU when the bf16 base fits the
  80 GB card, CPU otherwise. This keeps the policy uniform as models are added.
- In **v1, only Gemma implements `--merge`** (GPU merge). Its `final/merged/` holds
  the merged full model.
- **Qwen3-Omni ships adapter-only.** `--merge` with Qwen is rejected with a clear
  error; its merge is deferred to its own issue (the 70 GB upload and thinker/
  talker assembly are out of scope here). Eval loads Qwen via the adapter path.
- A requested merge that fails (OOM or otherwise) is **fail-loud** (non-zero
  exit), consistent with ADR-0005 — the operator asked for merged weights and
  should know they were not produced.

## Consequences

- The adapter path is the primary handoff for both models; `final/merged/` is a
  Gemma-only convenience.
- When Qwen merge is added later, its backend declares a CPU merge device and the
  runbook must size the VM's system RAM accordingly; no change to the CLI contract.
- "Merge failure is fatal" means a run that trained fine but OOMs at merge exits
  non-zero even though `final/adapter/` is already safely uploaded. The operator
  can re-merge offline from the uploaded adapter.
