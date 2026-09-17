# How do Trainer checkpoints sync to GCS and resume on a fresh VM?

Research for [#45](https://github.com/derkmed/universal_audio_instruct/issues/45), part of [#42 Finetune Gemma and Qwen on GCP](https://github.com/derkmed/universal_audio_instruct/issues/42). Written 2026-09-17.

Sources:
- **Transformers:** `huggingface/transformers` `main` at `da9d2c4` (version `5.18.0.dev0`, 2026-09-17). Line numbers below refer to that commit.
- **PEFT:** `huggingface/peft` `main` from the same day.
- **Google Cloud:** docs.cloud.google.com pages as served on 2026-09-17.

Placeholders: `<bucket>`, `<project>` and `<secret-id>`. No real ids appear in this note.

## Answer

- **Recommendation: a synchronous upload in `on_save`.**
  - Keep `output_dir` on the VM's local disk.
  - Add a `TrainerCallback` whose `on_save` uploads the new `checkpoint-<step>/` to `gs://<bucket>/finetunes/<model>/<run-id>/checkpoints/checkpoint-<step>/` with the `google-cloud-storage` client, then writes an empty `_COMPLETE` marker object **last**. After that it prunes remote checkpoints down to the latest 2 complete ones. It deletes a checkpoint's marker before the checkpoint's other objects.
  - On startup, the script finds the highest-step remote checkpoint that has a marker, downloads it into a fresh local `output_dir`, and calls `trainer.train(resume_from_checkpoint=<that local path>)`. It passes an explicit path, not `True`.
  - Preemption handling is not relied on. A preemption loses at most `save_steps` optimizer steps plus the upload in flight.
- **Do not use a Cloud Storage FUSE mount as `output_dir`.** Trainer writes checkpoint files in place, with no temp-dir-and-rename step. A preemption during a save leaves a `checkpoint-N/` directory without `optimizer.pt` or `scheduler.pt`. On resume, Trainer picks that directory (it has the highest step) and **silently** starts with a fresh optimizer and scheduler. Details below.
- **A background sync loop** (such as `gcloud storage rsync` on a timer) has the same partial-checkpoint problem. It also races with Trainer's local checkpoint rotation, and it has no natural point at which to write a completeness marker. Reject it.
- **Spot notice window:**
  - By default there is **0 s** of dedicated notice. The shutdown period is then "best effort and up to 30 seconds".
  - A 120 s notice is available in **Preview** via `gcloud beta compute instances create ... --preemption-notice-duration=120s`.
  - Do not build correctness on either.
- **A resumable PEFT checkpoint must contain:**
  - `adapter_config.json`
  - `adapter_model.safetensors`
  - `trainer_state.json`
  - `optimizer.pt`
  - `scheduler.pt`
  - `rng_state.pth`

  Only the adapter and `trainer_state.json` cause a hard error when missing. A missing optimizer, scheduler or RNG file only degrades the resume silently, so the completeness marker must be checked.
- **`save_total_limit`:** set it locally (2 is enough). Rotation runs inside `_save_checkpoint`, **before** `on_save` fires, and it always keeps the newest checkpoint. A synchronous upload in `on_save` therefore always finds its directory intact. Remote retention ("latest 2") is the callback's job, not Trainer's.
- **Deep Learning VM image:** use the `pytorch-2-9-cu129-ubuntu-2404-nvidia-580` image family (project `deeplearning-platform-release`). It ships PyTorch 2.9.0, CUDA 12.9, Python 3.12, NVIDIA driver 580 and Ubuntu 24.04, and is supported until August 4, 2028. This satisfies transformers `main`'s `torch>=2.5` and Python 3.10–3.14.
- **Secret Manager:**
  - The VM's service account needs `roles/secretmanager.secretAccessor` on the secret.
  - The VM needs the `cloud-platform` access scope, because the **default** scope set does not cover Secret Manager.
  - Read the secret with `gcloud secrets versions access latest --secret=<secret-id>` or with `SecretManagerServiceClient().access_secret_version(...)`.

## 1. What Trainer does on save and resume

### Save order

`Trainer._maybe_log_save_evaluate` (trainer.py:2231–2233):

```python
if self.control.should_save:
    self._save_checkpoint(model, trial)
    self.control = self.callback_handler.on_save(self.args, self.state, self.control)
```

`Trainer._save_checkpoint` (trainer.py:3229–3296) runs these steps in order:

1. `self.save_model(output_dir, _internal_call=True)` writes `output_dir = <run_dir>/checkpoint-<global_step>`. For a `PeftModel`, `_save` calls `save_pretrained` (trainer.py:4015–4032), which writes the adapter files. It also writes `training_args.bin` (trainer.py:4049). The processor is saved only if `processing_class` was passed to Trainer (trainer.py:4039). `train/main.py` does not pass it.
2. Unless `save_only_model` is set, it writes `optimizer.pt` and `scheduler.pt` (`_save_optimizer_and_scheduler`, trainer.py:3387), then `scaler.pt` if a grad scaler is in use, then `rng_state.pth` (`_save_rng_state`, trainer.py:3336).
3. It writes `trainer_state.json` (trainer.py:3283).
4. It calls `rotate_checkpoints(output_dir=run_dir, save_total_limit=..., best_model_checkpoint=..., use_mtime=True)` (trainer.py:3291).

It does **not** write to a temporary directory and rename it. Files appear in the final `checkpoint-N/` directory one at a time.

`save_only_model` "prevents resuming training from the checkpoint" (training_args.py:465–469), so it must stay `False`.

### PEFT file names

The PEFT file names come from `peft/src/peft/utils/constants.py:399–401`:

- `adapter_model.safetensors` (or the legacy `adapter_model.bin`)
- `adapter_config.json`

### Resume

`Trainer.train` (trainer.py:1526–1538):

- **`resume_from_checkpoint=True`:** it resolves the path with `get_last_checkpoint(args.output_dir)`, and raises `ValueError` if none is found.
- **Loading:** it then calls `_load_from_checkpoint(path)` and `TrainerState.load_from_json(<path>/trainer_state.json)`. A missing `trainer_state.json` therefore fails loudly.

`get_last_checkpoint` (trainer_utils.py:270–279) picks the `checkpoint-<n>` **directory** with the largest `n`. It does not check what the directory contains.

`_load_from_checkpoint` (trainer.py:3476 onward):

- **Missing files:** it raises `ValueError("Can't find a valid checkpoint at ...")` if none of the known weight or adapter files exist.
- **PEFT models:** it calls `model.load_adapter(resume_from_checkpoint, active_adapter, is_trainable=True)`.

`_load_optimizer_and_scheduler` (trainer.py:3756–3793) loads state **only if** `optimizer.pt` and `scheduler.pt` both exist (`if checkpoint_file_exists and os.path.isfile(<scheduler.pt>)`). Otherwise it does nothing and raises nothing. The run continues from step N with a fresh optimizer and a scheduler that restarts from zero, including warmup.

`_load_rng_state` (trainer.py:3711–3733) logs at `info` level and returns if `rng_state.pth` is missing.

### Mid-epoch resume

`_run_epoch` (trainer.py:1807–1814) calls `skip_first_batches(train_dataloader, steps_trained_in_current_epoch)` and restores RNG state after the skip.

A resumed run therefore only reproduces the original data order if the dataset (its rows **and** their order) is identical. The shuffled concatenation and the frozen run config from #42 must be deterministic and seeded, and must not depend on anything that varies between VMs.

### What a resumable single-GPU PEFT checkpoint needs

| File | Written by | If missing on resume |
|---|---|---|
| `adapter_config.json`, `adapter_model.safetensors` | `PeftModel.save_pretrained` | Hard error (`ValueError`) |
| `trainer_state.json` | `_save_checkpoint` | Hard error (`FileNotFoundError` from `load_from_json`) |
| `optimizer.pt`, `scheduler.pt` | `_save_optimizer_and_scheduler` | **Silent**: fresh optimizer, scheduler restarts |
| `rng_state.pth` | `_save_rng_state` | **Silent** (info log): RNG not restored |
| `scaler.pt` | `_save_scaler` | Only relevant with a fp16 grad scaler. `train/main.py` uses `bf16=True` |
| `training_args.bin` | `_save` | Not read on resume |

The base model is not in the checkpoint. On resume, `train/main.py` rebuilds it from the Hub, with the same 4-bit settings, before `load_adapter`.

### Newer `enable_jit_checkpoint` flag

`TrainingArguments.enable_jit_checkpoint` (training_args.py:488–493, `trainer_jit_checkpoint.py`) installs a SIGTERM handler:

- **When it saves:** after `kill_wait=3` s it sets a flag. At the next `on_step_begin`, `on_pre_optimizer_step`, `on_step_end` or `on_epoch_end` it calls `trainer._save_checkpoint(...)` directly and stops training.
- **Sentinel file:** it writes `checkpoint-is-incomplete.txt` into the directory while saving and removes it afterwards. **Nothing in Trainer reads that sentinel** on resume: a grep for `is-incomplete` finds only this file.
- **No `on_save`:** it calls `_save_checkpoint`, not the `on_save` path, so an upload callback's `on_save` would **not** fire for a JIT checkpoint.
- **Grace period:** the docs say the grace period must be "≥ longest iteration time + checkpoint save time".

Inference, not checked in GCP docs: GCP does not document whether the training process receives SIGTERM from the guest OS during the ACPI soft-off. JIT checkpointing would need a 120 s notice, a signal source, and an upload step of its own. It is not worth it for v1.

## 2. Spot preemption facts

From [Spot VMs](https://docs.cloud.google.com/compute/docs/instances/spot) (§ Preemption process):

- **Notice duration:**
  - Default: "0 seconds". There is "no dedicated delay between detecting preemption in metadata and the ACPI G2 Soft Off signal."
  - Optional: 120 seconds, in Preview, recommended "for any workloads that need a dedicated duration or longer than 30 seconds to handle preemption."
- **Shutdown period:** "The shutdown period for Spot VMs is best effort and up to 30 seconds". After it, Compute Engine sends ACPI G3 Mechanical Off.
- **Shutdown scripts:** the ACPI G2 Soft Off "triggers any shutdown script that you have configured".
- **Termination action:** with STOP (the default for Spot), "you can access and recover data from any persistent disks". With DELETE, the VM is deleted.
  - The fresh-VM design in #42 should assume the local disk is gone, so GCS is the only source of truth.
- **Runtime:** "Spot VMs don't have a minimum or maximum runtime unless you specifically limit the runtime."

From [Create and use Spot VMs](https://docs.cloud.google.com/compute/docs/instances/create-use-spot):

- **Enable the 120 s notice:** `gcloud beta compute instances create VM_NAME --provisioning-model=SPOT --preemption-notice-duration=120s --instance-termination-action=...`. The REST field is `scheduling.preemptionNoticeDuration: {seconds: 120}`.
- **Detect preemption:** `curl "http://metadata.google.internal/computeMetadata/v1/instance/preempted?wait_for_change=true" -H "Metadata-Flavor: Google"` blocks until preemption and then returns `TRUE`.

From [Shutdown scripts](https://docs.cloud.google.com/compute/docs/shutdownscript): "Compute Engine executes shutdown scripts only on a best-effort basis." Scripts still running when the instance stops are "forcefully" stopped.

## 3. GCS and gcsfuse facts that decide the comparison

- **Strong consistency** ([Consistency](https://docs.cloud.google.com/storage/docs/consistency)):
  - An object "is immediately available for reading ... as soon as you receive a success response".
  - Listing is strongly consistent.
  - A deleted object returns 404 immediately.

  So "list the checkpoints, then check for the marker" is reliable.
- **Uploads are all-or-nothing per object** ([Resumable uploads](https://docs.cloud.google.com/storage/docs/resumable-uploads)): "Only a completed resumable upload appears in your bucket". A *directory* of objects is never atomic, though. That is why a marker object is needed.
- **Parallel composite uploads.** [Parallel composite uploads](https://docs.cloud.google.com/storage/docs/parallel-composite-uploads) says `gcloud storage` performs them by default (property `storage/parallel_composite_upload_enabled` is `None`) for objects that meet the size criteria. After an aborted upload, "you should manually delete any temporary objects". So a preempted `gcloud storage rsync` can also leave temporary component objects behind. The Python client's `Blob.upload_from_filename` does not do this.
- **`gcloud storage rsync`** ([reference](https://docs.cloud.google.com/sdk/gcloud/reference/storage/rsync)):
  - It needs `--recursive`.
  - It compares mtimes by default. `--checksums-only` compares hashes instead.
  - `--delete-unmatched-destination-objects` "can delete data quickly if you specify the wrong source and destination combination".
  - `--exclude` takes a Python regex.
  - The page says nothing about atomicity.
- **Python client:** `google.cloud.storage.transfer_manager.upload_many_from_filenames(bucket, filenames, source_directory=..., blob_name_prefix=..., max_workers=8, worker_type="process", raise_exception=False, ...)` ([reference](https://docs.cloud.google.com/python/docs/reference/storage/latest/google.cloud.storage.transfer_manager)). Pass `raise_exception=True`, or check the results, before writing the marker.
- **Cloud Storage FUSE** ([overview](https://docs.cloud.google.com/storage/docs/cloud-storage-fuse/overview), [semantics](https://github.com/googlecloudplatform/gcsfuse/blob/master/docs/semantics.md)):
  - **Not POSIX:** "Cloud Storage FUSE is not POSIX compliant".
  - **Writes:** streaming writes are the default from v3.0. "The object will be finalized only when the file is closed." An interrupted file never appears, but the files already closed in the same `checkpoint-N/` directory do.
  - **Renaming directories:** "not supported" by default in flat-namespace buckets. It is atomic only in buckets with hierarchical namespace, so the usual "write to tmp, then rename" fix does not work on a normal bucket.
  - **Deleting directories:** "The complete unlink directory operation is not atomic". This matters because Trainer's rotation calls `shutil.rmtree`.
  - **mtime:** Trainer's rotation sorts by mtime and warns that "mtime is not reliable on some filesystems (e.g., cloud fuse filesystems)". It falls back to step order only when the mtimes of all checkpoints span less than 1 s (trainer_utils.py:313–320).

## 4. Comparison under Spot preemption

| | A. `on_save` callback uploads, then writes a marker (recommended) | B. gcsfuse mount as `output_dir` | C. Background sync loop |
|---|---|---|---|
| Where Trainer writes | Local disk (fast) | Straight to GCS through FUSE. Each file is streamed, and the object only appears on close | Local disk |
| Preempted mid-save | Local only. The remote copy of this step never gets a marker, so the previous marked checkpoint stays authoritative | `checkpoint-N/` exists remotely with only some files. `get_last_checkpoint` picks it. With no optimizer or scheduler, the resume is **silently wrong** | The loop may already have copied some files of `checkpoint-N/`. Same problem as B unless the loop checks for completeness |
| Preempted mid-upload | Some objects missing and no marker, so the checkpoint is ignored. The next run can delete or overwrite it | n/a | Some objects missing. rsync may also leave temporary composite objects |
| How completeness is marked | A `_COMPLETE` object written after every upload succeeds. It is exact, because the callback runs after `_save_checkpoint` returns | No hook. You would need `trainer_state.json` as a heuristic (it is written last, but `scaler`/`rng` order could change) or a custom marker from a callback anyway | No reliable point: the loop cannot tell when Trainer has finished a directory |
| Interaction with `save_total_limit` | Rotation runs before `on_save` and keeps the newest checkpoint, so the upload never races with deletion. Remote pruning is explicit | Rotation runs `rmtree` over FUSE, which is non-atomic. mtime-based ordering is unreliable on FUSE | Races: rotation can delete a directory mid-sync. `--delete-unmatched-destination-objects` then mirrors the deletions |
| Training throughput | The upload blocks the step. Adapter plus 8-bit optimizer state for a r=16 attention-only LoRA is small, so the upload should take seconds (inference, to be checked in the GPU check) | Every checkpoint write goes over the network. Also slower for the model and processor files | No blocking |
| Needs the 120 s notice? | No | No, and it would not help | No |
| Extra software | `google-cloud-storage` (pip) | gcsfuse on the image, a mount at boot, and a storage scope with write access | `gcloud` plus a supervisor loop |

A notice-driven upload (a shutdown script or a `preempted?wait_for_change=true` watcher uploading the latest local checkpoint) can be layered on top of A. With the default notice it gets at most about 30 best-effort seconds, and it only helps if the latest local checkpoint is newer than the latest marked remote one. That can happen only when an upload failed. So it adds little and is left out of v1.

## 5. Recommended design (for the spec)

1. **Settings.** `TrainingArguments(output_dir=<local disk>/checkpoints, save_strategy="steps", save_steps=<N>, save_total_limit=2, save_only_model=False)`. Choose `N` so that a lost `N` steps is an acceptable cost for a preemption.
2. **`GcsCheckpointCallback.on_save(args, state, control)`.** Runs only when `state.is_world_process_zero`.
   - Its source directory is `os.path.join(args.output_dir, f"checkpoint-{state.global_step}")`.
   - It uploads every file under that directory to `<prefix>/checkpoints/checkpoint-<step>/` (with `transfer_manager` or a plain `Blob.upload_from_filename` loop).
   - If any upload failed, it raises.
   - It then uploads `<prefix>/checkpoints/checkpoint-<step>/_COMPLETE`.
   - Finally it lists the marked checkpoints and, for each one older than the newest 2, deletes `_COMPLETE` first and then the rest.
3. **Resume on startup.**
   - Find the resume point: list `<prefix>/checkpoints/`, keep the steps that have `_COMPLETE`, and take the max.
   - If one exists, download it into `<local>/checkpoints/checkpoint-<step>/` and call `trainer.train(resume_from_checkpoint=<that path>)`.
   - Otherwise call `trainer.train()`.
   - Pass the explicit path rather than `True`, so that a stale local directory can never be picked.
4. **Picking the run.** The script must resume into the **same `<run-id>` prefix**. "Latest run for this model" needs a rule, such as the newest `<run-id>` without `final/adapter/`. Otherwise a new timestamp run-id is minted on each fresh VM and resume never triggers. See the map notes below.
5. **Recommended checks at startup.** Before calling `train()`, check that the downloaded checkpoint has `optimizer.pt`, `scheduler.pt`, `rng_state.pth`, `trainer_state.json` and the adapter files. The marker should guarantee this, but the check is cheap.

## 6. Deep Learning VM image

From [Choose an image](https://docs.cloud.google.com/deep-learning-vm/docs/images) (page last updated 2026-09-15):

- **Current PyTorch families:**
  - `pytorch-2-9-cu129-ubuntu-2404-nvidia-580`: PyTorch 2.9.0, CUDA 12.9, Python 3.12, Ubuntu 24.04. Support ends August 4, 2028.
  - `pytorch-2-9-cu129-ubuntu-2204-nvidia-580`: the same, on Ubuntu 22.04.
  - Both are in image project `deeplearning-platform-release` and ship NVIDIA driver 580.
- **Older family:** `pytorch-2-7-cu128-ubuntu-2204-nvidia-570` (PyTorch 2.7.1, Python 3.10). Its patch support ended April 13, 2026, and it is available until April 13, 2027. Avoid it.
- **Other notes from the page:**
  - "Debian images have all been deprecated."
  - "All active images support A3 Ultra GPU accelerators."
  - To pin an exact image, use a dated name such as `pytorch-2-9-cu129-ubuntu-2204-nvidia-580-v20260416`.
- **Fit with transformers `main`** (`setup.py`):
  - `torch>=2.5` → PyTorch 2.9 OK.
  - `SUPPORTED_PYTHON_VERSIONS = (10, 14)` → Python 3.12 OK.
  - `peft>=0.20.0` and `accelerate>=1.1.0` (dependency table).
- **Recommendation:** use `pytorch-2-9-cu129-ubuntu-2404-nvidia-580`. #42 installs a pinned venv, which pulls its own `torch` wheel. So what matters most from the image is driver 580 and CUDA 12.9. Pin `torch` in `requirements.txt` to a build for CUDA 12.x that driver 580 supports.

## 7. Reading a Secret Manager secret from the VM

- **IAM** ([Access a secret version](https://docs.cloud.google.com/secret-manager/docs/access-secret-version)): grant the VM's service account "the Secret Manager Secret Accessor (`roles/secretmanager.secretAccessor`) IAM role on a secret". Grant it on the one secret, not the whole project.
- **Scopes** ([Service accounts](https://docs.cloud.google.com/compute/docs/access/service-accounts)): "both access scopes and IAM roles must allow access". "The best practice is to set the full `cloud-platform` access scope on the instance, then control the service account's access using IAM roles."
  - [`gcloud compute instances create`](https://docs.cloud.google.com/sdk/gcloud/reference/compute/instances/create): with no `--scopes`, the `default` alias applies. It covers `devstorage.read_only`, `logging.write`, `monitoring.write`, `pubsub`, `service.management.readonly`, `servicecontrol` and `trace.append`.
  - That set covers **neither Secret Manager nor GCS writes**, so the runbook must pass `--service-account=<sa-email> --scopes=cloud-platform`.
- **GCS writes:** the same service account needs an object-write role on the bucket, such as `roles/storage.objectUser`, which also covers delete for pruning. This comes from general IAM knowledge and was not re-read for this note; confirm it against the Cloud Storage IAM roles page when writing the runbook.
- **Read it:**
  - Shell: `gcloud secrets versions access latest --secret=<secret-id>`.
  - Python (`google-cloud-secret-manager`):

    ```python
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()  # uses the VM's metadata-server credentials
    name = f"projects/<project>/secrets/<secret-id>/versions/latest"
    hf_token = client.access_secret_version(request={"name": name}).payload.data.decode("UTF-8")
    ```

  - Fall back to the `HF_TOKEN` environment variable on any exception, as #42 already decided.
  - Get `<project>` from the metadata server or from config, never from a checked-in constant.

## Notes for the map (#42)

- **transformers floor is wrong.** `requirements.txt` pins `transformers>=5.5.0`, but `train/main.py` relies on 5.15 behaviour: `warmup_ratio` was removed, and a float `warmup_steps` is read as a ratio. The floor should be `>=5.15` (or an exact pin).
- **New dependencies:** `google-cloud-storage` and `google-cloud-secret-manager`. PEFT needs `>=0.20.0` to match transformers `main`.
- **Run-id rule for resume.** A timestamp run-id is minted per launch, so resume needs a way to find the in-progress run: a `--run-id` flag, or "the newest run without `final/`". This is an open decision.
- **Deterministic data order is a resume requirement.** `skip_first_batches` assumes the same dataset order after restart.
- **Runbook flags:** `--provisioning-model=SPOT`, `--scopes=cloud-platform` and `--service-account`. `--instance-termination-action` has no effect on correctness under this design.
