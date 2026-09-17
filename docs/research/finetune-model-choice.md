# Which Gemma and Qwen models are best to finetune on one 80 GB GPU?

Research for [#43](https://github.com/derkmed/universal_audio_instruct/issues/43), part of the map [#42](https://github.com/derkmed/universal_audio_instruct/issues/42). Checked on 2026-09-17 against the Hugging Face Hub, `transformers` v5.17.0 source, `peft` v0.21.0 source and their release notes.

## Answer

- **Gemma: `google/gemma-4-12B-it`** (Gemma 4 12B Unified). It is the largest Gemma 4 model that accepts audio. It is Apache 2.0 and not gated. Its bf16 weights are 23.9 GB, so **bf16 LoRA** fits easily. Because it has no separate encoders, the repo's plain `q_proj/k_proj/v_proj/o_proj` list only matches the language model. It needs `transformers>=5.10` (5.11 or later recommended).
- **Qwen: `Qwen/Qwen3-Omni-30B-A3B-Instruct`** (the current default). It is still the strongest open-weights Qwen model that accepts audio: Qwen3.5 and Qwen3.6 do not accept audio, and no Qwen3.5-Omni weights are on the Hub. **But the repo's claim that "QLoRA brings it to ~15–18 GB" does not hold on `transformers` 5.x.** The MoE experts are stored as fused 3-D `nn.Parameter`s, and bitsandbytes 4-bit quantization only converts `nn.Linear` layers, so about 58 GB of expert weights stay in bf16. It fits one 80 GB GPU only if we **load just the thinker** (no talker or code2wav), at batch size 1 with gradient checkpointing, and even then it is tight (about 65–75 GB, estimated). Unless someone proves 4-bit expert quantization works, treat bf16 LoRA and "QLoRA" as roughly the same memory for this model. The fallback if it does not fit is `Qwen/Qwen2.5-Omni-7B`.

## Candidates

The VRAM columns are **estimates** (weights + LoRA r=16 adapters + AdamW states for the adapters + activations for one 30 s clip, batch size 1, gradient checkpointing on). They are not measured. The GPU check on the map should confirm them. Weight sizes come from safetensors sizes and parameter counts on the Hub.

| Model | Audio in? | Params (total) | License / gating | bf16 weights | bf16 LoRA (est.) | 4-bit QLoRA (est.) | Min `transformers` | LoRA targets matching `q/k/v/o_proj` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `google/gemma-4-E2B-it` | yes (≤30 s) | 5.1 B (2.3 B effective) | Apache 2.0, not gated | ~10 GB | ~15–20 GB | ~10–14 GB¹ | 5.5.0 | **Breaks as-is**²: also matches `Gemma4ClippableLinear` wrappers in the audio and vision towers |
| `google/gemma-4-E4B-it` | yes (≤30 s) | 8.0 B (4.5 B effective) | Apache 2.0, not gated | ~16 GB | ~22–30 GB | ~12–18 GB¹ | 5.5.0 | same as E2B² |
| **`google/gemma-4-12B-it`** | **yes (≤30 s)** | **11.96 B** | **Apache 2.0, not gated** | **23.9 GB** | **~30–40 GB** | ~12–20 GB | **5.10.0 (use ≥5.11.0)³** | **yes**: only `model.language_model.layers.*` has them (no encoder towers) |
| `google/gemma-4-26B-A4B-it` | **no** | 25.8 B | Apache 2.0 | ~52 GB | — | — | 5.5.0 | n/a (no audio) |
| `google/gemma-4-31B-it` | **no** | 31.3 B | Apache 2.0 | ~62 GB | — | — | 5.5.0 | n/a (no audio) |
| **`Qwen/Qwen3-Omni-30B-A3B-Instruct`** | **yes** | **35.3 B** (thinker ≈30 B, 3 B active) | **Apache 2.0, not gated** | **70.5 GB** (whole model) | full model: **does not fit**; thinker only: **~65–75 GB (tight)** | about the same as bf16 LoRA⁴ | **4.57.0** | yes, but they also match the audio tower, talker and code2wav unless only the thinker is loaded or a regex is used⁵ |
| `Qwen/Qwen3-Omni-30B-A3B-Thinking` | yes | 35.3 B | Apache 2.0 | 70.5 GB | as Instruct | as Instruct | 4.57.0 | as Instruct. It writes a reasoning trace first, which does not suit our short-answer targets. |
| `Qwen/Qwen3-Omni-30B-A3B-Captioner` | yes (audio only) | 35.3 B | Apache 2.0 | 70.5 GB | as Instruct | as Instruct | 4.57.0 | as Instruct. It is a captioning finetune that ignores text prompts, so it cannot follow instructions. |
| `Qwen/Qwen2.5-Omni-7B` | yes | 10.7 B | Apache 2.0, not gated | 22.4 GB (incl. talker/token2wav) | ~25–35 GB | ~12–18 GB | 4.52.0 | yes in `thinker.model` / `talker.model`, also `q/k/v_proj` in the audio tower⁵ |
| `Qwen/Qwen2-Audio-7B-Instruct` | yes | 8.4 B | Apache 2.0, not gated | ~17 GB | ~22–28 GB | ~10–14 GB | 4.45.0 | yes (LM and audio encoder) |
| Qwen3.5 / Qwen3.6 (0.8 B–397 B) | **no** (image-text-to-text) | — | — | — | — | — | — | n/a |
| `Qwen/Qwen3-ASR-*` | yes, but ASR only | 0.6 B / 1.7 B | — | — | — | — | — | not an instruct model |

`peft`: PyPI lists no minimum `transformers` version. `peft` 0.19.0 was the first release with default LoRA targets for `gemma4` (`.*language_model\..*\.(q_proj|v_proj)`), and 0.20.0 fixed Gemma 4 prefix-tuning bugs. **Pin `peft>=0.19.0`; 0.21.0 is current.** No `peft` release notes mention Qwen-Omni. The LoRA-on-`nn.Linear` path we use needs nothing specific to that model.

Notes:

1. **Gemma E2B/E4B:** bitsandbytes only quantizes `nn.Linear`. The large per-layer embedding tables stay in bf16, so 4-bit saves less than the parameter count suggests.
2. **`Gemma4ClippableLinear`:** in `modeling_gemma4.py` (v5.17.0) this class is an `nn.Module` that *wraps* an `nn.Linear`. The audio tower's `q_proj/k_proj/v_proj/post` layers and the vision tower's `q/k/v/o_proj` layers use it. The checkpoint keys show this: `model.audio_tower.layers.*.q_proj.linear`, `model.vision_tower.encoder.layers.*.q_proj.linear`. With `target_modules=["q_proj", ...]`, PEFT matches these wrappers, which are not supported layer types. PEFT's own default for `gemma4` is a regex limited to `language_model`, and that is the fix. This affects the current default `GEMMA-4` (E2B) in `train/`.
3. **Gemma 4 12B Unified:** `models/gemma4_unified` first appears in the v5.10.0 tag. The v5.10.1 notes announce it: an encoder-free model with "No Audio Tower", where raw 16 kHz waveform frames go through `RMSNorm → Linear`. v5.11.0 fixed "gemma4_unified conversion script and config bugs" (#46398). The checkpoint's linear layers are only `model.language_model.layers.*.{q,k,v,o,gate,up,down}_proj` plus `embed_audio/embed_vision.embedding_projection` and `vision_embedder.patch_dense`.
4. **Qwen3-Omni 4-bit:** `Qwen3OmniMoeThinkerTextExperts` holds `gate_up_proj` and `down_proj` as `nn.Parameter(num_experts, …)` (`modeling_qwen3_omni_moe.py` v5.17.0). `Bnb4BitHfQuantizer.param_needs_quantization` only returns true for `bnb.nn.Linear4bit`, and `replace_with_bnb_linear` only replaces `type(module) is nn.Linear`. Expert size from `config.json` (48 layers × 128 experts × 3 × 768 × 2048) is about 29 B params, or about 58 GB in bf16, and 4-bit does not touch it. Only attention (32 q heads / 4 kv heads, hidden 2048) and the encoders get quantized, which saves little memory.
5. **Qwen-Omni module names:** the checkpoint has `q/k/v/o_proj` in `thinker.model.layers`, `talker.model.layers`, `talker.code_predictor.model` and `code2wav.pre_transformer.layers`, and has `q/k/v_proj` (plus `out_proj`) in `thinker.audio_tower.layers`. Loading the full `Qwen3OmniMoeForConditionalGeneration` with the plain list puts adapters on the talker too, which wastes memory and never trains.

## Why these two

**Gemma 4 12B.**
- Only E2B, E4B and 12B accept audio; the model card says 26B-A4B and 31B have "No Audio". 12B is the biggest of the three, and the card's audio scores are the best of the three: CoVoST 38.5 vs 35.54 (E4B) and 33.47 (E2B); FLEURS 0.069 vs 0.08 and 0.09 (lower is better; the 12B numbers exclude Chinese).
- It has a 256K context, and the card says the encoder-free design allows "the entire model to be fine-tuned in one pass".
- 24 GB of bf16 weights leaves plenty of room on 80 GB, so use bf16 LoRA (no NF4 noise, faster steps). QLoRA is not needed.
- Maximum audio length is 30 s, which matches `max_audio_seconds=30` in the repo.

**Qwen3-Omni-30B-A3B-Instruct.**
- It is the only current Qwen model with audio input that is both instruction-tuned and larger than 7B.
- Qwen3.5 and Qwen3.6 are `image-text-to-text` on the Hub, and a Hub search finds no open Qwen3.5-Omni.
- The Omni card's benchmarks mention Qwen3-Omni-Flash, but the Hub has no open Flash weights.
- The catch is memory. The card lists 78.85 GB minimum for BF16 *inference* on a 15 s video with the full model. It also says `model.disable_talker()` saves "about 10GB".
- For training, load `Qwen3OmniMoeThinkerForConditionalGeneration` (the thinker only, about 30 B / 61 GB), freeze everything, and put LoRA on the thinker's attention.
- If the GPU check shows it does not fit, switch to `Qwen/Qwen2.5-Omni-7B` (Apache 2.0, 22 GB, `transformers>=4.52`) and load only its thinker as well.

## What `eval/` and `train/` would need to change

`eval/config.py` / `train/config.py`
- Change `DEFAULT_MODEL_PATHS["GEMMA-4"]` to `google/gemma-4-12B-it`, or add a `--model-path` override in the run config.
- The Qwen default is unchanged.
- `lora_target_modules` must be allowed to be a **regex string**, and each model needs its own default:
  - Gemma: `r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)"`. This is harmless on 12B and required on E2B/E4B.
  - Qwen: `r"^thinker\.model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"`, or the plain list once only the thinker is loaded.
- `peft` accepts a string `target_modules` as a full-match regex.

`requirements.txt` / `train/requirements.txt`
- Raise `transformers>=5.5.0` to `>=5.11.0` (needed for `gemma4_unified`).
- Pin `peft>=0.19.0`.
- Keep `bitsandbytes`, but know that it does not quantize the Qwen experts.

`eval/backends/gemma.py` (`GemmaBackend`)
- Probably works with 12B unchanged. `AutoModelForCausalLM` maps `gemma4_unified` to `Gemma4UnifiedForConditionalGeneration` (`MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` in v5.17.0). `AutoProcessor` resolves to `Gemma4UnifiedProcessor`, whose `__call__` takes `audio=`.
- The card uses `AutoModelForMultimodalLM` and puts the audio part *after* the text. Consider matching both.
- Nobody has run 12B through this backend yet.

`train/backends/gemma.py`
- Same loading path, so no structural change.
- For 12B, default to `--no-4bit`.

`eval/backends/qwen.py` (`QwenBackend`)
- Works for the base model today, since it loads the full model and generates text.
- For a finetune, it needs to load the adapter onto the thinker. PEFT key prefixes must match however `train/` loaded the model: training the thinker alone gives adapter keys without the `thinker.` prefix.
- Consider calling `model.disable_talker()` in eval as well. It saves about 10 GB, and eval only needs text output.

`train/backends/qwen.py` (`QwenTrainBackend`)
- Load `Qwen3OmniMoeThinkerForConditionalGeneration` (or the full model, then `disable_talker()` before PEFT wrapping) in bf16.
- Make `load_in_4bit=False` the Qwen default, or warn that it barely saves memory.
- Keep batch size 1 with gradient checkpointing.
- Check that the thinker's `forward` accepts the processor's `input_features`/`feature_attention_mask` and `labels`.

`FINETUNING.md`
- The "Qwen3-Omni-30B → QLoRA … ~15–18 GB" section and the E2B-based examples are out of date. Fix them in the docs sweep.

## Sources

- Hub model API (params, gating, license, `config.json`): `https://huggingface.co/api/models/<id>` and `https://huggingface.co/<id>/resolve/main/config.json` for every model in the table
- Gemma 4 12B card: https://huggingface.co/google/gemma-4-12B-it (audio on E2B/E4B/12B only, 30 s max, 256K context, CoVoST/FLEURS, encoder-free)
- Gemma 4 E4B card: https://huggingface.co/google/gemma-4-E4B-it (effective vs total params, ~300M audio encoder, 128K context, `AutoModelForMultimodalLM`)
- Gemma 4 license link: https://ai.google.dev/gemma/docs/gemma_4_license (the cards declare `license: apache-2.0`)
- Gemma safetensors headers (module names, 12B size 23,919,549,408 bytes): `https://huggingface.co/google/gemma-4-{E4B,12B}-it/resolve/main/model.safetensors`
- Qwen3-Omni card: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct (Apache 2.0, BF16 memory table, `disable_talker` saves ~10 GB, Captioner is a finetune of Instruct)
- Qwen3-Omni weight index (70,519,637,090 bytes, module names): https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/resolve/main/model.safetensors.index.json
- Qwen2.5-Omni-7B LICENSE (Apache 2.0) and index: https://huggingface.co/Qwen/Qwen2.5-Omni-7B
- Qwen model listing (Qwen3.5/3.6 are image-text-to-text, Qwen3-ASR): `https://huggingface.co/api/models?author=Qwen&search=...`
- transformers release notes: https://github.com/huggingface/transformers/releases/tag/v5.5.0 (Gemma4), https://github.com/huggingface/transformers/releases/tag/v5.10.1 (Gemma4 unified), https://github.com/huggingface/transformers/releases/tag/v5.11.0 (gemma4_unified fixes)
- transformers source, v5.17.0: `src/transformers/models/gemma4/modeling_gemma4.py` (`Gemma4ClippableLinear`), `models/qwen3_omni_moe/modeling_qwen3_omni_moe.py` (fused experts, `disable_talker`, thinker class), `models/auto/modeling_auto.py` (Auto mappings), `quantizers/quantizer_bnb_4bit.py` and `integrations/bitsandbytes.py` (only `nn.Linear` is quantized), `models/gemma4_unified/processing_gemma4_unified.py`
- First tags with each model directory (checked via raw.githubusercontent.com): `qwen2_audio` v4.45.0, `qwen2_5_omni` v4.52.0, `qwen3_omni_moe` v4.57.0, `gemma4` v5.5.0, `gemma4_unified` v5.10.0
- peft release notes: https://github.com/huggingface/peft/releases/tag/v0.19.0 (gemma4 default targets), https://github.com/huggingface/peft/releases/tag/v0.20.0 (Gemma 4 fixes); `src/peft/utils/constants.py` at v0.21.0
- This repo: `eval/config.py`, `eval/backends/{gemma,qwen}.py`, `train/config.py`, `train/backends/{base,gemma,qwen}.py`, `FINETUNING.md`, `requirements.txt`
