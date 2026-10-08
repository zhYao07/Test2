"""Qwen3.5 study classifier: full vision tower + text-only LoRA + 12-logit head."""

import hashlib
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
import numpy as np
import torch
from torch import nn

from rsna_data import LABELS, SLOTS


ARCHITECTURE = "baseline_v24_qwen35_2b_vision_full_text_lora"
DEFAULT_QWEN_MODEL_DIR = Path(__file__).resolve().parents[1] / "Qwen3.5-2B"
PROMPT_VERSION = "study_summary_v1"
MODEL_INPUT_KEYS = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw",
                    "mm_token_type_ids", "position_ids"}


def load_processor(model_dir, image_size):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    processor.tokenizer.padding_side = "right"
    patch = int(processor.image_processor.patch_size)
    merge = int(processor.image_processor.merge_size)
    if image_size < patch * merge or image_size % (patch * merge):
        raise ValueError(f"image_size must be a positive multiple of {patch * merge}")
    if processor.tokenizer.pad_token_id is None:
        raise ValueError("Qwen tokenizer must define pad_token_id")
    return processor


def processor_signature(processor):
    """Guard tokenizer/template/normalization changes between training and inference."""
    image = processor.image_processor
    backend = json.loads(processor.tokenizer.backend_tokenizer.to_str())
    # Padding/truncation are fixed by make_vlm_batch and may be mutated by calls.
    backend.pop("padding", None)
    backend.pop("truncation", None)
    contract = {
        "prompt_version": PROMPT_VERSION,
        "chat_template": processor.chat_template,
        "vocab": sorted(processor.tokenizer.get_vocab().items()),
        "tokenizer_backend": backend,
        "special_tokens": processor.tokenizer.special_tokens_map,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "image_token_id": processor.image_token_id,
        "patch_size": image.patch_size, "merge_size": image.merge_size,
        "temporal_patch_size": image.temporal_patch_size,
        "image_mean": image.image_mean, "image_std": image.image_std,
        "do_resize": False, "do_rescale": False, "do_normalize": True,
    }
    payload = json.dumps(contract, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_vlm_batch(processor, batch, use_metadata=True, max_tokens=16384):
    """Process v14 float [0,1] windows on CPU; retain processor-generated M-RoPE data."""
    images = batch["images"]
    if images.device.type != "cpu" or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Process packed [N,3,H,W] images on CPU before moving to GPU")
    if not len(images) or not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Expected finite nonempty v14 windows in [0,1]")
    texts, ordered_images, image_counts = [], [], []
    owners, slots, positions = (batch[key].numpy() for key in
                               ("window_batch_indices", "window_slot_indices", "window_positions"))
    size = len(batch["study_uid"])
    if any(values.shape != (len(images),) for values in (owners, slots, positions)):
        raise ValueError("Packed image maps differ in length")
    if np.any((owners < 0) | (owners >= size)) or np.any((slots < 0) | (slots >= len(SLOTS))):
        raise ValueError("Invalid study/slot image mapping")
    for study in range(size):
        content = [{"type": "text", "text":
                    "These are ordered MRI windows from one knee study. "
                    "The three image channels contain neighboring slices. "
                    "Window position is normalized within each series.\n"}]
        count = 0
        for slot, name in enumerate(SLOTS):
            selected = np.flatnonzero((owners == study) & (slots == slot))
            present = bool(batch["slot_mask"][study, slot])
            if present != bool(len(selected)):
                raise ValueError("Every present slot must have images, and absent slots must not")
            if not present:
                content.append({"type": "text", "text": f"\n{name}: unavailable.\n"})
                continue
            selected = selected[np.argsort(positions[selected], kind="stable")]
            description = f"\n{name}:"
            if use_metadata:
                flags = batch["series_features"][study, slot, :2].tolist()
                description += f" fluid_sensitive={int(flags[0] > 0.5)}, fat_suppression={int(flags[1] > 0.5)};"
            content.append({"type": "text", "text": description + "\n"})
            for index in selected:
                content.append({"type": "text", "text": f"position={positions[index]:+.3f}\n"})
                content.append({"type": "image"})
                ordered_images.append(images[index])
                count += 1
        if not count:
            raise ValueError("Study contains no images")
        content.append({"type": "text", "text":
                        "\nSummarize the evidence across all available series for these findings: "
                        + ", ".join(LABELS) + "."})
        texts.append(processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False))
        image_counts.append(count)
    # Inputs are already 384x384 and [0,1]. Apply only Qwen's normalization/patch packing.
    encoded = processor(text=texts, images=ordered_images, padding=True, truncation=False,
                        return_tensors="pt", do_resize=False, do_rescale=False, do_normalize=True)
    grid = encoded["image_grid_thw"]
    patch, merge = processor.image_processor.patch_size, processor.image_processor.merge_size
    expected_grid = torch.tensor([1, images.shape[-2] // patch, images.shape[-1] // patch])
    if grid.shape != (len(images), 3) or not torch.equal(grid.cpu(), expected_grid.expand_as(grid)):
        raise ValueError("Processor changed image geometry; expected one grid per v14 window")
    lengths = encoded["attention_mask"].sum(-1)
    if lengths.max().item() > max_tokens:
        raise ValueError(f"Prompt has {int(lengths.max())} tokens > max_tokens={max_tokens}; "
                         "reduce windows/image size or explicitly raise the budget (no truncation)")
    offset = 0
    for study, count in enumerate(image_counts):
        required = int((grid[offset:offset + count].prod(-1) // merge ** 2).sum())
        actual = int((encoded["input_ids"][study] == processor.image_token_id).sum())
        if actual != required:
            raise ValueError(f"Image/token mismatch in study {study}: {actual} != {required}")
        offset += count
    result = {key: value for key, value in encoded.items() if key in MODEL_INPUT_KEYS}
    for key in ("study_uid", "targets", "label_mask", "label_weight", "candidate_count"):
        if key in batch:
            result[key] = batch[key]
    result["window_count"] = len(images)
    return result


def _canonical_config(config):
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()
                    if k not in {"_name_or_path", "transformers_version", "architectures", "dtype", "torch_dtype",
                                 "_attn_implementation_autoset"}}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    return clean(config.to_dict())


class RSNAQwen35(nn.Module):
    def __init__(self, model_dir=DEFAULT_QWEN_MODEL_DIR, *, core=None, lora_rank=16, lora_alpha=32,
                 lora_dropout=0.05, head_dropout=0.1, gradient_checkpointing=True,
                 dtype=torch.bfloat16, attn_implementation="sdpa"):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3_5Model

        if lora_rank < 1 or lora_alpha < 1 or not 0 <= lora_dropout < 1 or not 0 <= head_dropout < 1:
            raise ValueError("Invalid LoRA/head configuration")
        if core is None:
            core, loading = Qwen3_5Model.from_pretrained(
                str(model_dir), local_files_only=True, dtype=dtype,
                attn_implementation=attn_implementation, output_loading_info=True)
            if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
                raise ValueError(f"Incomplete/incompatible Qwen snapshot: {loading}")
        self.vlm = core
        self.base_config = _canonical_config(core.config)
        self.model_settings = dict(lora_rank=lora_rank, lora_alpha=lora_alpha,
                                   lora_dropout=lora_dropout, head_dropout=head_dropout)
        core.requires_grad_(False)
        # Discover exact text-layer Linear paths, including hybrid DeltaNet projections.
        # Wrapping only language_model prevents accidental LoRA insertion into the vision tower.
        targets = [name for name, module in core.language_model.named_modules()
                   if name.startswith("layers.") and isinstance(module, nn.Linear)]
        if not targets:
            raise ValueError("No text-layer linear modules found for LoRA")
        core.language_model = get_peft_model(core.language_model, LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=targets, bias="none"))
        self.lora_targets = targets
        core.visual.requires_grad_(True)  # Includes the pretrained merger/projector.
        core.visual.float()
        for parameter in core.language_model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        core.config.text_config.use_cache = False
        if gradient_checkpointing:
            core.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.head = nn.Sequential(nn.LayerNorm(core.config.text_config.hidden_size),
                                  nn.Dropout(head_dropout),
                                  nn.Linear(core.config.text_config.hidden_size, len(LABELS)))
        self.assert_trainability()

    def assert_trainability(self):
        if not all(p.requires_grad for p in self.vlm.visual.parameters()):
            raise RuntimeError("Entire vision tower/merger must be trainable")
        active = [name for name, p in self.vlm.language_model.named_parameters() if p.requires_grad]
        if not active or any("lora_" not in name for name in active):
            raise RuntimeError("Only text LoRA parameters may be trainable")

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, **kwargs):
        inputs = dict(input_ids=input_ids, attention_mask=attention_mask,
                      pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        inputs.update({key: value for key, value in kwargs.items() if key in MODEL_INPUT_KEYS})
        # Use the base multimodal model, avoiding [B,T,248320] vocabulary logits.
        output = self.vlm(**inputs, use_cache=False, return_dict=True)
        locations = torch.arange(attention_mask.shape[1], device=attention_mask.device)
        last = locations.expand_as(attention_mask).masked_fill(~attention_mask.bool(), -1).max(-1).values
        if (last < 0).any():
            raise ValueError("Empty attention mask")
        pooled = output.last_hidden_state[torch.arange(len(last), device=last.device), last]
        return self.head(pooled.float()).float()

    def parameter_groups(self, vision_lr, lora_lr, head_lr, weight_decay):
        categories = {"vision": [], "lora": [], "head": []}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            category = "vision" if name.startswith("vlm.visual.") else "head" if name.startswith("head.") else "lora"
            categories[category].append((name, parameter))
        groups = []
        for category, lr in (("vision", vision_lr), ("lora", lora_lr), ("head", head_lr)):
            for decay in (True, False):
                params = [p for _, p in categories[category] if (p.ndim >= 2) == decay]
                if params:
                    groups.append(dict(params=params, lr=lr, weight_decay=weight_decay if decay else 0,
                                       name=f"{category}_{'decay' if decay else 'no_decay'}"))
        return groups

    def trainable_state_dict(self):
        return {name: p.detach().cpu().clone() for name, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state_dict(self, state):
        expected = {name: p for name, p in self.named_parameters() if p.requires_grad}
        if set(state) != set(expected):
            missing, extra = set(expected) - set(state), set(state) - set(expected)
            raise ValueError(f"Checkpoint trainable key mismatch: missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}")
        for name, parameter in expected.items():
            if state[name].shape != parameter.shape:
                raise ValueError(f"Checkpoint shape mismatch: {name}")
        with torch.no_grad():
            for name, parameter in expected.items():
                parameter.copy_(state[name].to(parameter))


def predict_batch(model, batch):
    return model(**{key: value for key, value in batch.items() if key in MODEL_INPUT_KEYS})


@contextmanager
def attention_context(model, implementation="sdpa", sdpa_backend="math"):
    """Select evaluation kernels and restore both backbones for the next training epoch."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    raw = model.module if hasattr(model, "module") else model
    core = raw.vlm
    previous = {"": core.config._attn_implementation,
                "text_config": core.config.text_config._attn_implementation,
                "vision_config": core.config.vision_config._attn_implementation}
    if sdpa_backend not in ("auto", "math"):
        raise ValueError("sdpa_backend must be auto or math")
    try:
        core.set_attn_implementation(implementation)
        for config in (core.config.text_config, core.config.vision_config):
            if config._attn_implementation != implementation:
                raise RuntimeError("Qwen attention backend did not switch; check Transformers version")
        kernels = sdpa_kernel(SDPBackend.MATH) if implementation == "sdpa" and sdpa_backend == "math" else nullcontext()
        with kernels:
            yield
    finally:
        core.set_attn_implementation(previous)


def validate_checkpoint(checkpoint, model, signature):
    from rsna_data import CANDIDATE_BUDGETS, validate_series_selection_config

    if (checkpoint.get("architecture") != ARCHITECTURE or checkpoint.get("labels") != LABELS
            or checkpoint.get("slots") != SLOTS or checkpoint.get("candidate_budgets") != CANDIDATE_BUDGETS):
        raise ValueError("Requires a matching Baseline_v24 checkpoint")
    validate_series_selection_config(checkpoint.get("series_selection"))
    if checkpoint.get("model_settings") != model.model_settings or checkpoint.get("lora_targets") != model.lora_targets:
        raise ValueError("LoRA/head configuration differs from checkpoint")
    if checkpoint.get("base_config") != model.base_config:
        raise ValueError("Qwen base configuration differs from checkpoint")
    if checkpoint.get("processor_signature") != signature:
        raise ValueError("Tokenizer/template/image normalization differs from training")


def load_classifier(checkpoint_path, model_dir, device, dtype=torch.bfloat16,
                    attn_implementation="sdpa"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Not a Baseline_v24 checkpoint")
    model = RSNAQwen35(model_dir, **checkpoint["model_settings"], gradient_checkpointing=False,
                      dtype=dtype, attn_implementation=attn_implementation)
    processor = load_processor(model_dir, checkpoint["args"]["image_size"])
    validate_checkpoint(checkpoint, model, processor_signature(processor))
    model.load_trainable_state_dict(checkpoint["trainable_model"])
    for key in ("trainable_model", "training_model", "ema", "optimizer", "scheduler", "scaler", "rng_by_rank"):
        checkpoint.pop(key, None)
    return model.to(device).eval(), processor, checkpoint
