"""Train v24 on weak labels, validating on the 58 official gold studies."""

import argparse
import atexit
from contextlib import contextmanager, nullcontext
from datetime import timedelta
import json
from itertools import islice
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from rsna_data import (DEFAULT_ROOT, LABELS, SLOTS, CANDIDATE_BUDGETS, QUALITY_COLUMNS,
                       add_series_quality, collate_studies, load_metadata,
                       make_series_selection_config, series_selection_report)
from vlm_data import VLMKneeDataset
from rsna_model import (ARCHITECTURE, DEFAULT_QWEN_MODEL_DIR, RSNAQwen35, load_processor, make_vlm_batch,
                        predict_batch, processor_signature, validate_checkpoint, attention_context, load_classifier)


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--labels-csv", type=Path, default=SCRIPT_DIR / "label.csv")
    parser.add_argument("--qwen-model-dir", type=Path, default=DEFAULT_QWEN_MODEL_DIR,
                        help="Defaults to ../Qwen3.5-2B next to Baseline_v24; no automatic downloads")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--cache-dir", type=Path, default=SCRIPT_DIR / "input_cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--crop-mm", type=float, default=140)
    parser.add_argument("--train-windows", type=int, default=24)
    parser.add_argument("--eval-windows", type=int, default=0, help="0 = all v14 candidates (up to 80)")
    parser.add_argument("--span-lo", type=float, default=0.02)
    parser.add_argument("--span-hi", type=float, default=0.98)
    parser.add_argument("--series-quality-workers", type=int, default=8)
    parser.add_argument("--coverage-quantile", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=16384)
    metadata = parser.add_mutually_exclusive_group()
    metadata.add_argument("--no-metadata", dest="no_metadata", action="store_true",
                          help="Default: omit fluid/fat-suppression flags; retain slot/plane and position")
    metadata.add_argument("--metadata", dest="no_metadata", action="store_false",
                          help="Include fluid/fat-suppression flags in the prompt")
    parser.set_defaults(no_metadata=True)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--vision-lr", type=float, default=1e-5)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1)
    parser.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--attn-implementation", choices=["sdpa", "eager", "flash_attention_2"], default="sdpa")
    parser.add_argument("--eval-attn-implementation", choices=["sdpa", "eager", "flash_attention_2"], default="sdpa",
                        help="Independent evaluation backend; default sdpa avoids external FlashAttention kernels")
    parser.add_argument("--eval-sdpa-backend", choices=["math", "auto"], default="math",
                        help="Default math disables fused SDPA kernels during evaluation")
    parser.add_argument("--skip-validation-preflight", action="store_true",
                        help="Skip the full validation check before training")
    parser.add_argument("--validation-only", action="store_true",
                        help="Run validation preflight and exit without training")
    parser.add_argument("--smoke-test", action="store_true",
                        help="One optimizer update, one full-window validation batch, checkpoint/resume/inference checks, then exit")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    weighting = parser.add_mutually_exclusive_group()
    weighting.add_argument("--pos-weight", dest="pos_weight", action="store_true",
                           help="Default: use v14 train-derived positive class weights")
    weighting.add_argument("--no-pos-weight", dest="pos_weight", action="store_false")
    parser.set_defaults(pos_weight=True)
    parser.add_argument("--ema-decay", type=float, default=0.9995,
                        help="EMA decay cap; v14 update-count warmup is applied")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA validation/inference weights")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    args = parser.parse_args()
    for key in ("epochs", "batch_size", "eval_batch_size", "accum_steps", "image_size", "series_quality_workers",
                "max_tokens", "lora_rank", "lora_alpha", "log_interval", "cpu_threads"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be >=1")
    if args.num_workers < 0 or args.train_windows < 5 or (args.eval_windows != 0 and args.eval_windows < 5):
        parser.error("Invalid worker/window count")
    if not 0 <= args.span_lo < args.span_hi <= 1 or not 0 <= args.coverage_quantile <= 1:
        parser.error("Invalid span/coverage quantile")
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.lora_dropout < 1 or not 0 <= args.head_dropout < 1:
        parser.error("Invalid warmup/dropout")
    if min(args.crop_mm, args.vision_lr, args.lora_lr, args.head_lr, args.grad_clip) <= 0 or args.weight_decay < 0:
        parser.error("Crop/lrs/grad-clip must be positive; decay must be nonnegative")
    if not 0 <= args.ema_decay < 1:
        parser.error("--ema-decay must be in [0, 1)")
    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint are mutually exclusive")
    if args.validation_only and args.skip_validation_preflight:
        parser.error("--validation-only cannot be combined with --skip-validation-preflight")
    if args.smoke_test and args.validation_only:
        parser.error("--smoke-test and --validation-only are mutually exclusive")
    return args


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_distributed():
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-process training requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
        atexit.register(cleanup_distributed)
    return distributed, local_rank, dist.get_rank() if distributed else 0, dist.get_world_size() if distributed else 1


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_label_column(series):
    text = series.astype(str).str.strip().str.upper()
    mapped = text.map({"YES": 1., "Y": 1., "TRUE": 1., "NO": 0., "N": 0., "FALSE": 0.,
                       "UNK": 0.5, "UNKNOWN": 0.5})
    values = mapped.where(mapped.notna(), pd.to_numeric(series, errors="coerce"))
    return values.where(values.between(0, 1), np.nan).astype(np.float32)


def prepare_studies(train_df, labels_csv):
    weak = pd.read_csv(labels_csv)
    required = ["StudyInstanceUID", *LABELS]
    missing = [c for c in required if c not in weak.columns]
    if missing:
        raise ValueError(f"Weak-label CSV missing columns: {missing}")
    if weak.StudyInstanceUID.duplicated().any() or train_df.StudyInstanceUID.duplicated().any():
        raise ValueError("Duplicate study UIDs in labels/train.csv")
    studies = train_df[required].rename(columns={label: f"gold__{label}" for label in LABELS})
    studies = studies.merge(weak[required], on="StudyInstanceUID", how="left", validate="one_to_one")
    for label in LABELS:
        studies[label] = parse_label_column(studies[label])
        gold = pd.to_numeric(studies[f"gold__{label}"], errors="coerce").astype(np.float32)
        if not gold.dropna().isin([0, 1]).all():
            raise ValueError(f"Official gold label must be binary: {label}")
        studies[f"gold__{label}"] = gold
        studies.loc[gold.notna(), label] = gold[gold.notna()]
    return studies[studies[LABELS].notna().any(axis=1)].reset_index(drop=True)


def calculate_pos_weight(studies):
    values = studies[LABELS].to_numpy(dtype=np.float32)
    confidence = np.where(np.isnan(values), 0, 2 * np.abs(values - 0.5))
    binary = np.nan_to_num(values, nan=0.5) > 0.5
    positives = (confidence * binary).sum(0)
    negatives = (confidence * ~binary).sum(0)
    return torch.tensor(np.clip(negatives / np.maximum(positives, 1), 1, 10), dtype=torch.float32)


def masked_bce(logits, targets, mask, label_weight, pos_weight):
    loss = F.binary_cross_entropy_with_logits(logits.float(), targets.float(),
                                             reduction="none", pos_weight=pos_weight)
    weight = mask.float() * label_weight.float()
    return (loss * weight).sum() / weight.sum().clamp_min(1)


def make_loader(dataset, batch_size, workers, sampler, shuffle):
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=shuffle and sampler is None, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      prefetch_factor=2 if workers else None, drop_last=False,
                      generator=torch.Generator(), collate_fn=collate_studies)


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def autocast_context(args, device):
    return torch.autocast(device_type=device.type,
                          dtype=torch.bfloat16 if args.amp == "bf16" else torch.float16,
                          enabled=device.type == "cuda" and args.amp != "none")


def make_scheduler(optimizer, total_steps, warmup_ratio):
    warmup = int(total_steps * warmup_ratio)

    def factor(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = min(1, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, factor)


class TrainableEMA:
    """EMA of vision, LoRA and head parameters; the frozen base is shared unchanged."""

    def __init__(self, model, decay=0.9995):
        raw = model.module if isinstance(model, DDP) else model
        self.shadow = {name: p.detach().clone() for name, p in raw.named_parameters() if p.requires_grad}
        self.decay = decay
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model):
        raw = model.module if isinstance(model, DDP) else model
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for name, parameter in raw.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach(), 1 - decay)

    def state_dict(self):
        return dict(model={name: p.detach().cpu().clone() for name, p in self.shadow.items()},
                    decay=self.decay, num_updates=self.num_updates)

    def load_state_dict(self, state):
        if state["decay"] != self.decay or set(state["model"]) != set(self.shadow):
            raise ValueError("EMA decay/parameter keys differ from checkpoint")
        for name, target in self.shadow.items():
            if state["model"][name].shape != target.shape:
                raise ValueError(f"EMA shape mismatch: {name}")
        with torch.no_grad():
            for name, target in self.shadow.items():
                target.copy_(state["model"][name].to(target))
        self.num_updates = int(state["num_updates"])

    @contextmanager
    def average_parameters(self, model):
        raw = model.module if isinstance(model, DDP) else model
        parameters = dict(raw.named_parameters())
        backup = {name: parameters[name].detach().clone() for name in self.shadow}
        try:
            with torch.no_grad():
                for name, value in self.shadow.items():
                    parameters[name].copy_(value)
            yield
        finally:
            with torch.no_grad():
                for name, value in backup.items():
                    parameters[name].copy_(value)


def train_epoch(model, processor, loader, sampler, optimizer, scheduler, scaler,
                pos_weight, device, args, epoch, rank, ema=None, max_batches=None):
    model.train()
    loader.dataset.set_epoch(epoch)
    loader.generator.manual_seed(args.seed + epoch)
    if sampler is not None:
        sampler.set_epoch(epoch)
    optimizer.zero_grad(set_to_none=True)
    losses, started = [], time.perf_counter()
    batch_count = len(loader) if max_batches is None else min(len(loader), max_batches)
    for step, raw_batch in enumerate(islice(loader, batch_count)):
        batch = move_batch(make_vlm_batch(processor, raw_batch, not args.no_metadata, args.max_tokens), device)
        group_start = (step // args.accum_steps) * args.accum_steps
        group_size = min(args.accum_steps, batch_count - group_start)
        update = step - group_start + 1 == group_size
        sync = model.no_sync() if isinstance(model, DDP) and not update else nullcontext()
        with sync:
            with autocast_context(args, device):
                logits = predict_batch(model, batch)
                loss = masked_bce(logits, batch["targets"], batch["label_mask"], batch["label_weight"], pos_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss / group_size).backward()
        if update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip,
                                          error_if_nonfinite=args.smoke_test)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
                if ema is not None:
                    ema.update(model)
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        if rank == 0 and ((step + 1) % args.log_interval == 0 or step + 1 == batch_count):
            elapsed = time.perf_counter() - started
            print(f"epoch={epoch + 1} step={step + 1}/{batch_count} loss={np.mean(losses):.5f} "
                  f"windows={batch['window_count']} tokens={int(batch['attention_mask'].sum(-1).max())} "
                  f"step_time={elapsed / (step + 1):.2f}s eta={elapsed / (step + 1) * (batch_count - step - 1) / 60:.1f}min",
                  flush=True)
    stats = torch.tensor([sum(losses), len(losses)], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(stats)
    return float(stats[0] / stats[1]), losses


@torch.no_grad()
def validate(model, processor, loader, device, args, world_size, max_batches=None):
    model.eval()
    rows = []
    batch_count = len(loader) if max_batches is None else min(len(loader), max_batches)
    for step, raw_batch in enumerate(islice(loader, batch_count)):
        batch = move_batch(make_vlm_batch(processor, raw_batch, not args.no_metadata, args.max_tokens), device)
        with autocast_context(args, device):
            pred = predict_batch(model, batch).sigmoid().cpu().numpy()
        for i, uid in enumerate(batch["study_uid"]):
            rows.append((uid, pred[i], batch["targets"][i].cpu().numpy(), batch["label_mask"][i].cpu().numpy()))
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"validation step={step + 1}/{batch_count} windows={batch['window_count']} "
                  f"tokens={int(batch['attention_mask'].sum(-1).max())}", flush=True)
    gathered = [None] * world_size
    if dist.is_initialized():
        dist.all_gather_object(gathered, rows)
    else:
        gathered = [rows]
    unique = {row[0]: row for part in gathered for row in part}
    if not unique:
        raise ValueError("Validation set is empty")
    ordered = [unique[uid] for uid in sorted(unique)]
    predictions = np.stack([row[1] for row in ordered])
    targets = np.stack([row[2] for row in ordered])
    masks = np.stack([row[3] for row in ordered])
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Nonfinite validation predictions")
    aucs = {}
    for i, label in enumerate(LABELS):
        valid = masks[:, i] > 0
        y = targets[valid, i]
        aucs[label] = float(roc_auc_score(y, predictions[valid, i])) if len(np.unique(y)) == 2 else None
    scores = [v for v in aucs.values() if v is not None]
    macro = float(np.mean(scores)) if scores else None
    frame = pd.DataFrame(predictions, columns=LABELS)
    frame.insert(0, "StudyInstanceUID", [row[0] for row in ordered])
    return macro, aucs, frame


def evaluate(model, processor, loader, device, args, world_size, ema=None, max_batches=None):
    with attention_context(model, args.eval_attn_implementation, args.eval_sdpa_backend):
        with ema.average_parameters(model) if ema is not None else nullcontext():
            return validate(model, processor, loader, device, args, world_size, max_batches)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    temporary.replace(path)


def capture_rng(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def checkpoint_payload(model, processor_hash, args, epoch, best_auc, history, ema=None):
    inference_state = ema.state_dict()["model"] if ema is not None else model.trainable_state_dict()
    return dict(architecture=ARCHITECTURE, labels=LABELS, slots=SLOTS, candidate_budgets=CANDIDATE_BUDGETS,
                series_selection=args.series_selection, model_settings=model.model_settings,
                lora_targets=model.lora_targets, base_config=model.base_config, processor_signature=processor_hash,
                trainable_model=inference_state, weights_type="ema" if ema is not None else "raw",
                ema_updates=ema.num_updates if ema is not None else 0, epoch=epoch, best_auc=best_auc,
                validation_history=history,
                args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})


def save_checkpoint(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_resume(checkpoint, args, *, allow_smoke_test=False):
    if checkpoint.get("smoke_test") and not allow_smoke_test:
        raise ValueError("Smoke-test checkpoints contain a partial epoch; start normal training without --resume")
    if checkpoint["series_selection"] != args.series_selection:
        raise ValueError("Resume series selection thresholds differ")
    saved = checkpoint["args"]
    for key in ("image_size", "crop_mm", "span_lo", "span_hi", "train_windows", "eval_windows", "no_metadata",
                "seed", "amp", "attn_implementation", "no_gradient_checkpointing", "batch_size", "accum_steps",
                "epochs", "world_size", "vision_lr", "lora_lr", "head_lr", "weight_decay", "warmup_ratio", "pos_weight",
                "max_tokens", "grad_clip", "no_ema", "ema_decay"):
        if saved.get(key) != getattr(args, key):
            raise ValueError(f"Resume configuration mismatch: {key}")
    if not args.no_ema:
        state = checkpoint.get("ema")
        if state is None or state.get("decay") != args.ema_decay or "training_model" not in checkpoint:
            raise ValueError("EMA resume requires last.pt with matching EMA and raw training parameters")
    for key in ("optimizer", "scheduler", "scaler", "rng_by_rank"):
        if key not in checkpoint:
            raise ValueError(f"Resume requires last.pt with {key}; best.pt is inference/init-only")


def smoke_test(model, processor, train_loader, valid_loader, train_sampler, optimizer, scheduler,
               scaler, pos_weight, device, args, rank, world_size, signature, ema, epoch=0):
    """Exercise the production training, evaluation, serialization and inference paths."""
    from inference import submission_frame

    raw_model = model.module if isinstance(model, DDP) else model
    directory = args.output_dir / "smoke_test"
    step_before = scheduler.last_epoch
    ema_before = ema.num_updates if ema is not None else 0
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "result.json", dict(status="running"))
        print(f"[smoke 1/4] one optimizer update ({args.accum_steps} micro-batches maximum)", flush=True)
    loss, _ = train_epoch(model, processor, train_loader, train_sampler, optimizer, scheduler, scaler,
                          pos_weight, device, args, epoch, rank, ema, max_batches=args.accum_steps)
    if scheduler.last_epoch != step_before + 1 or (ema is not None and ema.num_updates != ema_before + 1):
        raise RuntimeError("Smoke test requires exactly one successful optimizer/scheduler/EMA update")
    if rank == 0:
        print("[smoke 2/4] one validation batch per rank with configured evaluation windows", flush=True)
    _, _, predictions = evaluate(raw_model, processor, valid_loader, device, args, world_size, ema, max_batches=1)
    state = capture_rng(device)
    rng_by_rank = [None] * world_size
    if dist.is_initialized():
        dist.all_gather_object(rng_by_rank, state)
    else:
        rng_by_rank = [state]
    if rank == 0:
        print("[smoke 3/4] save and restore raw parameters, EMA, optimizer, scheduler and RNG", flush=True)
        payload = checkpoint_payload(raw_model, signature, args, epoch - 1, -1., [], ema)
        payload["smoke_test"] = True
        save_checkpoint(directory / "best.pt", payload)
        if ema is not None:
            payload.update(training_model=raw_model.trainable_state_dict(),
                           ema=dict(model=payload["trainable_model"], decay=ema.decay, num_updates=ema.num_updates))
        payload.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                       scaler=scaler.state_dict(), rng_by_rank=rng_by_rank)
        save_checkpoint(directory / "last.pt", payload)
        predictions.to_csv(directory / "valid_predictions.csv", index=False, lineterminator="\n")
        del payload
    if dist.is_initialized():
        dist.barrier()
    restored = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    validate_checkpoint(restored, raw_model, signature)
    validate_resume(restored, args, allow_smoke_test=True)
    # Compare before loading so an incorrect rank's parameters cannot be silently overwritten.
    for name, parameter in raw_model.named_parameters():
        if parameter.requires_grad:
            expected = restored.get("training_model", restored["trainable_model"])[name]
            torch.testing.assert_close(parameter.detach().cpu(), expected, rtol=0, atol=0)
    raw_model.load_trainable_state_dict(restored.get("training_model", restored["trainable_model"]))
    optimizer.state.clear()
    optimizer.load_state_dict(restored["optimizer"])
    scheduler.load_state_dict(restored["scheduler"])
    scaler.load_state_dict(restored["scaler"])
    if ema is not None:
        ema.load_state_dict(restored["ema"])
        if ema.num_updates != ema_before + 1:
            raise RuntimeError("EMA update count did not survive checkpoint roundtrip")
    restore_rng(restored["rng_by_rank"][rank], device)
    if scheduler.last_epoch != step_before + 1:
        raise RuntimeError("Scheduler step did not survive checkpoint roundtrip")
    del restored
    if rank == 0:
        print("[smoke 4/4] reload inference classifier and verify probabilities/submission CSV", flush=True)
        dtype = torch.bfloat16 if device.type == "cuda" and args.amp == "bf16" else torch.float32
        classifier, inference_processor, _ = load_classifier(
            directory / "best.pt", args.qwen_model_dir, device, dtype=dtype,
            attn_implementation=args.eval_attn_implementation)
        raw_batch = next(iter(valid_loader))
        batch = move_batch(make_vlm_batch(inference_processor, raw_batch, not args.no_metadata, args.max_tokens), device)
        with torch.no_grad(), attention_context(classifier, args.eval_attn_implementation, args.eval_sdpa_backend):
            with autocast_context(args, device):
                probabilities = predict_batch(classifier, batch).sigmoid().cpu().numpy()
        expected = predictions.set_index("StudyInstanceUID").loc[batch["study_uid"], LABELS].to_numpy()
        np.testing.assert_allclose(probabilities, expected, rtol=1e-4, atol=1e-5)
        frame = submission_frame(pd.DataFrame({"StudyInstanceUID": batch["study_uid"]}),
                                 dict(zip(batch["study_uid"], probabilities)))
        frame.to_csv(directory / "submission.csv", index=False, lineterminator="\n")
        reread = pd.read_csv(directory / "submission.csv", dtype={"StudyInstanceUID": str})
        if reread.columns.tolist() != ["StudyInstanceUID", *LABELS] or reread.StudyInstanceUID.tolist() != batch["study_uid"]:
            raise RuntimeError("Submission CSV columns/UIDs changed during roundtrip")
        np.testing.assert_allclose(reread[LABELS].to_numpy(), probabilities, rtol=1e-6, atol=1e-7)
        write_json(directory / "result.json", dict(status="passed", optimizer_updates=1, train_loss=loss,
                   validation_batches_per_rank=1, submission_studies=len(frame), world_size=world_size,
                   train_attention=args.attn_implementation, eval_attention=args.eval_attn_implementation,
                   eval_sdpa_backend=args.eval_sdpa_backend, eval_windows=args.eval_windows))
        del classifier, batch
        print(f"SMOKE TEST PASSED: {directory}; exiting without starting full training", flush=True)
    if dist.is_initialized():
        dist.barrier()


def main():
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    distributed, local_rank, rank, world_size = setup_distributed()
    args.world_size = world_size
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU does not support bf16; use --amp fp16")
        torch.backends.cuda.matmul.allow_tf32 = True
    seed_everything(args.seed + rank)
    args.qwen_model_dir = args.qwen_model_dir.expanduser().resolve()
    processor = load_processor(args.qwen_model_dir, args.image_size)
    signature = processor_signature(processor)
    quality_cache = None if args.no_cache else args.cache_dir / "series_quality"
    objects = [None]
    if rank == 0:
        metadata = list(load_metadata(args.data_root))
        metadata[1] = add_series_quality(metadata[1], quality_cache, args.series_quality_workers)
        objects[0] = metadata[:2]
    if distributed:
        dist.broadcast_object_list(objects, src=0)
    train_df, series_df = objects[0]
    studies = prepare_studies(train_df, args.labels_csv)
    gold = studies[[f"gold__{label}" for label in LABELS]].notna().all(axis=1)
    train_studies, valid_studies = studies[~gold].copy(), studies[gold].copy()
    confidence = np.where(train_studies[LABELS].notna(), 2 * np.abs(train_studies[LABELS] - 0.5), 0).sum(1)
    train_studies = train_studies.loc[confidence > 0].reset_index(drop=True)
    valid_studies = valid_studies.reset_index(drop=True)
    if len(valid_studies) != 58 or not len(train_studies):
        raise ValueError(f"Expected nonempty weak training and 58 gold studies; got {len(train_studies)}/{len(valid_studies)}")
    for label in LABELS:
        valid_studies[label] = valid_studies[f"gold__{label}"]
    args.series_selection = make_series_selection_config(
        series_df[series_df.StudyInstanceUID.isin(train_studies.StudyInstanceUID)], args.coverage_quantile)
    dataset_kwargs = dict(image_size=args.image_size, crop_mm=args.crop_mm, train_windows=args.train_windows,
                          eval_windows=args.eval_windows, span_lo=args.span_lo, span_hi=args.span_hi,
                          cache_dir=None if args.no_cache else args.cache_dir, seed=args.seed,
                          series_selection=args.series_selection)
    training = VLMKneeDataset(train_studies, series_df, train=True, **dataset_kwargs)
    validation = VLMKneeDataset(valid_studies, series_df, train=False, **dataset_kwargs)
    train_sampler = DistributedSampler(training, shuffle=True) if distributed else None
    valid_sampler = DistributedSampler(validation, shuffle=False) if distributed else None
    train_loader = make_loader(training, args.batch_size, args.num_workers, train_sampler, True)
    valid_loader = make_loader(validation, args.eval_batch_size, args.num_workers, valid_sampler, False)
    dtype = torch.bfloat16 if device.type == "cuda" and args.amp == "bf16" else torch.float32
    # fp16 autocast keeps frozen weights fp32, and trainable weights are always fp32.
    model = RSNAQwen35(args.qwen_model_dir, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                       lora_dropout=args.lora_dropout, head_dropout=args.head_dropout,
                       gradient_checkpointing=not args.no_gradient_checkpointing,
                       dtype=dtype, attn_implementation=args.attn_implementation)
    checkpoint = None
    if args.resume or args.init_checkpoint:
        checkpoint = torch.load(args.resume or args.init_checkpoint, map_location="cpu", weights_only=False)
        validate_checkpoint(checkpoint, model, signature)
        if args.resume:
            validate_resume(checkpoint, args)
        training_state = checkpoint.get("training_model", checkpoint["trainable_model"]) if args.resume else checkpoint["trainable_model"]
        model.load_trainable_state_dict(training_state)
    model = model.to(device)
    optimizer = AdamW(model.parameter_groups(args.vision_lr, args.lora_lr, args.head_lr, args.weight_decay))
    scheduler = make_scheduler(optimizer, args.epochs * math.ceil(len(train_loader) / args.accum_steps), args.warmup_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp == "fp16")
    start_epoch, best_auc, history = 0, -1., []
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, best_auc = checkpoint["epoch"] + 1, checkpoint["best_auc"]
        history = checkpoint["validation_history"]
    raw_model = model
    if distributed:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
    # DDP initialization synchronizes parameters before each rank initializes its EMA.
    ema = None if args.no_ema else TrainableEMA(raw_model, args.ema_decay)
    if args.resume and ema is not None:
        ema.load_state_dict(checkpoint["ema"])
    if args.resume:
        restore_rng(checkpoint["rng_by_rank"][rank], device)
    del checkpoint
    pos_weight = calculate_pos_weight(train_studies) if args.pos_weight else torch.ones(len(LABELS))
    pos_weight = pos_weight.to(device)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        processor.save_pretrained(args.output_dir / "processor")
        write_json(args.output_dir / "hyperparameters.json",
                   {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        report = series_selection_report(series_df, args.series_selection, args.crop_mm, args.image_size)
        splits = {**dict.fromkeys(train_studies.StudyInstanceUID.astype(str), "train"),
                  **dict.fromkeys(valid_studies.StudyInstanceUID.astype(str), "valid")}
        report["split"] = report.StudyInstanceUID.map(splits)
        report.to_csv(args.output_dir / "series_selection.csv", index=False, lineterminator="\n")
        series_df[["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", *QUALITY_COLUMNS]].to_csv(
            args.output_dir / "series_quality.csv", index=False, lineterminator="\n")
        write_json(args.output_dir / "series_selection_summary.json",
                   dict(config=args.series_selection, changed_slots=int(report.changed.sum()),
                        changed_studies=int(report.loc[report.changed.eq(1), "StudyInstanceUID"].nunique())))
        print(f"split: train={len(training)} | gold_valid={len(validation)} | windows={args.train_windows}/{args.eval_windows or 'all'}")
        print(f"Trainable={sum(p.numel() for p in raw_model.parameters() if p.requires_grad):,}; "
              f"text LoRA modules={len(raw_model.lora_targets)}; EMA={'off' if ema is None else args.ema_decay}; "
              f"pos_weight={args.pos_weight}", flush=True)
        print(f"attention: train={args.attn_implementation}; eval={args.eval_attn_implementation}; "
              f"eval_sdpa_backend={args.eval_sdpa_backend}", flush=True)
    if args.smoke_test:
        smoke_test(model, processor, train_loader, valid_loader, train_sampler, optimizer, scheduler,
                   scaler, pos_weight, device, args, rank, world_size, signature, ema, start_epoch)
        cleanup_distributed()
        return
    if not args.skip_validation_preflight:
        if rank == 0:
            print("Starting full validation preflight before training", flush=True)
        state = capture_rng(device)
        macro, _, _ = evaluate(raw_model, processor, valid_loader, device, args, world_size, ema)
        restore_rng(state, device)
        if rank == 0:
            print(f"Validation preflight passed: macro_auc={macro}", flush=True)
    if args.validation_only:
        cleanup_distributed()
        return
    for epoch in range(start_epoch, args.epochs):
        started = time.perf_counter()
        loss, step_losses = train_epoch(model, processor, train_loader, train_sampler, optimizer,
                                        scheduler, scaler, pos_weight, device, args, epoch, rank, ema)
        # DDP gradients are already synced; validation directly calls the raw model, avoiding extra collectives.
        macro, aucs, predictions = evaluate(raw_model, processor, valid_loader, device, args, world_size, ema)
        improved = macro is not None and macro > best_auc
        if improved:
            best_auc = macro
        history.append(dict(epoch=epoch + 1, train_loss=loss, train_loss_by_step=step_losses if rank == 0 else [],
                            val_macro_auc=macro, val_auc_by_label=aucs,
                            weights_type="ema" if ema is not None else "raw",
                            ema_updates=ema.num_updates if ema is not None else 0))
        state = capture_rng(device)
        rng_by_rank = [None] * world_size
        if distributed:
            dist.all_gather_object(rng_by_rank, state)
        else:
            rng_by_rank = [state]
        if rank == 0:
            print(f"epoch={epoch + 1} train_loss={loss:.5f} val_macro_auc={macro} "
                  f"elapsed={(time.perf_counter() - started) / 60:.1f}min\n{json.dumps(aucs)}", flush=True)
            write_json(args.output_dir / "validation_history.json", history)
            predictions.to_csv(args.output_dir / "valid_predictions_last.csv", index=False, lineterminator="\n")
            payload = checkpoint_payload(raw_model, signature, args, epoch, best_auc, history, ema)
            if improved:
                save_checkpoint(args.output_dir / "best.pt", payload)
                predictions.to_csv(args.output_dir / "valid_predictions_best.csv", index=False, lineterminator="\n")
            if ema is not None:
                payload.update(training_model=raw_model.trainable_state_dict(),
                               ema=dict(model=payload["trainable_model"], decay=ema.decay, num_updates=ema.num_updates))
            payload.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                           scaler=scaler.state_dict(), rng_by_rank=rng_by_rank)
            save_checkpoint(args.output_dir / "last.pt", payload)
            del payload
        if distributed:
            dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
