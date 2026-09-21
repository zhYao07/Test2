"""Train RSNA Knee DINOv2 model with single GPU or DistributedDataParallel."""

import atexit
import argparse
import json
import math
import os
import random
from pathlib import Path

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

from rsna_data import DEFAULT_CROP_MM, DEFAULT_IMAGE_SIZE, DEFAULT_ROOT, LABELS, KneeDataset, load_metadata
from rsna_model import RSNADINOv2


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--labels-csv", type=Path, default=SCRIPT_DIR / "llm_labels_v4_blend.csv", help="13列公开软标签文件")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1, help="每张GPU的batch size")
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4, help="每个DDP进程的worker数量")
    parser.add_argument("--num-slices", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--crop-mm", type=float, default=DEFAULT_CROP_MM, help="中心 physical crop 的边长（毫米）")
    parser.add_argument("--backbone", default="dinov2_vits14")
    parser.add_argument("--local-dinov2-repo", type=Path, default=SCRIPT_DIR / "dinov2", help="本地DINOv2源码目录，不访问GitHub")
    parser.add_argument("--backbone-weights", type=Path, default=SCRIPT_DIR / "dinov2_pretrain_weights" / "dinov2_vits14_pretrain.pth", help="本地DINOv2预训练权重")
    parser.add_argument("--backbone-mode", choices=["frozen", "last2", "last4", "full"], default="frozen")
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--encoder-chunk-size", type=int, default=24)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="只加载模型权重，用于冻结阶段后微调")
    parser.add_argument("--resume", type=Path, default=None, help="恢复同配置训练，包括优化器和调度器")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    return parser.parse_args()


def setup_distributed():
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
        atexit.register(cleanup_distributed)
        return True, local_rank, dist.get_rank(), dist.get_world_size()
    return False, 0, 0, 1


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed, rank=0):
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_label_column(series):
    text = series.astype(str).str.strip().str.upper()
    mapped = text.map({"YES": 1.0, "Y": 1.0, "TRUE": 1.0, "NO": 0.0, "N": 0.0, "FALSE": 0.0, "UNK": 0.5, "UNKNOWN": 0.5, "NAN": np.nan, "": np.nan})
    numeric = pd.to_numeric(series, errors="coerce")
    values = mapped.where(mapped.notna(), numeric)
    return values.where(values.between(0.0, 1.0), np.nan).astype(np.float32)


def resolve_labels_csv(path, data_root):
    candidates = [path, Path.cwd() / path.name, Path(data_root) / path.name, Path(data_root).parent / path.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot find {path.name}; pass its full path with --labels-csv")


def prepare_studies(train_df, labels_csv):
    labels_df = pd.read_csv(labels_csv)
    missing = [column for column in ["StudyInstanceUID", *LABELS] if column not in labels_df.columns]
    if missing:
        raise ValueError(f"Weak-label CSV missing columns: {missing}")
    studies = train_df[["StudyInstanceUID", *LABELS]].rename(columns={label: f"gold__{label}" for label in LABELS})
    studies = studies.merge(labels_df[["StudyInstanceUID", *LABELS]], on="StudyInstanceUID", how="inner")
    for label in LABELS:
        studies[label] = parse_label_column(studies[label])
        gold_column = f"gold__{label}"
        studies[gold_column] = pd.to_numeric(studies[gold_column], errors="coerce").astype(np.float32)
        has_gold = studies[gold_column].notna()
        studies.loc[has_gold, label] = studies.loc[has_gold, gold_column]
    return studies[studies[LABELS].notna().any(axis=1)].reset_index(drop=True)

def distributed_metadata(root, distributed, rank):
    objects = [load_metadata(root) if rank == 0 else None]
    if distributed:
        dist.broadcast_object_list(objects, src=0)
    return objects[0]


def make_loader(dataset, batch_size, workers, sampler, shuffle, drop_last):
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle if sampler is None else False,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None, drop_last=drop_last,
    )


def build_model(args, distributed, rank):
    kwargs = dict(
        backbone_name=args.backbone, pretrained=False,
        local_repo=str(args.local_dinov2_repo), weights_path=args.backbone_weights,
        num_slices=args.num_slices, freeze_backbone=True,
        encoder_chunk_size=args.encoder_chunk_size,
    )
    if distributed and rank != 0:
        dist.barrier()
    model = RSNADINOv2(**kwargs)
    if distributed and rank == 0:
        dist.barrier()

    if args.backbone_mode == "last2":
        model.unfreeze_last_blocks(2)
    elif args.backbone_mode == "last4":
        model.unfreeze_last_blocks(4)
    elif args.backbone_mode == "full":
        model.freeze_backbone(False)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint.get("model", checkpoint))
    return model


def make_optimizer(model, args):
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
    backbone = [parameter for parameter in model.parameters() if id(parameter) in backbone_ids and parameter.requires_grad]
    head = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids and parameter.requires_grad]
    groups = [{"params": head, "lr": args.head_lr}]
    if backbone:
        groups.append({"params": backbone, "lr": args.backbone_lr})
    return AdamW(groups, weight_decay=args.weight_decay)


def make_scheduler(optimizer, total_steps, warmup_ratio):
    warmup_steps = int(total_steps * warmup_ratio)

    def schedule(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    return LambdaLR(optimizer, schedule)


def calculate_pos_weight(studies):
    values = studies[LABELS].to_numpy(dtype=np.float32)
    confidence = np.where(np.isnan(values), 0, 2 * np.abs(values - 0.5))
    binary = np.nan_to_num(values, nan=0.5) > 0.5
    positives = np.sum(confidence * binary, axis=0)
    negatives = np.sum(confidence * ~binary, axis=0)
    return torch.tensor(np.clip(negatives / np.maximum(positives, 1), 1, 10), dtype=torch.float32)


def masked_bce(logits, targets, mask, label_weight, pos_weight):
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    weight = mask * label_weight
    return (loss * weight).sum() / weight.sum().clamp_min(1)


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def autocast_context(args, device):
    enabled = args.amp != "none" and device.type == "cuda"
    dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def train_epoch(model, loader, sampler, optimizer, scheduler, scaler, pos_weight, device, args, epoch, rank):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)
    optimizer.zero_grad(set_to_none=True)
    total_loss, steps = 0.0, 0

    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        with autocast_context(args, device):
            logits = model(batch["images"], batch["slot_mask"], batch["series_features"])
            loss = masked_bce(logits, batch["targets"], batch["label_mask"], batch["label_weight"], pos_weight)
            scaled_loss = loss / args.accum_steps
        scaler.scale(scaled_loss).backward()

        update = (step + 1) % args.accum_steps == 0 or step + 1 == len(loader)
        if update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.detach().item()
        steps += 1
        if rank == 0 and (step + 1) % args.log_interval == 0:
            print(f"epoch={epoch + 1} step={step + 1}/{len(loader)} loss={total_loss / steps:.5f}", flush=True)

    stats = torch.tensor([total_loss, steps], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (stats[0] / stats[1]).item()


@torch.no_grad()
def validate(model, loader, device, args, distributed, world_size):
    model.eval()
    local = {"uid": [], "pred": [], "target": [], "mask": [], "weight": []}
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast_context(args, device):
            logits = model(batch["images"], batch["slot_mask"], batch["series_features"])
        local["uid"].extend(batch["study_uid"])
        local["pred"].append(torch.sigmoid(logits).float().cpu().numpy())
        local["target"].append(batch["targets"].float().cpu().numpy())
        local["mask"].append(batch["label_mask"].float().cpu().numpy())
        local["weight"].append(batch["label_weight"].float().cpu().numpy())
    for key in ["pred", "target", "mask", "weight"]:
        local[key] = np.concatenate(local[key])

    gathered = [None] * world_size if distributed else [local]
    if distributed:
        dist.all_gather_object(gathered, local)

    merged = {}
    for part in gathered:
        for index, uid in enumerate(part["uid"]):
            merged[uid] = (part["pred"][index], part["target"][index], part["mask"][index], part["weight"][index])
    uids = list(merged)
    predictions = np.stack([value[0] for value in merged.values()])
    targets = np.stack([value[1] for value in merged.values()])
    masks = np.stack([value[2] for value in merged.values()])
    weights = np.stack([value[3] for value in merged.values()])

    aucs = {}
    for index, label in enumerate(LABELS):
        valid = (masks[:, index] > 0) & (weights[:, index] > 0)
        y_true, y_pred = (targets[valid, index] > 0.5).astype(np.int64), predictions[valid, index]
        aucs[label] = float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) == 2 else float("nan")
    macro_auc = float(np.nanmean(list(aucs.values())))

    return macro_auc, aucs

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_auc, args):
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save({
        "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "epoch": epoch, "best_auc": best_auc, "args": vars(args),
    }, path)


def main():
    args = parse_args()
    distributed, local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed, rank)
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"

    metadata = distributed_metadata(args.data_root, distributed, rank)
    train_df, train_series_df = metadata[0], metadata[1]
    args.labels_csv = resolve_labels_csv(args.labels_csv, args.data_root)
    args.local_dinov2_repo = args.local_dinov2_repo.expanduser().resolve()
    args.backbone_weights = args.backbone_weights.expanduser().resolve()
    if not args.local_dinov2_repo.is_dir():
        raise FileNotFoundError(f"Local DINOv2 repository not found: {args.local_dinov2_repo}")
    if not args.backbone_weights.is_file():
        raise FileNotFoundError(f"Local DINOv2 weights not found: {args.backbone_weights}")
    studies = prepare_studies(train_df, args.labels_csv)
    gold_columns = [f"gold__{label}" for label in LABELS]
    is_gold = studies[gold_columns].notna().all(axis=1)
    train_studies = studies[~is_gold].reset_index(drop=True)
    valid_studies = studies[is_gold].reset_index(drop=True)
    for label in LABELS:
        valid_studies[label] = valid_studies[f"gold__{label}"].astype(np.float32)
    if len(valid_studies) != 58:
        raise ValueError(f"Expected 58 fully labeled validation studies, found {len(valid_studies)}")
    if rank == 0:
        print(f"labels={args.labels_csv}")
        print(f"DINOv2 code={args.local_dinov2_repo}")
        print(f"DINOv2 weights={args.backbone_weights}")
        print(f"split: weak-label train={len(train_studies)} | gold-label valid={len(valid_studies)} | total={len(studies)}")
        print(f"input: crop={args.crop_mm:g} mm | resize={args.image_size}x{args.image_size} | slices={args.num_slices}")

    train_dataset = KneeDataset(train_studies, train_series_df, args.num_slices, args.image_size, args.crop_mm)
    valid_dataset = KneeDataset(valid_studies, train_series_df, args.num_slices, args.image_size, args.crop_mm)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False) if distributed else None
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False, drop_last=False) if distributed else None
    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, train_sampler, True, False)
    valid_loader = make_loader(valid_dataset, args.batch_size, args.num_workers, valid_sampler, False, False)

    model = build_model(args, distributed, rank).to(device)
    resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if resume_checkpoint:
        model.load_state_dict(resume_checkpoint["model"])
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    raw_model = model.module if isinstance(model, DDP) else model
    optimizer = make_optimizer(raw_model, args)
    updates_per_epoch = math.ceil(len(train_loader) / args.accum_steps)
    scheduler = make_scheduler(optimizer, args.epochs * updates_per_epoch, args.warmup_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16" and device.type == "cuda")
    start_epoch, best_auc = 0, -float("inf")
    if resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        scaler.load_state_dict(resume_checkpoint["scaler"])
        start_epoch, best_auc = resume_checkpoint["epoch"] + 1, resume_checkpoint["best_auc"]

    pos_weight = calculate_pos_weight(train_studies).to(device) if not args.no_pos_weight else torch.ones(len(LABELS), device=device)
    output_dir = args.output_dir
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.concat([
            train_studies[["StudyInstanceUID"]].assign(split="train"),
            valid_studies[["StudyInstanceUID"]].assign(split="valid"),
        ], ignore_index=True).to_csv(output_dir / "split.csv", index=False)
        print("trainable parameters:", sum(parameter.numel() for parameter in raw_model.parameters() if parameter.requires_grad))

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, train_sampler, optimizer, scheduler, scaler, pos_weight, device, args, epoch, rank)
        macro_auc, aucs = validate(model, valid_loader, device, args, distributed, world_size)
        if rank == 0:
            print(f"epoch={epoch + 1} train_loss={train_loss:.5f} val_macro_auc={macro_auc:.5f}")
            print(json.dumps(aucs, ensure_ascii=False, indent=2))
            improved = macro_auc > best_auc
            best_auc = max(best_auc, macro_auc)
            save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, best_auc, args)
            if improved:
                save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, best_auc, args)
        if distributed:
            dist.barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
