# -*- coding: utf-8 -*-
"""Train Baseline_v22: v14 data/EMA with DINOv2 attention-MIL head."""

import atexit
from copy import deepcopy
from datetime import timedelta
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from rsna_data import (DEFAULT_CROP_MM, DEFAULT_IMAGE_SIZE, DEFAULT_ROOT, LABELS, SLOTS,
                       CANDIDATE_BUDGETS, KneeDataset, collate_studies, load_metadata,
                       add_series_quality, make_series_selection_config, validate_series_selection_config,
                       series_selection_report, QUALITY_COLUMNS)
from rsna_model import ARCHITECTURE, RSNADINOv2, predict_batch


SCRIPT_DIR = Path(__file__).resolve().parent



class ModelEMA:
    """每次成功的 optimizer step 后更新独立、无梯度的 EMA 模型。"""

    def __init__(self, model, decay=0.999):
        raw_model = model.module if isinstance(model, DDP) else model
        self.model = deepcopy(raw_model).eval()
        self.model.requires_grad_(False)
        self.decay = decay
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model):
        raw_model = model.module if isinstance(model, DDP) else model
        self.num_updates += 1
        # 初始影子权重迅速跟上训练；随后逐步增大平滑强度。
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for name, parameter in raw_model.named_parameters():
            target = self.model.get_parameter(name)
            if parameter.requires_grad and target.is_floating_point():
                target.lerp_(parameter.detach(), 1 - decay)
            else:
                target.copy_(parameter.detach())
        # buffer 包含非持久化归一化常数；直接复制，整数计数器不做平均。
        for name, buffer in raw_model.named_buffers():
            self.model.get_buffer(name).copy_(buffer.detach())

    def state_dict(self):
        return {"model": self.model.state_dict(), "num_updates": self.num_updates, "decay": self.decay}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"], strict=True)
        self.num_updates = state["num_updates"]
        self.decay = state["decay"]


def format_duration(seconds):
    """将秒数格式化为适合训练日志阅读的 HH:MM:SS。"""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# 命令行参数集中在这里；下面不逐条解释 add_argument，重点注释实际执行流程。
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--labels-csv", type=Path, default=SCRIPT_DIR / "label.csv", help="训练软标签 CSV（含 StudyInstanceUID 和 12 个标签列）")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2, help="每张GPU的batch size")
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4, help="每个DDP进程的worker数量")
    parser.add_argument("--train-windows", type=int, default=24, help="每个 Study 每次训练的随机窗口总数；至少5")
    parser.add_argument("--span-lo", type=float, default=0.02)
    parser.add_argument("--span-hi", type=float, default=0.98)
    parser.add_argument("--cache-dir", type=Path, default=SCRIPT_DIR / "input_cache", help="唯一切片与窗口索引缓存，所有实验可共用")
    parser.add_argument("--no-cache", action="store_true", help="关闭磁盘缓存，每次重新读取 DICOM")
    parser.add_argument("--series-quality-workers", type=int, default=8, help="rank 0 的 header 质量扫描线程数")
    parser.add_argument("--coverage-quantile", type=float, default=0.2, help="仅由弱标签训练 Study 拟合的各平面 coverage 分位数")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--crop-mm", type=float, default=DEFAULT_CROP_MM, help="中心 physical crop 的边长（毫米）")
    parser.add_argument("--dinov2-model-dir", type=Path, default=SCRIPT_DIR.parent / "dinov2-pytorch-small-v1", help="Kaggle DINOv2 模型目录，包含 config.json 和 pytorch_model.bin")
    parser.add_argument("--backbone-mode", choices=["frozen", "last2", "last4", "last6", "full"], default="frozen")
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--encoder-chunk-size", type=int, default=24)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument("--no-metadata", action="store_true", default=True, help="兼容 v14 命令；v22 MIL 不使用 metadata")
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="只加载模型权重，用于冻结阶段后微调")
    parser.add_argument("--resume", type=Path, default=None, help="恢复同配置训练，包括优化器和调度器")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay 上限；早期按更新次数 warmup")
    parser.add_argument("--no-ema", action="store_true", help="关闭 EMA，使用原始训练权重验证/推理")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    args = parser.parse_args()
    if args.train_windows < len(SLOTS) or not 0 <= args.span_lo < args.span_hi <= 1:
        parser.error("--train-windows must be >=5, and 0 <= span-lo < span-hi <= 1")
    if not 0 <= args.ema_decay < 1:
        parser.error("--ema-decay must be in [0, 1)")
    if args.series_quality_workers < 1 or not 0 <= args.coverage_quantile <= 1:
        parser.error("--series-quality-workers must be >=1 and --coverage-quantile in [0,1]")
    if args.init_checkpoint and args.resume:
        parser.error("--init-checkpoint and --resume are mutually exclusive")
    return args


def setup_distributed():
    # torchrun 会注入 RANK/WORLD_SIZE/LOCAL_RANK；普通 python 启动则按单进程运行。
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        # 每个进程只使用自己的 GPU，并通过 NCCL 与其他进程通信。
        torch.cuda.set_device(local_rank)
        # 首次 rank 0 需要扫描全部序列 header；其余 rank 等待广播期间允许较长准备时间。
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"), timeout=timedelta(hours=2))
        atexit.register(cleanup_distributed)
        return True, local_rank, dist.get_rank(), dist.get_world_size()
    return False, 0, 0, 1


def cleanup_distributed():
    # 退出时释放进程组，防止分布式资源残留。
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed, rank=0):
    # 不同 rank 使用不同随机序列，同时同一配置重复运行仍可复现种子设置。
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_label_column(series):
    # 兼容 YES/NO/UNK 文本及 [0,1] 数值软标签；非法或越界值统一视为缺失。
    text = series.astype(str).str.strip().str.upper()
    mapped = text.map({"YES": 1.0, "Y": 1.0, "TRUE": 1.0, "NO": 0.0, "N": 0.0, "FALSE": 0.0, "UNK": 0.5, "UNKNOWN": 0.5, "NAN": np.nan, "": np.nan})
    numeric = pd.to_numeric(series, errors="coerce")
    values = mapped.where(mapped.notna(), numeric)
    return values.where(values.between(0.0, 1.0), np.nan).astype(np.float32)


def resolve_labels_csv(path, data_root):
    # 标签文件可能位于脚本目录、当前目录、数据目录或其上级，按顺序查找。
    candidates = [path, Path.cwd() / path.name, Path(data_root) / path.name, Path(data_root).parent / path.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot find {path.name}; pass its full path with --labels-csv")


def prepare_studies(train_df, labels_csv):
    # 将弱标签与官方训练表按 StudyInstanceUID 对齐，保留两者以便划分。
    labels_df = pd.read_csv(labels_csv)
    missing = [column for column in ["StudyInstanceUID", *LABELS] if column not in labels_df.columns]
    if missing:
        raise ValueError(f"Weak-label CSV missing columns: {missing}")
    studies = train_df[["StudyInstanceUID", *LABELS]].rename(columns={label: f"gold__{label}" for label in LABELS})
    studies = studies.merge(labels_df[["StudyInstanceUID", *LABELS]], on="StudyInstanceUID", how="inner")
    for label in LABELS:
        # 有官方真值时覆盖弱标签；否则保留弱标签作为训练目标。
        studies[label] = parse_label_column(studies[label])
        gold_column = f"gold__{label}"
        studies[gold_column] = pd.to_numeric(studies[gold_column], errors="coerce").astype(np.float32)
        has_gold = studies[gold_column].notna()
        studies.loc[has_gold, label] = studies.loc[has_gold, gold_column]
    # 完全没有可用标签的 Study 无法提供监督信号，直接剔除。
    return studies[studies[LABELS].notna().any(axis=1)].reset_index(drop=True)

def distributed_metadata(root, distributed, rank, quality_cache_dir=None, quality_workers=8):
    # 仅 rank 0 读取 CSV，再广播给其他进程，避免每张 GPU 重复读取元数据。
    objects = [load_metadata(root) if rank == 0 else None]
    if rank == 0:
        metadata = list(objects[0])
        metadata[1] = add_series_quality(metadata[1], quality_cache_dir, quality_workers)
        objects[0] = tuple(metadata)
    if distributed:
        dist.broadcast_object_list(objects, src=0)
    return objects[0]


def make_loader(dataset, batch_size, workers, sampler, shuffle, drop_last):
    # DDP 模式由 DistributedSampler 决定各进程样本，因此不能同时开启 DataLoader shuffle。
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle if sampler is None else False,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None, drop_last=drop_last,
        collate_fn=collate_studies,
    )


def build_model(args, distributed, rank):
    # 模型构造先从冻结 DINOv2 开始，随后按 backbone_mode 决定解冻范围。
    kwargs = dict(
        model_dir=str(args.dinov2_model_dir),
        freeze_backbone=True,
        encoder_chunk_size=args.encoder_chunk_size,
    )
    if distributed and rank != 0:
        # 让 rank 0 先完成本地权重加载；其余 rank 随后再加载同一模型目录。
        dist.barrier()
    model = RSNADINOv2(**kwargs)
    if distributed and rank == 0:
        dist.barrier()

    if args.backbone_mode == "last2":
        model.unfreeze_last_blocks(2)
    elif args.backbone_mode == "last4":
        model.unfreeze_last_blocks(4)
    elif args.backbone_mode == "last6":
        model.unfreeze_last_blocks(6)
    elif args.backbone_mode == "full":
        model.freeze_backbone(False)
    if args.init_checkpoint:
        # 阶段性微调仅继承模型权重；不会继承旧优化器状态或 epoch。
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("architecture") != ARCHITECTURE or checkpoint.get("slots") != SLOTS:
            raise ValueError("--init-checkpoint requires matching v22 attention-MIL weights")
        model.load_state_dict(checkpoint["model"], strict=True)
    return model


def make_optimizer(model, args):
    # backbone 和新加的聚合/预测模块分组，分别设置较小/较大的学习率。
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
    backbone = [parameter for parameter in model.parameters() if id(parameter) in backbone_ids and parameter.requires_grad]
    head = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids and parameter.requires_grad]
    groups = [{"params": head, "lr": args.head_lr}]
    if backbone:
        groups.append({"params": backbone, "lr": args.backbone_lr})
    return AdamW(groups, weight_decay=args.weight_decay)


def make_scheduler(optimizer, total_steps, warmup_ratio):
    # 学习率先线性 warmup，再按余弦曲线逐步衰减到零。
    warmup_steps = int(total_steps * warmup_ratio)

    def schedule(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    return LambdaLR(optimizer, schedule)


def calculate_pos_weight(studies):
    # 用训练集软标签的置信度估计各类正负比例，正类权重限制在 [1,10]。
    values = studies[LABELS].to_numpy(dtype=np.float32)
    confidence = np.where(np.isnan(values), 0, 2 * np.abs(values - 0.5))
    binary = np.nan_to_num(values, nan=0.5) > 0.5
    # 0.5 标签贡献零；靠近 0/1 的标签对统计贡献更大。
    positives = np.sum(confidence * binary, axis=0)
    negatives = np.sum(confidence * ~binary, axis=0)
    return torch.tensor(np.clip(negatives / np.maximum(positives, 1), 1, 10), dtype=torch.float32)


def masked_bce(logits, targets, mask, label_weight, pos_weight):
    # 逐标签 BCE；mask 排除缺失标签，label_weight 降低不确定软标签的影响。
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    weight = mask * label_weight
    # 按有效权重之和归一化；全零权重时 clamp 防止除零。
    return (loss * weight).sum() / weight.sum().clamp_min(1)


def move_batch(batch, device):
    # 只移动张量；Study UID 等字符串仍留在 CPU，供验证结果去重。
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def autocast_context(args, device):
    # CUDA 上可用 bf16/fp16 自动混合精度；CPU 或 amp=none 时禁用。
    enabled = args.amp != "none" and device.type == "cuda"
    dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def train_epoch(model, loader, sampler, optimizer, scheduler, scaler, pos_weight, device, args, epoch, rank, ema=None):
    # 一轮训练：前向 → 带掩码损失 → 梯度累积 → 裁剪 → 参数与学习率更新。
    model.train()
    loader.dataset.set_epoch(epoch)
    if sampler is not None:
        # 每轮改变 DDP 采样顺序，但各 rank 之间仍按规则分片。
        sampler.set_epoch(epoch)
    optimizer.zero_grad(set_to_none=True)
    total_loss, steps = 0.0, 0
    step_losses = []
    epoch_start = time.perf_counter()

    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        with autocast_context(args, device):
            # 统一传入窗口位置和 Study/slot 索引；--no-metadata 时不执行元数据分支。
            # 模型输出 [B,12] logits；targets/mask/权重与 12 个标签一一对应。
            logits = predict_batch(model, batch, use_metadata=not args.no_metadata)
            loss = masked_bce(logits, batch["targets"], batch["label_mask"], batch["label_weight"], pos_weight)
            # 累积多次小 batch 的梯度，近似更大的有效 batch。
            scaled_loss = loss / args.accum_steps
        scaler.scale(scaled_loss).backward()

        # 达到累积步数或来到 epoch 最后一个 batch 时才真正更新参数。
        update = (step + 1) % args.accum_steps == 0 or step + 1 == len(loader)
        if update:
            # fp16 下先还原梯度尺度，再裁剪；bf16/none 的 scaler 是空操作。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # fp16 溢出会跳过 optimizer step，此时 EMA 和 scheduler 均不推进。
            step_succeeded = scaler.get_scale() >= scale_before
            if step_succeeded and ema is not None:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)
            # 调度器按优化器更新次数推进，而不是按每个 DataLoader batch 推进。
            if step_succeeded:
                scheduler.step()

        loss_value = loss.detach().item()
        total_loss += loss_value
        steps += 1
        if rank == 0:
            step_losses.append(loss_value)
        if rank == 0 and ((step + 1) % args.log_interval == 0 or step + 1 == len(loader)):
            elapsed = time.perf_counter() - epoch_start
            seconds_per_step = elapsed / steps
            eta = seconds_per_step * (len(loader) - steps)
            print(
                f"epoch={epoch + 1} step={step + 1}/{len(loader)} "
                f"windows={len(batch['images'])} candidates={int(batch['candidate_count'].sum())} "
                f"loss={total_loss / steps:.5f} elapsed={format_duration(elapsed)} "
                f"step_time={seconds_per_step:.2f}s eta={format_duration(eta)}",
                flush=True,
            )

    stats = torch.tensor([total_loss, steps], device=device, dtype=torch.float64)
    if dist.is_initialized():
        # 合并各 rank 的 loss 总和与 batch 数，得到全局平均训练 loss。
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (stats[0] / stats[1]).item(), step_losses


@torch.no_grad()
def validate(model, loader, device, args, distributed, world_size):
    # 验证阶段不建梯度图；各进程先收集自己的 Study 预测，再汇总算 AUC。
    model.eval()
    local = {"uid": [], "pred": [], "target": [], "mask": [], "weight": []}
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast_context(args, device):
            logits = predict_batch(model, batch, use_metadata=not args.no_metadata)
        # AUC 使用概率排序；sigmoid 不改变同一标签内的排序。
        local["uid"].extend(batch["study_uid"])
        local["pred"].append(torch.sigmoid(logits).float().cpu().numpy())
        local["target"].append(batch["targets"].float().cpu().numpy())
        local["mask"].append(batch["label_mask"].float().cpu().numpy())
        local["weight"].append(batch["label_weight"].float().cpu().numpy())
    for key in ["pred", "target", "mask", "weight"]:
        local[key] = np.concatenate(local[key])

    gathered = [None] * world_size if distributed else [local]
    if distributed:
        # 每个 rank 都拿到完整验证预测；只有 rank 0 最终打印/保存结果。
        dist.all_gather_object(gathered, local)

    # DistributedSampler 可能补齐重复 Study，用 UID 去重，避免验证集重复计分。
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
        # 缺失/零置信度标签不参与该标签的 AUC；软标签以 >0.5 二值化。
        valid = (masks[:, index] > 0) & (weights[:, index] > 0)
        y_true, y_pred = (targets[valid, index] > 0.5).astype(np.int64), predictions[valid, index]
        aucs[label] = float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) == 2 else float("nan")
    # 某标签若只有单一类别，其 AUC 为 NaN；宏平均跳过 NaN。
    macro_auc = float(np.nanmean(list(aucs.values())))

    return macro_auc, aucs


def write_json(path, value):
    # 先写临时文件再替换目标文件，降低训练中断造成 JSON 半写入的风险。
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def plot_loss_curve(history, path):
    """绘制每个训练 step 的原始 loss 和滑动平均曲线。"""
    losses = [
        loss
        for entry in history
        for loss in entry.get("train_loss_by_step", [])
        if loss is not None
    ]
    if not losses:
        return

    steps = np.arange(1, len(losses) + 1)
    smooth_window = min(20, len(losses))
    smooth_losses = np.convolve(losses, np.ones(smooth_window) / smooth_window, mode="valid")
    smooth_steps = steps[smooth_window - 1:]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(steps, losses, linewidth=1, alpha=0.3, label="Step loss")
    axis.plot(smooth_steps, smooth_losses, linewidth=2, label=f"Moving average ({smooth_window} steps)")
    axis.set_xlabel("Global step")
    axis.set_ylabel("Loss")
    axis.set_title("Training Loss Curve")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    temporary_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary_path, dpi=150)
    plt.close(figure)
    temporary_path.replace(path)


def json_score(value):
    # JSON 不支持 NaN/Inf；此处把不可用分数记为 null。
    return float(value) if math.isfinite(value) else None


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_auc, args, history, ema=None):
    # DDP 外壳不属于模型权重，存内部 module 方便单卡/多卡读取。
    raw_model = model.module if isinstance(model, DDP) else model
    ema_state = ema.state_dict() if ema is not None else None
    training_state = raw_model.state_dict()
    torch.save({
        # model 始终对应实际验证/推理权重；续训单独恢复 training_model。
        "model": ema_state["model"] if ema_state is not None else training_state,
        "training_model": training_state, "ema": ema_state,
        "weights_type": "ema" if ema is not None else "raw",
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "epoch": epoch, "best_auc": best_auc, "args": vars(args),
        "validation_history": history,
        "architecture": ARCHITECTURE, "slots": SLOTS, "candidate_budgets": CANDIDATE_BUDGETS,
        "series_selection": args.series_selection,
    }, path)


def validate_resume_checkpoint(checkpoint, args):
    if (checkpoint.get("architecture") != ARCHITECTURE or checkpoint.get("slots") != SLOTS or
        checkpoint.get("candidate_budgets") != CANDIDATE_BUDGETS):
        raise ValueError("--resume requires a matching Baseline_v22 checkpoint")
    validate_series_selection_config(checkpoint.get("series_selection"))
    if checkpoint["series_selection"] != args.series_selection:
        raise ValueError("Resume series selection config differs from current training header statistics")
    saved = checkpoint.get("args", {})
    for name in ("train_windows", "span_lo", "span_hi", "image_size", "crop_mm", "backbone_mode", "no_metadata", "seed", "no_ema", "ema_decay"):
        if saved.get(name) != getattr(args, name):
            raise ValueError(f"Resume configuration mismatch: {name}")

    if "training_model" not in checkpoint:
        raise ValueError("Resume checkpoint missing original training weights")
    if not args.no_ema:
        state = checkpoint.get("ema")
        if state is None or state.get("decay") != args.ema_decay:
            raise ValueError("Resume checkpoint missing EMA state or decay mismatch")


def main():
    # 初始化运行环境：解析参数、选择单卡/DDP、设置每个进程的设备和随机种子。
    args = parse_args()
    distributed, local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed, rank)
    # 允许 CUDA 矩阵计算使用 TF32，以提高支持该格式的 GPU 上的吞吐。
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"

    # 读取影像序列元数据并定位标签 CSV / 本地 DINOv2 权重。
    quality_cache = None if args.no_cache else args.cache_dir / "series_quality"
    metadata = distributed_metadata(args.data_root, distributed, rank, quality_cache, args.series_quality_workers)
    train_df, train_series_df = metadata[0], metadata[1]
    args.labels_csv = resolve_labels_csv(args.labels_csv, args.data_root)
    args.dinov2_model_dir = args.dinov2_model_dir.expanduser().resolve()
    for filename in ("config.json", "pytorch_model.bin"):
        if not (args.dinov2_model_dir / filename).is_file():
            raise FileNotFoundError(f"DINOv2 model file not found: {args.dinov2_model_dir / filename}")
    # 官方 12 标签全部齐全的 Study 作为验证集；其余含弱标签的 Study 作为训练集。
    studies = prepare_studies(train_df, args.labels_csv)
    gold_columns = [f"gold__{label}" for label in LABELS]
    is_gold = studies[gold_columns].notna().all(axis=1)
    train_studies = studies[~is_gold].reset_index(drop=True)
    valid_studies = studies[is_gold].reset_index(drop=True)
    for label in LABELS:
        # 验证目标强制取官方真值，不用合并后的弱标签列。
        valid_studies[label] = valid_studies[f"gold__{label}"].astype(np.float32)
    # 明确约束验证集大小，防止标签文件变化造成无声的数据划分变化。
    if len(valid_studies) != 58:
        raise ValueError(f"Expected 58 fully labeled validation studies, found {len(valid_studies)}")
    # 验证 Study 的图像 header 可用于自身选序列，但不参与覆盖阈值拟合。
    threshold_series = train_series_df[train_series_df.StudyInstanceUID.isin(train_studies.StudyInstanceUID)]
    args.series_selection = make_series_selection_config(threshold_series, args.coverage_quantile)
    if rank == 0:
        print(f"labels={args.labels_csv}")
        print(f"DINOv2 Hugging Face model={args.dinov2_model_dir}")
        print(f"split: weak-label train={len(train_studies)} | gold-label valid={len(valid_studies)} | total={len(studies)}")
        print(f"input: crop={args.crop_mm:g} mm | resize={args.image_size}x{args.image_size} | candidates/slot={CANDIDATE_BUDGETS} | train windows/study={args.train_windows} | validation=all candidates | span={args.span_lo:g}-{args.span_hi:g}")
        print(f"input cache: {'disabled' if args.no_cache else args.cache_dir}")
        print(f"series selection: {args.series_selection}")
        print(f"header geometry usable: {int(train_series_df.quality_usable.sum())}/{len(train_series_df)} series")

    # Dataset 在取样时才加载 DICOM；验证与训练采用相同的图像预处理。
    dataset_kwargs = dict(image_size=args.image_size, crop_mm=args.crop_mm, train_windows=args.train_windows,
                          span_lo=args.span_lo, span_hi=args.span_hi,
                          cache_dir=None if args.no_cache else args.cache_dir, seed=args.seed,
                          series_selection=args.series_selection)
    train_dataset = KneeDataset(train_studies, train_series_df, train=True, **dataset_kwargs)
    valid_dataset = KneeDataset(valid_studies, train_series_df, train=False, **dataset_kwargs)
    # DDP 中每个 rank 只处理一部分 Study；drop_last=False 保证不丢弃尾部样本。
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False) if distributed else None
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False, drop_last=False) if distributed else None
    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, train_sampler, True, False)
    valid_loader = make_loader(valid_dataset, args.batch_size, args.num_workers, valid_sampler, False, False)

    # init_checkpoint 在 build_model 中仅加载权重；resume 则在这里恢复完整训练状态。
    model = build_model(args, distributed, rank).to(device)
    resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if resume_checkpoint:
        validate_resume_checkpoint(resume_checkpoint, args)
        model.load_state_dict(resume_checkpoint["training_model"])
    if distributed:
        # 包装为 DDP 后，各 rank 的梯度会在反向传播时同步。
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.backbone_mode == "full",
        )

    # 优化器使用原始模型参数；总调度步数按每轮实际参数更新次数计算。
    raw_model = model.module if isinstance(model, DDP) else model
    # DDP 构造已同步初始参数；每个 rank 独立维护相同的 EMA。
    ema = None if args.no_ema else ModelEMA(raw_model, args.ema_decay)
    if resume_checkpoint and ema is not None:
        ema.load_state_dict(resume_checkpoint["ema"])
    optimizer = make_optimizer(raw_model, args)
    updates_per_epoch = math.ceil(len(train_loader) / args.accum_steps)
    scheduler = make_scheduler(optimizer, args.epochs * updates_per_epoch, args.warmup_ratio)
    # GradScaler 仅在 fp16 CUDA 训练时启用；bf16 不需要缩放梯度。
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16" and device.type == "cuda")
    start_epoch, best_auc = 0, -float("inf")
    if resume_checkpoint:
        # 完整恢复优化器、调度器、scaler 与已完成轮次，从下一轮继续。
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        scaler.load_state_dict(resume_checkpoint["scaler"])
        start_epoch, best_auc = resume_checkpoint["epoch"] + 1, resume_checkpoint["best_auc"]

    # 类别正例权重只由训练集估计，避免使用验证集标签信息。
    pos_weight = calculate_pos_weight(train_studies).to(device) if not args.no_pos_weight else torch.ones(len(LABELS), device=device)
    output_dir = args.output_dir
    if rank == 0:
        # 仅主进程写文件，避免多卡同时写同一路径。
        output_dir.mkdir(parents=True, exist_ok=True)
        report = series_selection_report(train_series_df, args.series_selection, args.crop_mm, args.image_size)
        report["split"] = report.StudyInstanceUID.map(
            {**dict.fromkeys(train_studies.StudyInstanceUID.astype(str), "train"),
             **dict.fromkeys(valid_studies.StudyInstanceUID.astype(str), "valid")})
        report.to_csv(output_dir / "series_selection.csv", index=False)
        train_series_df[["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", *QUALITY_COLUMNS]].to_csv(
            output_dir / "series_quality.csv", index=False)
        summary = dict(config=args.series_selection, changed_slots=int(report.changed.sum()),
                       changed_studies=int(report.loc[report.changed.eq(1), "StudyInstanceUID"].nunique()),
                       present_slots=int(report.selected_series_uid.ne("").sum()),
                       studies=int(report.StudyInstanceUID.nunique()),
                       per_slot_changed={str(k): int(v) for k, v in report.groupby("slot").changed.sum().items()})
        write_json(output_dir / "series_selection_summary.json", summary)
        print(f"series replacements: {summary['changed_slots']}/{summary['present_slots']} slots, "
              f"{summary['changed_studies']}/{summary['studies']} studies", flush=True)
        hyperparameters = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        write_json(output_dir / "hyperparameters.json", hyperparameters)
        history_path = output_dir / "validation_history.json"
        if resume_checkpoint:
            # 优先从 checkpoint 取历史；旧 checkpoint 若没有该字段，则尝试旁边的 JSON。
            history = resume_checkpoint.get("validation_history")
            if history is None:
                history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
            # 丢弃超出当前续训起点的历史记录，避免重复或未来 epoch 混入。
            history = [entry for entry in history if entry["epoch"] <= start_epoch]
        else:
            history = []
        write_json(history_path, history)
        plot_loss_curve(history, output_dir / "loss_curve.png")
        print(f"validation weights: {'EMA' if ema is not None else 'raw'} | ema_decay={args.ema_decay} | ema_updates={ema.num_updates if ema is not None else 0}")
        print("trainable parameters:", sum(parameter.numel() for parameter in raw_model.parameters() if parameter.requires_grad))

    # 每轮训练后立即验证；按本地 Macro AUC 选择 best.pt。
    training_start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()
        train_loss, step_losses = train_epoch(
            model, train_loader, train_sampler, optimizer, scheduler, scaler,
            pos_weight, device, args, epoch, rank, ema,
        )
        train_seconds = time.perf_counter() - epoch_start
        validation_start = time.perf_counter()
        macro_auc, aucs = validate(ema.model if ema is not None else model, valid_loader, device, args, distributed, world_size)
        validation_seconds = time.perf_counter() - validation_start
        if rank == 0:
            epoch_seconds = time.perf_counter() - epoch_start
            total_seconds = time.perf_counter() - training_start
            print(
                f"epoch={epoch + 1} train_loss={train_loss:.5f} val_macro_auc={macro_auc:.5f} "
                f"train_time={format_duration(train_seconds)} "
                f"val_time={format_duration(validation_seconds)} "
                f"epoch_time={format_duration(epoch_seconds)} "
                f"total_time={format_duration(total_seconds)}",
                flush=True,
            )
            print(json.dumps(aucs, ensure_ascii=False, indent=2))
            improved = macro_auc > best_auc
            best_auc = max(best_auc, macro_auc)
            # 历史中同时保存总体 AUC 与每个标签的 AUC，方便分析类别间差异。
            history.append({
                "epoch": epoch + 1,
                "weights_type": "ema" if ema is not None else "raw",
                "ema_updates": ema.num_updates if ema is not None else 0,
                "train_loss": json_score(train_loss),
                "train_loss_by_step": [json_score(loss) for loss in step_losses],
                "val_macro_auc": json_score(macro_auc),
                "val_auc_by_label": {label: json_score(score) for label, score in aucs.items()},
            })
            write_json(history_path, history)
            plot_loss_curve(history, output_dir / "loss_curve.png")
            # last.pt 每轮覆盖；best.pt 仅在验证指标刷新时覆盖。
            save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, best_auc, args, history, ema)
            if improved:
                save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, best_auc, args, history, ema)
        if distributed:
            # 等待 rank 0 完成记录和 checkpoint，再让各 rank 同步进入下一轮。
            dist.barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
