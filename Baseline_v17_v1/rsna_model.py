# -*- coding: utf-8 -*-
"""v1 骨干及层次 Transformer，五slot原始相邻候选与逐标签聚合。"""

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel

from rsna_data import LABELS, SERIES_FEATURES, SLOTS

ARCHITECTURE = "baseline_v17_v1_80_candidates_original_augmented_ema"


# 整个模型的数据流：Study 内各序列的切片 → DINOv2 → 切片聚合 → 序列融合 → 12 个标签的 logit。
# 这里的“切片注意力”“序列 Transformer”“标签交叉注意力”是三个不同层级的操作。
def load_backbone(model_dir):
    """Load the local Kaggle/Hugging Face DINOv2 model without network access."""
    model_dir = Path(model_dir)
    # 先明确检查本地权重是否齐全，避免 from_pretrained 隐式尝试联网。
    required = [model_dir / "config.json", model_dir / "pytorch_model.bin"]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"DINOv2 model directory {model_dir} is missing {missing}")
    return AutoModel.from_pretrained(str(model_dir), local_files_only=True)


class RSNADINOv2(nn.Module):
    """原始相邻窗口 + 序列内物理位置 + 逐标签窗口/slot 聚合。"""

    def __init__(
        self,
        model_dir=None,
        hidden_dim=256,
        num_heads=8,
        slice_layers=2,
        slot_layers=1,
        dropout=0.1,
        freeze_backbone=True,
        encoder_chunk_size=24,
    ):
        super().__init__()
        # DINOv2 对每张 2.5D 图像独立编码；它不直接处理整段 MRI 序列。
        self.backbone = load_backbone(model_dir)

        if encoder_chunk_size < 1:
            raise ValueError("encoder_chunk_size must be positive")
        # 一次编码的图像张数，控制显存占用，不改变模型结构。
        self.encoder_chunk_size = encoder_chunk_size
        self.freeze_backbone(freeze_backbone)
        backbone_dim = self.backbone.config.hidden_size

        # 2.5D 三通道输入沿用 ImageNet 均值/标准差；buffer 会随模型移动设备，但不训练、不存 checkpoint。
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        # CLS 特征由 DINOv2 维度投影到后续 Transformer 使用的 hidden_dim。
        self.slice_projection = nn.Sequential(nn.LayerNorm(backbone_dim), nn.Linear(backbone_dim, hidden_dim))
        # 随机窗口数/位置可变，编码原始 anchor 物理深度，不使用随机抽取次序。
        self.window_position = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

        # 切片级自注意力：同一序列内的切片可以交换信息，输入形状为 [B×槽位数, S, hidden_dim]。
        slice_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout, batch_first=True, norm_first=True)
        self.slice_encoder = nn.TransformerEncoder(slice_layer, slice_layers, enable_nested_tensor=False)
        # 每个标签独立选择窗口；常数 bias 会被窗口 softmax 抵消。
        self.slice_attention = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, len(LABELS), bias=False))

        # 将 TE/TR、像素间距等 11 维序列元数据映射到和图像特征相同的维度。
        self.metadata_projection = nn.Sequential(
            nn.Linear(len(SERIES_FEATURES), hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
        )
        # 五个 slot 各有身份向量，区分方位与 Fluid Sensitive 选择偏好。
        self.slot_embedding = nn.Parameter(torch.zeros(1, len(SLOTS), hidden_dim))
        # 序列级自注意力：让同一 Study 中有效的不同序列相互交换信息。
        slot_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout, batch_first=True, norm_first=True)
        self.slot_encoder = nn.TransformerEncoder(slot_layer, slot_layers, enable_nested_tensor=False)

        # 每个异常标签都有独立 query，用交叉注意力从各序列读取与该标签有关的信息。
        self.label_queries = nn.Parameter(torch.empty(len(LABELS), hidden_dim))
        self.label_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        # 标签特征归一化后，与逐标签权重做点积，得到未经过 sigmoid 的 12 个 logit。
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.label_weight = nn.Parameter(torch.empty(len(LABELS), hidden_dim))
        self.label_bias = nn.Parameter(torch.zeros(len(LABELS)))
        nn.init.trunc_normal_(self.slot_embedding, std=0.02)
        nn.init.trunc_normal_(self.label_queries, std=0.02)
        nn.init.trunc_normal_(self.label_weight, std=0.02)

    def freeze_backbone(self, freeze=True):
        # 只改变 DINOv2 参数的 requires_grad；后续聚合与预测头始终可训练。
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not freeze

    def unfreeze_last_blocks(self, num_blocks=2):
        # 先冻结整个 backbone，再开放最后 num_blocks 个编码层和最终 LayerNorm。
        self.freeze_backbone(True)
        for block in self.backbone.encoder.layer[-num_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in self.backbone.layernorm.parameters():
            parameter.requires_grad = True

    def encode_images(self, images):
        use_grad = any(parameter.requires_grad for parameter in self.backbone.parameters())
        with nullcontext() if use_grad else torch.no_grad():
            chunks = [self.backbone(pixel_values=(chunk - self.image_mean) / self.image_std).last_hidden_state[:, 0]
                      for chunk in images.split(self.encoder_chunk_size)]
        return torch.cat(chunks)

    def aggregate_windows(self, features, positions, batch_indices, slot_indices, batch_size):
        """仅有效 slot 进入 slice Transformer，特征补齐处同时屏蔽 Transformer 和池化。"""
        # 输入 features: [N,256]；其余三个索引/位置张量均为 [N]，N 是整个 batch 的有效窗口总数。
        # 用 Study 编号和 slot 编号确定分组；不同 Study 或不同 slot 的窗口不在这里交换信息。
        group_ids = batch_indices * len(SLOTS) + slot_indices
        # 即使调用方重排窗口，也恢复序列内空间顺序。
        order = positions.argsort(stable=True)
        # 稳定排序先按物理位置、再按分组，使同组窗口连续排列并保留组内空间顺序。
        order = order[group_ids[order].argsort(stable=True)]
        group_ids = group_ids[order]
        # 原始 anchor 的归一化物理深度 [N,1] 经 MLP 变为 [N,256]，与图像特征相加。
        positioned = features + self.window_position(positions.to(features.dtype).unsqueeze(-1))
        positioned = positioned[order]
        # 每个 Study-slot 的窗口数可能不同；只让至少有一个窗口的 R 个分组进入 Transformer。
        counts = torch.bincount(group_ids, minlength=batch_size * len(SLOTS))
        active = counts.nonzero(as_tuple=False).squeeze(1)
        active_counts = counts[active]
        # W 是当前 batch 中单个 slot 的最大窗口数，仅把特征补齐到 W，不补齐或编码图像。
        width = int(active_counts.max())
        # 根据每组起点，将连续的窗口特征映射到二维的“分组行号、组内列号”。
        starts = active_counts.cumsum(0) - active_counts
        row_indices = torch.repeat_interleave(torch.arange(len(active), device=features.device), active_counts)
        local_indices = torch.arange(len(features), device=features.device) - torch.repeat_interleave(starts, active_counts)
        flat_indices = row_indices * width + local_indices
        # 补齐后的特征为 [R,W,256]；mask: [R,W]，True 表示真实窗口。
        padded = features.new_zeros(len(active) * width, features.shape[-1])
        padded = padded.index_copy(0, flat_indices, positioned).view(len(active), width, -1)
        mask = torch.arange(width, device=features.device)[None, :] < active_counts[:, None]
        # 两层 slice Transformer 在各 slot 内建模窗口关系；padding 不参与注意力读取。
        contextual = self.slice_encoder(padded, src_key_padding_mask=~mask)
        # 每个窗口输出 12 个标签分数 [R,W,12]；padding 设为 -inf，池化权重为零。
        scores = self.slice_attention(contextual).masked_fill(~mask[..., None], -torch.inf)
        # 每个标签独立在窗口维 W 上做 softmax，再加权求和，得到 [R,12,256]。
        # 同一 slot 因而有 12 个向量，而不是所有标签共用一个序列向量。
        pooled = torch.einsum("rwl,rwd->rld", scores.softmax(dim=1), contextual)
        # 将有效分组放回完整结构 [B,5,12,256]；缺失 slot 保留零特征并返回有效 mask [B,5]。
        slots = features.new_zeros(batch_size * len(SLOTS), len(LABELS), features.shape[-1])
        slots = slots.index_copy(0, active, pooled).view(batch_size, len(SLOTS), len(LABELS), -1)
        return slots, counts.view(batch_size, len(SLOTS)) > 0

    def forward(self, images, slot_mask, series_features=None, *, window_positions,
                window_batch_indices, window_slot_indices):
        # images: [N,3,H,W]，三通道是原始相邻切片；slot_mask: [B,5]；元数据: [B,5,11]。
        # 窗口以紧凑形式拼接，三个 [N] 张量分别记录物理位置、所属 Study 和所属 slot。
        if images.ndim != 4 or images.shape[1] != 3 or not len(images):
            raise ValueError("Expected nonempty packed windows [N,3,H,W]")
        if slot_mask.ndim != 2 or slot_mask.shape[1] != len(SLOTS):
            raise ValueError("Expected slot_mask [B,5]")
        batch_size = len(slot_mask)
        # 每个 Study 至少有一个有效 slot，且每张图像都必须有完整、合法的窗口映射。
        if not bool(slot_mask.bool().any(dim=1).all()):
            raise ValueError("Every study must have at least one valid slot")
        for values in (window_positions, window_batch_indices, window_slot_indices):
            if values.shape != (len(images),):
                raise ValueError("Window maps must match the packed images")
        if (bool((window_batch_indices < 0).any()) or bool((window_batch_indices >= batch_size).any()) or
            bool((window_slot_indices < 0).any()) or bool((window_slot_indices >= len(SLOTS)).any())):
            raise ValueError("Invalid window batch/slot index")
        # DINOv2 独立编码每个窗口的 CLS，再通过 LayerNorm + Linear 投影为 [N,256]。
        features = self.slice_projection(self.encode_images(images))
        # 序列内 Transformer + 逐标签窗口池化，得到 slot_features: [B,5,12,256]。
        slot_features, present = self.aggregate_windows(
            features, window_positions, window_batch_indices, window_slot_indices, batch_size
        )
        # 核对实际窗口分组与数据提供的 slot_mask，防止有效 slot 没有窗口或缺失 slot 混入窗口。
        if not torch.equal(present, slot_mask.bool()):
            raise ValueError("Every valid slot must have a window; invalid slots cannot contain windows")
        if series_features is not None:
            # 11 维元数据投影为 [B,5,256]，扩展标签维后加到该 slot 的全部 12 个标签特征上。
            slot_features = slot_features + self.metadata_projection(series_features).unsqueeze(2)
        # slot 身份向量区分五种序列角色，同一个 slot 的身份信息由各标签共享。
        slot_features = slot_features + self.slot_embedding.unsqueeze(2)
        # [B,5,12,256] → [B,12,5,256] → [B×12,5,256]，每个 Study-label 独立融合五个 slot。
        label_slots = slot_features.permute(0, 2, 1, 3).reshape(batch_size * len(LABELS), len(SLOTS), -1)
        # 缺失 slot 的 mask 扩展到所有标签，True 表示 Transformer/交叉注意力应忽略该 slot。
        label_mask = (~present)[:, None].expand(batch_size, len(LABELS), len(SLOTS))
        label_mask = label_mask.reshape(batch_size * len(LABELS), len(SLOTS))
        # 一层 slot Transformer：各标签的五个 slot 分别交换信息，但共用同一套网络参数。
        label_slots = self.slot_encoder(label_slots, src_key_padding_mask=label_mask)
        # 12 个 query 是可学习参数；每个标签取自己的 query，形成 [B×12,1,256]。
        queries = self.label_queries[None, :, None].expand(batch_size, -1, -1, -1)
        queries = queries.reshape(batch_size * len(LABELS), 1, -1)
        # 标签 query 通过交叉注意力读取自己的五个 slot，输出 [B×12,1,256] 标签特征。
        label_features, _ = self.label_attention(queries, label_slots, label_slots,
                                                key_padding_mask=label_mask, need_weights=False)
        # 恢复 [B,12,256] 并归一化，再与逐标签分类权重做点积，返回 logits [B,12]。
        label_features = self.output_norm(label_features.view(batch_size, len(LABELS), -1))
        return (label_features * self.label_weight[None]).sum(-1) + self.label_bias


def predict_batch(model, batch, use_metadata=True):
    return model(batch["images"], batch["slot_mask"], batch["series_features"] if use_metadata else None,
                 window_positions=batch["window_positions"], window_batch_indices=batch["window_batch_indices"],
                 window_slot_indices=batch["window_slot_indices"])


__all__ = ["ARCHITECTURE", "RSNADINOv2", "predict_batch"]
