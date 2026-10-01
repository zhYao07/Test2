"""沿用 v5 网络：slot 共享窗口预算、原始相邻三张输入、层次聚合。"""

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn

from rsna_data import LABELS, SERIES_FEATURES, SLOTS


ARCHITECTURE = "baseline_v6_slot_budget_adjacent"


def load_backbone(model_dir):
    from transformers import AutoModel

    model_dir = Path(model_dir)
    required = [model_dir / "config.json", model_dir / "pytorch_model.bin"]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"DINOv2 model directory {model_dir} is missing {missing}")
    return AutoModel.from_pretrained(str(model_dir), local_files_only=True)


class RSNADINOv2(nn.Module):
    def __init__(self, model_dir=None, hidden_dim=256, num_heads=8,
                 slice_layers=2, slot_layers=1, dropout=0.1,
                 freeze_backbone=True, encoder_chunk_size=24):
        super().__init__()
        if encoder_chunk_size < 1:
            raise ValueError("encoder_chunk_size must be positive")
        self.backbone = load_backbone(model_dir)
        self.encoder_chunk_size = encoder_chunk_size
        self.freeze_backbone(freeze_backbone)
        backbone_dim = self.backbone.config.hidden_size
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.slice_projection = nn.Sequential(nn.LayerNorm(backbone_dim), nn.Linear(backbone_dim, hidden_dim))
        # 输入所属序列内归一化的物理组中心，不依赖固定组数，也不把不同序列连成一条轴。
        self.group_position = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        slice_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4,
                                                dropout, batch_first=True, norm_first=True)
        self.slice_encoder = nn.TransformerEncoder(slice_layer, slice_layers, enable_nested_tensor=False)
        self.slice_attention = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.metadata_projection = nn.Sequential(
            nn.Linear(len(SERIES_FEATURES), hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
        )
        self.slot_embedding = nn.Parameter(torch.zeros(1, len(SLOTS), hidden_dim))
        slot_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4,
                                               dropout, batch_first=True, norm_first=True)
        self.slot_encoder = nn.TransformerEncoder(slot_layer, slot_layers, enable_nested_tensor=False)
        # 保留 v1 的逐标签 query、交叉注意力和输出头。
        self.label_queries = nn.Parameter(torch.empty(len(LABELS), hidden_dim))
        self.label_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.label_weight = nn.Parameter(torch.empty(len(LABELS), hidden_dim))
        self.label_bias = nn.Parameter(torch.zeros(len(LABELS)))
        for parameter in (self.slot_embedding, self.label_queries, self.label_weight):
            nn.init.trunc_normal_(parameter, std=0.02)

    def freeze_backbone(self, freeze=True):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not freeze

    def unfreeze_last_blocks(self, num_blocks=2):
        self.freeze_backbone(True)
        for block in self.backbone.encoder.layer[-num_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in self.backbone.layernorm.parameters():
            parameter.requires_grad = True

    def encode_images(self, images):
        """输入仅包含有效组 [总组数,3,H,W]；图像不存在 padding。"""
        use_grad = any(parameter.requires_grad for parameter in self.backbone.parameters())
        with nullcontext() if use_grad else torch.no_grad():
            # 每个 chunk 单独归一化，避免额外创建一整份归一化图像。
            chunks = [self.backbone(pixel_values=(chunk - self.image_mean) / self.image_std).last_hidden_state[:, 0]
                      for chunk in images.split(self.encoder_chunk_size)]
        return torch.cat(chunks)

    def aggregate_groups(self, features, group_counts, group_positions):
        """只对特征补齐；每行是一个真实序列，保证至少有一个有效组。"""
        if group_counts.ndim != 1 or not len(group_counts) or bool((group_counts <= 0).any()):
            raise ValueError("Every series must contain at least one group")
        if int(group_counts.sum()) != len(features) or len(group_positions) != len(features):
            raise ValueError("Group counts/positions do not match packed image features")
        series_count, max_groups = len(group_counts), int(group_counts.max())
        starts = group_counts.cumsum(0) - group_counts
        series_indices = torch.repeat_interleave(torch.arange(series_count, device=features.device), group_counts)
        local_indices = torch.arange(len(features), device=features.device) - torch.repeat_interleave(starts, group_counts)
        flat_indices = series_indices * max_groups + local_indices
        positioned = features + self.group_position(group_positions.to(features.dtype).unsqueeze(-1))
        padded = features.new_zeros(series_count * max_groups, features.shape[-1])
        padded = padded.index_copy(0, flat_indices, positioned).view(series_count, max_groups, -1)
        group_mask = torch.arange(max_groups, device=features.device)[None, :] < group_counts[:, None]
        padded = self.slice_encoder(padded, src_key_padding_mask=~group_mask)
        scores = self.slice_attention(padded).squeeze(-1).masked_fill(~group_mask, -torch.inf)
        attention = scores.softmax(dim=1)
        return (padded * attention.unsqueeze(-1)).sum(dim=1)

    @staticmethod
    def aggregate_series(series_features, series_batch_indices, series_slot_indices, batch_size):
        """同 slot 内按真实序列平均；每个序列先聚合为一个向量，长序列不会因组数多占权重。"""
        indices = series_batch_indices * len(SLOTS) + series_slot_indices
        sums = series_features.new_zeros(batch_size * len(SLOTS), series_features.shape[-1])
        sums = sums.index_add(0, indices, series_features)
        counts = series_features.new_zeros(batch_size * len(SLOTS))
        counts = counts.index_add(0, indices, series_features.new_ones(len(series_features)))
        return (sums / counts.clamp_min(1).unsqueeze(-1)).view(batch_size, len(SLOTS), -1)

    def forward(self, images, slot_mask, series_features=None, *, group_counts,
                group_positions, series_batch_indices, series_slot_indices):
        if slot_mask.ndim != 2 or slot_mask.shape[1] != len(SLOTS) or not bool(slot_mask.bool().any(dim=1).all()):
            raise ValueError("Every study must contain at least one valid slot")
        batch_size = len(slot_mask)
        features = self.slice_projection(self.encode_images(images))
        encoded_series = self.aggregate_groups(features, group_counts, group_positions)
        if series_features is not None:
            encoded_series = encoded_series + self.metadata_projection(series_features)
        slot_features = self.aggregate_series(encoded_series, series_batch_indices, series_slot_indices, batch_size)
        slot_features = self.slot_encoder(slot_features + self.slot_embedding,
                                         src_key_padding_mask=~slot_mask.bool())
        queries = self.label_queries.unsqueeze(0).expand(batch_size, -1, -1)
        label_features, _ = self.label_attention(queries, slot_features, slot_features,
                                                key_padding_mask=~slot_mask.bool(), need_weights=False)
        label_features = self.output_norm(label_features)
        return (label_features * self.label_weight.unsqueeze(0)).sum(dim=-1) + self.label_bias


def predict_batch(model, batch, use_metadata=True):
    """训练/验证/推理统一调用接口；batch 的张量应事先移动到目标设备。"""
    return model(batch["images"], batch["slot_mask"],
                 batch["series_features"] if use_metadata else None,
                 group_counts=batch["group_counts"], group_positions=batch["group_positions"],
                 series_batch_indices=batch["series_batch_indices"],
                 series_slot_indices=batch["series_slot_indices"])


__all__ = ["ARCHITECTURE", "RSNADINOv2", "predict_batch"]
