"""DINOv2-based hierarchical model for RSNA Knee."""

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel

from rsna_data import LABELS, SERIES_FEATURES, SLOTS


# A3 消融：切片聚合与槽位 Transformer 后，对有效槽位做掩码平均，再输出 12 个标签。
# 仅移除标签查询交叉注意力；切片注意力池化与槽位 Transformer 保持不变。
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
    """2.5D DINOv2 encoder + slice aggregation + label-specific slot fusion."""

    def __init__(
        self,
        model_dir=None,
        hidden_dim=256,
        num_slices=24,
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

        self.num_slices = num_slices
        # 一次编码的图像张数，控制显存占用，不改变模型结构。
        self.encoder_chunk_size = encoder_chunk_size
        self.freeze_backbone(freeze_backbone)
        backbone_dim = self.backbone.config.hidden_size

        # 2.5D 三通道输入沿用 ImageNet 均值/标准差；buffer 会随模型移动设备，但不训练、不存 checkpoint。
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        # CLS 特征由 DINOv2 维度投影到后续 Transformer 使用的 hidden_dim。
        self.slice_projection = nn.Sequential(nn.LayerNorm(backbone_dim), nn.Linear(backbone_dim, hidden_dim))
        # 可学习的位置编码让模型知道切片在采样序列中的相对位置。
        self.slice_position = nn.Parameter(torch.zeros(1, num_slices, hidden_dim))

        # 切片级自注意力：同一序列内的切片可以交换信息，输入形状为 [B×槽位数, S, hidden_dim]。
        slice_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout, batch_first=True, norm_first=True)
        self.slice_encoder = nn.TransformerEncoder(slice_layer, slice_layers, enable_nested_tensor=False)
        # 切片注意力池化：为每张切片打一个分数，softmax 后加权求和成一个序列向量。
        self.slice_attention = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

        # 将 TE/TR、像素间距等 11 维序列元数据映射到和图像特征相同的维度。
        self.metadata_projection = nn.Sequential(
            nn.Linear(len(SERIES_FEATURES), hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
        )
        # 每个序列槽位各有一个可学习身份向量，用于区分方位和是否脂肪抑制。
        self.slot_embedding = nn.Parameter(torch.zeros(1, len(SLOTS), hidden_dim))
        # 序列级自注意力：让同一 Study 中有效的不同序列相互交换信息。
        slot_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout, batch_first=True, norm_first=True)
        self.slot_encoder = nn.TransformerEncoder(slot_layer, slot_layers, enable_nested_tensor=False)

        # A3：不建立逐标签 query 和标签交叉注意力，改用共享的 Study 特征。
        # 标签特征归一化后，与逐标签权重做点积，得到未经过 sigmoid 的 12 个 logit。
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.label_weight = nn.Parameter(torch.empty(len(LABELS), hidden_dim))
        self.label_bias = nn.Parameter(torch.zeros(len(LABELS)))
        nn.init.trunc_normal_(self.slice_position, std=0.02)
        nn.init.trunc_normal_(self.slot_embedding, std=0.02)
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

    @staticmethod
    def make_2p5d(images):
        """把前一张、当前张、后一张切片组成 DINOv2 的三个输入通道。"""
        # 输入 [B, 6, S, 1, H, W]；去掉单通道维度后为 [B, 6, S, H, W]。
        gray = images.squeeze(3)
        # 首尾切片没有更外侧的邻居，因此重复边界切片。
        previous = torch.cat([gray[:, :, :1], gray[:, :, :-1]], dim=2)
        following = torch.cat([gray[:, :, 1:], gray[:, :, -1:]], dim=2)
        return torch.stack([previous, gray, following], dim=3)

    def encode_images(self, images, slot_mask):
        # 仅编码实际存在的槽位；空槽位保留为零，避免浪费 DINOv2 计算量。
        batch, slots, slices, _, height, width = images.shape
        # 将 B×槽位×切片展平成普通图像 batch，供二维视觉骨干处理。
        images = self.make_2p5d(images).reshape(batch * slots * slices, 3, height, width)
        # 一个槽位如果有效，其所有采样切片都有效；掩码展平后对应上面的图像顺序。
        valid = slot_mask[:, :, None].expand(batch, slots, slices).reshape(-1).bool()
        valid_indices = valid.nonzero(as_tuple=False).squeeze(1)
        images = images[valid_indices]
        images = (images - self.image_mean) / self.image_std

        # 冻结骨干时关闭梯度图，节省显存；last6 等微调模式则保留梯度。
        use_grad = any(parameter.requires_grad for parameter in self.backbone.parameters())
        context = nullcontext() if use_grad else torch.no_grad()
        with context:
            # 取每张图像的 CLS token，形状 [有效切片数, backbone_dim]。
            chunks = [self.backbone(pixel_values=chunk).last_hidden_state[:, 0] for chunk in images.split(self.encoder_chunk_size)]
        encoded = torch.cat(chunks)
        # 把编码结果放回原来的 B×槽位×切片位置；无效槽位保持零。
        all_encoded = encoded.new_zeros(batch * slots * slices, encoded.shape[-1])
        all_encoded = all_encoded.index_copy(0, valid_indices, encoded)
        return all_encoded.view(batch, slots, slices, -1)

    def forward(self, images, slot_mask, series_features=None):
        # images: [B, 6, S, 1, H, W]；slot_mask: [B, 6]；series_features: [B, 6, 11]。
        batch, slots, slices = images.shape[:3]
        if slices != self.num_slices:
            raise ValueError(f"Expected {self.num_slices} slices, got {slices}")

        features = self.encode_images(images, slot_mask)
        # 每个槽位的 S 张切片做自注意力（相当于序列级），形状 [B×6, S, hidden_dim]。
        features = self.slice_projection(features).view(batch * slots, slices, -1)
        features = self.slice_encoder(features + self.slice_position)
        # 对 S 张切片归一化权重并求和，得到 [B, 6, hidden_dim]。
        attention = self.slice_attention(features).softmax(dim=1)
        slot_features = (features * attention).sum(dim=1).view(batch, slots, -1)

        if series_features is not None:
            # 元数据作为加性特征融入对应序列，而不是直接输入 DINOv2。
            slot_features = slot_features + self.metadata_projection(series_features)
        # padding mask 中 True 表示忽略缺失槽位，与 slot_mask 的语义相反。 做slot级注意力
        slot_features = self.slot_encoder(slot_features + self.slot_embedding, src_key_padding_mask=~slot_mask.bool())

        # A3：只平均真实存在的槽位，缺失槽位不参与 Study 特征。
        valid_slots = slot_mask.to(dtype=slot_features.dtype).unsqueeze(-1)
        study_features = (slot_features * valid_slots).sum(dim=1)
        study_features = study_features / valid_slots.sum(dim=1).clamp_min(1)
        study_features = self.output_norm(study_features)
        # 共享 Study 特征与 12 组逐标签权重做点积，返回 [B, 12] logits。
        return (study_features.unsqueeze(1) * self.label_weight.unsqueeze(0)).sum(dim=-1) + self.label_bias


__all__ = ["RSNADINOv2"]
