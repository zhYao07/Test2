"""DINOv2-based hierarchical model for RSNA Knee."""

from contextlib import nullcontext
import warnings

import torch
import torch.nn as nn

from rsna_data import LABELS, SERIES_FEATURES, SLOTS


class RSNADINOv2(nn.Module):
    """2.5D DINOv2 encoder + slice aggregation + label-specific slot fusion."""

    def __init__(
        self,
        backbone_name="dinov2_vits14",
        pretrained=True,
        local_repo=None,
        weights_path=None,
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
        source = "local" if local_repo else "github"
        repo = local_repo or "facebookresearch/dinov2"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"xFormers is available.*", category=UserWarning)
            self.backbone = torch.hub.load(repo, backbone_name, source=source, pretrained=pretrained and weights_path is None)
        if weights_path:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state)

        self.num_slices = num_slices
        self.encoder_chunk_size = encoder_chunk_size
        self.freeze_backbone(freeze_backbone)
        backbone_dim = self.backbone.embed_dim

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.slice_projection = nn.Sequential(nn.LayerNorm(backbone_dim), nn.Linear(backbone_dim, hidden_dim))
        self.slice_position = nn.Parameter(torch.zeros(1, num_slices, hidden_dim))

        slice_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout, batch_first=True, norm_first=True)
        self.slice_encoder = nn.TransformerEncoder(slice_layer, slice_layers, enable_nested_tensor=False)
        self.slice_attention = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

        self.metadata_projection = nn.Sequential(
            nn.Linear(len(SERIES_FEATURES), hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
        )
        self.slot_embedding = nn.Parameter(torch.zeros(1, len(SLOTS), hidden_dim))
        slot_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout, batch_first=True, norm_first=True)
        self.slot_encoder = nn.TransformerEncoder(slot_layer, slot_layers, enable_nested_tensor=False)

        self.label_queries = nn.Parameter(torch.empty(len(LABELS), hidden_dim))
        self.label_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.label_weight = nn.Parameter(torch.empty(len(LABELS), hidden_dim))
        self.label_bias = nn.Parameter(torch.zeros(len(LABELS)))
        nn.init.trunc_normal_(self.slice_position, std=0.02)
        nn.init.trunc_normal_(self.slot_embedding, std=0.02)
        nn.init.trunc_normal_(self.label_queries, std=0.02)
        nn.init.trunc_normal_(self.label_weight, std=0.02)

    def freeze_backbone(self, freeze=True):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not freeze

    def unfreeze_last_blocks(self, num_blocks=2):
        self.freeze_backbone(True)
        for block in self.backbone.blocks[-num_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in self.backbone.norm.parameters():
            parameter.requires_grad = True

    @staticmethod
    def make_2p5d(images):
        """把前一张、当前张、后一张切片组成 DINOv2 的三个输入通道。"""
        gray = images.squeeze(3)
        previous = torch.cat([gray[:, :, :1], gray[:, :, :-1]], dim=2)
        following = torch.cat([gray[:, :, 1:], gray[:, :, -1:]], dim=2)
        return torch.stack([previous, gray, following], dim=3)

    def encode_images(self, images, slot_mask):
        batch, slots, slices, _, height, width = images.shape
        images = self.make_2p5d(images).reshape(batch * slots * slices, 3, height, width)
        valid = slot_mask[:, :, None].expand(batch, slots, slices).reshape(-1).bool()
        valid_indices = valid.nonzero(as_tuple=False).squeeze(1)
        images = images[valid_indices]
        images = (images - self.image_mean) / self.image_std

        use_grad = any(parameter.requires_grad for parameter in self.backbone.parameters())
        context = nullcontext() if use_grad else torch.no_grad()
        with context:
            chunks = [self.backbone(chunk) for chunk in images.split(self.encoder_chunk_size)]
        encoded = torch.cat(chunks)
        all_encoded = encoded.new_zeros(batch * slots * slices, encoded.shape[-1])
        all_encoded = all_encoded.index_copy(0, valid_indices, encoded)
        return all_encoded.view(batch, slots, slices, -1)

    def forward(self, images, slot_mask, series_features=None):
        batch, slots, slices = images.shape[:3]
        if slices != self.num_slices:
            raise ValueError(f"Expected {self.num_slices} slices, got {slices}")

        features = self.encode_images(images, slot_mask)
        features = self.slice_projection(features).view(batch * slots, slices, -1)
        features = self.slice_encoder(features + self.slice_position)
        attention = self.slice_attention(features).softmax(dim=1)
        slot_features = (features * attention).sum(dim=1).view(batch, slots, -1)

        if series_features is not None:
            slot_features = slot_features + self.metadata_projection(series_features)
        slot_features = self.slot_encoder(slot_features + self.slot_embedding, src_key_padding_mask=~slot_mask.bool())

        queries = self.label_queries.unsqueeze(0).expand(batch, -1, -1)
        label_features, _ = self.label_attention(queries, slot_features, slot_features, key_padding_mask=~slot_mask.bool())
        label_features = self.output_norm(label_features)
        return (label_features * self.label_weight.unsqueeze(0)).sum(dim=-1) + self.label_bias


__all__ = ["RSNADINOv2"]
