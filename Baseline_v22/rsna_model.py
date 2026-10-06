"""v14 DINOv2 CLS backbone with train_knee.py per-diagnosis attention MIL."""
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModel
from rsna_data import LABELS, SLOTS

ARCHITECTURE = "baseline_v22_dinov2_attention_mil_ema"

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
    """All valid windows of one study form a single bag, across all five slots."""
    def __init__(self, model_dir=None, dropout=0.2, freeze_backbone=True,
                 encoder_chunk_size=24):
        super().__init__()
        self.backbone = load_backbone(model_dir)
        if encoder_chunk_size < 1:
            raise ValueError("encoder_chunk_size must be positive")
        self.encoder_chunk_size = encoder_chunk_size
        self.freeze_backbone(freeze_backbone)
        feature_dim = self.backbone.config.hidden_size
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        # Same head dimensions/operations as RaptorClassifier; F_dim is DINOv2's CLS width.
        self.norm = nn.LayerNorm(feature_dim)
        self.att = nn.Sequential(nn.Linear(feature_dim, 256), nn.Tanh(), nn.Dropout(dropout),
                                 nn.Linear(256, len(LABELS)))
        self.clsW = nn.Parameter(torch.zeros(len(LABELS), feature_dim))
        self.clsb = nn.Parameter(torch.zeros(len(LABELS)))
        nn.init.trunc_normal_(self.clsW, std=0.02)

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

    def head(self, feats, mask=None):
        """One softmax over the complete study bag, after all encoder chunks."""
        h = self.norm(feats)
        a = self.att(h)
        if mask is not None:
            a = a.masked_fill(~mask[..., None], -torch.inf)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb

    def forward(self, images, slot_mask, series_features=None, *, window_positions,
                window_batch_indices, window_slot_indices):
        # Keep v14's packed-window input contract; position/metadata do not enter the MIL head.
        if images.ndim != 4 or images.shape[1] != 3 or not len(images):
            raise ValueError("Expected nonempty packed windows [N,3,H,W]")
        if slot_mask.ndim != 2 or slot_mask.shape[1] != len(SLOTS):
            raise ValueError("Expected slot_mask [B,5]")
        batch_size = len(slot_mask)
        if not bool(slot_mask.bool().any(dim=1).all()):
            raise ValueError("Every study must have at least one valid slot")
        for values in (window_positions, window_batch_indices, window_slot_indices):
            if values.shape != (len(images),):
                raise ValueError("Window maps must match the packed images")
        if window_batch_indices.dtype != torch.long or window_slot_indices.dtype != torch.long:
            raise ValueError("Window indices must be int64")
        if (bool((window_batch_indices < 0).any()) or bool((window_batch_indices >= batch_size).any()) or
                bool((window_slot_indices < 0).any()) or bool((window_slot_indices >= len(SLOTS)).any())):
            raise ValueError("Invalid window batch/slot index")
        groups = window_batch_indices * len(SLOTS) + window_slot_indices
        present = torch.bincount(groups, minlength=batch_size * len(SLOTS)).view(batch_size, len(SLOTS)) > 0
        if not torch.equal(present, slot_mask.bool()):
            raise ValueError("Every valid slot must have a window; invalid slots cannot contain windows")
        features = self.encode_images(images)
        # Pad only feature bags; never encode padded images or mix studies during pooling.
        order = window_batch_indices.argsort(stable=True)
        counts = torch.bincount(window_batch_indices, minlength=batch_size)
        width = int(counts.max())
        starts = counts.cumsum(0) - counts
        columns = torch.arange(len(images), device=features.device) - torch.repeat_interleave(starts, counts)
        rows = torch.repeat_interleave(torch.arange(batch_size, device=features.device), counts)
        padded = features.new_zeros(batch_size * width, features.shape[-1])
        padded = padded.index_copy(0, rows * width + columns, features[order]).view(batch_size, width, -1)
        mask = torch.arange(width, device=features.device)[None] < counts[:, None]
        return self.head(padded, mask)


def predict_batch(model, batch, use_metadata=False):
    return model(batch["images"], batch["slot_mask"],
                 window_positions=batch["window_positions"], window_batch_indices=batch["window_batch_indices"],
                 window_slot_indices=batch["window_slot_indices"])


__all__ = ["ARCHITECTURE", "RSNADINOv2", "predict_batch"]
