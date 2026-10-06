"""MIL math, variable packed bags, chunking, gradients and checkpoint smoke checks."""
from pathlib import Path
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
import rsna_model as models
import train

ROOT = Path(__file__).resolve().parent


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.projection = nn.Linear(3, 8)

    def forward(self, pixel_values):
        return SimpleNamespace(last_hidden_state=self.projection(pixel_values.mean((-1, -2)))[:, None])


def make_model():
    with patch.object(models, "load_backbone", return_value=TinyBackbone()):
        return models.RSNADINOv2(dropout=0., freeze_backbone=False, encoder_chunk_size=3)


def sample():
    # Deliberately interleave two studies with different bag lengths and missing slots.
    batch_ids = torch.tensor([1, 0, 1, 0, 0, 1, 0])
    slot_ids = torch.tensor([4, 0, 4, 1, 0, 4, 1])
    slot_mask = torch.zeros(2, 5, dtype=torch.bool)
    slot_mask[batch_ids, slot_ids] = True
    return dict(images=torch.rand(7, 3, 14, 14), slot_mask=slot_mask,
                series_features=torch.rand(2, 5, 11), window_positions=torch.rand(7),
                window_batch_indices=batch_ids, window_slot_indices=slot_ids)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_complete_bag_matches_reference_formula_and_gradients(self):
        torch.manual_seed(22)
        model = make_model()
        batch = sample()
        actual = models.predict_batch(model, batch)
        feats = model.encode_images(batch["images"])
        expected = []
        # Literal train_knee.py operations, separately on each unpadded study bag.
        for study in range(2):
            h = model.norm(feats[batch["window_batch_indices"] == study][None])
            a = torch.softmax(model.att(h), dim=1)
            pooled = torch.einsum("bkn,bkf->bnf", a, h)
            expected.append((pooled * model.clsW).sum(-1) + model.clsb)
        torch.testing.assert_close(actual, torch.cat(expected), atol=1e-7, rtol=1e-5)
        actual_grads = torch.autograd.grad(actual.sum(), tuple(model.parameters()), retain_graph=True)
        expected_grads = torch.autograd.grad(torch.cat(expected).sum(), tuple(model.parameters()))
        for first, second in zip(actual_grads, expected_grads):
            torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-4)

    def test_order_metadata_positions_and_chunk_size_do_not_change_eval(self):
        model = make_model().eval()
        batch = sample()
        with torch.no_grad():
            original = models.predict_batch(model, batch)
            order = torch.randperm(7)
            altered = {name: (value[order] if name in ("images", "window_positions", "window_batch_indices", "window_slot_indices") else value)
                       for name, value in batch.items()}
            altered["window_positions"] = torch.zeros(7)
            altered["series_features"] = torch.zeros(2, 5, 11)
            model.encoder_chunk_size = 7
            torch.testing.assert_close(original, models.predict_batch(model, altered, use_metadata=True), atol=1e-7, rtol=1e-5)

    def test_notebook_model_matches_source(self):
        notebook = json.loads((ROOT / "Kaggle_Inference.ipynb").read_text(encoding="utf-8"))
        source = (ROOT / "rsna_model.py").read_text(encoding="utf-8").replace("from rsna_data import LABELS, SLOTS\n", "")
        self.assertEqual(source, "".join(notebook["cells"][2]["source"]))
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell-{i}", "exec")

    def test_invalid_missing_slot_and_legacy_resume_rejected(self):
        model = make_model()
        batch = sample()
        batch["slot_mask"][0, 4] = True
        with self.assertRaisesRegex(ValueError, "Every valid slot"):
            models.predict_batch(model, batch)
        with self.assertRaisesRegex(ValueError, "matching Baseline_v22"):
            train.validate_resume_checkpoint({"architecture": "baseline_v14_80_candidates_quality_selection_ema"}, None)

    def test_real_dinov2_training_ema_and_state_roundtrip(self):
        torch.manual_seed(22)
        model = models.RSNADINOv2(model_dir=ROOT.parent / "dinov2-pytorch-small-v1", encoder_chunk_size=2)
        model.unfreeze_last_blocks(6)
        model.train()
        ema = train.ModelEMA(model)
        batch = sample()
        batch["images"] = torch.rand(7, 3, 28, 28)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        before = model.clsW.detach().clone()
        loss = F.binary_cross_entropy_with_logits(models.predict_batch(model, batch), torch.rand(2, 12))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
        optimizer.step()
        ema.update(model)
        self.assertEqual(ema.num_updates, 1)
        self.assertFalse(torch.equal(before, model.clsW))
        restored = models.RSNADINOv2(model_dir=ROOT.parent / "dinov2-pytorch-small-v1", encoder_chunk_size=3).eval()
        restored.load_state_dict(ema.model.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(models.predict_batch(restored, batch), models.predict_batch(ema.model, batch), atol=1e-6, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
