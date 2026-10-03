"""CPU checks: python -m unittest discover -s Baseline_v15 -p test_rope.py -v."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

import rsna_model
from rsna_data import CANDIDATE_BUDGETS, LABELS, SLOTS
from rsna_model import (ARCHITECTURE, RSNADINOv2, RoPEWindowEncoderLayer,
                        make_position_encoding_config)
import train


ROOT = Path(__file__).resolve().parent


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=384)
        self.projection = nn.Linear(3, 384)

    def forward(self, pixel_values):
        return SimpleNamespace(last_hidden_state=self.projection(pixel_values.mean((-2, -1)))[:, None])


def small_model(**kwargs):
    with patch.object(rsna_model, "load_backbone", return_value=TinyBackbone()):
        return RSNADINOv2(hidden_dim=32, num_heads=4, dropout=0.0, **kwargs)


def training_args():
    with patch.object(sys, "argv", ["train.py"]):
        args = train.parse_args()
    args.series_selection = dict(version="header_quality_v1", coverage_quantile=0.2,
                                 coverage_thresholds=dict(Sagittal=50.0, Coronal=40.0, Axial=30.0))
    return args


class RoPEChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)
        self.layer = RoPEWindowEncoderLayer(32, 4, 0.0, 10000.0, 16.0).eval()
        self.features = torch.randn(2, 5, 32)
        self.positions = torch.tensor([[-1.0, -0.6, -0.1, 0.4, 0.9], [-0.9, -0.4, 0.0, 0.6, 1.0]])
        self.valid = torch.tensor([[True, True, True, True, True], [True, True, False, False, False]])

    def test_rotation_preserves_norm_and_depends_on_relative_distance(self):
        query, key = torch.randn(2, 4, 5, 8), torch.randn(2, 4, 5, 8)
        qr, kr = self.layer.rotate_queries_and_keys(query, key, self.positions)
        torch.testing.assert_close(qr.square().sum(-1), query.square().sum(-1))
        torch.testing.assert_close(kr.square().sum(-1), key.square().sum(-1))
        # Independent pair-wise formula: (R(a)q).(R(b)k) = q.(R(b-a)k).
        frequencies = 10000.0 ** (-torch.arange(0, 8, 2, dtype=torch.float64) / 8)
        angles = (self.positions[:, None, :] - self.positions[:, :, None]).double() * 16.0
        angles = angles[:, None, :, :, None] * frequencies
        qe, qo = query[..., 0::2].double().unsqueeze(-2), query[..., 1::2].double().unsqueeze(-2)
        ke, ko = key[..., 0::2].double().unsqueeze(-3), key[..., 1::2].double().unsqueeze(-3)
        expected = ((qe * ke + qo * ko) * angles.cos() + (qo * ke - qe * ko) * angles.sin()).sum(-1)
        torch.testing.assert_close(qr @ kr.transpose(-1, -2), expected.float(), atol=8e-6, rtol=1e-5)

    def test_zero_positions_match_standard_transformer(self):
        standard = nn.TransformerEncoderLayer(32, 4, 128, dropout=0.0,
                                              batch_first=True, norm_first=True).eval()
        standard.load_state_dict(self.layer.state_dict(), strict=True)
        with torch.no_grad():
            actual = self.layer(self.features, torch.zeros_like(self.positions), self.valid)
            expected = standard(self.features, src_key_padding_mask=~self.valid)
        torch.testing.assert_close(actual[self.valid], expected[self.valid], atol=1e-6, rtol=1e-5)

    def test_padding_cannot_affect_real_windows_or_receive_gradients(self):
        features = self.features.clone().requires_grad_()
        expected = self.layer(features, self.positions, self.valid)
        altered = self.features.clone()
        altered[~self.valid] = 100 * torch.randn_like(altered[~self.valid])
        positions = self.positions.clone()
        positions[~self.valid] = 123.0
        actual = self.layer(altered, positions, self.valid)
        torch.testing.assert_close(actual[self.valid], expected[self.valid])
        expected[self.valid].square().mean().backward()
        self.assertTrue(torch.equal(features.grad[~self.valid], torch.zeros_like(features.grad[~self.valid])))

    def test_translation_invariance_and_position_sensitivity(self):
        expected = self.layer(self.features, self.positions, self.valid)
        shifted = self.layer(self.features, self.positions + 0.3, self.valid)
        torch.testing.assert_close(expected[self.valid], shifted[self.valid], atol=2e-6, rtol=1e-5)
        collapsed = self.layer(self.features, torch.zeros_like(self.positions), self.valid)
        self.assertGreater((expected[self.valid] - collapsed[self.valid]).abs().max().item(), 1e-3)

    def test_rotation_uses_original_positions_for_subsets(self):
        query, key = torch.randn(2, 4, 5, 8), torch.randn(2, 4, 5, 8)
        full_query, full_key = self.layer.rotate_queries_and_keys(query, key, self.positions)
        indices = torch.tensor([0, 2, 4])
        subset_query, subset_key = self.layer.rotate_queries_and_keys(query[:, :, indices], key[:, :, indices],
                                                                      self.positions[:, indices])
        torch.testing.assert_close(subset_query, full_query[:, :, indices])
        torch.testing.assert_close(subset_key, full_key[:, :, indices])

    def test_bf16_forward_and_backward_are_finite(self):
        self.layer.train()
        features = self.features.clone().requires_grad_()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = self.layer(features, self.positions, self.valid)
            loss = output[self.valid].square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(features.grad).all())
        for parameter in self.layer.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_eval_disables_attention_dropout(self):
        layer = RoPEWindowEncoderLayer(32, 4, 0.4, 10000.0, 16.0).eval()
        with torch.no_grad():
            first = layer(self.features, self.positions, self.valid)
            second = layer(self.features, self.positions, self.valid)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_invalid_configuration_is_rejected(self):
        for base, scale in [(1, 16), (float("nan"), 16), (10000, 0), (10000, float("inf"))]:
            with self.assertRaises(ValueError):
                make_position_encoding_config(base, scale)
        with self.assertRaisesRegex(ValueError, "even head"):
            RoPEWindowEncoderLayer(28, 4, 0.0, 10000.0, 16.0)

    def test_variable_groups_are_isolated_and_permutation_invariant(self):
        model = small_model().eval()
        features = torch.randn(8, 32)
        # Unequal groups, duplicate depths, absent slots and two Studies.
        positions = torch.tensor([-0.8, -0.1, -0.1, 0.7, -0.2, 0.6, -0.9, 0.4])
        batch = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])
        slots = torch.tensor([0, 0, 0, 0, 2, 2, 4, 4])
        with torch.no_grad():
            expected, present = model.aggregate_windows(features, positions, batch, slots, 2)
            order = torch.randperm(8)
            actual, actual_present = model.aggregate_windows(features[order], positions[order], batch[order], slots[order], 2)
            torch.testing.assert_close(actual, expected)
            self.assertTrue(torch.equal(present, actual_present))
            for study, slot in [(0, 0), (0, 2), (1, 4)]:
                chosen = (batch == study) & (slots == slot)
                individual, _ = model.aggregate_windows(features[chosen], positions[chosen], torch.zeros(chosen.sum(), dtype=torch.long),
                                                         slots[chosen], 1)
                torch.testing.assert_close(individual[0, slot], expected[study, slot], atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.equal(expected[~present], torch.zeros_like(expected[~present])))

    def test_full_candidate_forward_backward_and_ema(self):
        model = small_model()
        # One study with 80 candidates; another with a single valid window.
        slots = torch.cat([torch.full((count,), slot) for slot, count in enumerate(CANDIDATE_BUDGETS)] + [torch.tensor([0])])
        batch = torch.cat([torch.zeros(sum(CANDIDATE_BUDGETS), dtype=torch.long), torch.ones(1, dtype=torch.long)])
        positions = torch.cat([torch.linspace(-0.98, 0.98, count) for count in CANDIDATE_BUDGETS] + [torch.tensor([0.3])])
        mask = torch.tensor([[True] * 5, [True, False, False, False, False]])
        metadata = torch.randn(2, 5, 11)
        ema = train.ModelEMA(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        output = model(torch.rand(81, 3, 28, 28), mask, metadata,
                       window_positions=positions, window_batch_indices=batch, window_slot_indices=slots)
        self.assertEqual(output.shape, (2, len(LABELS)))
        F.binary_cross_entropy_with_logits(output, torch.rand_like(output)).backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        ema.update(model)
        self.assertEqual(ema.num_updates, 1)
        self.assertFalse(any(name.startswith("window_position.") for name in model.state_dict()))

    def test_checkpoint_roundtrip_and_configuration_guards(self):
        args = training_args()
        args.rope_base, args.rope_position_scale = 5000.0, 12.0
        model = small_model(rope_base=args.rope_base, rope_position_scale=args.rope_position_scale)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = train.make_scheduler(optimizer, 4, 0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        ema = train.ModelEMA(model)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "smoke.pt"
            train.save_checkpoint(path, model, optimizer, scheduler, scaler, 0, 0.5, args, [], ema)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        train.validate_resume_checkpoint(checkpoint, args)
        restored = small_model(rope_base=args.rope_base, rope_position_scale=args.rope_position_scale)
        restored.load_state_dict(checkpoint["model"], strict=True)
        restored_ema = train.ModelEMA(restored)
        restored_ema.load_state_dict(checkpoint["ema"])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
        for change in ({"architecture": "baseline_v14_80_candidates_quality_selection_ema"},
                       {"position_encoding": make_position_encoding_config()}, {"position_encoding": None}):
            with self.assertRaises(ValueError):
                train.validate_resume_checkpoint({**checkpoint, **change}, args)
        wrong_args = deepcopy(args)
        wrong_args.rope_position_scale = 16.0
        with self.assertRaisesRegex(ValueError, "RoPE"):
            train.validate_resume_checkpoint(checkpoint, wrong_args)

    def test_init_checkpoint_rejects_v14_and_wrong_rope(self):
        args = training_args()
        args.init_checkpoint = Path("dummy.pt")
        with patch.object(rsna_model, "load_backbone", return_value=TinyBackbone()):
            model = RSNADINOv2()
        checkpoint = dict(architecture=ARCHITECTURE, slots=SLOTS, model=model.state_dict(),
                          position_encoding=model.position_encoding)
        with patch.object(rsna_model, "load_backbone", return_value=TinyBackbone()), patch.object(torch, "load", return_value=checkpoint):
            loaded = train.build_model(args, distributed=False, rank=0)
            self.assertEqual(loaded.position_encoding, model.position_encoding)
            checkpoint["architecture"] = "baseline_v14_80_candidates_quality_selection_ema"
            with self.assertRaisesRegex(ValueError, "v15 RoPE"):
                train.build_model(args, distributed=False, rank=0)
            checkpoint["architecture"] = ARCHITECTURE
            checkpoint["position_encoding"] = make_position_encoding_config(position_scale=8.0)
            with self.assertRaisesRegex(ValueError, "RoPE configuration"):
                train.build_model(args, distributed=False, rank=0)

    def test_notebook_matches_training_modules_and_predictions(self):
        notebook = json.loads((ROOT / "Kaggle_Inference.ipynb").read_text(encoding="utf-8"))
        code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(code[0], (ROOT / "rsna_data.py").read_text(encoding="utf-8"))
        expected_model = (ROOT / "rsna_model.py").read_text(encoding="utf-8").replace(
            "from rsna_data import LABELS, SERIES_FEATURES, SLOTS\n", "")
        self.assertEqual(code[1], expected_model)
        for source in code:
            compile(source, "notebook", "exec")
        namespace = {"__name__": "v15_notebook_test"}
        exec(code[0], namespace)
        exec(code[1], namespace)
        namespace["load_backbone"] = lambda model_dir: TinyBackbone()
        actual_model = namespace["RSNADINOv2"](hidden_dim=32, num_heads=4, dropout=0.0).eval()
        expected_model = small_model().eval()
        actual_model.load_state_dict(expected_model.state_dict(), strict=True)
        kwargs = dict(window_positions=torch.tensor([-0.8, 0.2, 0.9]),
                      window_batch_indices=torch.tensor([0, 0, 0]),
                      window_slot_indices=torch.tensor([0, 0, 0]))
        images = torch.rand(3, 3, 28, 28)
        mask = torch.tensor([[True, False, False, False, False]])
        with torch.no_grad():
            torch.testing.assert_close(actual_model(images, mask, **kwargs), expected_model(images, mask, **kwargs), rtol=0, atol=0)

    @unittest.skipUnless((ROOT.parent / "dinov2-pytorch-small-v1" / "pytorch_model.bin").is_file(),
                         "Local DINOv2 weights are unavailable")
    def test_real_dinov2_last6_without_metadata(self):
        args = training_args()
        args.backbone_mode, args.no_metadata = "last6", True
        model = train.build_model(args, distributed=False, rank=0)
        model.train()
        optimizer = train.make_optimizer(model, args)
        ema = train.ModelEMA(model)
        output = model(torch.rand(4, 3, 28, 28),
                       torch.tensor([[True, False, False, False, False], [False, False, False, False, True]]),
                       window_positions=torch.tensor([-0.7, 0.1, 0.8, 0.3]),
                       window_batch_indices=torch.tensor([0, 0, 0, 1]),
                       window_slot_indices=torch.tensor([0, 0, 0, 4]))
        loss = F.binary_cross_entropy_with_logits(output, torch.rand_like(output))
        loss.backward()
        self.assertEqual(output.shape, (2, len(LABELS)))
        self.assertTrue(torch.isfinite(loss))
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.slice_encoder.layers[0].self_attn.in_proj_weight.grad.abs().sum().item(), 0)
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum().item() > 0
                            for parameter in model.backbone.encoder.layer[-1].parameters()))
        optimizer.step()
        ema.update(model)
        self.assertEqual(ema.num_updates, 1)


if __name__ == "__main__":
    unittest.main()
