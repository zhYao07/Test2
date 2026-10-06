"""CPU regression checks for v21 weight loading, fine-tuning and checkpoint inference."""

import ast
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torchvision.models import resnet50

import rsna_data as data
import rsna_model as models
import train


ROOT = Path(__file__).resolve().parent
SELECTION = dict(version=data.SERIES_SELECTION_VERSION, coverage_quantile=0.2,
                 coverage_thresholds=dict(Sagittal=40.0, Coronal=40.0, Axial=40.0))


def batch():
    # Two studies with different missing slots, and a singleton encoder chunk.
    return dict(images=torch.rand(5, 3, 32, 32),
                slot_mask=torch.tensor([[True, True, False, False, False],
                                        [False, False, True, False, True]]),
                window_positions=torch.tensor([0.2, -0.7, 0.1, -0.3, 0.8]),
                window_batch_indices=torch.tensor([0, 0, 0, 1, 1]),
                window_slot_indices=torch.tensor([0, 0, 1, 2, 4]),
                series_features=torch.zeros(2, 5, len(data.SERIES_FEATURES)))


class RadImageNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(21)
        cls.temporary = TemporaryDirectory()
        cls.path = Path(cls.temporary.name) / "official_layout.pt"
        # Reference follows the official notebook, independently of our key conversion.
        cls.reference = nn.Sequential(*list(resnet50(weights=None).children())[:9]).eval()
        cls.state = {"backbone." + k: v for k, v in cls.reference.state_dict().items()}
        torch.save(cls.state, cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def model(self):
        return models.RSNARadImageNet(self.path, hidden_dim=16, num_heads=4,
                                      dropout=0.0, encoder_chunk_size=2)

    def test_official_layout_matches_reference_and_rejects_bad_weights(self):
        backbone = models.load_backbone(self.path).eval()
        with torch.no_grad():
            for size in (32, 336):
                image = torch.rand(1, 3, size, size) * 2 - 1
                torch.testing.assert_close(backbone(image), self.reference(image).flatten(1))
        with self.assertRaises(FileNotFoundError):
            models.load_backbone(Path(self.temporary.name) / "missing.pt")
        invalid = Path(self.temporary.name) / "invalid.pt"
        state = dict(self.state)
        state.pop("backbone.0.weight")
        torch.save(state, invalid)
        with self.assertRaisesRegex(ValueError, "missing"):
            models.load_backbone(invalid)
        state["backbone.0.weight"] = torch.zeros(1)
        torch.save(state, invalid)
        with self.assertRaises(RuntimeError):
            models.load_backbone(invalid)
        # Classifier tensors must never be silently ignored.
        state = dict(self.state, **{"fc.weight": torch.zeros(1)})
        torch.save(state, invalid)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            models.load_backbone(invalid)

    def test_modes_gradients_bn_and_chunk_invariance(self):
        model = self.model()
        packed = batch()
        for mode, prefixes in (("frozen", ()), ("layer4", ("layer4.",)),
                               ("layer3+layer4", ("layer3.", "layer4.")),
                               ("full", ("",))):
            model.set_backbone_mode(mode)
            model.train()
            for name, parameter in model.backbone.named_parameters():
                self.assertEqual(parameter.requires_grad, name.startswith(prefixes))
            self.assertTrue(all(not m.training for m in model.backbone.modules()
                                if isinstance(m, nn.BatchNorm2d)))
            buffers = {k: v.clone() for k, v in model.backbone.named_buffers()}
            model.zero_grad(set_to_none=True)
            logits = models.predict_batch(model, packed)
            self.assertEqual(logits.shape, (2, 12))
            loss = train.masked_bce(logits, torch.zeros_like(logits), torch.ones_like(logits),
                                    torch.ones_like(logits), torch.ones(12))
            loss.backward()
            for name, parameter in model.backbone.named_parameters():
                self.assertEqual(parameter.grad is not None, parameter.requires_grad, name)
            for name, value in model.backbone.named_buffers():
                torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
            self.assertTrue(torch.isfinite(loss))
        model.eval()
        with torch.no_grad():
            model.encoder_chunk_size = 2
            first = models.predict_batch(model, packed)
            model.encoder_chunk_size = 5
            torch.testing.assert_close(models.predict_batch(model, packed), first, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(model.encode_images(packed["images"]),
                                       self.reference(packed["images"] * 2 - 1).flatten(1))

    def test_optimizer_ema_checkpoint_roundtrip_and_v14_rejection(self):
        model = self.model()
        model.set_backbone_mode("layer4")
        args = SimpleNamespace(head_lr=3e-4, backbone_lr=1e-5, weight_decay=0.05,
                               train_windows=24, span_lo=0.02, span_hi=0.98, image_size=336,
                               crop_mm=140.0, backbone_mode="layer4", no_metadata=False,
                               seed=42, no_ema=False, ema_decay=0.999, radimagenet_sha256="test",
                               series_selection=SELECTION)
        optimizer = train.make_optimizer(model, args)
        scheduler = train.make_scheduler(optimizer, 4, 0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        ema = train.ModelEMA(model)
        packed = batch()
        old = model.backbone.layer4[0].conv1.weight.detach().clone()
        models.predict_batch(model, packed).square().mean().backward()
        optimizer.step()
        scheduler.step()
        ema.update(model)
        self.assertFalse(torch.equal(old, model.backbone.layer4[0].conv1.weight))
        path = Path(self.temporary.name) / "best.pt"
        train.save_checkpoint(path, model, optimizer, scheduler, scaler, 0, 0.5, args, [], ema)
        checkpoint = torch.load(path, weights_only=False)
        train.validate_resume_checkpoint(checkpoint, args)
        restored = models.RSNARadImageNet(load_pretrained=False, hidden_dim=16,
                                         num_heads=4, dropout=0.0).eval()
        restored.load_state_dict(checkpoint["model"], strict=True)
        with torch.no_grad():
            torch.testing.assert_close(models.predict_batch(restored, packed),
                                       models.predict_batch(ema.model, packed), rtol=1e-4, atol=1e-5)
        bad = dict(checkpoint, architecture="baseline_v14_80_candidates_quality_selection_ema")
        with self.assertRaisesRegex(ValueError, "Baseline_v21"):
            train.validate_resume_checkpoint(bad, args)
        bad = deepcopy(args)
        bad.radimagenet_sha256 = "different"
        with self.assertRaisesRegex(ValueError, "radimagenet_sha256"):
            train.validate_resume_checkpoint(checkpoint, bad)
        init_args = SimpleNamespace(radimagenet_weights=self.path, encoder_chunk_size=2,
                                    backbone_mode="frozen", init_checkpoint=path, no_metadata=False)
        invalid = Path(self.temporary.name) / "wrong_backbone.pt"
        torch.save(dict(checkpoint, backbone_spec={}), invalid)
        init_args.init_checkpoint = invalid
        with self.assertRaisesRegex(ValueError, "matching v21"):
            train.build_model(init_args, False, 0)

    def test_v14_data_fusion_and_notebook_consistency(self):
        self.assertEqual((ROOT / "rsna_data.py").read_bytes(),
                         (ROOT.parent / "Baseline_v14/rsna_data.py").read_bytes())
        before = ast.parse((ROOT.parent / "Baseline_v14/rsna_model.py").read_text(encoding="utf-8"))
        after = ast.parse((ROOT / "rsna_model.py").read_text(encoding="utf-8"))
        def methods(tree):
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
            return {node.name: ast.dump(node) for node in cls.body if isinstance(node, ast.FunctionDef)}
        for name in ("aggregate_windows", "forward"):
            self.assertEqual(methods(before)[name], methods(after)[name])
        notebook = json.loads((ROOT / "Kaggle_Inference.ipynb").read_text(encoding="utf-8"))
        for index, filename in ((1, "rsna_data.py"), (2, "rsna_model.py")):
            expected = (ROOT / filename).read_text(encoding="utf-8").replace("# -*- coding: utf-8 -*-\n", "")
            expected = expected.replace("from rsna_data import LABELS, SERIES_FEATURES, SLOTS\n", "")
            if filename == "rsna_data.py":
                expected = expected.replace("pydicom.dcmread(file, stop_before_pixels=True)",
                                            "inference_read_header(file, stop_before_pixels=True)")
                expected = expected.replace("pydicom.dcmread(path)", "inference_read_dataset(path)")
            actual = "".join(notebook["cells"][index]["source"])
            self.assertEqual(ast.dump(ast.parse(actual)), ast.dump(ast.parse(expected)))
        source = next("".join(cell["source"]) for cell in notebook["cells"]
                      if "def inference_worker(" in "".join(cell["source"]))
        self.assertNotIn("DINOV2_MODEL_DIR", source)
        self.assertIn("load_pretrained=False", source)
        compile(source, "inference", "exec")

    def test_swa_export_loads_into_notebook_model(self):
        model = self.model()
        ema = train.ModelEMA(model)
        tracker = train.Top3EMA()
        with torch.no_grad():
            # Raw bias differs deliberately, so averaging the raw model would be detected.
            model.label_bias.fill_(100)
            for epoch, value in enumerate((-2.0, 0.0, 2.0)):
                ema.model.label_bias.fill_(value)
                tracker.update(ema, epoch, 0.8 + epoch * 0.01)
        state = tracker.average()
        torch.testing.assert_close(state["label_bias"], torch.zeros(12))
        config = SimpleNamespace(image_size=224, crop_mm=140.0, span_lo=0.02, span_hi=0.98,
                                 no_metadata=True, series_selection=SELECTION, swa=True)
        path = Path(self.temporary.name) / "swa.pt"
        train.save_inference_checkpoint(path, state, config, swa_sources=tracker.sources(), val_macro_auc=None)
        checkpoint = torch.load(path, weights_only=False)
        self.assertEqual(checkpoint["backbone_spec"], models.BACKBONE_SPEC)
        self.assertEqual(checkpoint["architecture"], models.ARCHITECTURE)
        notebook = json.loads((ROOT / "Kaggle_Inference.ipynb").read_text(encoding="utf-8"))
        namespace = {}
        for index in (1, 2):
            exec(compile("".join(notebook["cells"][index]["source"]), f"cell-{index}", "exec"), namespace)
        restored = namespace["RSNARadImageNet"](load_pretrained=False, hidden_dim=16,
                                               num_heads=4, dropout=0.0).eval()
        restored.load_state_dict(checkpoint["model"], strict=True)
        ema.model.load_state_dict(state, strict=True)
        packed = batch()
        with torch.no_grad():
            result = namespace["predict_batch"](restored, packed, use_metadata=False)
            torch.testing.assert_close(result, models.predict_batch(ema.model, packed, use_metadata=False),
                                       rtol=1e-4, atol=1e-5)
            self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
