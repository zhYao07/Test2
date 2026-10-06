"""CPU regression tests for selection, sampling, acquisition isolation and fusion."""

import ast
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

import rsna_data as data
import rsna_model as model_module
import train


CONFIG = dict(version=data.SERIES_SELECTION_VERSION, coverage_quantile=0.2,
              coverage_thresholds=dict(Sagittal=40.0, Coronal=40.0, Axial=40.0))


def rows():
    records = []
    for slot, (plane, fluid, _) in enumerate(data.SLOT_SPECS):
        for rank in range(3):
            uid = f"{slot}_{rank}"
            records.append(dict(StudyInstanceUID="study", SeriesInstanceUID=uid,
                                SeriesPath=Path(uid), Anatomical_Plane=plane,
                                Fluid_Sensitive=fluid or 0, Fat_Suppression=0, NumSlices=30,
                                quality_usable=1, quality_header_errors=0, quality_geometry_errors=0,
                                quality_unique_slices=30, quality_coverage_mm=60.0,
                                quality_pixel_spacing_mm=0.5 + rank * 0.1,
                                quality_slice_spacing_mm=2.0, quality_fov_mm=160.0))
    return pd.DataFrame(records)


def fake_series(path, count, image_size, *args):
    identity = int(Path(path).name.split("_")[0]) * 3 + int(Path(path).name.split("_")[1]) + 1
    pixels = torch.full((count + 2, image_size, image_size), float(identity))
    indices = torch.arange(count)[:, None] + torch.arange(3)[None]
    return pixels, indices, torch.linspace(-1, 1, count), torch.full((11,), float(identity))


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.projection = nn.Linear(3, 8)

    def forward(self, pixel_values):
        return SimpleNamespace(last_hidden_state=self.projection(pixel_values.mean(dim=(2, 3)))[:, None])


def small_model():
    with patch.object(model_module, "load_backbone", return_value=TinyBackbone()):
        return model_module.RSNADINOv2(hidden_dim=16, num_heads=4, dropout=0.0)


def dataset(training=False, frame=None, cache=None):
    studies = pd.DataFrame([dict(StudyInstanceUID="study", **{label: 0.5 for label in data.LABELS})])
    return data.KneeDataset(studies, rows() if frame is None else frame, image_size=4,
                            train=training, cache_dir=cache, series_selection=CONFIG)


class MultiSeriesTests(unittest.TestCase):
    def test_quality_gate_and_preserved_top1(self):
        frame = rows()
        original = data.select_top1_slot_series(frame, CONFIG)
        chosen = data.select_slot_series(frame, CONFIG)
        self.assertEqual([r.SeriesInstanceUID for r in original], [r[0].SeriesInstanceUID for r in chosen])
        self.assertEqual([len(r) for r in chosen], [2] * 5)
        uids = [r.SeriesInstanceUID for slot in chosen for r in slot]
        self.assertEqual(len(set(uids)), 10)
        for column, value in (("quality_coverage_mm", 20), ("quality_fov_mm", 100), ("quality_usable", 0)):
            bad = frame.copy()
            bad.loc[bad.SeriesInstanceUID.isin(["0_1", "0_2"]), column] = value
            self.assertEqual(len(data.select_slot_series(bad, CONFIG)[0]), 1)
        # Incomplete top1 retains v14 fallback but does not admit a second acquisition.
        frame.loc[frame.Anatomical_Plane.eq("Axial"), "quality_usable"] = 0
        axial = data.select_slot_series(frame, CONFIG)[4]
        self.assertEqual([r.SeriesInstanceUID for r in axial], ["4_0"])

    def test_reserve_later_slot_and_contrast(self):
        frame = rows().iloc[:3].copy()  # All sagittal fluid, so second slot falls back.
        chosen = data.select_slot_series(frame, CONFIG)
        self.assertEqual([r.SeriesInstanceUID for r in chosen[0]], ["0_0", "0_2"])
        self.assertEqual([r.SeriesInstanceUID for r in chosen[1]], ["0_1"])
        frame = rows()
        frame = frame[~frame.SeriesInstanceUID.isin(["0_1", "0_2"])]
        # Do not pull a non-fluid acquisition into the fluid slot as top2.
        self.assertEqual(len(data.select_slot_series(frame, CONFIG)[0]), 1)

    def test_budgets_sampling_bank_and_collation(self):
        self.assertEqual([data.split_series_budget(slot, 2) for slot in range(5)],
                         [[23, 12], [17, 8], [15, 8], [10, 5], [15, 7]])
        with patch.object(data, "load_series", side_effect=fake_series):
            ds = dataset(True)
            bank = ds.candidate_bank("study")
            self.assertEqual(len(bank["window_indices"]), 120)
            self.assertEqual(data.CANDIDATE_BUDGETS, [35, 25, 23, 15, 22])
            groups = bank["window_slot_indices"] * 2 + bank["window_series_indices"]
            self.assertEqual(torch.bincount(groups).tolist(), [23, 12, 17, 8, 15, 8, 10, 5, 15, 7])
            self.assertEqual(tuple(bank["series_features"].shape), (5, 2, 11))
            sample = ds[0]
            self.assertEqual(len(sample["images"]), 24)
            groups = sample["window_slot_indices"] * 2 + sample["window_series_indices"]
            self.assertEqual(len(groups.unique()), 10)
            torch.testing.assert_close(ds[0]["images"], sample["images"])
            ds.set_epoch(1)
            self.assertFalse(torch.equal(ds[0]["window_positions"], sample["window_positions"]))
            for image in sample["images"]:
                self.assertEqual(len(image.unique()), 1)  # Adjacent channels cannot cross acquisitions.
            valid = dataset()[0]
            self.assertEqual(len(valid["images"]), 120)
            batch = data.collate_studies([sample, valid])
            self.assertEqual(len(batch["images"]), 144)
            self.assertEqual(tuple(batch["series_mask"].shape), (2, 5, 2))
            self.assertEqual(int(batch["window_batch_indices"].sum()), 120)
            single = rows().query("SeriesInstanceUID.str.endswith('_0')", engine="python")
            self.assertEqual(len(dataset(frame=single)[0]["images"]), 80)
            mixed = rows()
            mixed.loc[mixed.SeriesInstanceUID.isin(["1_1", "1_2"]), "quality_coverage_mm"] = 20
            self.assertEqual(len(dataset(frame=mixed)[0]["images"]), 112)
            # Every top1 retains the same windows/pixels/positions/metadata as the single-series bank.
            single_bank = dataset(frame=single).candidate_bank("study")
            for slot in range(5):
                current = (bank["window_slot_indices"] == slot) & (bank["window_series_indices"] == 0)
                previous = single_bank["window_slot_indices"] == slot
                torch.testing.assert_close(bank["window_positions"][current], single_bank["window_positions"][previous])
                torch.testing.assert_close(bank["slice_images"][bank["window_indices"][current]],
                                           single_bank["slice_images"][single_bank["window_indices"][previous]])
                torch.testing.assert_close(bank["series_features"][slot, 0], single_bank["series_features"][slot, 0])
        groups = np.repeat(np.arange(10), 8)
        for seed in range(20):
            chosen = data.sample_training_windows(groups, 24, np.random.default_rng(seed))
            self.assertEqual(len(np.unique(chosen)), 24)
            self.assertEqual(len(np.unique(groups[chosen])), 10)
        with self.assertRaises(ValueError):
            data.sample_training_windows(groups, 9, np.random.default_rng(0))
        self.assertEqual(len(data.sample_training_windows([0, 1], 24, np.random.default_rng(0))), 24)

    def test_cache_identity_and_report(self):
        with TemporaryDirectory() as tmp:
            frame = rows()
            for i, record in frame.iterrows():
                directory = Path(tmp) / record.SeriesInstanceUID
                directory.mkdir()
                (directory / "1.dcm").write_bytes(b"synthetic")
                frame.at[i, "SeriesPath"] = directory
            ds = dataset(frame=frame, cache=Path(tmp) / "cache")
            selected = data.select_slot_series(frame, CONFIG)
            first = ds._cache_path("study", selected)
            swapped = [list(slot) for slot in selected]
            swapped[0].reverse()
            self.assertNotEqual(first, ds._cache_path("study", swapped))
            (selected[0][1].SeriesPath / "1.dcm").write_bytes(b"changed content")
            self.assertNotEqual(first, ds._cache_path("study", selected))
            with patch.object(data, "load_series", side_effect=fake_series) as loader:
                bank = ds.candidate_bank("study")
                self.assertEqual(loader.call_count, 10)
                ds.candidate_bank("study")
                self.assertEqual(loader.call_count, 10)
                self.assertEqual(bank["cache_schema"], 20)
        report = data.series_selection_report(rows(), CONFIG)
        self.assertEqual(int(report.selected_series_count.sum()), 10)
        self.assertEqual(int(report.changed.sum()), 0)
        self.assertTrue(report.second_series_uid.ne("").all())

    def test_slice_transformer_acquisition_and_batch_isolation(self):
        model = small_model().eval()
        features = torch.randn(7, 16)
        positions = torch.tensor([0.9, -0.8, 0.1, 0.7, -0.1, -0.5, 0.5])
        batches = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        slots = torch.zeros(7, dtype=torch.long)
        series = torch.tensor([0, 1, 0, 1, 0, 0, 1])
        pooled, mask = model.aggregate_windows(features, positions, batches, slots, series, 2)
        altered = features.clone()
        altered[(batches == 0) & (series == 1)] += torch.randn(2, 16) * 10
        changed, _ = model.aggregate_windows(altered, positions, batches, slots, series, 2)
        torch.testing.assert_close(pooled[0, 0, 0], changed[0, 0, 0])
        torch.testing.assert_close(pooled[1], changed[1])
        self.assertFalse(torch.allclose(pooled[0, 0, 1], changed[0, 0, 1]))
        perm = torch.tensor([6, 3, 1, 0, 5, 4, 2])
        reordered, _ = model.aggregate_windows(features[perm], positions[perm], batches[perm],
                                                slots[perm], series[perm], 2)
        torch.testing.assert_close(pooled, reordered)
        self.assertEqual(int(mask.sum()), 4)

    def test_fusion_masks_label_specific_weights_and_gradient(self):
        model = small_model()
        features = torch.randn(2, 5, 2, 12, 16, requires_grad=True)
        mask = torch.zeros(2, 5, 2, dtype=torch.bool)
        mask[0, 0] = True
        mask[1, 1, 0] = True
        fused = model.fuse_series(features, mask)
        self.assertTrue(torch.isfinite(fused).all())
        torch.testing.assert_close(fused[1, 1], features[1, 1, 0])
        self.assertEqual(int(torch.count_nonzero(fused[0, 1:])), 0)
        scores = model.series_attention(features)[0, 0, :, :, 0].softmax(0)
        self.assertFalse(torch.allclose(scores[:, 0], scores[:, 1]))
        fused.square().sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertEqual(int(torch.count_nonzero(features.grad[~mask])), 0)
        self.assertGreater(float(model.series_attention[1].weight.grad.abs().sum()), 0)

    def test_forward_backward_and_invalid_mask(self):
        with patch.object(data, "load_series", side_effect=fake_series):
            complete = dataset(True)[0]
            sparse = dataset(True, rows().query("SeriesInstanceUID == '4_0'"))[0]
        batch = data.collate_studies([complete, sparse])
        model = small_model()
        for metadata in (False, True):
            model.zero_grad(set_to_none=True)
            logits = model_module.predict_batch(model, batch, use_metadata=metadata)
            self.assertEqual(tuple(logits.shape), (2, 12))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.rand_like(logits))
            loss.backward()
            self.assertTrue(torch.isfinite(logits).all())
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(model.series_attention[1].weight.grad.abs().sum()), 0)
        batch["series_mask"][0, 0, 1] = 0
        with self.assertRaisesRegex(ValueError, "Every valid series"):
            model_module.predict_batch(model, batch)

    def test_single_series_matches_v14(self):
        spec = importlib.util.spec_from_file_location("v14_model", Path(__file__).parents[1] / "Baseline_v14/rsna_model.py")
        v14 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v14)
        model = small_model().eval()
        with patch.object(v14, "load_backbone", return_value=TinyBackbone()):
            old = v14.RSNADINOv2(hidden_dim=16, num_heads=4, dropout=0.0).eval()
        old.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("series_attention.")})
        with patch.object(data, "load_series", side_effect=fake_series):
            frame = rows()[rows().SeriesInstanceUID.str.endswith("_0")]
            batch = data.collate_studies([dataset(frame=frame)[0]])
        for metadata in (False, True):
            current = model_module.predict_batch(model, batch, metadata)
            previous = old(batch["images"], batch["slot_mask"], batch["series_features"][:, :, 0] if metadata else None,
                           window_positions=batch["window_positions"], window_batch_indices=batch["window_batch_indices"],
                           window_slot_indices=batch["window_slot_indices"])
            torch.testing.assert_close(current, previous)

    def test_checkpoint_ema_and_resume_guards(self):
        model = small_model()
        ema = train.ModelEMA(model)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        args = SimpleNamespace(series_selection=CONFIG, train_windows=24, span_lo=0.02, span_hi=0.98,
                               image_size=4, crop_mm=140, backbone_mode="frozen", no_metadata=True,
                               seed=42, no_ema=False, ema_decay=0.999)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            train.save_checkpoint(path, model, optimizer, scheduler, scaler, 0, 0.5, args, [], ema)
            checkpoint = torch.load(path, weights_only=False)
            train.validate_resume_checkpoint(checkpoint, args)
            restored = small_model()
            restored.load_state_dict(checkpoint["model"], strict=True)
            ema.update(model)
            ema.load_state_dict(checkpoint["ema"])
            checkpoint["multi_series"] = {}
            with self.assertRaisesRegex(ValueError, "multi-series"):
                train.validate_resume_checkpoint(checkpoint, args)
            checkpoint["multi_series"] = data.MULTI_SERIES_CONFIG
            checkpoint["multi_series"] = dict(data.MULTI_SERIES_CONFIG, version="v19_top2_v1")
            with self.assertRaisesRegex(ValueError, "multi-series"):
                train.validate_resume_checkpoint(checkpoint, args)
            checkpoint["multi_series"] = data.MULTI_SERIES_CONFIG
            checkpoint["series_selection"] = dict(CONFIG, coverage_quantile=0.3)
            with self.assertRaisesRegex(ValueError, "selection config"):
                train.validate_resume_checkpoint(checkpoint, args)
            checkpoint["series_selection"] = CONFIG
            checkpoint["architecture"] = "baseline_v14_80_candidates_quality_selection_ema"
            with self.assertRaisesRegex(ValueError, "matching"):
                train.validate_resume_checkpoint(checkpoint, args)
            json_path = Path(tmp) / "summary.json"
            train.write_json(json_path, {"multi_series": data.MULTI_SERIES_CONFIG})
            self.assertNotIn(b"\r", json_path.read_bytes())

    def test_v14_initialization_only_allows_new_attention_parameters(self):
        model = small_model()
        state = {k: v for k, v in model.state_dict().items() if not k.startswith("series_attention.")}
        args = SimpleNamespace(dinov2_model_dir="unused", encoder_chunk_size=24,
                               backbone_mode="frozen", no_metadata=True)
        with TemporaryDirectory() as tmp:
            args.init_checkpoint = Path(tmp) / "v14.pt"
            checkpoint = dict(architecture="baseline_v14_80_candidates_quality_selection_ema",
                              slots=data.SLOTS, model=state)
            torch.save(checkpoint, args.init_checkpoint)
            with patch.object(train, "RSNADINOv2", side_effect=lambda **kwargs: small_model()):
                initialized = train.build_model(args, False, 0)
                torch.testing.assert_close(initialized.slot_embedding, model.slot_embedding)
                self.assertTrue(all(not p.requires_grad for p in initialized.metadata_projection.parameters()))
                checkpoint["model"].pop("label_bias")
                torch.save(checkpoint, args.init_checkpoint)
                with self.assertRaisesRegex(ValueError, "Incompatible v14"):
                    train.build_model(args, False, 0)

    def test_notebook_source_matches_modules(self):
        root = Path(__file__).parent
        notebook = json.loads((root / "Kaggle_Inference.ipynb").read_text(encoding="utf-8"))
        for i, filename in ((1, "rsna_data.py"), (2, "rsna_model.py")):
            source = (root / filename).read_text(encoding="utf-8")
            expected = ast.parse(source)
            expected.body = [node for node in expected.body
                             if not isinstance(node, ast.ImportFrom) or node.module != "rsna_data"]
            actual = ast.parse("".join(notebook["cells"][i]["source"]))
            self.assertEqual(ast.dump(expected), ast.dump(actual))
        ast.parse("".join(notebook["cells"][3]["source"]))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
