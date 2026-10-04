"""Checks for training-only augmentation, reproducibility and v14 compatibility."""

import ast
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import rsna_data as data


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v14_reference_data", ROOT / "Baseline_v14" / "rsna_data.py")
v14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v14)


def selection_config():
    return dict(version=data.SERIES_SELECTION_VERSION, coverage_quantile=0.2,
                coverage_thresholds=dict(Sagittal=80.0, Coronal=80.0, Axial=80.0))


def synthetic_data():
    studies = pd.DataFrame([
        dict(StudyInstanceUID=uid, **{label: 0.8 for label in data.LABELS})
        for uid in ("study-a", "study-b")
    ])
    series = pd.DataFrame(dict(StudyInstanceUID=studies.StudyInstanceUID))
    generator = torch.Generator().manual_seed(5)
    slots = torch.repeat_interleave(torch.arange(5), torch.tensor(data.CANDIDATE_BUDGETS))
    centers = torch.arange(80) * 3 + 1
    bank = dict(cache_schema=data.CACHE_SCHEMA, slice_images=torch.rand(240, 14, 14, generator=generator),
                window_indices=torch.stack((centers - 1, centers, centers + 1), dim=1),
                window_positions=torch.cat([torch.linspace(-0.96, 0.96, n) for n in data.CANDIDATE_BUDGETS]),
                window_slot_indices=slots, series_features=torch.zeros(5, len(data.SERIES_FEATURES)),
                slot_mask=torch.ones(5))
    bank["slice_images"][:, 0, 0] = 0
    return studies, series, bank


class BankDataset(data.KneeDataset):
    """Picklable fixture usable by Windows spawn workers."""

    def __init__(self, config, train=True):
        studies, series, self.bank = synthetic_data()
        super().__init__(studies, series, train=train, series_selection=selection_config(),
                         intensity_augmentation=config)

    def candidate_bank(self, uid):
        return self.bank


class IntensityTests(unittest.TestCase):
    def setUp(self):
        self.config = data.make_intensity_augmentation_config(1)
        ramp = torch.linspace(0, 1, 64).reshape(1, 1, 8, 8)
        self.images = ramp.expand(10, 3, 8, 8).clone()
        self.slots = torch.arange(5).repeat_interleave(2)

    def augment(self, images=None, slots=None, epoch=0, uid="study", seed=42, config=None):
        return data.augment_series_intensity(
            self.images if images is None else images, self.slots if slots is None else slots,
            uid, epoch, seed, self.config if config is None else config)

    def test_series_and_channel_consistency(self):
        output = self.augment()
        for slot in range(5):
            torch.testing.assert_close(output[slot * 2], output[slot * 2 + 1], rtol=0, atol=0)
            for channel in (1, 2):
                torch.testing.assert_close(output[slot * 2, 0], output[slot * 2, channel], rtol=0, atol=0)
        self.assertFalse(torch.equal(output[0], output[2]))

    def test_known_formula_monotonicity_and_background(self):
        config = data.make_intensity_augmentation_config(1, (0.9, 0.9), (1.05, 1.05))
        output = self.augment(config=config)
        torch.testing.assert_close(output, (self.images.pow(0.9) * 1.05).clamp(0, 1), rtol=0, atol=0)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(((output >= 0) & (output <= 1)).all())
        self.assertTrue((output[..., 0, 0] == 0).all())
        self.assertTrue((output[0, 0].flatten().diff() >= 0).all())

    def test_input_is_not_modified(self):
        before = self.images.clone()
        output = self.augment()
        torch.testing.assert_close(before, self.images, rtol=0, atol=0)
        self.assertNotEqual(output.data_ptr(), self.images.data_ptr())

    def test_disabled_and_identity_are_exact(self):
        for config in (data.make_intensity_augmentation_config(0),
                       data.make_intensity_augmentation_config(1, (1, 1), (1, 1))):
            torch.testing.assert_close(self.augment(config=config), self.images, rtol=0, atol=0)

    def test_reproducibility_and_independent_random_stream(self):
        before = np.random.get_state()
        output = self.augment()
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        torch.testing.assert_close(output, self.augment(), rtol=0, atol=0)
        for kwargs in (dict(epoch=1), dict(uid="another"), dict(seed=2026)):
            self.assertFalse(torch.equal(output, self.augment(**kwargs)))

    def test_reordering_and_missing_slots_preserve_parameters(self):
        output = self.augment()
        permutation = torch.tensor([8, 3, 1, 5, 7, 2, 9, 0, 6, 4])
        shuffled = self.augment(self.images[permutation], self.slots[permutation])
        torch.testing.assert_close(shuffled, output[permutation], rtol=0, atol=0)
        keep = self.slots.eq(3)
        torch.testing.assert_close(self.augment(self.images[keep], self.slots[keep]), output[keep], rtol=0, atol=0)

    def test_probability_is_per_slot(self):
        config = data.make_intensity_augmentation_config(0.5, (0.9, 0.9), (1.0, 1.0))
        changed = []
        for epoch in range(60):
            output = self.augment(epoch=epoch, config=config)
            for slot in range(5):
                changed.append(not torch.equal(output[2 * slot], self.images[2 * slot]))
                torch.testing.assert_close(output[2 * slot], output[2 * slot + 1], rtol=0, atol=0)
        self.assertGreater(sum(changed), 100)
        self.assertLess(sum(changed), 200)

    def test_invalid_config_and_shape_are_rejected(self):
        for config in (None, {}, dict(self.config, probability=-0.1), dict(self.config, probability=1.1),
                       dict(self.config, probability=float("nan")), dict(self.config, gamma_range=[0, 1]),
                       dict(self.config, gain_range=[1.1, 0.9]), dict(self.config, gain_range=[None, 1]),
                       dict(self.config, gamma_range=[1]), dict(self.config, version="other")):
            with self.subTest(config=config), self.assertRaises(ValueError):
                data.validate_intensity_augmentation_config(config)
        with self.assertRaises(ValueError):
            self.augment(images=self.images[:, :1])


class DatasetTests(unittest.TestCase):
    def test_disabled_matches_v14_including_window_selection(self):
        studies, series, bank = synthetic_data()
        baseline = v14.KneeDataset(studies, series, train=True, series_selection=selection_config())
        baseline.candidate_bank = lambda uid: bank
        disabled = BankDataset(data.make_intensity_augmentation_config(0))
        augmented = BankDataset(data.make_intensity_augmentation_config(1))
        for epoch in (0, 1, 5):
            for dataset in (baseline, disabled, augmented):
                dataset.set_epoch(epoch)
            expected, actual, transformed = baseline[0], disabled[0], augmented[0]
            for key in expected:
                if isinstance(expected[key], torch.Tensor):
                    torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
                    if key != "images":
                        torch.testing.assert_close(expected[key], transformed[key], rtol=0, atol=0)
                else:
                    self.assertEqual(expected[key], actual[key])
            self.assertEqual(actual["images"].shape, (24, 3, 14, 14))
            self.assertFalse(torch.equal(actual["images"], transformed["images"]))

    def test_validation_is_unaugmented_all80_and_cache_unchanged(self):
        dataset = BankDataset(data.make_intensity_augmentation_config(1), train=False)
        before = dataset.bank["slice_images"].clone()
        sample = dataset[0]
        self.assertEqual(sample["images"].shape, (80, 3, 14, 14))
        expected = dataset.bank["slice_images"][dataset.bank["window_indices"]]
        torch.testing.assert_close(sample["images"], expected, rtol=0, atol=0)
        dataset.train = True
        dataset[0]
        torch.testing.assert_close(dataset.bank["slice_images"], before, rtol=0, atol=0)

    def test_mmap_v14_cache_is_reused_without_modification(self):
        studies, _, bank = synthetic_data()
        with tempfile.TemporaryDirectory() as directory:
            rows = []
            for i, (plane, preference, _) in enumerate(data.SLOT_SPECS):
                row = dict(StudyInstanceUID="study-a", SeriesInstanceUID=str(i), Anatomical_Plane=plane,
                           Fluid_Sensitive=preference or 0, Fat_Suppression=preference or 0,
                           NumSlices=1, SeriesPath=Path(directory) / str(i))
                row.update({column: 0 for column in data.QUALITY_COLUMNS})
                rows.append(row)
            series = pd.DataFrame(rows)
            kwargs = dict(train=True, cache_dir=directory, series_selection=selection_config())
            baseline = v14.KneeDataset(studies.iloc[:1], series, **kwargs)
            current = data.KneeDataset(studies.iloc[:1], series, intensity_augmentation=data.make_intensity_augmentation_config(1), **kwargs)
            selected = data.select_slot_series(series, selection_config())
            cache_path = baseline._cache_path("study-a", selected)
            self.assertEqual(cache_path, current._cache_path("study-a", selected))
            torch.save(bank, cache_path)
            before = cache_path.read_bytes()
            with patch.object(current, "_build_bank", side_effect=AssertionError("Should reuse v14 cache")):
                current[0]
            self.assertEqual(before, cache_path.read_bytes())
            cached = torch.load(cache_path, weights_only=True, mmap=True)
            torch.testing.assert_close(cached["slice_images"], bank["slice_images"], rtol=0, atol=0)

    def test_spawn_workers_and_persistent_epoch_updates(self):
        dataset = BankDataset(data.make_intensity_augmentation_config(1))
        loader = DataLoader(dataset, batch_size=2, num_workers=2, persistent_workers=True,
                            multiprocessing_context="spawn", collate_fn=data.collate_studies)
        try:
            for epoch in (0, 1):
                dataset.set_epoch(epoch)
                expected = data.collate_studies([dataset[i] for i in range(len(dataset))])
                actual = list(loader)[0]
                for key in ("images", "window_positions", "window_slot_indices"):
                    torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
        finally:
            if loader._iterator is not None:
                loader._iterator._shutdown_workers()


class IntegrationTests(unittest.TestCase):
    def test_cli_checkpoint_roundtrip_and_resume_guards(self):
        import train
        with patch.object(sys, "argv", ["train.py", "--epochs", "20", "--ema-decay", "0.9995"]):
            args = train.parse_args()
        self.assertEqual(args.intensity_augmentation, data.make_intensity_augmentation_config())
        args.series_selection = selection_config()
        model = torch.nn.Linear(3, 12)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        ema = train.ModelEMA(model, args.ema_decay)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            train.save_checkpoint(path, model, optimizer, scheduler, scaler, 0, 0.5, args, [], ema)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["intensity_augmentation"], args.intensity_augmentation)
        train.validate_resume_checkpoint(checkpoint, args)
        changed = copy.deepcopy(checkpoint)
        changed["intensity_augmentation"]["probability"] = 0
        old = dict(checkpoint, architecture="baseline_v14_80_candidates_quality_selection_ema")
        for incompatible in (changed, old, {k: v for k, v in checkpoint.items() if k != "intensity_augmentation"}):
            with self.assertRaises(ValueError):
                train.validate_resume_checkpoint(incompatible, args)
        for argv in (["--intensity-aug-prob", "nan"], ["--intensity-gamma-range", "1.1", "0.9"]):
            with patch.object(sys, "argv", ["train.py", *argv]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    train.parse_args()

    def test_source_model_split_sampler_and_notebook_parity(self):
        def tree(version, name):
            return ast.parse((ROOT / version / name).read_text(encoding="utf-8"))

        def node(module, name):
            return next(item for item in module.body if isinstance(item, (ast.FunctionDef, ast.ClassDef)) and item.name == name)

        before, after = tree("Baseline_v14", "rsna_data.py"), tree("Baseline_v17", "rsna_data.py")
        for name in ("select_slot_series", "load_series", "sample_training_windows", "physical_center_crop"):
            self.assertEqual(ast.dump(node(before, name)), ast.dump(node(after, name)), name)
        for name in ("_cache_path", "_build_bank", "candidate_bank"):
            self.assertEqual(ast.dump(node(node(before, "KneeDataset"), name)), ast.dump(node(node(after, "KneeDataset"), name)), name)
        before_train, after_train = tree("Baseline_v14", "train.py"), tree("Baseline_v17", "train.py")
        for name in ("prepare_studies", "ModelEMA", "train_epoch", "validate", "masked_bce"):
            self.assertEqual(ast.dump(node(before_train, name)), ast.dump(node(after_train, name)), name)

        def split_statements(module):
            statements = node(module, "main").body
            first = next(i for i, item in enumerate(statements) if isinstance(item, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "studies" for target in item.targets))
            last = next(i for i, item in enumerate(statements) if isinstance(item, ast.Assign) and any(isinstance(target, ast.Attribute) and target.attr == "series_selection" for target in item.targets))
            return [ast.dump(item) for item in statements[first:last + 1]]

        self.assertEqual(split_statements(before_train), split_statements(after_train))
        before_model, after_model = tree("Baseline_v14", "rsna_model.py"), tree("Baseline_v17", "rsna_model.py")
        for module in (before_model, after_model):
            module.body = [item for item in module.body if not (isinstance(item, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "ARCHITECTURE" for target in item.targets))]
        self.assertEqual(ast.dump(before_model), ast.dump(after_model))
        self.assertEqual((ROOT / "Baseline_v14/label.csv").read_bytes(), (ROOT / "Baseline_v17/label.csv").read_bytes())
        notebook = json.loads((ROOT / "Baseline_v17/Kaggle_Inference.ipynb").read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"notebook {index}", "exec")
        self.assertEqual(ast.dump(ast.parse("".join(notebook["cells"][1]["source"]))), ast.dump(after))
        model_source = tree("Baseline_v17", "rsna_model.py")
        model_source.body = [item for item in model_source.body if not (isinstance(item, ast.ImportFrom) and item.module == "rsna_data")]
        self.assertEqual(ast.dump(ast.parse("".join(notebook["cells"][2]["source"]))), ast.dump(model_source))
        worker = node(ast.parse("".join(notebook["cells"][3]["source"])), "inference_worker")
        dataset_call = next(item for item in ast.walk(worker) if isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == "KneeDataset")
        self.assertIs(next(keyword.value.value for keyword in dataset_call.keywords if keyword.arg == "train"), False)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
