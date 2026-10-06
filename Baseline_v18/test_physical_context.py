"""Physical 2.5D input, cache, sampling and training/inference consistency checks."""

import ast
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rsna_data as data
import train
from audit_physical_context import audit_series, summarize
import audit_physical_context as audit

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("reference_v14", HERE.parent / "Baseline_v14/rsna_data.py")
v14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v14)


def selection_config():
    return dict(version=data.SERIES_SELECTION_VERSION, coverage_quantile=.2,
                coverage_thresholds=dict(Sagittal=20.0, Coronal=20.0, Axial=20.0))


def write_series(path, spacing):
    path.mkdir(parents=True)
    for index in range(13):
        uid = generate_uid()
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = uid
        file = FileDataset(str(path / f"{12 - index:02d}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        file.SOPClassUID, file.SOPInstanceUID = MRImageStorage, uid
        file.ImageOrientationPatient = [1, 0, 0, 0, -1, 0]  # Negative normal is canonicalized.
        file.ImagePositionPatient = [0, 0, index * spacing - 100]
        file.PixelSpacing = [1, 1]
        file.Rows, file.Columns = 14, 14
        file.SamplesPerPixel = 1
        file.PhotometricInterpretation = "MONOCHROME2"
        file.BitsAllocated, file.BitsStored, file.HighBit, file.PixelRepresentation = 16, 16, 15, 0
        file.EchoTime, file.RepetitionTime = 30, 2000
        pixels = (np.arange(196).reshape(14, 14) + 1 + index * 100).astype(np.uint16)
        file.PixelData = pixels.tobytes()
        file.save_as(file.filename, enforce_file_format=True)


class PhysicalNeighbors(unittest.TestCase):
    def test_regular_spacing(self):
        for spacing, expected in ((1, [3, 6, 9]), (2, [5, 6, 7]),
                                  (3, [5, 6, 7]), (4, [5, 6, 7]), (7, [6, 6, 6])):
            with self.subTest(spacing=spacing):
                actual = data.physical_neighbor_indices(np.arange(13) * spacing, [6])
                np.testing.assert_array_equal(actual, [expected])

    def test_ties_symmetric_and_closer_to_anchor(self):
        np.testing.assert_array_equal(data.physical_neighbor_indices(np.arange(13) * 2, [6]), [[5, 6, 7]])
        np.testing.assert_array_equal(data.physical_neighbor_indices(np.arange(13) * 6, [6]), [[6, 6, 6]])

    def test_boundaries_singleton_and_irregular(self):
        np.testing.assert_array_equal(data.physical_neighbor_indices([0, 1, 2], [0, 2]), [[0, 0, 2], [0, 2, 2]])
        np.testing.assert_array_equal(data.physical_neighbor_indices([5], [0]), [[0, 0, 0]])
        np.testing.assert_array_equal(data.physical_neighbor_indices([-10, -6, -1, 0, 4, 9], [3]), [[2, 3, 4]])

    def test_duplicate_positions_preserve_anchor(self):
        np.testing.assert_array_equal(data.physical_neighbor_indices([0, 3, 3, 6, 6, 9], [2]), [[0, 2, 3]])
        np.testing.assert_array_equal(data.physical_neighbor_indices([0, 0, 0], [1]), [[1, 1, 1]])

    def test_translation_invariance(self):
        positions = np.array([0, 1.2, 3, 4.6, 7.8, 13])
        np.testing.assert_array_equal(data.physical_neighbor_indices(positions, [1, 3, 5]),
                                      data.physical_neighbor_indices(positions + 150.25, [1, 3, 5]))

    def test_invalid_geometry_and_configuration(self):
        for positions, anchors in (([], [0]), ([0, np.nan], [0]), ([2, 1], [0]),
                                   ([0, 1], [-1]), ([0, 1], [2]), ([0, 1], [0.5])):
            with self.assertRaises(ValueError):
                data.physical_neighbor_indices(positions, anchors)
        for value in (0, -3, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                data.make_physical_context_config(value)
        with self.assertRaises(ValueError):
            data.validate_physical_context_config(dict(version="old", context_mm=3))


class PipelineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        records = []
        for index, (plane, fluid, _) in enumerate(data.SLOT_SPECS):
            path = cls.root / "train_series" / "study-a" / str(index)
            write_series(path, [1, 2, 3, 4, 7][index])
            records.append(dict(StudyInstanceUID="study-a", SeriesInstanceUID=str(index),
                                SeriesPath=path, Anatomical_Plane=plane, Fluid_Sensitive=fluid or 0,
                                Fat_Suppression=fluid or 0, NumSlices=13))
        cls.series = data.add_series_quality(pd.DataFrame(records), workers=1)
        cls.studies = pd.DataFrame([dict(StudyInstanceUID="study-a", **{label: .8 for label in data.LABELS})])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def dataset(self, train_mode=False, **kwargs):
        return data.KneeDataset(self.studies, self.series, train=train_mode, image_size=14, crop_mm=14,
                                series_selection=selection_config(), **kwargs)

    def test_dicom_physical_order_and_center_unchanged(self):
        path = self.series.iloc[0].SeriesPath
        _, positions, _ = data.get_sorted_dicom_info(path)
        np.testing.assert_allclose(np.diff(positions), 1)
        old = v14.load_series(path, 23, image_size=14, crop_mm=14)
        new = data.load_series(path, 23, image_size=14, crop_mm=14)
        old_images, new_images = old[0][old[1]], new[0][new[1]]
        torch.testing.assert_close(old_images[:, 1], new_images[:, 1], rtol=0, atol=0)
        self.assertFalse(torch.equal(old_images[:, 0], new_images[:, 0]))
        for index in (2, 3):
            torch.testing.assert_close(old[index], new[index], rtol=0, atol=0)

    def test_3mm_series_matches_v14_exactly(self):
        path = self.series.iloc[2].SeriesPath
        old = v14.load_series(path, 15, image_size=14, crop_mm=14)
        new = data.load_series(path, 15, image_size=14, crop_mm=14)
        for a, b in zip(old, new):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_cache_reuse_isolation_and_context_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.dataset(cache_dir=directory)
            selected = data.select_slot_series(self.series, selection_config(), 14, 14)
            path = dataset._cache_path("study-a", selected)
            old = v14.KneeDataset(self.studies, self.series, image_size=14, crop_mm=14,
                                  cache_dir=directory, series_selection=selection_config())
            self.assertNotEqual(path, old._cache_path("study-a", selected))
            other = self.dataset(cache_dir=directory, physical_context=data.make_physical_context_config(4))
            self.assertNotEqual(path, other._cache_path("study-a", selected))
            bank = dataset.candidate_bank("study-a")
            self.assertEqual(bank["physical_context"], data.make_physical_context_config())
            with patch.object(data, "get_sorted_dicom_info", side_effect=AssertionError("Cache should prevent header reads")):
                hit = dataset.candidate_bank("study-a")
            torch.testing.assert_close(bank["slice_images"], hit["slice_images"], rtol=0, atol=0)
            del hit  # Release the Windows mmap handle before deliberately corrupting the fixture.
            bank["physical_context"] = data.make_physical_context_config(4)
            torch.save(bank, path)
            with self.assertRaisesRegex(ValueError, "Invalid input cache"):
                dataset.candidate_bank("study-a")

    def test_shared_quality_cache_migrates_legacy_without_header_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "input_cache" / "series_quality"
            shared = Path(directory) / "series_quality_cache"
            path = self.series.iloc[0].SeriesPath
            expected = data._cached_series_quality(path, legacy)
            legacy_file = next(legacy.glob("*.json"))
            original_bytes = legacy_file.read_bytes()
            with patch.object(data, "read_series_quality", side_effect=AssertionError("Valid legacy cache must prevent scans")):
                migrated = data._cached_series_quality(path, shared, legacy)
                self.assertEqual(migrated, expected)
                self.assertEqual(data._cached_series_quality(path, shared), expected)
            self.assertEqual(legacy_file.read_bytes(), original_bytes)
            self.assertEqual(len(list(shared.glob("*.json"))), 1)
            self.assertNotIn(b"\r\n", next(shared.glob("*.json")).read_bytes())
            # The new directory keeps the same source-file invalidation rules.
            source = next(path.glob("*.dcm"))
            stat = source.stat()
            try:
                os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
                with patch.object(data, "read_series_quality", return_value=expected) as reader:
                    data._cached_series_quality(path, shared, legacy)
                    reader.assert_called_once_with(path)
                self.assertEqual(len(list(shared.glob("*.json"))), 2)
            finally:
                os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    def test_build_datasets_keeps_quality_outside_image_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            images, quality = Path(directory) / "input_cache", Path(directory) / "series_quality_cache"
            metadata = (self.studies, self.series, self.studies, self.series, self.studies)
            with patch.object(data, "load_metadata", return_value=metadata):
                data.build_datasets(cache_dir=images, series_quality_cache_dir=quality,
                                    series_selection=selection_config())
            self.assertEqual(len(list(quality.glob("*.json"))), 5)
            self.assertFalse(images.exists())

    def test_training24_validation80_and_sampler_unchanged(self):
        dataset = self.dataset(train_mode=True)
        first = dataset[0]
        self.assertEqual(first["images"].shape, (24, 3, 14, 14))
        self.assertEqual(first["candidate_count"].item(), 80)
        self.assertEqual(len(torch.unique(first["window_slot_indices"])), 5)
        torch.testing.assert_close(first["images"], dataset[0]["images"], rtol=0, atol=0)
        dataset.set_epoch(1)
        self.assertFalse(torch.equal(first["window_positions"], dataset[0]["window_positions"]))
        valid = self.dataset()[0]
        self.assertEqual(valid["images"].shape, (80, 3, 14, 14))
        self.assertNotIn("augmented_images", first)
        batch = data.collate_studies([first, first])
        self.assertEqual(batch["images"].shape[0], 48)
        self.assertEqual(batch["targets"].shape, (2, 12))
        self.assertEqual(batch["window_batch_indices"].tolist(), [0] * 24 + [1] * 24)

    def test_audit_matches_actual_bank(self):
        record = dict(StudyInstanceUID="study-a", slot=data.SLOTS[0], selected_series_uid="0", split="valid")
        result = audit_series(record, self.root, 3, .02, .98)
        self.assertEqual(result[0]["error"], "")
        self.assertEqual(result[0]["windows"], 23)
        self.assertGreater(result[0]["changed_windows"], 0)
        bank = self.dataset().candidate_bank("study-a")
        self.assertEqual(len(bank["window_indices"][bank["window_slot_indices"] == 0]), 23)
        summary = summarize([result], data.make_physical_context_config())
        self.assertEqual(summary["candidate_windows"], 23)
        failed = audit_series(dict(record, selected_series_uid="absent"), self.root, 3, .02, .98)
        self.assertEqual(summarize([failed], data.make_physical_context_config())["failed_slots"], 1)

    def test_notebook_preprocessing_matches_dataset(self):
        notebook = json.loads((HERE / "Kaggle_Inference.ipynb").read_text(encoding="utf-8"))
        namespace = {"__name__": "notebook_test"}
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                compile(source, "notebook", "exec")
                if "def get_sorted_dicom_info" in source:
                    exec(source, namespace)
        dataset = namespace["KneeDataset"](self.studies, self.series, image_size=14, crop_mm=14,
                                            series_selection=selection_config(), train=False)
        expected, actual = self.dataset()[0], dataset[0]
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, actual[key], rtol=0, atol=0)

    def test_audit_cli_writes_complete_validation_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            report = data.series_selection_report(self.series, selection_config(), 14, 14)
            report["split"] = "valid"
            report_path = Path(directory) / "selection.csv"
            with report_path.open("w", encoding="utf-8", newline="\n") as handle:
                report.to_csv(handle, index=False, lineterminator="\n")
            output = Path(directory) / "audit"
            argv = ["audit.py", "--data-root", str(self.root), "--series-selection-csv", str(report_path),
                    "--output-dir", str(output), "--split", "valid", "--workers", "2"]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                audit.main()
            summary = json.loads((output / "physical_context_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["candidate_windows"], 80)
            self.assertEqual(summary["successful_slots"], 5)
            self.assertEqual(summary["failed_slots"], 0)
            self.assertEqual(len(pd.read_csv(output / "physical_context.csv")), 5)
            for path in output.iterdir():
                self.assertNotIn(b"\r\n", path.read_bytes())


class ConfigurationAndRegression(unittest.TestCase):
    def test_cli_defaults_and_reject_invalid_context(self):
        with patch.object(sys, "argv", ["train.py"]):
            args = train.parse_args()
        self.assertEqual(args.context_mm, 3)
        self.assertEqual(args.physical_context, data.make_physical_context_config())
        with patch.object(sys, "argv", ["train.py", "--context-mm", "0"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                train.parse_args()

    def test_quality_cache_cli_is_independent_of_preprocessing(self):
        for flags in ([], ["--no-cache"], ["--no-series-quality-cache"],
                      ["--no-cache", "--no-series-quality-cache"]):
            with patch.object(sys, "argv", ["train.py", "--cache-dir", "custom_images", *flags]):
                args = train.parse_args()
            image_cache, quality_cache = train.resolve_cache_directories(args)
            self.assertEqual(image_cache, None if "--no-cache" in flags else Path("custom_images"))
            self.assertEqual(quality_cache, None if "--no-series-quality-cache" in flags else HERE.parent / "series_quality_cache")
        with patch.object(sys, "argv", ["train.py", "--series-quality-cache-dir", "custom_quality"]):
            args = train.parse_args()
        self.assertEqual(train.resolve_cache_directories(args)[1], Path("custom_quality"))

    def test_resume_context_guard(self):
        with patch.object(sys, "argv", ["train.py"]):
            args = train.parse_args()
        args.series_selection = selection_config()
        checkpoint = dict(architecture=train.ARCHITECTURE, slots=data.SLOTS,
                          candidate_budgets=data.CANDIDATE_BUDGETS, args=vars(args).copy(),
                          series_selection=selection_config(), physical_context=args.physical_context.copy(),
                          training_model={}, ema=dict(decay=args.ema_decay))
        train.validate_resume_checkpoint(checkpoint, args)
        bad = copy.deepcopy(checkpoint)
        bad["physical_context"]["context_mm"] = 4
        with self.assertRaisesRegex(ValueError, "physical context"):
            train.validate_resume_checkpoint(bad, args)
        bad = copy.deepcopy(checkpoint)
        bad.pop("physical_context")
        with self.assertRaises(ValueError):
            train.validate_resume_checkpoint(bad, args)

    def test_unchanged_v14_components_and_labels(self):
        def definitions(path):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for filename, names in (
            ("rsna_data.py", ("sample_training_windows", "collate_studies", "select_slot_series",
                              "physical_center_crop", "read_dicom", "_read_sampled_slices",
                              "get_sorted_dicom_info", "make_series_selection_config", "read_series_quality")),
            ("train.py", ("ModelEMA", "masked_bce", "train_epoch", "validate", "prepare_studies",
                          "make_optimizer", "make_scheduler", "calculate_pos_weight"))):
            old, new = definitions(HERE.parent / "Baseline_v14" / filename), definitions(HERE / filename)
            for name in names:
                self.assertEqual(old[name], new[name], name)
        self.assertEqual((HERE / "label.csv").read_bytes(), (HERE.parent / "Baseline_v14/label.csv").read_bytes())

    def test_notebook_embeds_exact_sources_and_context_forwarding(self):
        notebook = json.loads((HERE / "Kaggle_Inference.ipynb").read_text(encoding="utf-8"))
        data_source = (HERE / "rsna_data.py").read_text(encoding="utf-8").replace("# -*- coding: utf-8 -*-\n", "", 1)
        model_source = (HERE / "rsna_model.py").read_text(encoding="utf-8").replace("# -*- coding: utf-8 -*-\n", "", 1)
        model_source = model_source.replace("from rsna_data import LABELS, SERIES_FEATURES, SLOTS\n", "")
        self.assertEqual("".join(notebook["cells"][1]["source"]), data_source)
        self.assertEqual("".join(notebook["cells"][2]["source"]), model_source)
        inference = "".join(notebook["cells"][3]["source"])
        self.assertIn('validate_physical_context_config(physical_context)', inference)
        self.assertIn('physical_context=physical_context', inference)
        self.assertIn('span_lo, span_hi, series_selection, physical_context)', inference)


if __name__ == "__main__":
    unittest.main()
