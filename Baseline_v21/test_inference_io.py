"""Exact CPU checks for v21 notebook I/O acceleration against original v14 data."""

import ast
from contextlib import nullcontext, redirect_stdout
import io
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, RLELossless, generate_uid
import torch

import rsna_data as reference
import rsna_model as models

ROOT = Path(__file__).resolve().parent
SELECTION = dict(version=reference.SERIES_SELECTION_VERSION, coverage_quantile=.2,
                 coverage_thresholds=dict(Sagittal=40., Coronal=40., Axial=40.))


def write_series(path, variant="normal", spacing=7.):
    path.mkdir(parents=True)
    for index in range(7):
        uid = generate_uid()
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = uid
        file = FileDataset(str(path / f"{6 - index:02d}.dcm"), {},
                           file_meta=meta, preamble=b"\0" * 128)
        file.SOPClassUID, file.SOPInstanceUID = MRImageStorage, uid
        file.ImageOrientationPatient = [1, 0, 0, 0, -1, 0]
        file.ImagePositionPatient = [0, 0, index * spacing - 100]
        file.PixelSpacing = [.8, 1.2]
        file.Rows, file.Columns = 18, 24
        file.SamplesPerPixel = 1
        file.PhotometricInterpretation = "MONOCHROME1" if variant == "mono1" else "MONOCHROME2"
        file.BitsAllocated, file.BitsStored, file.HighBit, file.PixelRepresentation = 16, 16, 15, 0
        file.EchoTime, file.RepetitionTime = 30, 2000
        file.RescaleSlope, file.RescaleIntercept = 1.5, -200
        pixels = (np.arange(432).reshape(18, 24) + 1 + index * 100).astype(np.uint16)
        if variant == "byte_fallback":
            # Competition-style byte payload with nominal 16-bit tags.
            pixels = (pixels % 255).astype(np.uint8)
        file.PixelData = pixels.tobytes()
        if variant == "bad_pixel" and index == 3:
            file.PixelData = b"\0\0"  # Triggers unchanged nearest-readable-slice fallback.
        if variant == "rle":
            file.compress(RLELossless)
        file.save_as(file.filename, enforce_file_format=True)


class InferenceIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        notebook = json.loads((ROOT / "Kaggle_Inference.ipynb").read_text(encoding="utf-8"))
        cls.namespace = {"__name__": "notebook_test"}
        for index in (1, 2, 3, 4):
            exec(compile("".join(notebook["cells"][index]["source"]), f"cell-{index}", "exec"), cls.namespace)
        cls.temporary = TemporaryDirectory()
        cls.root = Path(cls.temporary.name) / "test_series"
        cls.studies = pd.DataFrame({"StudyInstanceUID": ["full", "sparse", "fallback"]})
        records = []
        specs = [
            ("full", "sag_short", "Sagittal", 1, "normal", 2.),
            ("full", "sag_fluid", "Sagittal", 1, "mono1", 7.),
            ("full", "sag_second", "Sagittal", 0, "normal", 7.),
            ("full", "cor_fluid", "Coronal", 1, "bad_pixel", 7.),
            ("full", "cor_second", "Coronal", 0, "normal", 7.),
            ("full", "axial", "Axial", 0, "rle", 7.),
            ("sparse", "axial", "Axial", 0, "rle", 7.),
            ("fallback", "sag", "Sagittal", 1, "byte_fallback", 7.),
            ("fallback", "sag_bad", "Sagittal", 1, "normal", 7.),
            ("fallback", "sag_second", "Sagittal", 0, "normal", 7.),
        ]
        for uid, series, plane, fluid, variant, spacing in specs:
            directory = cls.root / uid / series
            write_series(directory, variant, spacing)
            if series == "sag_bad":
                (directory / "00.dcm").write_bytes(b"invalid DICOM header")
            records.append(dict(StudyInstanceUID=uid, SeriesInstanceUID=series,
                                Anatomical_Plane=plane, Fluid_Sensitive=fluid,
                                Fat_Suppression=fluid))
        cls.raw_rows = reference._prepare_series_df(pd.DataFrame(records), cls.root)
        cls.quality_rows = reference.add_series_quality(cls.raw_rows, workers=2)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def datasets(self, pool, image_size=224):
        kwargs = dict(image_size=image_size, crop_mm=140., train=False,
                      span_lo=.02, span_hi=.98, cache_dir=None, series_selection=SELECTION)
        original = reference.KneeDataset(self.studies, self.quality_rows, **kwargs)
        fast = self.namespace["FastInferenceDataset"](self.studies, self.raw_rows,
                                                      io_pool=pool, **kwargs)
        return original, fast

    def assert_sample_equal(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for name in first:
            if isinstance(first[name], torch.Tensor):
                self.assertEqual(first[name].dtype, second[name].dtype, name)
                self.assertEqual(first[name].shape, second[name].shape, name)
                self.assertTrue(torch.equal(first[name], second[name]), name)
            else:
                self.assertEqual(first[name], second[name], name)

    def test_all_inputs_selection_and_one_disk_read_are_exact(self):
        reads, lock, selected_rows = Counter(), threading.Lock(), []
        real_read = Path.read_bytes
        select = self.namespace["select_slot_series"]
        def counted(path):
            with lock:
                reads[path] += 1
            return real_read(path)
        def capture(rows, *args):
            selected_rows.append(rows.copy())
            return select(rows, *args)
        with ThreadPoolExecutor(max_workers=16) as pool:
            original, fast = self.datasets(pool)
            with patch.object(Path, "read_bytes", counted), patch.dict(self.namespace, select_slot_series=capture):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    for index in range(len(original)):
                        self.assert_sample_equal(original[index], fast[index])
                        self.assertFalse(hasattr(self.namespace["_study_io"], "records"))
            self.assertEqual(int(original[0]["candidate_count"]), 80)
        expected_files = list(self.root.glob("*/*/*.dcm"))
        self.assertEqual(set(reads), set(expected_files))
        self.assertTrue(all(count == 1 for count in reads.values()), reads)
        # Delaying quality computation to each study must preserve ranking and ties.
        for rows in selected_rows:
            uid = rows.StudyInstanceUID.iloc[0]
            expected = self.quality_rows[self.quality_rows.StudyInstanceUID.eq(uid)]
            pd.testing.assert_frame_equal(rows[reference.QUALITY_COLUMNS].reset_index(drop=True),
                                          expected[reference.QUALITY_COLUMNS].reset_index(drop=True),
                                          check_dtype=False)
            selected = select(rows, SELECTION, 140., 224)
            baseline = reference.select_slot_series(expected, SELECTION, 140., 224)
            identifiers = lambda slots: [None if row is None else str(row.SeriesInstanceUID) for row in slots]
            self.assertEqual(identifiers(selected), identifiers(baseline))
        self.assertEqual(selected_rows[0].StudyInstanceUID.iloc[0], "full")

    def test_prefetch_preserves_order_and_every_tensor(self):
        # Force study 1 to finish before study 0 to exercise ordered consumption.
        second_done = threading.Event()
        with ThreadPoolExecutor(max_workers=16) as io_pool, ThreadPoolExecutor(max_workers=2) as study_pool:
            original, fast = self.datasets(io_pool, image_size=32)
            class OutOfOrder:
                def __len__(self):
                    return len(fast)
                def __getitem__(self, index):
                    if index == 0 and not second_done.wait(15):
                        raise RuntimeError("Prefetch did not start the second study")
                    sample = fast[index]
                    if index == 1:
                        second_done.set()
                    return sample
            # Only page locking is unavailable on CPU; values and collate are real.
            with patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor):
                actual = list(self.namespace["prefetched_batches"](OutOfOrder(), study_pool))
            expected = [reference.collate_studies([original[index]]) for index in range(len(original))]
            for before, after in zip(expected, actual):
                self.assert_sample_equal(before, after)
            self.assertEqual([batch["study_uid"] for batch in actual], [[uid] for uid in self.studies.StudyInstanceUID])

    def test_disabled_path_uses_original_parser_and_inputs(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            original, _ = self.datasets(pool, image_size=32)
            rows = self.namespace["add_series_quality"](self.raw_rows, workers=2)
            disabled = self.namespace["KneeDataset"](self.studies, rows, image_size=32,
                       crop_mm=140., train=False, span_lo=.02, span_hi=.98,
                       cache_dir=None, series_selection=SELECTION)
            with patch.object(Path, "read_bytes", side_effect=AssertionError("Fast read used while disabled")):
                for index in range(len(original)):
                    self.assert_sample_equal(original[index], disabled[index])

    def test_raw_bytes_released_even_when_preprocessing_fails(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            _, fast = self.datasets(pool, image_size=32)
            with patch.object(fast, "_build_bank", side_effect=RuntimeError("preprocessing failure")):
                with self.assertRaisesRegex(RuntimeError, "preprocessing failure"):
                    fast[0]
            self.assertFalse(hasattr(self.namespace["_study_io"], "records"))
            self.assertEqual(fast[0]["study_uid"], "full")
            # If the original selected series is unreadable, preserve its failure.
            studies = self.studies[self.studies.StudyInstanceUID.eq("fallback")]
            rows = self.raw_rows[self.raw_rows.SeriesInstanceUID.eq("sag_bad")]
            quality = self.quality_rows[self.quality_rows.SeriesInstanceUID.eq("sag_bad")]
            baseline = reference.KneeDataset(studies, quality, image_size=32, train=False,
                                             cache_dir=None, series_selection=SELECTION)
            broken = self.namespace["FastInferenceDataset"](studies, rows, image_size=32,
                       train=False, cache_dir=None, series_selection=SELECTION, io_pool=pool)
            for dataset in (baseline, broken):
                with self.assertRaises(pydicom.errors.InvalidDicomError):
                    dataset[0]
            self.assertFalse(hasattr(self.namespace["_study_io"], "records"))

    def test_single_model_logits_and_probabilities_are_bitwise_equal(self):
        torch.manual_seed(21)
        model = models.RSNARadImageNet(load_pretrained=False, hidden_dim=16, num_heads=4,
                                       dropout=0., encoder_chunk_size=16).eval()
        with ThreadPoolExecutor(max_workers=16) as pool, torch.inference_mode():
            original, fast = self.datasets(pool, image_size=32)
            # Sparse study still exercises missing slots and a partial encoder chunk.
            before = reference.collate_studies([original[1]])
            after = self.namespace["collate_studies"]([fast[1]])
            for use_metadata in (False, True):
                first = models.predict_batch(model, before, use_metadata=use_metadata)
                second = self.namespace["predict_batch"](model, after, use_metadata=use_metadata)
                self.assertTrue(torch.equal(first, second))
                self.assertTrue(torch.equal(torch.sigmoid(first.float()), torch.sigmoid(second.float())))
            # Exercise both complete worker branches on CPU. CUDA transfer/autocast
            # are adapted only for this test; their production AST is checked below.
            tensor_to = torch.Tensor.to
            def cpu_to(tensor, *args, **kwargs):
                if args and isinstance(args[0], torch.device) and args[0].type == "cuda":
                    args = (torch.device("cpu"), *args[1:])
                return tensor_to(tensor, *args, **kwargs)
            with ThreadPoolExecutor(max_workers=2) as study_pool:
                with patch.object(torch.cuda, "set_device"), patch.object(torch.Tensor, "to", cpu_to), \
                     patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor), \
                     patch.object(torch.amp, "autocast", lambda *args, **kwargs: nullcontext()), \
                     redirect_stdout(io.StringIO()), warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    partials = []
                    for enabled in (False, True):
                        with patch.dict(self.namespace, FAST_IO=enabled):
                            partials.append(self.namespace["inference_worker"](
                                0, model, self.studies.iloc[[1]].reset_index(drop=True),
                                self.raw_rows, 32, 140., self.root.parent, True, .02, .98,
                                SELECTION, pool, study_pool))
            pd.testing.assert_frame_equal(partials[0], partials[1], check_exact=True)
            np.testing.assert_array_equal(partials[1][reference.LABELS].to_numpy(),
                                          torch.sigmoid(first.float()).numpy())

    def test_gpu_numerical_steps_match_original_notebook(self):
        notebook = json.loads((ROOT / "Kaggle_Inference.ipynb").read_text(encoding="utf-8"))
        old = json.loads((ROOT.parent / "Baseline_v14/Kaggle_Inference.ipynb").read_text(encoding="utf-8"))
        def numerical_steps(document):
            source = next("".join(cell["source"]) for cell in document["cells"]
                          if "def inference_worker(" in "".join(cell["source"]))
            function = next(node for node in ast.parse(source).body
                            if isinstance(node, ast.FunctionDef) and node.name == "inference_worker")
            loop = next(node for node in function.body if isinstance(node, ast.For))
            # Transfer -> autocast/forward -> fp32 sigmoid -> UID/probability storage.
            return [ast.dump(node) for node in loop.body[1:5]]
        self.assertEqual(numerical_steps(notebook), numerical_steps(old))


if __name__ == "__main__":
    unittest.main()
