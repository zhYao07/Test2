"""Regression checks for top3 EMA selection, averaging, resume and inference export."""

import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

import train
from rsna_data import SERIES_SELECTION_VERSION


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([100.0]))
        self.register_buffer("floating_buffer", torch.tensor([50.0]))
        self.register_buffer("counter", torch.tensor(99, dtype=torch.long))


def args():
    return SimpleNamespace(train_windows=24, span_lo=0.02, span_hi=0.98, image_size=224,
                           crop_mm=140.0, backbone_mode="layer3+layer4", no_metadata=True,
                           seed=42, no_ema=False, ema_decay=0.9995, radimagenet_sha256="test",
                           swa=True, series_selection=dict(version=SERIES_SELECTION_VERSION,
                           coverage_quantile=0.2, coverage_thresholds=dict(Sagittal=40.0, Coronal=40.0, Axial=40.0)))


def candidates():
    raw = TinyModel()
    ema = train.ModelEMA(raw, decay=0.9995)
    tracker = train.Top3EMA()
    with torch.no_grad():
        for epoch, (auc, value) in enumerate(((0.65, 1), (0.95, 4), (0.70, 7), (0.90, 10), (0.85, 13))):
            ema.model.weight.fill_(value)
            ema.model.floating_buffer.fill_(value * 2)
            ema.model.counter.fill_(epoch)
            tracker.update(ema, epoch, auc)
    return raw, ema, tracker


class SWATests(unittest.TestCase):
    def test_selection_uses_top3_ema_not_raw_and_snapshots_are_independent(self):
        raw, ema, tracker = candidates()
        self.assertEqual([entry["epoch"] for entry in tracker.entries], [1, 3, 4])
        averaged = tracker.average()
        torch.testing.assert_close(averaged["weight"], torch.tensor([9.0]))
        torch.testing.assert_close(averaged["floating_buffer"], torch.tensor([18.0]))
        self.assertEqual(averaged["counter"].item(), 1)
        self.assertEqual(raw.weight.item(), 100)
        self.assertTrue(all(source["weight"] == 1 / 3 for source in tracker.sources()))
        self.assertEqual([source["epoch"] for source in tracker.sources()], [2, 4, 5])
        with torch.no_grad():
            ema.model.weight.fill_(-999)
        torch.testing.assert_close(tracker.average()["weight"], averaged["weight"])
        with self.assertRaisesRegex(TypeError, "ModelEMA"):
            tracker.update(raw, 5, 0.99)
        self.assertFalse(tracker.update(ema, 5, float("nan")))
        self.assertFalse(tracker.update(ema, 5, float("inf")))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            tracker.update(ema, 1, 0.99)

    def test_ties_insufficient_candidates_and_corrupted_states(self):
        ema = train.ModelEMA(TinyModel())
        tracker = train.Top3EMA()
        for epoch in range(3):
            tracker.update(ema, epoch, 0.8)
        self.assertFalse(tracker.update(ema, 3, 0.8))
        self.assertEqual([e["epoch"] for e in tracker.entries], [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "three"):
            train.Top3EMA().average()
        malformed = deepcopy(tracker.entries)
        malformed[1]["epoch"] = malformed[0]["epoch"]
        with self.assertRaisesRegex(ValueError, "epochs"):
            tracker.load_state_dict(malformed)
        malformed = deepcopy(tracker.entries)
        malformed[1]["model"].pop("counter")
        tracker.load_state_dict(malformed)
        with self.assertRaisesRegex(ValueError, "keys"):
            tracker.average()
        _, _, tracker = candidates()
        tracker.entries[1]["model"]["weight"] = torch.zeros(2)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            tracker.average()

    def test_checkpoint_resume_retains_candidates_and_continues_ranking(self):
        raw, ema, tracker = candidates()
        config = args()
        optimizer = torch.optim.SGD(raw.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            train.save_checkpoint(path, raw, optimizer, scheduler, scaler, 4, 0.95, config, [], ema, tracker)
            checkpoint = torch.load(path, weights_only=False)
            train.validate_resume_checkpoint(checkpoint, config)
            # Training checkpoint retains the current EMA, not the fused state.
            self.assertEqual(checkpoint["model"]["weight"].item(), 13)
            resumed = train.Top3EMA()
            resumed.load_state_dict(checkpoint["swa_top3"])
            torch.testing.assert_close(resumed.average()["weight"], tracker.average()["weight"])
            with torch.no_grad():
                ema.model.weight.fill_(16)
            resumed.update(ema, 5, 0.98)
            self.assertEqual([entry["epoch"] for entry in resumed.entries], [5, 1, 3])
            torch.testing.assert_close(resumed.average()["weight"], torch.tensor([10.0]))
            missing = dict(checkpoint, swa_top3=None)
            with self.assertRaisesRegex(ValueError, "requires saved"):
                train.validate_resume_checkpoint(missing, config)
            disabled = deepcopy(config)
            disabled.swa = False
            with self.assertRaisesRegex(ValueError, "mismatch: swa"):
                train.validate_resume_checkpoint(checkpoint, disabled)
            old = dict(checkpoint, args={k: v for k, v in checkpoint["args"].items() if k != "swa"})
            old.pop("swa_top3")
            train.validate_resume_checkpoint(old, disabled)

    def test_finalize_exports_validated_fusion_and_handles_missing_top3(self):
        _, ema, tracker = candidates()
        config = args()
        def validate(model, *unused):
            self.assertEqual(model.weight.item(), 9)
            return 0.91, {"ACL": 0.92}
        with TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(train, "validate", side_effect=validate) as evaluate:
                train.finalize_swa(tracker, ema, [], torch.device("cpu"), config, False, 1, 0, output)
                evaluate.assert_called_once()
            checkpoint = torch.load(output / "swa.pt", weights_only=False)
            self.assertEqual(checkpoint["weights_type"], "swa_ema_top3")
            self.assertEqual(checkpoint["val_macro_auc"], 0.91)
            self.assertEqual(checkpoint["args"]["image_size"], 224)
            self.assertEqual(checkpoint["swa_sources"], tracker.sources())
            self.assertNotIn("training_model", checkpoint)
            torch.testing.assert_close(checkpoint["model"]["weight"], torch.tensor([9.0]))
            metrics = json.loads((output / "swa_validation.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["val_macro_auc"], 0.91)
            with patch.object(train, "validate") as evaluate:
                train.finalize_swa(train.Top3EMA(), ema, [], torch.device("cpu"), config, False, 1, 0, output)
                evaluate.assert_not_called()

    def test_distributed_finalize_broadcasts_the_average_before_validation(self):
        _, ema, tracker = candidates()
        with TemporaryDirectory() as directory:
            events = []
            def broadcast(value, src):
                events.append("broadcast")
                self.assertEqual(src, 0)
            def evaluate(model, *unused):
                events.append("validate")
                self.assertEqual(model.weight.item(), 9)
                return 0.9, {}
            with patch.object(train.dist, "broadcast", side_effect=broadcast), patch.object(train, "validate", side_effect=evaluate):
                train.finalize_swa(tracker, ema, [], torch.device("cpu"), args(), True, 2, 0, Path(directory))
            self.assertEqual(events, ["broadcast"] * (1 + len(ema.model.state_dict())) + ["validate"])
            # A worker without candidate snapshots receives the rank-0 model tensors.
            _, worker, _ = candidates()
            average = tracker.average()
            values = iter(average.values())
            def receive(value, src):
                if value.shape == torch.Size([]) and value.dtype == torch.int64 and value.item() == 0:
                    value.fill_(1)
                else:
                    value.copy_(next(values))
            with patch.object(train.dist, "broadcast", side_effect=receive), patch.object(train, "validate", return_value=(0.9, {})) as evaluate:
                train.finalize_swa(None, worker, [], torch.device("cpu"), args(), True, 2, 1, Path(directory))
                evaluate.assert_called_once()
                self.assertEqual(worker.model.weight.item(), 9)

    def test_cli_switch_requires_ema(self):
        for cli, enabled in (([], False), (["--swa"], True), (["--no-swa"], False)):
            with patch.object(sys, "argv", ["train.py", *cli]):
                self.assertEqual(train.parse_args().swa, enabled)
        with patch.object(sys, "argv", ["train.py", "--swa", "--no-ema"]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                train.parse_args()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
