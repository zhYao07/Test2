"""CPU integration tests for gold-selected five-fold v21 training."""

import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import train
from cv_results import summarize
from cv_split import make_manifest, fold_metadata, validate_fold_metadata
from rsna_data import LABELS, QUALITY_COLUMNS, SERIES_SELECTION_VERSION


def synthetic_studies():
    rows = []
    for i in range(78):
        gold = i < 58
        value = float(i % 2)
        rows.append(dict(StudyInstanceUID=f"s{i:03}",
                         **{label: value for label in LABELS},
                         **{f"gold__{label}": value if gold else np.nan for label in LABELS}))
    return pd.DataFrame(rows)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(1, len(LABELS))

    def forward(self, images):
        return self.backbone(images)


class TinyDataset:
    def __init__(self, studies, series, **kwargs):
        self.studies = studies

    def __len__(self):
        return len(self.studies)

    def set_epoch(self, epoch):
        pass

    def __getitem__(self, index):
        row = self.studies.iloc[index]
        return dict(study_uid=str(row.StudyInstanceUID),
                    images=torch.tensor([float(row[LABELS[0]])]),
                    targets=torch.tensor(row[LABELS].to_numpy(dtype=np.float32)),
                    label_mask=torch.ones(len(LABELS)),
                    label_weight=torch.ones(len(LABELS)), candidate_count=torch.tensor(1))


def config():
    return dict(version=SERIES_SELECTION_VERSION, coverage_quantile=0.2,
                coverage_thresholds=dict(Sagittal=40.0, Coronal=40.0, Axial=40.0))


class CrossValidationTests(unittest.TestCase):
    def test_real_split_reproducible_and_gold_excluded(self):
        root = Path(__file__).resolve().parent.parent
        official = pd.read_csv(root / "data/train.csv", dtype={"StudyInstanceUID": str})
        studies = train.prepare_studies(official, Path(__file__).parent / "label.csv")
        table, manifest = make_manifest(studies, LABELS)
        _, shuffled = make_manifest(studies.sample(frac=1, random_state=3), LABELS)
        self.assertEqual(manifest, shuffled)
        self.assertEqual(len(manifest["gold_uids"]), 58)
        metadata = [fold_metadata(manifest, fold) for fold in range(5)]
        validate_fold_metadata(metadata)
        self.assertLessEqual(table[table.fold.ge(0)].groupby("fold").size().max()
                             - table[table.fold.ge(0)].groupby("fold").size().min(), 1)
        bad = deepcopy(metadata)
        bad[0]["train_uids"].append(bad[0]["gold_uids"][0])
        with self.assertRaises(ValueError):
            validate_fold_metadata(bad)

    def test_cli_rejects_unsafe_resume_and_warm_start(self):
        for argv in (("--resume", "last.pt"), ("--fold", "0"),
                     ("--init-checkpoint", "best.pt")):
            with patch("sys.argv", ["train.py", *argv]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    train.parse_args()
        with patch("sys.argv", ["train.py", "--fold", "5"]):
            self.assertEqual(train.parse_args().fold, 5)

    def test_five_fold_training_gold_selection_resume_and_ensemble(self):
        torch.set_num_threads(1)
        studies = synthetic_studies()
        table, manifest = make_manifest(studies, LABELS)
        series = pd.DataFrame({"StudyInstanceUID": studies.StudyInstanceUID,
                               "SeriesInstanceUID": studies.StudyInstanceUID,
                               "Anatomical_Plane": "Sagittal", "quality_usable": 1})
        for column in QUALITY_COLUMNS:
            if column not in series:
                series[column] = 1.0
        observed_validations = []
        thresholds = []
        original_validate = train.validate

        def checked_validate(model, loader, *args, **kwargs):
            if not kwargs.get("return_predictions", False):
                observed_validations.append(set(loader.dataset.studies.StudyInstanceUID))
            return original_validate(model, loader, *args, **kwargs)

        def selection(rows, quantile):
            thresholds.append(set(rows.StudyInstanceUID))
            return config()

        def report(rows, *args):
            return pd.DataFrame(dict(StudyInstanceUID=rows.StudyInstanceUID,
                                     changed=0, selected_series_uid=rows.SeriesInstanceUID, slot="Sagittal"))

        with TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            with patch("sys.argv", ["train.py", "--epochs", "3", "--batch-size", "16", "--num-workers", "0", "--amp", "none", "--swa"]):
                args = train.parse_args()
            args.output_dir = Path(directory)
            args.radimagenet_sha256 = "synthetic"
            train.write_cv_manifest(args.output_dir, studies, table, manifest)
            stack.enter_context(patch.object(train, "KneeDataset", TinyDataset))
            stack.enter_context(patch.object(train, "make_loader", lambda dataset, batch, *unused: DataLoader(dataset, batch_size=batch)))
            stack.enter_context(patch.object(train, "build_model", lambda *unused: TinyModel()))
            stack.enter_context(patch.object(train, "predict_batch", lambda model, batch, **kwargs: model(batch["images"])))
            stack.enter_context(patch.object(train, "make_series_selection_config", selection))
            stack.enter_context(patch.object(train, "series_selection_report", report))
            stack.enter_context(patch.object(train, "plot_loss_curve", lambda *unused: None))
            stack.enter_context(patch.object(train, "validate", checked_validate))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            for fold in range(5):
                train.train_fold(args, False, 0, 0, 1, series, studies, manifest, fold)
            gold = set(manifest["gold_uids"])
            self.assertEqual(len(observed_validations), 20)  # 3 epochs + final SWA per fold
            self.assertTrue(all(uids == gold for uids in observed_validations))
            for fold in range(5):
                self.assertEqual(thresholds[fold], set(fold_metadata(manifest, fold)["train_uids"]))
                path = args.output_dir / f"fold{fold + 1}.pt"
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                self.assertEqual(checkpoint["cross_validation"]["fold"], fold)
                self.assertEqual(checkpoint["validation_set"], "gold58")
                self.assertTrue((args.output_dir / f"fold_{fold}" / "swa.pt").is_file())
                history = json.loads((args.output_dir / f"fold_{fold}" / "validation_history.json").read_text())
                self.assertEqual(checkpoint["best_auc"], max(item["val_macro_auc"] for item in history))
            result = summarize(args.output_dir, studies, manifest)
            self.assertEqual(result["weak_oof"]["studies"], 20)
            self.assertEqual(result["gold_holdout_ensemble"]["studies"], 58)
            args.resume = args.output_dir / "fold_0/last.pt"
            train.train_fold(args, False, 0, 0, 1, series, studies, manifest, 0)
            args.resume = args.output_dir / "fold1.pt"
            with self.assertRaisesRegex(ValueError, "fold membership"):
                train.train_fold(args, False, 0, 0, 1, series, studies, manifest, 1)

    def test_notebook_loads_only_complete_matching_v21_folds(self):
        notebook = json.loads((Path(__file__).parent / "Kaggle_Inference.ipynb").read_text(encoding="utf-8"))
        code = "".join(notebook["cells"][4]["source"])
        namespace = {"__name__": "notebook_test", "N_FOLDS": 5,
                     "ARCHITECTURE": train.ARCHITECTURE, "BACKBONE_SPEC": train.BACKBONE_SPEC,
                     "SLOTS": train.SLOTS, "CANDIDATE_BUDGETS": train.CANDIDATE_BUDGETS,
                     "validate_series_selection_config": train.validate_series_selection_config,
                     "validate_fold_metadata": validate_fold_metadata}
        exec(compile(code, "notebook", "exec"), namespace)
        self.assertEqual(namespace["CHECKPOINT_FILENAMES"], {"best.pt", *[f"fold{i}.pt" for i in range(1, 6)]})
        _, manifest = make_manifest(synthetic_studies(), LABELS)
        with TemporaryDirectory() as directory:
            paths = []
            for fold in range(5):
                path = Path(directory) / f"fold{fold + 1}.pt"
                checkpoint = dict(architecture=train.ARCHITECTURE, backbone_spec=train.BACKBONE_SPEC,
                                  slots=train.SLOTS, candidate_budgets=train.CANDIDATE_BUDGETS,
                                  series_selection=config(), model=TinyModel().state_dict(),
                                  cross_validation=fold_metadata(manifest, fold),
                                  args=dict(image_size=224, crop_mm=140, span_lo=0.02, span_hi=0.98, no_metadata=True))
                torch.save(checkpoint, path)
                paths.append(path)
            bundles, signature = namespace["load_fold_bundles"](paths[::-1])
            self.assertEqual([item["cross_validation"]["fold"] for item in bundles], list(range(5)))
            self.assertEqual(signature[0], 224)
            with self.assertRaises(ValueError):
                namespace["load_fold_bundles"](paths[:4])
            scores = [np.full((2, 12), i / 4, dtype=np.float32) for i in range(5)]
            np.testing.assert_array_equal(namespace["mean_fold_probabilities"](scores), np.full((2, 12), 0.5))


if __name__ == "__main__":
    unittest.main()
