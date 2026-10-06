"""Validate and aggregate completed OOF/gold predictions; no model loading."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cv_split import N_FOLDS, data_fingerprint, fold_metadata, validate_fold_metadata
from rsna_data import LABELS


def auc_report(studies, predictions):
    targets = studies.set_index("StudyInstanceUID").loc[predictions.StudyInstanceUID, LABELS].to_numpy(dtype=float)
    scores = predictions[LABELS].to_numpy(dtype=float)
    aucs, counts = {}, {}
    for index, label in enumerate(LABELS):
        usable = np.isfinite(targets[:, index]) & (targets[:, index] != 0.5)
        binary = targets[usable, index] > 0.5
        counts[label] = dict(positive=int(binary.sum()), negative=int((~binary).sum()), excluded=int((~usable).sum()))
        aucs[label] = float(roc_auc_score(binary, scores[usable, index])) if len(np.unique(binary)) == 2 else None
    valid = [value for value in aucs.values() if value is not None]
    return dict(studies=len(predictions), macro_auc=float(np.mean(valid)) if valid else None,
                auc_by_label=aucs, label_counts=counts)


def load_predictions(path, expected_uids, fold):
    frame = pd.read_csv(path, dtype={"StudyInstanceUID": str}, float_precision="round_trip")
    if list(frame.columns) != ["StudyInstanceUID", "fold", *LABELS]:
        raise ValueError(f"Unexpected prediction columns: {path}")
    if frame.StudyInstanceUID.duplicated().any() or frame.StudyInstanceUID.isna().any():
        raise ValueError(f"Duplicate/null predictions: {path}")
    if set(frame.StudyInstanceUID) != set(expected_uids) or not frame.fold.eq(fold).all():
        raise ValueError(f"Prediction membership/fold mismatch: {path}")
    scores = frame[LABELS].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError(f"Invalid probabilities: {path}")
    return frame


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def summarize(output_dir, studies, manifest):
    output_dir = Path(output_dir)
    if data_fingerprint(studies, LABELS) != manifest["data_sha256"]:
        raise ValueError("CV target snapshot differs from the manifest label fingerprint")
    expected = [fold_metadata(manifest, fold) for fold in range(N_FOLDS)]
    validate_fold_metadata(expected)
    oof, gold, metrics = [], [], []
    for fold, metadata in enumerate(expected):
        directory = output_dir / f"fold_{fold}"
        actual = json.loads((directory / "cv.json").read_text(encoding="utf-8"))
        if actual != metadata:
            raise ValueError(f"Stale or different fold outputs: {directory}")
        report = json.loads((directory / "fold_metrics.json").read_text(encoding="utf-8"))
        if report.get("cross_validation") != metadata:
            raise ValueError(f"Metrics metadata mismatch: {directory}")
        oof.append(load_predictions(directory / "oof_predictions.csv", metadata["valid_uids"], fold))
        gold.append(load_predictions(directory / "gold_predictions.csv", metadata["gold_uids"], fold))
        metrics.append(report)
    oof = pd.concat(oof, ignore_index=True).sort_values("StudyInstanceUID").reset_index(drop=True)
    if oof.StudyInstanceUID.duplicated().any():
        raise ValueError("OOF studies appear in more than one fold")
    gold_uids = manifest["gold_uids"]
    gold_scores = np.stack([frame.set_index("StudyInstanceUID").loc[gold_uids, LABELS].to_numpy(dtype=np.float32) for frame in gold])
    ensemble = pd.DataFrame(gold_scores.mean(axis=0, dtype=np.float64).astype(np.float32), columns=LABELS)
    ensemble.insert(0, "StudyInstanceUID", gold_uids)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False, lineterminator="\n")
    ensemble.to_csv(output_dir / "gold_ensemble_predictions.csv", index=False, lineterminator="\n")
    report = dict(method=manifest["method"], manifest_sha256=manifest["manifest_sha256"],
                  weak_oof=auc_report(studies, oof), gold_holdout_ensemble=auc_report(studies, ensemble),
                  fold_weak_macro_auc=[item["weak_validation"]["macro_auc"] for item in metrics],
                  fold_gold_macro_auc=[item["gold_holdout"]["macro_auc"] for item in metrics],
                  note="Weak OOF is agreement with weak labels. Gold58 selects the best epoch in each fold; its AUC is a validation score, not an independent test score. Ensemble weights are fixed equal.")
    write_json(output_dir / "cv_metrics.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.output_dir / "cv_manifest.json").read_text(encoding="utf-8"))
    studies = pd.read_csv(args.output_dir / "cv_targets.csv", dtype={"StudyInstanceUID": str}, float_precision="round_trip")
    print(json.dumps(summarize(args.output_dir, studies, manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
