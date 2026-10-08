"""Five weak-label folds with a fixed gold validation set for epoch selection."""

import hashlib
import json

import numpy as np
import pandas as pd

CV_METHOD = "v21_5fold_weak5_iterative_gold58_selection_v1"
N_FOLDS = 5
GOLD_COUNT = 58


def label_features(studies, labels):
    values = studies[labels].to_numpy(dtype=float)
    usable = np.isfinite(values) & (values != 0.5)
    # Stratification changes membership only, never the soft training targets.
    return np.concatenate([usable & (values > 0.5), usable & (values < 0.5), ~usable], axis=1)


def iterative_folds(features, seed):
    """Assign rare label evidence first, using remaining label/size quotas.

    Based on first-order iterative multi-label stratification. Hard capacities
    keep fold sizes within one study; UID sorting happens before this function.
    """
    features = np.asarray(features, dtype=bool)
    count = len(features)
    if count < N_FOLDS:
        raise ValueError("At least five weak-label studies are required")
    rng = np.random.RandomState(seed)
    capacities = np.array([count // N_FOLDS + (fold < count % N_FOLDS) for fold in range(N_FOLDS)])
    remaining = capacities.copy()
    desired = capacities[:, None] / count * features.sum(axis=0)[None, :]
    unassigned = np.ones(count, dtype=bool)
    assignments = np.full(count, -1, dtype=np.int64)
    while unassigned.any():
        totals = features[unassigned].sum(axis=0)
        nonzero = np.flatnonzero(totals)
        if len(nonzero):
            rare = nonzero[totals[nonzero] == totals[nonzero].min()]
            label = int(rng.choice(rare))
            candidates = np.flatnonzero(unassigned & features[:, label])
        else:
            label = None
            candidates = np.flatnonzero(unassigned)
        rng.shuffle(candidates)
        for index in candidates:
            available = np.flatnonzero(remaining > 0)
            if label is not None:
                needs = desired[available, label]
                available = available[needs == needs.max()]
            available = available[remaining[available] == remaining[available].max()]
            fold = int(rng.choice(available))
            assignments[index] = fold
            remaining[fold] -= 1
            desired[fold] -= features[index]
            unassigned[index] = False
    return assignments


def data_fingerprint(studies, labels):
    columns = [*labels, *[f"gold__{label}" for label in labels]]
    records = []
    for _, row in studies.sort_values("StudyInstanceUID").iterrows():
        values = [float(np.float32(row[column])) if pd.notna(row[column]) else None for column in columns]
        records.append([str(row.StudyInstanceUID), values])
    raw = json.dumps(records, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def make_manifest(studies, labels, seed=42):
    studies = studies.copy()
    if studies.StudyInstanceUID.isna().any():
        raise ValueError("Null StudyInstanceUID")
    studies["StudyInstanceUID"] = studies.StudyInstanceUID.astype(str)
    if studies.StudyInstanceUID.duplicated().any():
        raise ValueError("Duplicate StudyInstanceUID")
    studies = studies.sort_values("StudyInstanceUID").reset_index(drop=True)
    gold_columns = [f"gold__{label}" for label in labels]
    is_gold = studies[gold_columns].notna().all(axis=1)
    if int(is_gold.sum()) != GOLD_COUNT:
        raise ValueError(f"Expected {GOLD_COUNT} complete gold studies, found {int(is_gold.sum())}")
    if studies.loc[~is_gold, gold_columns].notna().any().any():
        raise ValueError("Partial gold studies require an explicit holdout policy")
    table = studies[["StudyInstanceUID"]].copy()
    table["fold"] = -1
    table.loc[~is_gold, "fold"] = iterative_folds(label_features(studies.loc[~is_gold], labels), seed)
    table["split"] = np.where(is_gold, "gold_holdout", "weak")
    manifest = dict(method=CV_METHOD, n_splits=N_FOLDS, split_seed=int(seed),
                    data_sha256=data_fingerprint(studies, labels),
                    gold_uids=table.loc[is_gold, "StudyInstanceUID"].tolist(),
                    folds={str(fold): table.loc[table.fold.eq(fold), "StudyInstanceUID"].tolist()
                           for fold in range(N_FOLDS)})
    raw = json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    manifest["manifest_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return table, manifest


def fold_metadata(manifest, fold):
    if fold not in range(N_FOLDS):
        raise ValueError("Fold must be 0..4")
    return dict(method=CV_METHOD, n_splits=N_FOLDS, split_seed=manifest["split_seed"], fold=int(fold),
                manifest_sha256=manifest["manifest_sha256"], data_sha256=manifest["data_sha256"],
                train_uids=sorted(uid for other in range(N_FOLDS) if other != fold
                                  for uid in manifest["folds"][str(other)]),
                valid_uids=manifest["folds"][str(fold)], gold_uids=manifest["gold_uids"])


def validate_fold_metadata(metadata):
    """Reject missing/duplicate folds, mixed experiments and holdout leakage."""
    if len(metadata) != N_FOLDS or any(not isinstance(item, dict) for item in metadata):
        raise ValueError("Exactly five v21_5fold cross-validation checkpoints are required")
    if {item.get("fold") for item in metadata} != set(range(N_FOLDS)):
        raise ValueError("Checkpoints must contain each fold 0..4 exactly once")
    first = metadata[0]
    for item in metadata:
        if item.get("method") != CV_METHOD or item.get("n_splits") != N_FOLDS:
            raise ValueError("Checkpoint is not a v21_5fold weak-five-fold/gold-selection model")
        for name in ("split_seed", "manifest_sha256", "data_sha256", "gold_uids"):
            if name not in first or item.get(name) != first[name]:
                raise ValueError(f"Cross-validation metadata mismatch: {name}")
        for name in ("train_uids", "valid_uids", "gold_uids"):
            uids = item.get(name)
            if not isinstance(uids, list) or not uids or any(not isinstance(uid, str) for uid in uids) or len(set(uids)) != len(uids):
                raise ValueError(f"Invalid checkpoint membership: {name}")
        train, valid, gold = (set(item[name]) for name in ("train_uids", "valid_uids", "gold_uids"))
        if len(gold) != GOLD_COUNT or train & valid or gold & (train | valid):
            raise ValueError("Training/validation/gold holdout overlap or wrong gold count")
    all_valid = set().union(*(set(item["valid_uids"]) for item in metadata))
    if len(all_valid) != sum(len(item["valid_uids"]) for item in metadata):
        raise ValueError("Validation studies overlap between folds")
    for item in metadata:
        if set(item["train_uids"]) != all_valid - set(item["valid_uids"]):
            raise ValueError("Each fold must train on exactly the other four weak folds")


def distribution_table(studies, table, labels):
    aligned = studies.set_index("StudyInstanceUID").loc[table.StudyInstanceUID]
    rows = []
    for fold in [-1, *range(N_FOLDS)]:
        values = aligned.loc[table.fold.to_numpy() == fold, labels].to_numpy(dtype=float)
        for index, label in enumerate(labels):
            usable = np.isfinite(values[:, index]) & (values[:, index] != 0.5)
            rows.append(dict(fold=fold, label=label, studies=len(values),
                             positive=int((usable & (values[:, index] > 0.5)).sum()),
                             negative=int((usable & (values[:, index] < 0.5)).sum()),
                             excluded=int((~usable).sum())))
    return pd.DataFrame(rows)
