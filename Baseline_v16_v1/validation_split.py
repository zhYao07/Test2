"""Deterministic label-only validation selection; no imaging/training dependency."""

import numpy as np
import pandas as pd

SPLIT_METHOD = "gold58_multilabel_confidence_case_mix_441_v1"


def selection_features(studies, labels):
    values = studies[labels].to_numpy(dtype=np.float64)
    usable = np.isfinite(values) & (values != 0.5)
    positive = usable & (values > 0.5)
    negative = usable & (values < 0.5)
    confidence = np.where(np.isfinite(values), 2 * np.abs(values - 0.5), 0)
    matrices = [positive, negative, ~usable]
    names = ([f"positive:{label}" for label in labels]
             + [f"negative:{label}" for label in labels]
             + [f"excluded:{label}" for label in labels])
    # Separate polarity and confidence: high confidence negatives cannot replace positives.
    bins = [(confidence < 0.4), (confidence >= 0.4) & (confidence < 0.8), confidence >= 0.8]
    for polarity, mask in (("positive", positive), ("negative", negative)):
        for name, band in zip(("low", "medium", "high"), bins):
            matrices.append(mask & band)
            names.extend(f"{polarity}_{name}:{label}" for label in labels)
    count = positive.sum(axis=1)
    for lower, upper in ((0, 0), (1, 1), (2, 2), (3, 4), (5, len(labels))):
        matrices.append(((count >= lower) & (count <= upper))[:, None])
        names.append(f"case_positive_count:{lower}-{upper}")
    # Common co-occurrences; no target selected from model validation results.
    pairs = [(int((positive[:, i] & positive[:, j]).sum()), i, j)
             for i in range(len(labels)) for j in range(i + 1, len(labels))]
    for total, i, j in sorted(pairs, key=lambda item: (-item[0], item[1], item[2]))[:12]:
        if total:
            matrices.append((positive[:, i] & positive[:, j])[:, None])
            names.append(f"pair:{labels[i]}+{labels[j]}")
    features = np.concatenate(matrices, axis=1).astype(np.float64)
    quality = confidence.mean(axis=1) * usable.mean(axis=1)
    return features, names, quality


def select_validation(studies, labels, seed=42, valid_count=441, locked_uids=()):
    """Greedy quota selection followed by deterministic swap refinement.

    Positive counts have strongest priority, especially rare labels. Other
    features retain polarity/confidence and case mix. Locked studies are
    selected first and cannot be removed by swap refinement. Confidence is a weak
    preference, not a claim of label correctness. Counts are approximate.
    """
    studies = studies.sort_values("StudyInstanceUID").reset_index(drop=True)
    if studies.StudyInstanceUID.isna().any() or studies.StudyInstanceUID.duplicated().any():
        raise ValueError("Split requires non-null, unique StudyInstanceUID values")
    if not 0 < valid_count < len(studies):
        raise ValueError("Validation count must leave at least one training Study")
    locked_uids = set(locked_uids)
    available = set(studies.StudyInstanceUID)
    if not locked_uids <= available:
        raise ValueError("Locked validation Study UID missing from labeled data")
    if len(locked_uids) > valid_count:
        raise ValueError("Locked validation studies exceed validation count")
    locked = studies.StudyInstanceUID.isin(locked_uids).to_numpy()
    features, names, quality = selection_features(studies, labels)
    target = features.sum(axis=0) * valid_count / len(studies)
    label_count = len(labels)
    target[:label_count] = np.rint(target[:label_count])
    priority = np.ones(len(names))
    priority[:label_count] = 12  # Prioritize positive quotas without sacrificing case mix.
    priority[label_count:3 * label_count] = 3
    priority[[name.startswith("case_") for name in names]] = 3
    weights = priority / np.maximum(target, 1)
    rng = np.random.RandomState(seed)
    tie = rng.uniform(0, 1e-9, len(studies))
    preference = 0.01 * quality + tie
    weighted = features * weights
    norm = (features * weighted).sum(axis=1)
    selected = locked.copy()
    counts = features[locked].sum(axis=0)
    for _ in range(valid_count - int(locked.sum())):
        delta = 2 * weighted @ (counts - target) + norm - preference
        delta[selected] = np.inf
        index = int(np.argmin(delta))
        selected[index] = True
        counts += features[index]

    # Evaluate every possible swap in blocks. Stop only at a local optimum or
    # the fixed iteration limit; deterministic for a given sorted table/seed.
    for _ in range(200):
        inside, outside = np.flatnonzero(selected & ~locked), np.flatnonzero(~selected)
        if not len(inside):
            break
        residual = counts - target
        add_cost = 2 * weighted[outside] @ residual + norm[outside] - preference[outside]
        remove_cost = -2 * weighted[inside] @ residual + norm[inside] + preference[inside]
        best_delta, best_pair = -1e-8, None
        for start in range(0, len(inside), 64):
            block = inside[start:start + 64]
            delta = (remove_cost[start:start + 64, None] + add_cost[None, :]
                     - 2 * weighted[block] @ features[outside].T)
            i, j = np.unravel_index(np.argmin(delta), delta.shape)
            if delta[i, j] < best_delta:
                best_delta, best_pair = float(delta[i, j]), (block[i], outside[j])
        if best_pair is None:
            break
        old, new = best_pair
        selected[old], selected[new] = False, True
        counts += features[new] - features[old]
    return studies.loc[~selected].reset_index(drop=True), studies.loc[selected].reset_index(drop=True)


def distribution_report(studies, valid, labels):
    full, names, full_quality = selection_features(studies, labels)
    part, _, part_quality = selection_features(valid, labels)
    # Pair feature order must come from full data, not re-ranked validation data.
    valid_values = valid[labels].to_numpy(dtype=float)
    valid_pos = valid_values > 0.5
    pair_start = next((i for i, name in enumerate(names) if name.startswith("pair:")), len(names))
    part = part[:, :pair_start]
    counts = list(part.sum(axis=0))
    for name in names[pair_start:]:
        # Names are descriptive only; labels may themselves include punctuation.
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if name == f"pair:{labels[i]}+{labels[j]}":
                    counts.append(int((valid_pos[:, i] & valid_pos[:, j]).sum()))
    rows = []
    for name, total, count in zip(names, full.sum(axis=0), counts):
        target = float(total * len(valid) / len(studies))
        rows.append(dict(feature=name, full_count=int(total), valid_count=int(count),
                         target_count=target, deviation=float(count - target),
                         full_fraction=float(total / len(studies)), valid_fraction=float(count / len(valid))))
    return dict(method=SPLIT_METHOD, full_studies=len(studies), valid_studies=len(valid),
                full_quality_mean=float(full_quality.mean()), valid_quality_mean=float(part_quality.mean()),
                confidence_bins=dict(low="<0.4", medium="[0.4,0.8)", high=">=0.8"), features=rows)
