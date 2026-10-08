"""Use v14 unchanged; cap evaluation windows before sending them to the VLM."""

import numpy as np
import torch

from rsna_data import KneeDataset, SLOTS


def evaluation_indices(slots, positions, limit):
    """Deterministic proportional slot allocation, evenly spaced within each slot."""
    slots, positions = np.asarray(slots), np.asarray(positions)
    if slots.shape != positions.shape or slots.ndim != 1 or not len(slots):
        raise ValueError("Expected nonempty matching slot/position arrays")
    if limit < 0 or (limit and limit < len(SLOTS)):
        raise ValueError("eval_windows must be 0 (all) or >=5")
    if not limit or limit >= len(slots):
        return np.lexsort((np.arange(len(slots)), positions, slots))
    active, counts = np.unique(slots, return_counts=True)
    quotas = np.ones(len(active), dtype=np.int64)
    desired = counts / counts.sum() * limit
    while quotas.sum() < limit:
        deficits = np.where(quotas < counts, desired - quotas, -np.inf)
        quotas[np.argmax(deficits)] += 1
    chosen = []
    for slot, quota in zip(active, quotas):
        indices = np.flatnonzero(slots == slot)
        indices = indices[np.argsort(positions[indices], kind="stable")]
        # Cell midpoints cover the whole series; no duplicate candidate indices.
        offsets = np.floor((np.arange(quota) + 0.5) * len(indices) / quota).astype(int)
        chosen.extend(indices[offsets])
    chosen = np.asarray(chosen, dtype=np.int64)
    return chosen[np.lexsort((chosen, positions[chosen], slots[chosen]))]


class VLMKneeDataset(KneeDataset):
    def __init__(self, *args, eval_windows=0, **kwargs):
        if eval_windows < 0 or (eval_windows and eval_windows < len(SLOTS)):
            raise ValueError("eval_windows must be 0 or >=5")
        super().__init__(*args, **kwargs)
        self.eval_windows = eval_windows

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if not self.train:
            chosen = evaluation_indices(sample["window_slot_indices"].numpy(),
                                        sample["window_positions"].numpy(), self.eval_windows)
            chosen = torch.as_tensor(chosen, dtype=torch.long)
            for key in ("images", "window_positions", "window_slot_indices"):
                sample[key] = sample[key][chosen]
        return sample
