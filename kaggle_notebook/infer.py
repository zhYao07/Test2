"""Kaggle submission inference for checkpoints trained by this package."""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rsna_data import LABELS, KneeDataset, _prepare_series_df
from rsna_model import RSNADINOv2


# Kaggle input paths confirmed by the successful run; edit here if mounts change.
DATA_ROOT = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
CHECKPOINT_PATH = "/kaggle/input/datasets/yzzzh7/baseline-v1-weights/best.pt"
DINOV2_MODEL_DIR = "/kaggle/input/models/metaresearch/dinov2/pytorch/small/1"
OUTPUT_PATH = Path("/kaggle/working/submission.csv")
BATCH_SIZE = 1
NUM_WORKERS = 0
ENCODER_CHUNK_SIZE = 16
NUM_GPUS = 2


def find_one(candidates, description):
    paths = sorted(set(Path(path) for path in candidates))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {description}, found {len(paths)}: {paths}. Set its path explicitly above.")
    return paths[0]


def input_files():
    """Find small input metadata without descending into millions of DICOM files."""
    for directory, subdirs, files in os.walk("/kaggle/input"):
        subdirs[:] = [name for name in subdirs if name not in {"train_series", "test_series"}]
        here = Path(directory)
        for name in files:
            if name in {"test_series.csv", "best.pt", "config.json"}:
                yield here / name


def resolve_inputs():
    found = list(input_files()) if not all((DATA_ROOT, CHECKPOINT_PATH, DINOV2_MODEL_DIR)) else []
    root = Path(DATA_ROOT) if DATA_ROOT else find_one(
        (p.parent for p in found if p.name == "test_series.csv"
         if (p.parent / "test.csv").is_file()
         and (p.parent / "sample_submission.csv").is_file()
         and (p.parent / "test_series").is_dir()),
        "competition data directory",
    )
    checkpoint = Path(CHECKPOINT_PATH) if CHECKPOINT_PATH else find_one(
        (p for p in found if p.name == "best.pt"), "best.pt checkpoint"
    )
    model_dir = Path(DINOV2_MODEL_DIR) if DINOV2_MODEL_DIR else find_one(
        (p.parent for p in found if p.name == "config.json"
         if (p.parent / "pytorch_model.bin").is_file()),
        "Hugging Face DINOv2 directory",
    )
    for path in (root / "test.csv", root / "test_series.csv", root / "sample_submission.csv",
                 root / "test_series", checkpoint, model_dir / "config.json",
                 model_dir / "pytorch_model.bin"):
        if not path.exists():
            raise FileNotFoundError(path)
    return root, checkpoint, model_dir


@torch.inference_mode()
def inference_worker(rank, model, local_test_df, series_df, num_slices, image_size, root):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    started = time.monotonic()
    series_df = series_df[series_df.StudyInstanceUID.isin(local_test_df.StudyInstanceUID)]
    print(f"GPU {rank}: indexing {len(series_df)} local series", flush=True)
    series_df = _prepare_series_df(series_df, root / "test_series")
    missing = set(local_test_df.StudyInstanceUID) - set(series_df.StudyInstanceUID)
    if missing:
        raise ValueError(f"No series metadata for {len(missing)} test studies")
    dataset = KneeDataset(local_test_df, series_df, num_slices=num_slices, image_size=image_size)
    print(f"GPU {rank}: {len(dataset)} studies assigned", flush=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        persistent_workers=NUM_WORKERS > 0)

    predictions = {}
    for index, batch in enumerate(loader, start=1):
        if index == 1:
            print(f"GPU {rank}: first study loaded after {time.monotonic() - started:.1f}s", flush=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(batch["images"].to(device, non_blocking=True),
                           batch["slot_mask"].to(device, non_blocking=True),
                           batch["series_features"].to(device, non_blocking=True))
        scores = torch.sigmoid(logits.float()).cpu().numpy()
        for uid, score in zip(batch["study_uid"], scores):
            predictions[str(uid)] = score
        if index % 5 == 0 or index == len(loader):
            print(f"GPU {rank}: {min(index * BATCH_SIZE, len(dataset))}/{len(dataset)} studies, "
                  f"elapsed {time.monotonic() - started:.1f}s", flush=True)

    if len(predictions) != len(dataset):
        raise ValueError(f"Expected {len(dataset)} predictions, got {len(predictions)}")
    values = np.stack([predictions[uid] for uid in local_test_df.StudyInstanceUID])
    if values.shape != (len(dataset), len(LABELS)) or not np.isfinite(values).all():
        raise ValueError("Prediction shape or finiteness check failed")
    if not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Predictions are outside [0, 1]")
    partial = pd.DataFrame(values, columns=LABELS)
    partial.insert(0, "StudyInstanceUID", local_test_df.StudyInstanceUID.to_numpy())
    return partial


def main():
    root, checkpoint_path, model_dir = resolve_inputs()
    print(f"data={root}\ncheckpoint={checkpoint_path}\nDINOv2={model_dir}", flush=True)
    if torch.cuda.device_count() < NUM_GPUS:
        raise RuntimeError(f"Enable Kaggle 2 x T4 GPU accelerator; found {torch.cuda.device_count()} GPU(s).")
    test_df = pd.read_csv(root / "test.csv", dtype={"StudyInstanceUID": str})
    series_df = pd.read_csv(root / "test_series.csv", dtype={
        "StudyInstanceUID": str, "SeriesInstanceUID": str,
    })
    sample = pd.read_csv(root / "sample_submission.csv", dtype={"StudyInstanceUID": str})
    expected = ["StudyInstanceUID", *LABELS]
    if list(sample.columns) != expected:
        raise ValueError(f"Submission columns differ from the training model: {list(sample.columns)}")
    if test_df.StudyInstanceUID.duplicated().any() or sample.StudyInstanceUID.duplicated().any():
        raise ValueError("Duplicate StudyInstanceUID in test or sample_submission")
    if set(test_df.StudyInstanceUID) != set(sample.StudyInstanceUID):
        raise ValueError("test.csv and sample_submission.csv contain different studies")

    print("Loading checkpoint", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    if any(key.startswith("backbone.patch_embed.") for key in state):
        raise ValueError("This best.pt uses the old torch.hub DINOv2 backbone, not the Hugging Face model")
    num_slices = int(state["slice_position"].shape[1])
    args = checkpoint.get("args", {}) if "model" in checkpoint else {}
    image_size = int(args.get("image_size", 224))
    if int(args.get("num_slices", num_slices)) != num_slices:
        raise ValueError("Checkpoint args and slice_position disagree about num_slices")
    models = []
    for rank in range(NUM_GPUS):
        model = RSNADINOv2(model_dir=str(model_dir), num_slices=num_slices,
                           encoder_chunk_size=ENCODER_CHUNK_SIZE, freeze_backbone=True)
        model.load_state_dict(state, strict=True)
        model.to(torch.device(f"cuda:{rank}")).eval()
        models.append(model)
        print(f"GPU {rank}: model ready", flush=True)
    del checkpoint, state

    # A thread per GPU keeps all computation inside this notebook process.
    # The models and studies are disjoint; no gradient or cross-GPU sync is needed.
    with ThreadPoolExecutor(max_workers=NUM_GPUS) as pool:
        futures = [pool.submit(inference_worker, rank, models[rank],
                               test_df.iloc[rank::NUM_GPUS].reset_index(drop=True),
                               series_df, num_slices, image_size, root)
                   for rank in range(NUM_GPUS)]
        partials = [future.result() for future in futures]
    predictions = pd.concat(partials, ignore_index=True)
    if predictions.StudyInstanceUID.duplicated().any():
        raise ValueError("Duplicate predictions across GPUs")
    if set(predictions.StudyInstanceUID) != set(sample.StudyInstanceUID):
        raise ValueError("GPU predictions do not cover the complete sample submission")
    submission = sample[["StudyInstanceUID"]].merge(predictions, on="StudyInstanceUID",
                                                       how="left", validate="one_to_one")
    if list(submission.columns) != ["StudyInstanceUID", *LABELS]:
        raise ValueError("Incorrect submission column order")
    if not np.isfinite(submission[LABELS].to_numpy()).all():
        raise ValueError("Non-finite value in merged submission")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}: {submission.shape[0]} rows, {submission.shape[1]} columns")
    print(submission.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
