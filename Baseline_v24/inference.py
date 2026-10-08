"""Offline v24 inference, on one GPU or multiple GPUs via torchrun."""

import argparse
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from rsna_data import LABELS, _prepare_series_df, add_series_quality, collate_studies
from rsna_model import DEFAULT_QWEN_MODEL_DIR, load_classifier, make_vlm_batch, predict_batch, attention_context
from vlm_data import VLMKneeDataset


def submission_frame(test_df, predictions):
    uids = test_df.StudyInstanceUID.astype(str).tolist()
    if len(set(uids)) != len(uids) or set(uids) != set(predictions):
        raise ValueError("Predicted UIDs must match test.csv exactly, without duplicates")
    values = np.asarray([predictions[uid] for uid in uids], dtype=np.float32)
    if not len(uids):
        values = np.empty((0, len(LABELS)), dtype=np.float32)
    if values.shape != (len(uids), len(LABELS)) or not np.isfinite(values).all():
        raise ValueError("Predictions must be finite [num_studies,12]")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Predictions must be probabilities in [0,1]")
    frame = pd.DataFrame(values, columns=LABELS)
    frame.insert(0, "StudyInstanceUID", uids)
    return frame


@torch.inference_mode()
def run_inference(args):
    torch.set_num_threads(args.cpu_threads)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not torch.cuda.is_available():
        raise RuntimeError("torchrun inference requires CUDA")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group("nccl")
    try:
        amp = args.amp
        if amp == "auto":
            amp = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp16"
        if device.type == "cpu":
            amp = "none"
        if amp == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Use --amp fp16 on a GPU without bf16 support")
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": torch.float32}[amp]
        model, processor, checkpoint = load_classifier(
            args.checkpoint, args.qwen_model_dir, device, dtype=dtype,
            attn_implementation=args.attn_implementation)
        saved = checkpoint["args"]
        test_df = pd.read_csv(args.data_root / "test.csv")
        local_studies = test_df.iloc[rank::world_size].reset_index(drop=True)
        series_df = pd.read_csv(args.data_root / "test_series.csv")
        series_df = series_df[series_df.StudyInstanceUID.isin(local_studies.StudyInstanceUID)].copy()
        series_df = _prepare_series_df(series_df, args.data_root / "test_series")
        series_df = add_series_quality(series_df, cache_dir=None, workers=args.series_quality_workers)
        eval_windows = saved["eval_windows"] if args.eval_windows is None else args.eval_windows
        dataset = VLMKneeDataset(
            local_studies, series_df, train=False, image_size=saved["image_size"], crop_mm=saved["crop_mm"],
            train_windows=saved["train_windows"], eval_windows=eval_windows,
            span_lo=saved["span_lo"], span_hi=saved["span_hi"], seed=saved["seed"],
            series_selection=checkpoint["series_selection"], cache_dir=None)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=device.type == "cuda", collate_fn=collate_studies)
        predictions = {}
        for step, raw in enumerate(loader):
            batch = make_vlm_batch(processor, raw, not saved["no_metadata"], saved["max_tokens"])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with attention_context(model, args.attn_implementation, args.sdpa_backend):
                with torch.autocast(device_type=device.type, dtype=dtype if amp != "none" else torch.bfloat16,
                                    enabled=device.type == "cuda" and amp != "none"):
                    probabilities = predict_batch(model, batch).sigmoid().cpu().numpy()
            predictions.update(zip(batch["study_uid"], probabilities))
            if (step + 1) % args.log_interval == 0 or step + 1 == len(loader):
                print(f"rank={rank} studies={len(predictions)}/{len(local_studies)} "
                      f"tokens={int(batch['attention_mask'].sum(-1).max())}", flush=True)
        gathered = [None] * world_size
        if distributed:
            dist.all_gather_object(gathered, predictions)
        else:
            gathered = [predictions]
        if rank == 0:
            merged = {}
            for part in gathered:
                if merged.keys() & part.keys():
                    raise ValueError("Duplicate study prediction across ranks")
                merged.update(part)
            frame = submission_frame(test_df, merged)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.output, index=False, lineterminator="\n")
            print(f"Saved {args.output}: {frame.shape}", flush=True)
            return frame
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-model-dir", type=Path, default=DEFAULT_QWEN_MODEL_DIR,
                        help="Defaults to ../Qwen3.5-2B next to Baseline_v24")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--series-quality-workers", type=int, default=4)
    parser.add_argument("--eval-windows", type=int, default=None)
    parser.add_argument("--amp", choices=["auto", "bf16", "fp16", "none"], default="auto")
    parser.add_argument("--attn-implementation", choices=["sdpa", "eager", "flash_attention_2"], default="sdpa")
    parser.add_argument("--sdpa-backend", choices=["math", "auto"], default="math",
                        help="Default math avoids fused attention kernels for full-window inference")
    parser.add_argument("--log-interval", type=int, default=20)
    args = parser.parse_args()
    if min(args.batch_size, args.series_quality_workers, args.log_interval, args.cpu_threads) < 1 or args.num_workers < 0:
        parser.error("Invalid batch/worker/log count")
    if args.eval_windows is not None and args.eval_windows != 0 and args.eval_windows < 5:
        parser.error("eval-windows must be 0 or >=5")
    return args


if __name__ == "__main__":
    run_inference(parse_args())
