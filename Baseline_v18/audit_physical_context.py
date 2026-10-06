"""Audit v18 channel changes using an existing v14/v18 series_selection.csv."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pydicom.errors import InvalidDicomError

from rsna_data import (DEFAULT_ROOT, DEFAULT_CONTEXT_MM, SLOTS, CANDIDATE_BUDGETS,
                       get_sorted_dicom_info, _sample_by_position,
                       physical_neighbor_indices, make_physical_context_config)


def audit_series(record, root, context_mm, span_lo, span_hi):
    result = {key: record[key] for key in ("StudyInstanceUID", "slot", "selected_series_uid", "split")}
    path = root / "train_series" / record["StudyInstanceUID"] / record["selected_series_uid"]
    try:
        _, positions, _ = get_sorted_dicom_info(path)
        first = int(len(positions) * span_lo)
        last = max(first, int(len(positions) * span_hi) - 1)
        budget = CANDIDATE_BUDGETS[SLOTS.index(record["slot"])]
        anchors = first + _sample_by_position(positions[first:last + 1], budget)
        old = np.clip(anchors[:, None] + [-1, 0, 1], 0, len(positions) - 1)
        new = physical_neighbor_indices(positions, anchors, context_mm)
        old_offsets = positions[old[:, [0, 2]]] - positions[anchors, None]
        new_offsets = positions[new[:, [0, 2]]] - positions[anchors, None]
        result.update(windows=budget, changed_windows=int(np.any(old != new, axis=1).sum()),
                      changed_side_channels=int((old[:, [0, 2]] != new[:, [0, 2]]).sum()),
                      old_anchor_side_channels=int((old[:, [0, 2]] == anchors[:, None]).sum()),
                      new_anchor_side_channels=int((new[:, [0, 2]] == anchors[:, None]).sum()),
                      new_same_position_side_channels=int((np.abs(new_offsets) <= 1e-4).sum()),
                      median_unique_spacing_mm=float(np.median(np.diff(np.unique(positions))))
                      if len(np.unique(positions)) > 1 else 0.0, error="")
        for prefix, offsets in (("old", old_offsets), ("new", new_offsets)):
            result[f"{prefix}_mean_abs_offset_mm"] = float(np.abs(offsets).mean())
        return result, old_offsets, new_offsets
    except (OSError, ValueError, AttributeError, IndexError, EOFError, InvalidDicomError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
        return result, None, None


def summarize(results, context):
    successful = [item for item in results if not item[0]["error"]]
    windows = sum(item[0]["windows"] for item in successful)
    changed = sum(item[0]["changed_windows"] for item in successful)
    collapsed = sum(item[0]["new_same_position_side_channels"] for item in successful)
    summary = dict(physical_context=context, selected_slots=len(results),
                   successful_slots=len(successful), failed_slots=len(results) - len(successful),
                   candidate_windows=windows, changed_windows=changed,
                   changed_window_fraction=changed / windows if windows else None,
                   new_same_position_side_channel_fraction=collapsed / (2 * windows) if windows else None)
    if successful:
        for prefix, index in (("old", 1), ("new", 2)):
            offsets = np.concatenate([item[index] for item in successful])
            summary[f"{prefix}_signed_offset_quantiles_mm"] = {
                side: dict(zip(("p0", "p10", "p50", "p90", "p100"),
                               np.quantile(offsets[:, channel], [0, .1, .5, .9, 1]).tolist()))
                for channel, side in enumerate(("negative", "positive"))}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--series-selection-csv", type=Path, required=True,
                        help="Use the actual v14/v18 selection report; do not refit quality thresholds")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/physical_context_audit"))
    parser.add_argument("--split", choices=["all", "train", "valid"], default="all")
    parser.add_argument("--context-mm", type=float, default=DEFAULT_CONTEXT_MM)
    parser.add_argument("--span-lo", type=float, default=.02)
    parser.add_argument("--span-hi", type=float, default=.98)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    context = make_physical_context_config(args.context_mm)
    if args.workers < 1 or not 0 <= args.span_lo < args.span_hi <= 1:
        parser.error("workers must be positive and span must satisfy 0 <= lo < hi <= 1")
    report = pd.read_csv(args.series_selection_csv, dtype=str, keep_default_na=False)
    required = {"StudyInstanceUID", "slot", "selected_series_uid", "split"}
    if not required.issubset(report.columns) or not set(report.slot).issubset(SLOTS):
        raise ValueError("Invalid series selection report")
    report = report[report.selected_series_uid.ne("")]
    if args.split != "all":
        report = report[report.split.eq(args.split)]
    if report.empty or report.duplicated(["StudyInstanceUID", "slot"]).any():
        raise ValueError("Selection report is empty or contains duplicate Study-slot rows")
    records = report.to_dict("records")
    results = []
    print(f"Reading headers for {len(records)} selected slots; no pixel decoding", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        work = pool.map(lambda row: audit_series(row, args.data_root, args.context_mm,
                                                args.span_lo, args.span_hi), records)
        for index, result in enumerate(work, 1):
            results.append(result)
            if index % 500 == 0 or index == len(records):
                print(f"Audited {index}/{len(records)} slots", flush=True)
    frame = pd.DataFrame([item[0] for item in results])
    summary = summarize(results, context)
    summary.update(span=[args.span_lo, args.span_hi], split=args.split,
                   per_slot={slot: summarize([r for r in results if r[0]["slot"] == slot], context)
                             for slot in SLOTS})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "physical_context.csv").open("w", encoding="utf-8", newline="\n") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")
    with (args.output_dir / "physical_context_summary.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_slot"}, ensure_ascii=False, indent=2))
    if summary["failed_slots"]:
        raise SystemExit("Some slots failed; inspect the CSV error column before interpreting the summary")


if __name__ == "__main__":
    main()
