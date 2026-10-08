"""Compare existing labels by UID, without API calls or changing label sources."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from extract import ROOT, HERE, TARGETS, write_csv, write_json, write_text


def read_index(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    index = {row["StudyInstanceUID"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"Duplicate study UID in {path}")
    return index


def number(value):
    return None if value is None or not value.strip() else float(value)


def metrics(rows):
    scores = [row for row in rows if row["new_score"] is not None]
    return {
        "total_items": len(rows), "paired_scored_items": len(scores),
        "new_masked_items": len(rows) - len(scores),
        "direction_disagreements_at_0_5": sum(row["direction_disagreement"] for row in scores),
        "mean_absolute_score_difference": statistics.mean(abs(row["delta"]) for row in scores) if scores else None,
        "note": "0.5 is a diagnostic cutoff only; agreement is not accuracy or calibration.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=ROOT / "Baseline_v22/label.csv")
    parser.add_argument("--blend", type=Path, default=ROOT / "labels_blend.csv")
    parser.add_argument("--run", type=Path, default=HERE / "output/pilot20_soft")
    parser.add_argument("--gold", type=Path, default=ROOT / "data/train.csv")
    args = parser.parse_args()
    old, blend = read_index(args.old), read_index(args.blend)
    new = read_index(args.run / "labels.csv")
    reports = read_index(args.run / "selected_studies.csv")
    with (args.run / "labels_detail.csv").open(encoding="utf-8-sig", newline="") as stream:
        detail = {(row["StudyInstanceUID"], row["target"]): row for row in csv.DictReader(stream)}
    comparison = []
    for index, (uid, row) in enumerate(new.items(), 1):
        if uid not in old or uid not in blend:
            raise ValueError(f"Study missing in reference labels: {uid}")
        for target in TARGETS:
            previous, current = number(old[uid][target]), number(row[target])
            if previous is None:
                raise ValueError(f"Missing old score: {uid}, {target}")
            item = detail[uid, target]
            comparison.append({
                "study_index": index, "StudyInstanceUID": uid, "target": target,
                "old_score": previous, "blend_score": number(blend[uid][target]),
                "new_score": current, "delta": None if current is None else current - previous,
                "direction_disagreement": None if current is None else int((previous >= 0.5) != (current >= 0.5)),
                "new_mask": row[target + "__mask"], "new_weight": row[target + "__weight"],
                "report_status": item["report_status"], "evidence_strength": item["evidence_strength"],
                "evidence": item["evidence"], "score_reason": item["score_reason"],
                "uncertainty_reason": item["uncertainty_reason"], "Report": reports[uid]["Report"],
            })
    gold = [row for row in read_index(args.gold).values() if all(row[t].strip() in {"0", "1", "0.0", "1.0"} for t in TARGETS)]
    summary = {
        "sources": {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for name, path in {"old": args.old, "blend": args.blend, "new": args.run / "labels.csv", "gold": args.gold}.items()},
        "pilot_studies": len(new), "overall": metrics(comparison),
        "per_target": {target: metrics([row for row in comparison if row["target"] == target]) for target in TARGETS},
        "complete_gold_studies": len(gold),
        "old_vs_gold_direction_disagreements": {target: sum((float(row[target]) >= 0.5) != (float(old[row["StudyInstanceUID"]][target]) >= 0.5) for row in gold) for target in TARGETS},
        "gold_note": "Descriptive comparison of legacy scores, not an independent evaluation of the new extractor; old-label provenance is unknown.",
    }
    output = args.run / "legacy_comparison"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "all_240_items.csv", list(comparison[0]), comparison)
    write_json(output / "summary.json", summary)
    lines = ["# 原有标签与新软标签的逐 study 对照", "", f"主参照：`{args.old}`；辅助：`{args.blend}`。按 StudyInstanceUID 对齐，不按文件行号匹配。",
             "", "原有标签不是金标。0.5 仅用于定位分歧，不转换保存的软标签，也不把一致率称为准确率。UNKNOWN 表示新标签被屏蔽。", "",
             "| study | 类别 | 原有 | blend | 新软分数 | 证据强度 |", "|---|---|---:|---:|---:|---|"]
    focus = {"Medial OA", "Lateral OA", "PF OA", "Effusion"}
    for row in comparison:
        if row["target"] in focus:
            score = "UNKNOWN" if row["new_score"] is None else f"{row['new_score']:.3f}"
            lines.append(f"| {row['study_index']} | {row['target']} | {row['old_score']:.3f} | {row['blend_score']:.3f} | {score} | {row['evidence_strength']} |")
    lines += ["", "完整 240 项、原始报告、原句和理由见 all_240_items.csv；逐类别统计和来源哈希见 summary.json。", ""]
    write_text(output / "comparison_ZH.md", "\n".join(lines))
    print(json.dumps({"output": str(output), **summary["overall"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
