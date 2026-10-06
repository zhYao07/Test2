"""Compare the supplied blend with saved soft statistics and original labels."""
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_label_distribution import ROOT, read_table, summary


def main():
    original_path = ROOT / "Baseline_v21/label.csv"
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == "f02b6b8e1a53f902d89afc4a93e57bd6c3d68512a31367565d492f9478074406"
    old = read_table(original_path)
    blend = read_table(ROOT / "labels_blend.csv")
    gold = read_table(ROOT / "report_labels/L1_20261005_v5/gold_original.csv")
    saved = json.loads((ROOT / "analysis/label_comparison_20261005/statistics.json").read_text(encoding="utf-8"))
    labels = [r["label"] for r in saved["per_class"]]
    ids = sorted(set(old) - set(gold))
    assert set(old) == set(blend) and len(ids) == 4349
    a = np.array([[float(old[u][k]) for k in labels] for u in ids])
    b = np.array([[float(blend[u][k]) for k in labels] for u in ids])
    for u in gold:
        assert all(float(blend[u][k]) == float(gold[u][k]) for k in labels)
    # The original soft file was moved/deleted. This is an inferred reconstruction,
    # not an independent row-by-row observation of that file.
    s = np.round((b - .2 * a) / .8, 10)
    assert np.all((s >= 0) & (s <= 1))
    for j, reference in enumerate(saved["per_class"]):
        actual, expected = summary(s[:, j]), reference["new"]
        for key in ["positive", "negative", "missing", "neutral"]:
            assert actual[key] == expected[key], (reference["label"], key)
        for key in ["mean", "weight_sum", "confidence_mean", "uncertain_pct", "quantiles"]:
            assert np.allclose(actual[key], expected[key], rtol=0, atol=1e-7), (reference["label"], key)
        for value, count in expected["common_values"]:
            assert int(np.isclose(s[:, j], value, rtol=0, atol=1e-8).sum()) == count
    rows = []
    for j, label in enumerate(labels):
        ss, bb, aa = s[:, j], b[:, j], a[:, j]
        changes = (ss > .5) != (bb > .5)
        rows.append(dict(label=label, soft=summary(ss), blend=summary(bb), old=summary(aa),
                         soft_negative_blend_positive=int(((ss < .5) & (bb > .5)).sum()),
                         soft_positive_blend_negative=int(((ss > .5) & (bb < .5)).sum()),
                         soft_blend_flips=int(changes.sum()),
                         inferred_soft_old_mae=float(np.abs(ss-aa).mean()),
                         blend_old_mae=float(np.abs(bb-aa).mean()),
                         soft_old_binary_disagreements=int(((ss > .5) != (aa > .5)).sum()),
                         blend_old_binary_disagreements=int(((bb > .5) != (aa > .5)).sum())))
    changes = (s > .5) != (b > .5)
    valid_old = a != .5
    overview = dict(total=4407, weak=4349, gold=58,
                    soft_source="Inferred (blend-0.2*old)/0.8; per-class saved summary checks all pass; source soft file unavailable",
                    soft_blend_flips=int(changes.sum()),
                    soft_blend_flip_pct=float(changes.mean()*100),
                    studies_with_flip=int(changes.any(axis=1).sum()),
                    soft_old_binary_disagreements=int((((s>.5)!=(a>.5)) & valid_old).sum()),
                    blend_old_binary_disagreements=int((((b>.5)!=(a>.5)) & valid_old).sum()),
                    soft_old_mae=float(np.abs(s-a).mean()),
                    blend_old_mae=float(np.abs(b-a).mean()),
                    soft_positive_per_study=float((s>.5).sum(axis=1).mean()),
                    blend_positive_per_study=float((b>.5).sum(axis=1).mean()),
                    soft_mean_confidence=float(np.abs(2*s-1).mean()),
                    blend_mean_confidence=float(np.abs(2*b-1).mean()),
                    blend_neutral_count=int((b==.5).sum()))
    result = dict(overview=overview, per_class=rows)
    out = ROOT / "analysis/blend_comparison_20261005"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "statistics.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    lines = ["# 融合标签与 soft 标签对比", "",
             "当前融合 CSV 名称为 labels_blend.csv，4,349 条原始记录已与下载 Parquet 按 UID 逐值核对一致；58 条追加 gold 与官方标签一致。主统计排除 gold。",
             "原 soft CSV 及其源 Parquet 当前不在原路径。soft 按 (blend−0.2×原label)/0.8 推算；推算结果的逐类阳性/阴性数、均值、权重和、分位数、不确定项比例及已保存高频值计数均与上次 soft 统计吻合。这支持 80% soft＋20% 原label 的融合关系，但不是对原 soft 文件的独立逐行复核。", "",
             "| 类别 | soft 阳性率 | blend 阳性率 | 原label 阳性率 | soft负→blend正 | soft正→blend负 | soft pos_weight | blend pos_weight |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['label']} | {r['soft']['positive_pct']:.2f}% | {r['blend']['positive_pct']:.2f}% | {r['old']['positive_pct']:.2f}% | {r['soft_negative_blend_positive']} | {r['soft_positive_blend_negative']} | {r['soft']['pos_weight']:.3f} | {r['blend']['pos_weight']:.3f} |")
    lines += ["", "## 整体统计", "", "```json", json.dumps(overview, ensure_ascii=False, indent=2), "```", "",
              "## 解释", "",
              "融合只将新标签数值向旧 label 拉回约 20%，不代表纠正了 20% 的错标签。原label也不是训练病例的真值。没有融合模型线上/线下预测，不能据分布保证分数提升。",
              "阳性判定和 AUC 的阈值仅用于统计，训练仍使用连续 target。接近 0.5 的项可能穿过阈值，远离 0.5 的大分歧通常保留 soft 的方向。v14/v21 中接近 0.5 的项还会被 2*abs(target-0.5) 降权。",
              "需要特别关注 Effusion、MCL、OA 和 Baker's：即使数值更接近旧 label，阳性率未必恢复。半月板 soft=0.45 的项目若旧值较高则较容易变为 >0.5。",
              "例如旧 Effusion=0.91、soft=0.05，blend=0.222，仍为阈值阴性；它在 v14/v21 中的 label_weight 由 0.90 降至 0.556，减少该分歧项的负向置信。",
              "建议在同一验证 UID、同一训练配置下比较；将原 label 的模型作为基线，并保存逐病例预测。不能将阈值阳性率变化等同于模型召回率或 AUC 变化。", ""]
    with (out / "report.md").open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
