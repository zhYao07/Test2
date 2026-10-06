"""Compare label files by UID; inspect checkpoint metadata without loading tensors."""
import collections
import csv
import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "label_comparison_20261005"


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["StudyInstanceUID"] for row in rows]
    assert len(ids) == len(set(ids)), f"Duplicate UIDs: {path}"
    return {row["StudyInstanceUID"]: row for row in rows}


class Opaque:
    """Placeholder: never invoke a callable supplied by a pickle."""
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        self.args = args

    def __setstate__(self, state):
        self.state = state


class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        return Opaque

    def persistent_load(self, pid):
        return Opaque()


def primitive(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): primitive(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [primitive(v) for v in value]
    if isinstance(value, Opaque):
        return {"opaque_args": primitive(value.args)}
    return str(type(value))


def checkpoint_metadata(path):
    with zipfile.ZipFile(path) as archive:
        member = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
        data = MetadataReader(io.BytesIO(archive.read(member))).load()
    keys = ["epoch", "best_auc", "args", "validation_history", "weights_type",
            "swa_sources", "validation_set", "architecture"]
    result = {key: primitive(data[key]) for key in keys if key in data}
    if "validation_history" in result:
        result["validation_history"] = [
            {k: v for k, v in epoch.items()
             if k in ["epoch", "val_macro_auc", "val_auc_by_label", "train_loss", "ema_updates"]}
            for epoch in result["validation_history"]
        ]
    return result


def summary(v):
    good = np.isfinite(v)
    w = np.where(good, 2 * np.abs(v - 0.5), 0)
    pos = good & (v > 0.5)
    neg = good & (v < 0.5)
    # Same float32 arithmetic and clipping as Baseline_v14/v21.
    f = v.astype(np.float32)
    c = np.where(np.isnan(f), 0, 2 * np.abs(f - 0.5))
    b = np.nan_to_num(f, nan=0.5) > 0.5
    pos_weight = np.clip(np.sum(c * ~b) / max(float(np.sum(c * b)), 1), 1, 10)
    return dict(mean=float(np.nanmean(v)), positive=int(pos.sum()),
                positive_pct=float(pos.mean() * 100), negative=int(neg.sum()),
                missing=int((~good).sum()), neutral=int((v == 0.5).sum()),
                uncertain_pct=float(((v >= 0.35) & (v <= 0.65)).mean() * 100),
                confidence_mean=float(w.mean()), weight_sum=float(w.sum()),
                positive_weight=float(w[pos].sum()), negative_weight=float(w[neg].sum()),
                weighted_soft_positive=float(np.nansum(w * v) / w.sum()),
                pos_weight=float(pos_weight),
                quantiles=np.nanquantile(v, [0, .1, .25, .5, .75, .9, 1]).tolist(),
                common_values=collections.Counter(v[good].tolist()).most_common(6))


def main():
    old = read_table(ROOT / "Baseline_v21" / "label.csv")
    new = read_table(ROOT / "labels_llm_soft.csv")
    official = read_table(ROOT / "data" / "train.csv")
    gold = read_table(ROOT / "report_labels/L1_20261005_v5/gold_original.csv")
    labels = [k for k in next(iter(old.values())) if k != "StudyInstanceUID"]
    assert set(old) == set(new) == set(official)
    assert len(gold) == 58
    weak_ids = sorted(set(old) - set(gold))

    def matrix(table, ids):
        return np.array([[float(table[uid][k]) if table[uid][k] else np.nan
                          for k in labels] for uid in ids])

    a, b = matrix(old, weak_ids), matrix(new, weak_ids)
    g = matrix(gold, sorted(gold))
    old_gold = matrix(old, sorted(gold))
    assert np.array_equal(matrix(new, sorted(gold)), g)
    official_gold_ids = {uid for uid, row in official.items() if all(row[k] != "" for k in labels)}
    assert official_gold_ids == set(gold)
    assert np.array_equal(matrix(official, sorted(gold)), g)
    partial = sum(any(row[k] for k in labels) for uid, row in official.items() if uid not in gold)
    assert partial == 0, "Need account for partial official label overrides"
    per_class = []
    examples = []
    for j, label in enumerate(labels):
        x, y = a[:, j], b[:, j]
        comparable = np.isfinite(x) & np.isfinite(y) & (x != .5) & (y != .5)
        up = comparable & (x < .5) & (y > .5)
        down = comparable & (x > .5) & (y < .5)
        per_class.append(dict(label=label, old=summary(x), new=summary(y),
                              gold_positive=int(g[:, j].sum()),
                              gold_positive_pct=float(g[:, j].mean() * 100),
                              comparable=int(comparable.sum()), old_negative_new_positive=int(up.sum()),
                              old_positive_new_negative=int(down.sum()),
                              flip_pct=float((up | down).sum() / comparable.sum() * 100),
                              old_high_new_low=int(((x >= .8) & (y <= .2)).sum()),
                              mean_abs_difference=float(np.nanmean(np.abs(x-y)))))
        changed = np.flatnonzero(up | down)
        changed = sorted(changed, key=lambda i: -abs(x[i]-y[i]))[:5]
        for i in changed:
            examples.append(dict(uid=weak_ids[i], label=label, old=float(x[i]), new=float(y[i]),
                                 report=official[weak_ids[i]]["Report"]))
    usable = np.isfinite(a) & np.isfinite(b) & (a != .5) & (b != .5)
    flips = usable & ((a > .5) != (b > .5))
    overview = dict(total=len(old), weak=len(weak_ids), gold=len(gold),
                    all_uid_sets_equal=True, new_gold_equals_official=True,
                    old_gold_different_cells=int((old_gold != g).sum()),
                    old_gold_threshold_disagreements=int(((old_gold > .5) != (g > .5)).sum()),
                    comparable_cells=int(usable.sum()), flip_cells=int(flips.sum()),
                    flip_cell_pct=float(flips.sum()/usable.sum()*100),
                    studies_with_flip=int(flips.any(axis=1).sum()),
                    old_positive_per_study=float((a > .5).sum(axis=1).mean()),
                    new_positive_per_study=float((b > .5).sum(axis=1).mean()),
                    old_weight_sum=float(np.nansum(2 * np.abs(a-.5))),
                    new_weight_sum=float(np.nansum(2 * np.abs(b-.5))),
                    old_missing=int(np.isnan(a).sum()), new_missing=int(np.isnan(b).sum()),
                    old_neutral=int((a == .5).sum()), new_neutral=int((b == .5).sum()))
    checkpoints = {name: checkpoint_metadata(ROOT / f"Baseline_v21/{folder}/best.pt")
                   for name, folder in [("old", "weights"), ("new", "weights_label_soft")]}
    best_epochs = {k: max(meta["validation_history"], key=lambda e: e["val_macro_auc"])
                   for k, meta in checkpoints.items()}
    j_eff, j_syn = labels.index("Effusion"), labels.index("Synovitis")
    pairs = {k: dict(collections.Counter(
        f"Effusion={bool(row[j_eff] > .5)},Synovitis={bool(row[j_syn] > .5)}"
        for row in values)) for k, values in [("old", a), ("new", b)]}
    manifest = read_table(ROOT / "report_labels/L1_20261005_v5/reports_manifest.csv")
    assert set(manifest) == set(old)
    language_stats = []
    for lang in sorted({row["language"] for row in manifest.values()}):
        idx = [i for i, uid in enumerate(weak_ids) if manifest[uid]["language"] == lang]
        mask = usable[idx]
        language_stats.append(dict(language_hint=lang, weak_count=len(idx),
                                   flip_pct=float(flips[idx].sum()/mask.sum()*100),
                                   old_effusion_positive_pct=float((a[idx,j_eff]>.5).mean()*100),
                                   new_effusion_positive_pct=float((b[idx,j_eff]>.5).mean()*100)))
    arg_diff = {k: {"old": checkpoints["old"].get("args", {}).get(k),
                    "new": checkpoints["new"].get("args", {}).get(k)}
                for k in set(checkpoints["old"].get("args", {})) | set(checkpoints["new"].get("args", {}))
                if checkpoints["old"].get("args", {}).get(k) != checkpoints["new"].get("args", {}).get(k)}
    result = dict(overview=overview, per_class=per_class, checkpoint_arg_differences=arg_diff,
                  checkpoints=checkpoints, best_epochs=best_epochs,
                  effusion_synovitis_pairs=pairs, language_stats=language_stats,
                  high_disagreement_examples=examples)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "statistics.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    lines = ["# 标签分布对比", "", "按 StudyInstanceUID 对齐；主统计排除 58 条 gold，使用 4,349 条弱标签。阳性定义为 >0.5；0.5 单独作为零权重项。旧 label 不是这 4,349 条记录的真值，分歧不等于新标签错误。", "",
             "| 类别 | 旧阳性率 | 新阳性率 | gold 阳性数/58 | 旧负→新正 | 旧正→新负 | 翻转比例 | 旧均值 | 新均值 | 旧 pos_weight | 新 pos_weight |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in per_class:
        o, n = r["old"], r["new"]
        lines.append(f"| {r['label']} | {o['positive_pct']:.2f}% | {n['positive_pct']:.2f}% | {r['gold_positive']} | {r['old_negative_new_positive']} | {r['old_positive_new_negative']} | {r['flip_pct']:.2f}% | {o['mean']:.4f} | {n['mean']:.4f} | {o['pos_weight']:.3f} | {n['pos_weight']:.3f} |")
    lines += ["", "## 置信度与损失权重", "", "v14/v21 的 label_weight=2*abs(target-0.5)。下表是弱标签全集统计；每批 loss 还会按权重总和归一化。总权重表示相对监督分配，不等于未经归一化的整体梯度倍率。", "",
              "| 类别 | 旧平均权重 | 新平均权重 | 新/旧正样本权重和 | 新/旧负样本权重和 | 旧0.5数量 | 新0.5数量 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in per_class:
        o,n=r["old"],r["new"]
        lines.append(f"| {r['label']} | {o['confidence_mean']:.3f} | {n['confidence_mean']:.3f} | {n['positive_weight']/o['positive_weight']:.3f} | {n['negative_weight']/o['negative_weight']:.3f} | {o['neutral']} | {n['neutral']} |")
    lines += ["", "## 覆盖与整体差异", "", "```json", json.dumps(overview, ensure_ascii=False, indent=2), "```", "",
              "## v21 checkpoint 元数据", "", "只解析 data.pkl 中的元数据；所有外部构造器均替换为占位符，不加载张量。以下 epoch 为 checkpoint 原始字段（通常从 0 开始）。", ""]
    for name, meta in checkpoints.items():
        lines += [f"### {name}", "", "```json", json.dumps({k:v for k,v in meta.items() if k not in ['args','validation_history']}, ensure_ascii=False, indent=2), "```", ""]
    lines += ["参数差异：", "", "```json", json.dumps(arg_diff, ensure_ascii=False, indent=2), "```", "",
              "## v21 最佳 epoch 逐类 gold AUC", "", "| 类别 | 旧 | 新 | 差值 |",
              "|---|---:|---:|---:|"]
    for label in labels:
        old_auc = best_epochs["old"]["val_auc_by_label"][label]
        new_auc = best_epochs["new"]["val_auc_by_label"][label]
        lines.append(f"| {label} | {old_auc:.5f} | {new_auc:.5f} | {new_auc-old_auc:+.5f} |")
    lines += ["", "## 解释与实验建议", "",
              "用户报告 v21 线上旧标签 0.928、新标签 0.925；v14 也同方向。当前 v21 checkpoint 的最佳 gold Macro AUC 分别为 0.909726（epoch 18）与 0.914651（epoch 14）。",
              "v14/v21 按官方 train.csv 中完整 gold 划分 58 条验证集，并覆盖 CSV 中 gold 的弱标签值。因此旧 CSV 的 58 条预测值与官方值不同不会改变这两版的实际验证真值；追加新 CSV 的 58 条 gold 也不会使这些病例进入训练。",
              "新旧 UID 集合完全一致，主要变化是标签定义/打分与训练权重。新标签在 11 类上减少阳性，只在 Synovitis 增加阳性。不能把旧弱标签当真值，不能据分歧数直接宣称新标签错误。",
              "Effusion 有 1477 条旧阳性变新阴性，其中 1437 条变为 0.2。短报告含 Mild effusion 的一例从 0.91 变为 0.05，提示需要重点核查程度阈值、未提及/否定的处理是否符合比赛标注定义；当前无法直接判定影像真值。",
              "新 Synovitis 的 1065 条阳性中 1028 条也是 Effusion 阳性；新 Effusion 阳性的 92.45% 同时 Synovitis 阳性，旧文件相应为 21.01%。两类训练关联显著改变，但仅凭该统计不能判定因果或提取器具体规则。",
              "loss 同时依赖 target、2*abs(target-0.5) 和自动 pos_weight。软标签替换改变三者。比如 Baker's 的 pos_weight 从 2.33 增到 8.10；weighted BCE 在单个 target=0.2 的常数预测最优点为 p*w/(1-p+p*w)，这里约 0.669。这个值是损失最优点，不是模型预测或诊断概率。",
              "58 条 gold 是小样本；MCL 仅 9 个阳性，Fracture 18 个。它们的阳性分布与弱标签集明显不同，这既可能来自样本选择也可能来自标注差异，不能据此反推线上阳性率。按该集合选 epoch 和反复调参可能产生选择偏差；当前没有两模型逐病例预测，因此不能计算配对 bootstrap 或断言分差显著。",
              "两 checkpoint 核心训练参数一致，但 cache 路径不同、旧文件未保存 swa 字段；不能证明缓存内容、所有训练代码和线上推理流程完全相同。",
              "建议以线上更好的原 label 为基线，先固定旧 pos_weight 比较新标签，分离目标变化和权重变化；再只替换一个类别/类别组，优先 Effusion、Baker's、MCL、OA；对分歧报告按语言及程度分层复核，并保留显式 target/weight/mask。固定验证 UID，保存每个 epoch 的逐病例预测，配对比较 AUC 与区间，避免只依赖 58 条 best 分数。",
              "AUC 依据排序。简单统一调整输出阈值或单调校准不能修复排序退化；实际训练目标变化可能改变排序。", "",
              "技术参考：[PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)、[scikit-learn roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)。", "",
              "## 报告语言提示分组", "", "语言为已有 manifest 的启发式提示，未经校准，不代表可靠语言识别。", "",
              "| 语言提示 | 弱标签病例数 | 项翻转率 | 旧 Effusion 阳性率 | 新 Effusion 阳性率 |",
              "|---|---:|---:|---:|---:|"]
    for r in language_stats:
        lines.append(f"| {r['language_hint']} | {r['weak_count']} | {r['flip_pct']:.2f}% | {r['old_effusion_positive_pct']:.2f}% | {r['new_effusion_positive_pct']:.2f}% |")
    lines += ["",
              "## 待复核的高分歧样本", "", "这是按数值差异选出的报告原文，用于定位提取差别，不代表任何一方已经被判定为正确。", ""]
    for e in examples:
        lines += [f"- {e['label']}：{e['old']} → {e['new']}；UID `{e['uid']}`", f"  报告：{e['report'].replace(chr(10),' ').replace(chr(13),' ')}", ""]
    with (OUT / "report.md").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"overview":overview,"language_stats":language_stats,"arg_diff":arg_diff,
                      "checkpoint_scores":{k:{f:v for f,v in m.items() if f not in ['args','validation_history']} for k,m in checkpoints.items()}}))


if __name__ == "__main__":
    main()
