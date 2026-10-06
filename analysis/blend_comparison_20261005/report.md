# 融合标签与 soft 标签对比

当前融合 CSV 名称为 labels_blend.csv，4,349 条原始记录已与下载 Parquet 按 UID 逐值核对一致；58 条追加 gold 与官方标签一致。主统计排除 gold。
原 soft CSV 及其源 Parquet 当前不在原路径。soft 按 (blend−0.2×原label)/0.8 推算；推算结果的逐类阳性/阴性数、均值、权重和、分位数、不确定项比例及已保存高频值计数均与上次 soft 统计吻合。这支持 80% soft＋20% 原label 的融合关系，但不是对原 soft 文件的独立逐行复核。

| 类别 | soft 阳性率 | blend 阳性率 | 原label 阳性率 | soft负→blend正 | soft正→blend负 | soft pos_weight | blend pos_weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACL | 12.99% | 12.99% | 20.44% | 0 | 0 | 6.788 | 6.521 |
| MCL | 4.92% | 4.92% | 15.13% | 0 | 0 | 10.000 | 10.000 |
| Medial Meniscus | 38.19% | 40.86% | 40.35% | 116 | 0 | 1.554 | 1.479 |
| Lateral Meniscus | 13.45% | 15.11% | 15.47% | 72 | 0 | 7.000 | 6.711 |
| Medial OA | 24.65% | 24.65% | 37.20% | 0 | 0 | 3.450 | 3.081 |
| Lateral OA | 14.92% | 14.92% | 27.20% | 0 | 0 | 6.595 | 5.935 |
| PF OA | 33.55% | 33.55% | 45.83% | 0 | 0 | 2.125 | 1.895 |
| Effusion | 25.57% | 25.57% | 59.44% | 0 | 0 | 2.862 | 2.249 |
| Synovitis | 24.49% | 24.49% | 12.49% | 0 | 0 | 1.745 | 2.226 |
| Baker's | 12.23% | 12.23% | 24.70% | 0 | 0 | 8.104 | 7.033 |
| Contusion | 15.20% | 18.28% | 17.11% | 134 | 0 | 6.392 | 6.403 |
| Fracture | 3.66% | 3.66% | 6.83% | 0 | 0 | 10.000 | 10.000 |

## 整体统计

```json
{
  "total": 4407,
  "weak": 4349,
  "gold": 58,
  "soft_source": "Inferred (blend-0.2*old)/0.8; per-class saved summary checks all pass; source soft file unavailable",
  "soft_blend_flips": 322,
  "soft_blend_flip_pct": 0.6170000766459722,
  "studies_with_flip": 308,
  "soft_old_binary_disagreements": 6352,
  "blend_old_binary_disagreements": 6030,
  "soft_old_mae": 0.1506895742316241,
  "blend_old_mae": 0.12055165938529931,
  "soft_positive_per_study": 2.238215681765923,
  "blend_positive_per_study": 2.3122556909634397,
  "soft_mean_confidence": 0.7516417567256842,
  "blend_mean_confidence": 0.7261590403924274,
  "blend_neutral_count": 5
}
```

## 解释

融合只将新标签数值向旧 label 拉回约 20%，不代表纠正了 20% 的错标签。原label也不是训练病例的真值。没有融合模型线上/线下预测，不能据分布保证分数提升。
阳性判定和 AUC 的阈值仅用于统计，训练仍使用连续 target。接近 0.5 的项可能穿过阈值，远离 0.5 的大分歧通常保留 soft 的方向。v14/v21 中接近 0.5 的项还会被 2*abs(target-0.5) 降权。
需要特别关注 Effusion、MCL、OA 和 Baker's：即使数值更接近旧 label，阳性率未必恢复。半月板 soft=0.45 的项目若旧值较高则较容易变为 >0.5。
例如旧 Effusion=0.91、soft=0.05，blend=0.222，仍为阈值阴性；它在 v14/v21 中的 label_weight 由 0.90 降至 0.556，减少该分歧项的负向置信。
建议在同一验证 UID、同一训练配置下比较；将原 label 的模型作为基线，并保存逐病例预测。不能将阈值阳性率变化等同于模型召回率或 AUC 变化。
