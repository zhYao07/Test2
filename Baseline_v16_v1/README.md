# Baseline_v16_v1：固定 58 条真值 + 383 条分层弱标签验证

基于 v16，继承 v14 的模型、header 质量选序列、损失、随机 24 个训练窗口、验证/推理全部候选和 EMA。仅改变验证集构建及相关审计记录。label1 指 v14 的 label.csv；本目录 label.csv 与它逐字节一致，不改软标签值。

## 固定真值与配额选择

验证集固定 441 条，必须包含官方 train.csv 中 12 标签完整的全部 58 个 Study；其余选择 383 条，剩下 3966 条训练。若合并后完整真值数量不是 58，直接报错，不默默丢弃真值数据。真值仍按原逻辑覆盖对应弱标签。

按 StudyInstanceUID 排序，固定 seed=42。先锁定 58 条 gold，再根据以下特征选择和优化另外 383 条；gold 不参与移出验证集的替换操作：

- 12 标签的正例、负例、缺失/0.5 数量。>0.5 为正，<0.5 为负；缺失和 0.5 不计 AUC。
- 每标签、每种正负极性的低/中/高置信度数量。置信度为 2×|p−0.5|，分档为 <0.4、[0.4,0.8)、>=0.8。
- 每 Study 的阳性标签数量分组：0、1、2、3–4、5–12。
- 全量数据中最常见的 12 个标签阳性共现组合。

目标数量按全量数据比例乘 441 计算，正例目标四舍五入。使用归一化配额误差平方和：正例权重 12、负例/排除/病例数量权重 3、其余特征权重 1，再除以 max(目标数量,1)，提高稀有标签的关注度。

先贪心补足 383 条，再最多做 200 次改善目标的单样本交换；均值置信度×有效标签比例只以 0.01 的系数作为弱偏好，seed 用于极小的平局打破。配额是近似目标，不保证所有分布完全匹配；固定真值的约束优先。

## 当前构造结果

本地 4407 条数据，训练 3966、验证 441（58 gold + 383 weak）。默认种子下的验证正例数：

| 标签 | 全量比例对应目标 | 实际 |
| --- | ---: | ---: |
| ACL | 91.36 | 91 |
| MCL | 66.75 | 67 |
| Medial Meniscus | 178.22 | 178 |
| Lateral Meniscus | 69.65 | 70 |
| Medial OA | 163.41 | 164 |
| Lateral OA | 119.48 | 119 |
| PF OA | 201.54 | 202 |
| Effusion | 262.18 | 263 |
| Synovitis | 57.04 | 57 |
| Baker's | 108.67 | 109 |
| Contusion | 76.35 | 76 |
| Fracture | 31.52 | 32 |

12 标签均有正负例，正例数量均与目标相差不到 1 条。其他特征的目标、实际和偏差详见 split_preview/split_distribution.json。全量/验证的置信度代理均值分别为 0.7842/0.7910；它不是经过校准的标签正确率。

## 训练与输出

从 Baseline_v16_v1 目录运行：

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --data-root /root/RSNA/rsna-knee-abnormality-detection \
  --output-dir ./outputs/last6_3e-4_quality_selection_gold58_balanced383 \
  --cache-dir ../Baseline_v10/input_cache \
  --crop-mm 140 \
  --image-size 336 \
  --train-windows 24 \
  --span-lo 0.02 \
  --span-hi 0.98 \
  --batch-size 2 \
  --accum-steps 2 \
  --backbone-mode last6 \
  --epochs 15 \
  --head-lr 3e-4 \
  --backbone-lr 1e-5 \
  --ema-decay 0.999 \
  --seed 42 \
  --coverage-quantile 0.2 \
  --series-quality-workers 8 \
  --amp bf16 \
  --no-metadata
```

默认使用本目录 label.csv；--labels-csv 可指定标签路径。必须与所保存的划分标签指纹一致才能续训。coverage 阈值与类别权重仍仅用训练部分估计。建议从原始 DINOv2 预训练权重重新训练，避免用见过新验证 Study 的旧模型初始化。

split_preview/ 已包含当前数据的实际划分清单、完整配置、标签分布及选择特征偏差。训练启动时重新执行相同构造过程，并在输出目录保存：

- data_split.csv：Study UID、train/valid、gold_label_count。
- data_split.json：方法版本、seed、58 条锁定 UID、完整划分成员及目标标签 SHA256。
- split_label_summary.json：每标签训练/验证正负例及排除数量。
- split_distribution.json：置信度分档、病例组合、共现目标与实际偏差。

checkpoint 保存划分配置，--resume 检查成员、标签指纹和原训练配置，且只接受 v16_v1。使用本目录 Kaggle_Inference.ipynb 及 v16_v1 best.pt；模型架构标记为 baseline_v16_v1_80_candidates_quality_selection_ema。

## 检查与限制

已检查固定 58 条全部进入验证、441/3966 大小、无重叠及完整覆盖、选择不改目标标签、乱序可复现、无效输入拒绝、锁定样本不可移出及各标签正负例。单元测试：python -m unittest discover -s Baseline_v16_v1 -p test_validation_split.py（需 NumPy/Pandas）。

此验证集仍有 383 条弱标签，不能保证其正确，也不能直接与只用 58 条真值的 AUC 比较。当前按 Study 隔离，未核实跨 Study 的患者重复。当前本地环境缺少 PyTorch，未运行完整训练、真实影像前向或四卡验证。
