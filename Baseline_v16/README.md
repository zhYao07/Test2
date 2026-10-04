# Baseline_v16：90/10 Study 随机划分

基于 Baseline_v14，仅调整数据划分与相关记录、续训校验。保留 header 质量排序、DINOv2 Small CLS、训练随机 24 个窗口、验证/推理全部候选（最多 80 个）、EMA、损失与标签聚合方式。原 v14 文件不修改。

## 数据划分

- 使用官方 train.csv 与本目录 label.csv 按 StudyInstanceUID 一对一合并后的全部可用 Study。官方真值存在时覆盖弱标签，否则沿用弱标签；完全无标签的 Study 剔除。
- 原 58 条完整 gold Study 也纳入统一划分，不再固定作为验证集。
- 按 StudyInstanceUID 排序后，用 np.random.RandomState(seed) 随机排列；验证数量为 ceil(N × 0.1)，其余训练。默认 seed=42；所有 DDP rank 使用相同划分，不受 CSV 行顺序影响。
- 当前本地数据共 4407 条，默认分为训练 3966、验证 441；原 58 条 gold 中训练 56、验证 2。不同数据文件或 seed 会改变结果。
- 同一 Study 的所有序列/切片只进入一个集合。当前未按患者 ID 分组；若同一患者有多个 Study，仍可能跨集合，应在有可靠患者标识时进一步按患者分组。
- coverage 阈值与类别权重仅从训练部分估计。验证图像 header 仅用于自身序列选择。

## 验证分数如何理解

9:1 可用于更大样本量的本地实验，但此验证集主要是弱标签，衡量的是与弱标签的一致性，不能直接与 v14 的 58 条真值验证 AUC 比较。弱标签来源或系统性误差也可能影响它与线上分数的相关性。

沿用 v14 的 AUC 实现：缺失或 0.5 标签不计分，其他软标签以 >0.5 二值化。单一类别的标签 AUC 为 NaN，macro AUC 跳过 NaN；启动时打印各标签正负例数量并提示不可计分标签。当前 seed=42 的 441 条验证数据中，全部 12 标签均有正负例，Fracture 正例 28 条，仍需留意稀有标签波动。

## 四卡训练

从 Baseline_v16 目录运行：

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --data-root /root/RSNA/rsna-knee-abnormality-detection \
  --output-dir ./outputs/last6_3e-4_quality_selection_split90_10 \
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

标签默认使用本目录 label.csv，也可用 --labels-csv 指定。建议从 DINOv2 预训练权重重新训练；旧 v14 的已训练权重可能看过新的验证 Study，用它初始化会影响验证独立性。CLI 保留 --init-checkpoint 用于明确设计的微调实验。

## 输出与推理

新增输出：

- data_split.csv：Study UID、train/valid 与官方标签数量。
- data_split.json：完整划分、种子、标签 SHA256 指纹。
- split_label_summary.json：训练/验证每标签正负例及排除数量。

checkpoint 保存 data_split；--resume 只接受 v16，并校验划分成员、标签指纹与原有训练/选择配置。数据或种子改变时拒绝续训，避免沿用旧验证历史。series_selection.csv 保留实际 train/valid 标记。

使用本目录 Kaggle_Inference.ipynb 和 v16 best.pt；架构标记为 baseline_v16_80_candidates_quality_selection_ema。模型结构与 v14 一致，推理不需要训练集划分文件；coverage 阈值使用 checkpoint 保存值，不在测试集拟合。

## 本地检查

用真实 train.csv / label.csv 检查划分大小、集合互斥与覆盖、固定种子及 CSV 乱序可复现性、不同种子、非法/重复 UID 拒绝、标签指纹变化、续训划分拒绝、每标签正负例数量。检查 Python 与 notebook 代码语法，并核对 notebook 内嵌模块与独立模块一致（省略编码声明）。

当前环境缺少 PyTorch，未运行完整训练、DICOM 前向或四卡验证；真实训练效果与线上分数仍需在训练服务器确认。
