# Baseline_v14：Series selection 质量排序实验

基于 Baseline_v10，仅修改序列选择：同一候选集合中，按 DICOM header 的几何质量排序，替换直接取 CSV 首个序列的方式。保留 DINOv2 Small CLS（384 → 256）、随机抽 24 个训练窗口、EMA、原有损失和 slice/slot/label 聚合结构，不加入 v12 patch mean、v13 分层抽样或 SWA。尚未完成真实训练和线上评估。

## 排序规则

先沿用 v10 的平面、Fluid Sensitive 偏好、`NumSlices > 0` 和不重复 Series UID 的规则，再在这个候选集合内部比较。

- 有完整可读 header、有效 PixelSpacing/图像尺寸、至少两个不同切片位置且法向一致的序列优先。只有一个候选时仍选该候选；全部候选缺少有效几何信息时退回 CSV 首个，不改变 slot 缺失策略。
- 覆盖参考阈值按 Sagittal/Coronal/Axial 分别计算：默认取 **4349 个弱标签训练 Study** 有效序列 coverage 的第 20 百分位。58 个 gold 验证 Study 和测试集不参与拟合。这是预先固定的几何启发式，不表示医学上的充分覆盖，也未通过验证标签调参。
- 按以下 tuple 从大到小排序，不使用多项加权总分：

```text
(min(z_coverage / plane_threshold, 1),
 min(min_in_plane_fov / crop_mm, 1),
 -max(max(row_spacing, col_spacing), crop_mm / image_size),
 -median_unique_slice_spacing,
 number_of_unique_slice_positions)
```

coverage 和视野达到参考值后不再奖励更大范围，再比较平面采样密度、层间距和有效切片数。同分保留原 CSV 顺序；某平面没有可用训练几何数据时阈值为 0，此时不比较该平面的覆盖比例。

切片位置由 `ImagePositionPatient` 投影到 `ImageOrientationPatient` 法向得到；相同位置按 0.0001 mm 精度合并。层间距来自不同物理位置的差值，不用 SliceThickness 替代。PixelSpacing 同时考虑行/列；336/140 mm 下对低于约 0.417 mm 的间距不继续奖励。FOV 用中央切片行列尺寸和对应 PixelSpacing 计算。

本版只读 header，不检测运动伪影、信噪比或实际像素解码失败率。`quality_usable` 指几何信息可用于排序，不保证像素一定可解码；原有像素解码和损坏切片恢复逻辑沿用 v10。TE/TR 不参与新排序，也不改变对比类别划分。

## 准备、缓存与诊断

训练首次启动时仅 rank 0 使用 8 个线程扫描全部训练序列 header，并向其他 rank 广播结果。准备阶段可能明显长于 v10；DDP 等待超时设为两小时以允许首次扫描。

几何统计保存在 `cache-dir/series_quality/*.json`，key 包括规则版本、绝对序列路径和 DICOM 文件名/大小/修改时间。之后启动读取小型统计缓存；`--series-quality-workers` 控制线程数，`--no-cache` 同时禁用统计和输入缓存。

输入候选窗口的预处理、缓存格式和 key 保留 v10。仍选同一批序列时可复用 v10 输入缓存；序列更换时自动生成新 key。训练随机抽 24，验证/推理仍使用所选序列的全部有效候选，最多 80 个。

输出目录新增：

- `series_quality.csv`：各训练序列的 header 质量指标。
- `series_selection.csv`：每 Study-slot 的 v10 首选、v14 选择、是否更换及双方指标，标记 train/valid 划分。两个策略各自按不重复规则执行，因此前一个 slot 的选择也可能影响后一个 slot。
- `series_selection_summary.json`：规则配置、实际 coverage 阈值、更换 Study/slot 数及各 slot 更换数。

checkpoint 的 `series_selection` 保存阈值和规则版本。Kaggle 每张 GPU 只扫描分配给自己的 Study 的序列 header，使用保存的训练阈值，不在测试集拟合，不写大型输入缓存。

## 四卡训练

从 `Baseline_v14` 目录运行，与 v10 最佳配置独立对照：

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --data-root /root/RSNA/rsna-knee-abnormality-detection \
  --output-dir ./outputs/last6_3e-4_quality_selection \
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

DINOv2 默认使用上级 `dinov2-pytorch-small-v1`，标签为本目录 `label.csv`。默认训练窗口改为 24，其余原有 CLI 默认继承 v10；最佳配置以完整命令为准。

## 权重与对照

- 从同一份 DINOv2 预训练权重重新训练，比较除序列选择之外条件一致的 v10/v14。先检查更换比例和指标，再看逐标签验证 AUC 与线上分数。
- 架构标记 `baseline_v14_80_candidates_quality_selection_ema`。模型参数结构与 v10 相同，`--init-checkpoint` 可加载匹配 slot 的 v8/v9/v10/v14 CLS 权重；继续微调应作为单独实验。
- `--resume` 只接受 v14，且当前训练几何统计拟合的选择配置必须与 checkpoint 相同。v12 的投影维度不同，不能直接加载。
- `best.pt` 的 `model` 已是实际 EMA 验证/推理权重，原始权重与 EMA 状态按 v10 保存。
- 使用本目录 `Kaggle_Inference.ipynb`，挂载 v14 `best.pt`、DINOv2 和比赛数据。多份权重时明确填写 `CHECKPOINT_PATH`；notebook 检查 v14 架构和选择配置。

目录只包含六个文件，不复制 v10 权重或缓存。

## 本地验证

已通过合成 DICOM 的几何统计、异常 header、排序/回退、统计缓存失效、所选 Series UID 的输入缓存隔离检查；训练随机 24 / 验证全部 80、epoch 可复现性及 notebook 与训练代码一致性均已验证。预处理、随机采样、CLS 模型、loss 和 EMA 实现与 v10 逐项进行 AST 比对，未改变。

使用本地真实 DINOv2 Small 权重在 CPU 上完成 28×28 小图的 last6 + no-metadata 前向、BCE 反向和优化器更新，验证 EMA checkpoint 保存/恢复及选择配置不一致时的续训拒绝。尚未验证四卡运行、336×336 显存、完整训练、真实全量数据的更换比例和分数；这些需在训练服务器上运行本实验确认。
