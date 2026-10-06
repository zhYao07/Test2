# Baseline_v18：physical 2.5D ±3 mm

基于 v14，只修改每个窗口的两侧通道：`nearest(z - 3 mm)`、原 anchor、`nearest(z + 3 mm)`。使用已有患者坐标排序后的切片位置，在同一序列内选择最近的真实切片，不做 Z 插值。训练、验证和 Kaggle 推理使用同一规则。

保留 v14 的序列质量排序、CLS 模型、label query、随机 24 个训练窗口、五 slot 候选配额 23/17/15/10/15、140 mm 中心裁剪、336 图像、损失、EMA 和原来的 58 个 gold 验证 Study。没有 v16 验证划分、v17 增强或 v17_v1 双视图。

## 邻片规则

- 中间通道始终是原 anchor 索引，即使存在同位置切片也不重新查找。
- 两侧按物理距离找最近切片；距离在 0.0001 mm 内相同，优先取更靠近 anchor 的物理位置。重复位置优先选择索引更靠近 anchor 的切片，仍同分则取较小索引。
- 目标超出覆盖范围时选最近边界切片；允许两侧与 anchor 重复。不会为了避免重复而改成索引邻片。
- 例如 1 mm 间距通常由 ±1 mm 变成 ±3 mm；3 mm 和 4 mm 间距通常保持原邻片；2 mm 间距遇到 ±2/±4 mm 同分时选 ±2 mm。极粗间距可能使两侧都选回 anchor。
- 仍从原 2%–98% 索引范围选择 anchor，邻片可使用整个原序列。原来的 anchor 归一化、物理位置编码和坏像素读取恢复逻辑保持不变。

这统一的是目标邻片距离，实际偏移依然取决于真实切片位置。收益需通过本次训练和线上提交确认。

## 四卡训练

从 `Baseline_v18` 目录运行。使用 v14 最佳训练参数，从相同 DINOv2 Small 预训练权重重新训练：

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --data-root /root/RSNA/rsna-knee-abnormality-detection \
  --output-dir ./outputs/last6_3e-4_9995_physical3mm \
  --cache-dir ./input_cache \
  --series-quality-cache-dir /root/RSNA/series_quality_cache \
  --crop-mm 140 \
  --image-size 336 \
  --train-windows 24 \
  --span-lo 0.02 \
  --span-hi 0.98 \
  --context-mm 3 \
  --batch-size 2 \
  --accum-steps 2 \
  --backbone-mode last6 \
  --epochs 20 \
  --head-lr 3e-4 \
  --backbone-lr 1e-5 \
  --amp bf16 \
  --no-metadata \
  --ema-decay 0.9995 \
  --coverage-quantile 0.2 \
  --series-quality-workers 8 \
  --seed 42
```

`--context-mm` 默认 3；原训练命令可以使用，但请给 v18 单独的输出目录。本次只测试 ±3 mm。不要加 `--init-checkpoint`，这样才能与从相同预训练初始化的 v14 对照。DINOv2 默认位于上级 `dinov2-pytorch-small-v1`，训练标签用本目录的相同 `label.csv`。

每个 Study 训练仍输入 24 张三通道图，验证/推理最多 80 张；没有双视图，也不增加模型前向窗口数。首次构建缓存时，实际解码的唯一切片数可能变化。

## 缓存、权重与推理

header 质量缓存与图像预处理缓存独立：

```text
/root/RSNA/
  series_quality_cache/       # 跨版本共用的 header 质量 JSON
  Baseline_v18/
    input_cache/              # 本版本图像预处理 .pt
    outputs/                  # 权重、所选序列 CSV、验证记录
```

`--series-quality-cache-dir` 默认是当前版本目录的上一级 `series_quality_cache`，因此以后复制 v18 为相邻的新版本目录时，默认仍共用这一个缓存。`--cache-dir` 只控制图像预处理目录；`--no-cache` 只关闭图像缓存，header 缓存仍保留。需要重新扫描 header 时使用独立的 `--no-series-quality-cache`。`build_datasets` 也提供独立的 `series_quality_cache_dir` 参数，显式传 `None` 可关闭 header 缓存。

若 `--cache-dir` 指向原有缓存目录，启动会尝试读取其中旧 `series_quality/*.json`，把有效命中的记录复制到新共享目录；不会删除旧文件，也不会为这些命中重新扫描 header。如果本次图像目录是新目录，可先在服务器上把旧 JSON 复制到共享目录：

```bash
mkdir -p /root/RSNA/series_quality_cache
cp -n /root/RSNA/Baseline_v14/input_cache/series_quality/*.json /root/RSNA/series_quality_cache/
```

把旧目录改成实际已有缓存的位置。质量 JSON 的文件名和失效规则未改：规则版本、DICOM 绝对路径、文件名、大小、修改时间相同即可复用；后续启动仍枚举文件并检查这些信息，有效缓存命中时不重新读取 header。

这些 JSON 保存的是序列几何质量统计。最终所选 Series UID 仍按原有规则快速排序得出，写入每次输出目录的 `series_selection.csv`。只增加模型版本、改变增强或物理邻片参数，不会影响选序列；修改选择规则、coverage 配置、XY 裁剪/图像尺寸、训练 Study 集合或原数据则可能影响结果。因此不把所选序列永久锁死。

图像缓存仍采用 schema 2，key 包含 `context_mm` 和邻片规则版本，不会读取或覆盖 v14 的旧图像缓存。Kaggle notebook 继续按当前 GPU 分配的测试 Study 读取 header，不依赖训练服务器的共享目录。

checkpoint 顶层和 `args` 保存 `physical_context`，顶层同时保存原有 `series_selection`。`--resume` 只接受匹配的 v18 架构、物理邻片配置和其余续训配置；不能把 v14 作为 v18 的完整续训状态。模型参数结构不变，另行微调时可用 `--init-checkpoint` 加载 v14 权重。

使用本目录 `Kaggle_Inference.ipynb` 和 v18 `best.pt`。notebook 检查架构、物理邻片版本、毫米距离及预处理参数一致性，使用 checkpoint 保存的序列选择阈值和全部候选。`best.pt` 的 `model` 仍是实际 EMA 验证/推理权重。

## 训练前检查实际改变比例

可选：用 v14 最佳运行输出的 `series_selection.csv`，按实际选中的序列扫描 header，检查有多少候选窗口改变，以及是否出现过多 anchor 重复。只读 header，不解码像素，不重拟合序列选择阈值。先看 58 个验证 Study：

```bash
python audit_physical_context.py \
  --data-root /root/RSNA/rsna-knee-abnormality-detection \
  --series-selection-csv ../Baseline_v14/outputs/last6_3e-4_9995/series_selection.csv \
  --output-dir ./outputs/physical_context_audit_valid \
  --split valid \
  --workers 8
```

把 `--series-selection-csv` 改成实际 v14 输出路径。去掉 `--split valid` 可检查全部 Study。若旧实验的 span 不同，审计也要传入对应 `--span-lo/--span-hi`。

输出 `physical_context.csv` 和 `physical_context_summary.json`，包含窗口变化比例、两侧实际偏移分位数、与 anchor 同位置的比例和失败序列。统计按完整候选计算，不是某轮训练随机 24 的统计。它需要重新读取选中序列的切片位置 header：v14 质量缓存只存汇总指标，没有全部位置。

## 验证

```bash
python -m unittest discover -s . -p test_physical_context.py -v
```

测试覆盖物理邻片、同距规则、边界/重复位置、合成 DICOM 实际预处理、图像缓存隔离、训练/验证采样、审计输出及 notebook 一致性。已使用本地真实 DINOv2 Small 在 CPU、28×28 输入下完成 last6 + no-metadata 前向、BCE 反向、优化器更新、EMA 和 checkpoint 保存/续训配置检查；notebook 加载 EMA 权重后的 80 窗口预测与训练模块完全一致。

完整四卡训练、336 图像显存、全量输入变化比例和竞赛分数需要在服务器确认。
