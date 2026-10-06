# Baseline_v20：弱标签五折 + 58 个金标留出

直接在 v20 上实现五折交叉验证。沿用 v14 的 DINOv2 Small CLS 模型、五个 slot、原始相邻切片、140 mm physical crop、训练随机 24 窗口、验证/推理全部 80 候选、损失、优化器、调度器和 EMA。`rsna_data.py`、`rsna_model.py`、`label.csv` 与 v14 逐字节相同；训练入口、CV 工具、README 和推理 notebook 已更新。

## 固定划分

4349 个弱标签 Study 做五折；官方 12 标签完整的 58 个金标 Study 始终留出，既不训练，也不参与选 epoch 或融合权重。

先按 StudyInstanceUID 排序，默认 split seed=42。采用一阶多标签迭代分层：根据 12 标签的正例（>0.5）、负例（<0.5）、缺失/0.5 的证据，优先分配稀有类别，再根据剩余标签配额和折大小分配。只改变成员，不修改软标签和置信度权重；NaN/0.5 仍不计 AUC。无需额外安装分层库。[算法参考](https://github.com/trent-b/iterative-stratification)

| 折 | 弱标签训练 | 弱标签验证 | Fracture 验证正例 |
| --- | ---: | ---: | ---: |
| 0 | 3479 | 870 | 60 |
| 1 | 3479 | 870 | 60 |
| 2 | 3479 | 870 | 59 |
| 3 | 3479 | 870 | 59 |
| 4 | 3480 | 869 | 59 |

每个弱标签 Study 恰好在一个折验证，并在其他四折训练。模型初始化和采样的 `--seed` 与成员划分的 `--split-seed` 分开；默认均为 42。不同折默认使用同一模型种子，不额外改变训练配置。当前按 Study 隔离，官方 CSV 不含可靠患者 ID，尚未验证跨 Study 的患者重复。

训练根据当前标签生成成员、标签快照和分布；可用下文的 `--split-only` 命令单独生成预览。

## 训练

从 `Baseline_v20` 目录运行。原 v14 的训练参数默认值保留；下面沿用 v14 README 的 last6 配置。若你原来的最佳实验使用 `--no-pos-weight`，在同一命令追加该选项。

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --data-root /root/RSNA/rsna-knee-abnormality-detection \
  --output-dir ./outputs/cv5_last6 \
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
  --split-seed 42 \
  --coverage-quantile 0.2 \
  --series-quality-workers 8 \
  --amp bf16 \
  --no-metadata
```

不传 `--fold` 时顺序训练 fold 0..4，DDP 进程组和 header 统计只准备一次，每折重新初始化模型、优化器、调度器和 EMA。所有折从原始 DINOv2 预训练权重开始。禁止 `--init-checkpoint`，防止使用已经见过折内验证样本的 v14 权重。

每折的 coverage 阈值和类别正例权重仅用该折训练 Study 计算。序列 header 质量缓存默认统一放在 `/root/RSNA/series_quality_cache/*.json`，不存在时由 rank 0 自动创建，五折及使用同一规则的版本可以共享。可用 `--series-quality-cache-dir` 修改位置；`--cache-dir` 仅指定预处理输入 `.pt` 缓存，默认本目录的 `input_cache/`。`--no-cache` 同时禁用两类磁盘缓存。

两类缓存继续沿用 v14 的格式和 key。旧 header 缓存若位于其他目录，需将其中的 JSON 放入共享目录才能复用；DICOM 的绝对路径、文件大小及修改时间必须一致。输入缓存包含实际选中序列及预处理配置，选择相同序列时可以复用。

只训练某折：在上面的完整命令追加 `--fold 2`。中断后续训：用相同训练参数追加 `--fold 2 --resume ./outputs/cv5_last6/fold_2/last.pt`。续训检查成员、软/金标签指纹、训练参数及 coverage 配置。不会自动跳过或覆盖已有折；已完成部分折后，请明确选择下一折，或恢复中断折。

仅预览划分，不扫描 DICOM、不加载权重：

```bash
python train.py --data-root /root/RSNA/rsna-knee-abnormality-detection \
  --output-dir ./split_preview --split-only
```

## checkpoint、OOF 和金标评估

各折每个 epoch 只验证自己的弱标签验证集，以 weak macro AUC 选 `best.pt`。训练结束后加载该最佳权重，导出该折 OOF，再对全部 58 个金标评估一次。金标不用于 epoch 选择，也不调五折权重。

```text
outputs/cv5_last6/
  cv_manifest.json             # 划分方法、seed、成员和数据指纹
  folds.csv                    # fold 0..4；金标 fold=-1
  cv_targets.csv               # 原始解析后的软/金标签快照
  split_distribution.csv       # 各折逐标签正负例和排除数量
  fold_0/ ... fold_4/
    best.pt / last.pt
    cv.json / data_split.csv
    oof_predictions.csv        # 只包含本折弱标签验证 Study
    gold_predictions.csv       # 固定 58 个金标 Study
    fold_metrics.json
    validation_history.json / loss_curve.png
    series_quality.csv / series_selection.csv / series_selection_summary.json
  oof_predictions.csv          # 五折完成后，合并全部 4349 个折外预测
  gold_ensemble_predictions.csv
  cv_metrics.json
```

五折完成后自动检查完整覆盖和无重叠，并生成汇总。也可手动运行：

```bash
python cv_results.py --output-dir ./outputs/cv5_last6
```

汇总直接用合并后的 OOF 计算 weak macro/逐标签 AUC，不用五个 AUC 的均值替代。金标按 Study UID 对齐，平均五折的 sigmoid 概率（float64 累加，输出 float32），计算完整 58 个 Study 的 macro/逐标签 AUC。缺失/0.5 标签排除，单一类别的 AUC 记录 null。

弱标签 OOF 反映对弱标签的泛化，不能当作金标准确率。训练、OOF、金标评估沿用 `--amp`；Kaggle 使用原 v14 的 fp16 autocast，不同精度或运行环境可能产生数值差异。

模型结构标记继续使用 v14；checkpoint 新增 `cross_validation` 保存 v20 方法、折号、成员和指纹。新 notebook 必须加载完整五折 v20 checkpoint，旧的单份 v14 权重不能代替。无需改标签或模型结构，但需要重新训练五个模型。

## Kaggle 推理

使用本目录 `Kaggle_Inference.ipynb`，可将五折各自的 `best.pt` 重命名为 `fold0.pt` 至 `fold4.pt`，放在同一个 Kaggle Dataset 文件夹下；也支持保留 `fold_0/best.pt` 到 `fold_4/best.pt` 的结构。挂载该权重 Dataset、DINOv2 和比赛数据，默认自动发现恰好五份权重，`CHECKPOINT_PATHS = []` 无需修改。挂载有多余权重或使用其他文件名时，在 `CHECKPOINT_PATHS` 填写全部五个实际路径，顺序任意；真实折号仍根据 checkpoint 内的元数据校验。

检查 fold 0..4 恰好各一份、相同划分/标签指纹、金标完全留出、相同图像预处理和 metadata 设置。按折号排序，每折使用自己的训练 coverage 阈值。每张 GPU 放五个独立模型，按原 v14 的 Study 分片方式并行推理。

保留 v14 的 float32 权重和输入、fp16 autocast、batch size=1、encoder chunk size=16、窗口/顺序、分位数、归一化及 sigmoid。五折在 sigmoid 后等权平均；融合后的输出与旧单模型不同。

每个 Study 的 DICOM 字节只从磁盘读取一次，16 个 I/O 线程和最多 2 个 Study 预取；header 和像素仍使用 pydicom 和 v14 的异常切片恢复。五个 slot 选中的 Series UID 全部相同时才复用预处理。没有半精度权重转换、uint8 量化、分位数近似或自定义解码。

五模型增加计算量，需要在 Kaggle 实际确认显存、运行时间和最终分数。

## 验证

```bash
python tools/check_line_endings.py
```

换行检查从仓库根目录运行。开发阶段已通过 9 项测试，验证真实划分可复现性/稀有类别、留出隔离、错误 checkpoint/续训拒绝、OOF 汇总和概率融合、不同折选择不同序列、DICOM 单次读取、原 v14 与加速五折的 CPU 推理一致性，以及小型合成模型的一折优化器/EMA/保存/续训/预测导出。模型、原始预处理和训练数学函数与 v14 的一致性也已逐项检查。

本地没有 CUDA；尚未完成真实五折训练、多卡 DDP 或 Kaggle T4 全量推理。测试中的小型模型不代表真实 DINOv2 的训练效果和性能。
