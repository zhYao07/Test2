# Baseline_v17：v14 + 序列一致的强度增强

直接基于 Baseline_v14。保留原来的 58 个 gold Study 验证集、弱标签训练集、header 质量选序列、DINOv2 Small CLS、随机抽 24 个训练窗口、验证/推理最多 80 个候选、loss 和 EMA。采用原版 `label.csv`。模型参数结构与 v14 相同。

## 首轮增强

训练时，每个 Study-slot 独立以 50% 概率触发增强；触发后从 `[0.9,1.1]` 分别均匀采样 Gamma 和 gain，二者共同应用：

```python
images = (images.pow(gamma) * gain).clamp(0, 1)
```

同一 slot 内所有窗口、三个通道、重复出现的切片共享同一组参数。零背景保持为零。第一轮仅测试强度增强，参数范围是预先固定的实验起点。

接入位置：读取原始候选缓存 → 沿用 v14 随机抽 24 个窗口 → 组装三通道图像 → 训练强度增强 → 模型内部 ImageNet 标准化 → DINOv2。验证和推理直接使用原图。

增强随机种子由 `seed|epoch|StudyUID|slot|intensity` 的 BLAKE2b hash 决定，独立于原有窗口抽样随机流，不依赖 DDP rank 或 worker 数。同一 epoch 可复现，不同 epoch 重新生成参数；各 slot 独立采样，不受其他 slot 缺失或窗口重排影响。

## 缓存与配置

输入缓存始终保存未增强图像，增强不写入或修改缓存。候选缓存格式/key、质量统计缓存和序列选择阈值规则与 v14 完全相同，数据绝对路径和文件状态相同时可复用。

新参数：

| 参数 | 默认值 | 用途 |
|---|---|---|
| `--intensity-aug-prob` | `0.5` | 每个训练序列触发联合增强的概率；`0` 完全关闭 |
| `--intensity-gamma-range LOW HIGH` | `0.9 1.1` | Gamma 均匀采样范围 |
| `--intensity-gain-range LOW HIGH` | `0.9 1.1` | gain 均匀采样范围 |

增强配置保存到 `hyperparameters.json` 和 checkpoint 的 `intensity_augmentation`，包含规则版本 `series_gamma_gain_v1`、概率与参数范围。`--resume` 要求 v17 架构、序列选择配置、增强配置及原有训练配置匹配。改变增强参数应开启独立实验。

`--init-checkpoint` 可读取 v8/v9/v10/v14/v17 的匹配 CLS 权重，仍采用严格参数加载；首轮对照应从同一份 DINOv2 预训练权重重新训练。Kaggle Notebook 检查 v17 checkpoint，记录训练增强配置，但不在推理时执行增强。

## 与 v14 最佳配置对照的四卡命令

从 `Baseline_v17` 目录运行。20 epochs、EMA 0.9995、last6、336/140 mm、随机 24 和关闭 metadata 与用户的 v14 对照保持一致；其他原有 CLI 默认继承 v14，以完整命令为准。

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --data-root /root/RSNA/rsna-knee-abnormality-detection \
  --output-dir ./outputs/last6_3e-4_9995_intensity_aug \
  --cache-dir ../Baseline_v14/input_cache \
  --crop-mm 140 \
  --image-size 336 \
  --train-windows 24 \
  --span-lo 0.02 \
  --span-hi 0.98 \
  --batch-size 2 \
  --accum-steps 2 \
  --backbone-mode last6 \
  --epochs 20 \
  --head-lr 3e-4 \
  --backbone-lr 1e-5 \
  --amp bf16 \
  --no-metadata \
  --ema-decay 0.9995 \
  --seed 42 \
  --coverage-quantile 0.2 \
  --series-quality-workers 8 \
  --intensity-aug-prob 0.5 \
  --intensity-gamma-range 0.9 1.1 \
  --intensity-gain-range 0.9 1.1
```

若服务器上的 v14 缓存不在上述路径，修改 `--cache-dir` 指向实际目录。目录只复制代码、Notebook 和原始标签，不复制已有权重或缓存。

关闭增强对照：将概率改为 `--intensity-aug-prob 0`，使用独立输出目录。后续消融可将 gain 范围设为 `1 1` 测试 Gamma-only，或将 Gamma 范围设为 `1 1` 测试 gain-only。

## 检查

```bash
python -m unittest discover -s Baseline_v17 -p test_intensity_augmentation.py -v
```

从项目根目录执行，使用安装了 PyTorch、pandas、pydicom、transformers 等训练依赖的环境。

本地已通过全部 14 项检查：变换公式、同序列/三通道一致性、零背景和数值范围、独立随机流、窗口重排及缺失 slot、不修改输入缓存、关闭增强与 v14 完全一致、验证全部 80 个原始窗口、多 spawn worker 与 persistent worker 跨 epoch 一致性、checkpoint 保存和续训配置保护，以及原验证划分/模型/Notebook 源码对照。

另使用本地真实 DINOv2 Small 权重在 CPU 上完成 28×28 小图的增强随机 24、last6 + no-metadata 前向、BCE 反向、优化器与 EMA 更新；Notebook 严格加载同一权重后，对未增强的全部 80 个候选给出的预测与训练模块完全一致。尚未运行 336×336 四卡 CUDA 完整训练或线上评估。
