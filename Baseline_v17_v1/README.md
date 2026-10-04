# Baseline_v17_v1：原图 + 增强图共同训练

基于 Baseline_v17，每个 Study 沿用原抽样随机流选择 24 个窗口，同时返回这批窗口的原图和增强视图。两组作为两个独立 Study 视图参与训练，分别输出 12 个标签，采用同一标签、mask 和权重：

```text
loss = 0.5 * (BCE_original + BCE_augmented)
```

每个视图的窗口/slot 注意力由独立的 batch 索引隔离。原图和增强图各自预测，两个 loss 等权平均；不新增一致性损失。优化器、梯度累积、学习率调度和 EMA 的更新次数沿用 v17。

## 图片数量与增强

`--train-windows 24` 表示每个视图使用 24 个窗口：**每个 Study 共 24 张原图 + 24 张增强分支图，即 48 张三通道输入**。四卡、每卡 batch size 2 时，每个 micro-batch 为 8 个真实 Study、16 个视图、384 张输入；accum steps 2 时每次更新对应 16 个真实 Study、32 个视图、768 张输入。

增强分支沿用 v17：每个 slot 以 50% 概率应用联合 Gamma/gain，二者均从 `[0.9,1.1]` 均匀采样。同序列所有窗口、三个通道及重复切片共享参数。未触发的 slot 在增强分支也保持原图；希望整个增强分支都应用变换时，可另做 `--intensity-aug-prob 1` 实验。

增强在缓存读取和原窗口抽样之后执行，不修改原图或缓存。沿用 `seed|epoch|StudyUID|slot|intensity` 的独立随机流，不受 DDP rank、worker 数及窗口重排影响。

训练的模型输入数量是 v17 的两倍，计算量和显存需求增加，实际耗时/显存增幅需在训练服务器测量。`--batch-size` 仍表示每 GPU 的真实 Study 数。如果每卡 batch 2 显存不足，可改为 batch 1、accum steps 4，以保持每次更新的真实 Study 数。

## 验证、缓存与权重

保留 v14/v17 的原版标签和 58 个 gold Study 验证集；验证/推理每个 Study 只用原图，最多 80 个候选窗口，输出一次。header 质量选序列、输入预处理、模型参数结构、候选缓存格式/key 和质量统计缓存均沿用 v17。

checkpoint 保存 `training_views=["original","augmented"]`、`training_loss_reduction="equal_view_masked_bce_mean"`，以及原增强配置、序列选择配置和训练参数。架构标记为 `baseline_v17_v1_80_candidates_original_augmented_ema`。`--resume` 仅接受匹配的 v17_v1 完整状态，核对视图/损失约定和增强参数。`--init-checkpoint` 可严格加载兼容的 v8/v9/v10/v14/v17/v17_v1 CLS 权重；标准对照使用相同 DINOv2 预训练初始化。

使用本目录 `Kaggle_Inference.ipynb` 配合 v17_v1 的 `best.pt`。推理核对 checkpoint 的训练视图记录，但只执行原图视图。

## 四卡训练

从 `Baseline_v17_v1` 目录运行。学习率、20 epochs、EMA 0.9995、last6、336/140 mm、随机 24 和 no-metadata 保持原配置，缓存指向已有 v17 的目录：

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --data-root /root/RSNA/rsna-knee-abnormality-detection \
  --output-dir ./outputs/last6_3e-4_9995_original_augmented \
  --cache-dir ../Baseline_v17/input_cache \
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
  --coverage-quantile 0.2 \
  --series-quality-workers 8 \
  --seed 42 \
  --intensity-aug-prob 0.5 \
  --intensity-gamma-range 0.9 1.1 \
  --intensity-gain-range 0.9 1.1
```

缓存目录应改为服务器上的实际路径。关闭增强时，`--intensity-aug-prob 0` 仍训练两个相同的原图视图，模型前向数量仍为两倍。

## 检查

从项目根目录执行：

```bash
python -m unittest discover -s Baseline_v17_v1 -p test_intensity_augmentation.py -v
```

目录仅复制代码、标签和 Notebook，不复制已有权重或缓存。

本地已通过全部 17 项检查，覆盖原图/增强图配对、视图索引隔离、标签与 mask 复制、loss 和梯度等权平均、优化器/调度器/EMA 更新次数、多 worker 跨 epoch 可复现性、缓存复用且不被修改、原 58 个 gold 验证划分及 checkpoint/Notebook 一致性。

另使用真实本地 DINOv2 Small 权重在 CPU 上完成 28×28 小图的原图 24 + 增强图 24 成对前向、视图隔离、last6/no-metadata BCE 反向、优化器与 EMA 更新。Notebook 严格加载同一权重后，对单个未增强视图的全部 80 个候选给出的预测与训练模块完全一致。尚未运行 336×336 四卡完整训练或线上评估。
