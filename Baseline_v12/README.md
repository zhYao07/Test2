# Baseline_v12：CLS + patch mean 特征实验

基于 Baseline_v10，仅将每个窗口的 DINOv2 特征从 CLS 改为 CLS 与全部 patch token 均值拼接。用于和 v10 的线上 0.928 配置作单因素对照，尚未完成真实训练和线上评估。

## 特征变化

```text
原始相邻三张切片 [N,3,336,336]
  → DINOv2 Small last_hidden_state [N,577,384]
  → concat(CLS, mean(patch tokens)) [N,768]
  → LayerNorm(768) + Linear(768,256)
  → v10 原有 slice Transformer / 逐标签窗口池化 / slot Transformer / label query
  → 12 个 logits
```

336/14=24，因此每个窗口有 576 个 patch token；均值只取 `tokens[:,1:]`，不含 CLS。不增加 patch attention、top-k 池化或 backbone 编码次数。默认 256 维投影比 v10 多 99,072 个参数，后续网络形状保持一致。分块编码、冻结/部分解冻和梯度传播逻辑沿用 v10。

## 对照配置

保持 v10 最佳实验的 24 个训练窗口、336 尺寸、140 mm 物理裁剪、2%-98% 覆盖、last6、15 epochs、有效 batch 16、关闭 metadata 和默认 EMA。数据文件、标签和损失、五个 slot、23/17/15/10/15 候选预算、训练随机抽样、最多 80 个候选验证/推理、4349/58 划分均沿用 v10，不加入 SWA。

`train.py` 的 `--train-windows` 默认改为 24，其余 CLI 默认值继承 v10；完整最佳配置以以下显式命令为准。

从 `Baseline_v12` 目录运行：

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --data-root /root/RSNA/rsna-knee-abnormality-detection \
  --output-dir ./outputs/last6_3e-4_cls_patch_mean \
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
  --amp bf16 \
  --no-metadata
```

默认 DINOv2 从上级目录 `dinov2-pytorch-small-v1` 加载。缓存逻辑与 v10 完全相同，可直接复用既有输入缓存；其他缓存位置请调整 `--cache-dir`。服务器若没有 v10 缓存，可指定 `./input_cache`。

建议从同一份 DINOv2 预训练权重重新训练，以比较特征设计。保存逐标签 AUC、最佳 epoch、本地 Macro AUC 和线上分数；其余实验条件应与 v10 对齐。

## Checkpoint 与 Kaggle 推理

- 架构标记为 `baseline_v12_80_candidates_cls_patch_mean_ema`。
- 投影从 384 维变为 768 维，v10/v11 完整权重不能直接加载；`--init-checkpoint` 和 `--resume` 仅接受匹配的 v12 权重。不自动迁移旧投影。
- EMA 沿用 v10：checkpoint 的 `model` 保存实际验证/推理权重，`training_model` 与 `ema` 用于恢复训练。默认以 EMA 验证并保存 `best.pt`。
- 使用本目录的 `Kaggle_Inference.ipynb`，挂载 v12 的 `best.pt`、DINOv2 和比赛数据；训练模型与 notebook 内嵌模型采用相同的 CLS+patch mean 实现。
- `CHECKPOINT_PATH` 留空时寻找唯一 `best.pt`；有多份权重时明确填写 v12 的路径。Notebook 检查 v12 架构标记并严格加载权重。
- 预处理、metadata 开关和覆盖范围读取 checkpoint 参数，推理使用全部有效候选，支持最多两张 GPU，不写大型输入缓存，不加入 TTA 或排名转换。

目录只包含数据、模型、训练脚本、标签、README 和 Kaggle notebook，不复制 v10 权重或缓存。

## 本地验证

已使用本地 DINOv2 Small 真实权重完成 CPU 检查：336 输入的 576 个 patch token、CLS/patch 均值与分块编码结果、含缺失 slot 的两 Study 前向、训练与 notebook 预测一致、冻结与 last6 反向传播、EMA 更新及旧 v10 续训权重拒绝。Python 与 notebook 代码均通过语法检查，训练模型与 notebook 模型的 AST 一致，数据模块与标签文件和 v10 逐字节一致。

尚未运行服务器四卡完整训练或 Kaggle 线上评分；模型收益需由本次实验确认。
