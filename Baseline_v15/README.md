# Baseline_v15：连续物理深度 RoPE 实验

基于 Baseline_v14，把窗口 Transformer 的可学习加性位置 MLP 替换为 RoPE。沿用 v14 的 header 几何质量选序列、DINOv2 Small CLS（384 → 256）、随机抽 24 个训练窗口、最多 80 个候选的验证/推理、逐标签窗口池化、slot/label 聚合、loss 和 EMA。

## 位置编码

`rsna_data.py` 与 v14 完全相同。窗口中心的原始物理深度来自 `ImagePositionPatient` 在切片法向上的投影，再按整个序列范围归一化：

```text
p = 2 * (anchor_depth - series_min_depth) / series_coverage - 1
t = p * rope_position_scale
theta_j = t * rope_base ** (-2*j / head_dim)
```

默认 `rope_base=10000`、`rope_position_scale=16`，即整个序列的旋转坐标范围为 [-16,16]。这是固定的实验起点，并非验证集调优结果。尺度不随抽样窗口数、候选编号、slot 或 batch 改变，同一个窗口在训练和推理时使用相同位置。

- 移除 `window_position` MLP，不再向窗口图像特征相加位置向量。
- 两层窗口 Transformer 的每个 attention head，对 Q/K 的相邻通道对应用旋转；默认 head_dim=32，全部 32 维旋转。V 不旋转。
- 保留 v14 的 pre-norm、FFN、残差和 dropout。SDPA 在训练时使用 attention dropout，eval 时为 0。
- 变长窗口的特征和坐标按相同的 Study-slot 分组/排序/补齐处理。padding 不参与注意力读取或逐标签池化，不同 Study/slot 的窗口不相互读取。
- 角度和 sin/cos 使用 float32 构造，再转换为 Q/K dtype，支持 bf16/fp16 AMP。训练和推理都显式走 RoPE 路径。
- DINOv2 内部二维位置编码、五个 slot 身份向量和 label query 沿用 v14。

该 RoPE 注意力主要利用窗口间的相对深度差；统一平移一组窗口的位置不改变注意力分数。与 v14 的加性位置 MLP 相比，不再显式提供“扫描范围中的绝对位置”向量。仍使用归一化深度，因此相同的坐标差并不代表跨序列相同的毫米距离。

参考：[RoFormer](https://arxiv.org/abs/2104.09864)，[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。

## v14 数据流程

在原有平面、Fluid Sensitive 偏好和 Series UID 不重复规则所确定的候选集合内，按 header 几何质量选序列：优先可用几何，再比较达到参考阈值的 coverage/FOV、平面采样密度、层间距和唯一切片数。同分沿用 CSV 顺序；全部不可用时回退首个候选。

各平面的 coverage 参考阈值仍只由弱标签训练 Study 拟合，默认第 20 百分位。58 个 gold 验证 Study 和测试集不参与拟合。完整规则见 [v14 README](../Baseline_v14/README.md)。

候选预算仍为 `[23,17,15,10,15]`。训练每个 Study 随机抽 24 个有效窗口；验证/推理用全部候选。几何统计和输入缓存格式/key 沿用 v14/v10，所选序列相同时可复用缓存。继续输出 `series_quality.csv`、`series_selection.csv` 和 `series_selection_summary.json`。

## 四卡训练

从 `Baseline_v15` 目录运行：

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --data-root /root/RSNA/rsna-knee-abnormality-detection \
  --output-dir ./outputs/last6_3e-4_quality_rope \
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
  --rope-base 10000 \
  --rope-position-scale 16 \
  --amp bf16 \
  --no-metadata
```

DINOv2 默认使用上级 `dinov2-pytorch-small-v1`，标签默认本目录 `label.csv`；另外复制 v14 的 `label2.csv`、`label3.csv` 供显式指定。目录不复制已有权重和缓存。

## 权重与推理

- 架构标记：`baseline_v15_80_candidates_quality_selection_rope_ema`。checkpoint 新增 `position_encoding`，记录编码类型、坐标约定、通道配对、base 和 position_scale；命令行参数也保存到 `args`。
- `--init-checkpoint` 和 `--resume` 只接受相同 RoPE 配置的 v15。v14 使用加性位置 MLP，不能直接加载或续训。默认从原有 DINOv2 预训练骨干开始训练，用相同配置与 v14 对照。
- `--resume` 还核对原有训练配置和 series selection 阈值，并恢复原始训练权重、优化器、调度器和 EMA。
- `best.pt` 的 `model` 对应实际验证/推理权重：默认 EMA，`--no-ema` 时为原始权重。
- 使用本目录 `Kaggle_Inference.ipynb`，挂载 v15 `best.pt`、DINOv2 和比赛数据。notebook 的 data/model cell 与训练模块同步，从 checkpoint 读取 RoPE 参数，并拒绝旧架构或参数不一致的权重。

## 本地验证

运行 `python -m unittest discover -s Baseline_v15 -p test_rope.py -v`。本地已通过全部 14 项检查：旋转公式/范数、零位置与标准 Transformer 等价、相对平移不变性、原始坐标在子采样中的一致性、padding 屏蔽、不同分组隔离、重排不变性、CPU bf16 反向、eval dropout、80 个候选加单窗口 Study 的前向/反向及 EMA、checkpoint 保护和 notebook 预测一致性。另以真实本地 DINOv2 Small 权重在 CPU 上完成 28×28 小图的 last6 + no-metadata 前向、BCE 反向、优化器更新和 EMA 更新。

尚未进行完整训练、四卡 CUDA 测试、336×336 显存测试或线上评估；RoPE 对分数的影响需通过 v14/v15 对照实验判断。
