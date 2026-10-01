# Baseline_v6：每个 slot 共享 32 个原始相邻窗口

基于 Baseline_v5，只改变输入窗口的分配与构造。网络结构、组→序列→slot 聚合、序列元数据、标签 query 预测头、loss、训练超参数、标签和 58 例验证划分均沿用 v5。默认仍为 336 px、140 mm crop。

## 窗口分配

每个 slot 的全部序列共享 `--slot-window-budget`，默认最多 32 个有效窗口，**不是每个序列 32 个**。

序列先按 SeriesInstanceUID 排序，再轮流分配一个窗口，直到预算或原始切片容量耗尽。每个序列的窗口数不超过原始切片数；短序列未用完的份额自动分给其他序列。所有序列至少保留一个窗口。若同 slot 的序列数大于预算，会明确报错，需增加预算，避免静默丢弃序列。

| 同一 slot 的原始切片数 | 各序列窗口配额 |
|---|---|
| 30、30 | 16、16 |
| 30、30、30 | 11、11、10 |
| 8、30 | 8、24 |
| 30 | 30 |
| 4、7 | 4、7 |

总切片数不足预算时只使用实际数量，不重复 anchor 凑足 32 个。六个 slot 均存在时，每个 Study 最多编码 192 个窗口。

## 2.5D 输入

每个序列独立按患者物理位置排序，在全序列范围内设置均匀物理目标，并在保持索引递增、为后续 anchor 留足切片的位置范围内选最近切片，以获得指定数量的不同 anchor。一个窗口时取物理中点附近；若全部物理位置相同，则按原始索引均匀选择。

每个 anchor `i` 对应原始排序序列中的 `[i-1, i, i+1]`，序列边界取 `[0,0,1]` 或 `[N-2,N-1,N-1]`。邻近窗口可以共享图片，但同一序列的 anchor 不重复，且窗口绝不跨序列构造。坏 DICOM 继续沿用 v5 的最近可读切片替代策略。

沿用 **v5 整个序列裁剪后的 0.5%/99.5% 强度分位数**，避免采样改变归一化统计；因此仍会读取全部原始切片用于归一化，但只 resize/编码选中的窗口。位置编码使用 anchor 在所属序列内的归一化物理深度 `[-1,1]`。

```text
slot 内的全部序列
  → 共用最多 32 个窗口的配额
  → 每个序列独立选不同 anchor，取原始相邻三张
  → 共享 DINOv2 + 256 维投影和物理位置编码
  → 每个序列独立的组 Transformer + masked attention pooling
  → 序列向量加入对应元数据，同 slot 内平均
  → v5 的 slot Transformer 和标签交叉注意力头
  → [B,12] logits
```

沿用紧凑图像布局 `[总有效窗口数,3,H,W]`，由 `group_counts`、序列所属 Study 和 slot 保存边界。仅组特征补齐并加 mask，缺失 slot 继续用 `slot_mask` 排除。

## 训练与推理

依赖与 v5 相同。默认标签为本目录 `label.csv`（复制自 v5），预训练模型默认位于项目根目录 `dinov2-pytorch-small-v1/`，输出为 `Baseline_v6/outputs/`。

```bash
python Baseline_v6/train.py --data-root data --slot-window-budget 32 --backbone-mode frozen --amp bf16
```

Linux 四卡、与 v5 相同的最后四层微调：

```bash
torchrun --standalone --nproc_per_node=4 Baseline_v6/train.py --data-root data --slot-window-budget 32 --backbone-mode last4 --amp fp16
```

`--encoder-chunk-size` 只控制一次编码多少个窗口，与 slot 总预算不同。`--num-slices` 不适用于 v6。

模型参数名称和形状与 v5 相同，`--init-checkpoint` 可以继承 v5 或 v6 模型权重。独立比较输入方案时，应使用与 v5 相同的初始化方式和训练参数。`--resume` 只接受 v6 检查点，并要求窗口预算一致。

Kaggle 使用本目录 `Kaggle_Inference.ipynb`，附加比赛数据、DINOv2 Small 和 v6 的 `best.pt`。图像尺寸、crop、slot 窗口预算和元数据开关均从检查点读取；检查点用 v6 标识区分输入流程。支持一张或两张 GPU。

本目录仅保留与 v1 相同的六个文件，不保留额外测试或辅助脚本。
