# Baseline_v5：全序列、不重叠 2.5D、层次聚合

以当前 `baseline_v1` 为基础，保留其弱标签训练、58 例真值验证、DDP、微调模式和逐标签 query 交叉注意力预测头。仅在 v5 中改变输入与聚合，v1 文件不修改。

每个 slot 的全部序列均参与处理。每个序列按患者物理位置排序，读取全部原始切片，沿用 140 mm 中心裁剪、整序列强度归一化、336 px resize。随后按 `[0,1,2]`、`[3,4,5]` 等构造不重叠三通道输入；末尾不足三张时重复最后一张补齐。损坏 DICOM 仍按 v1 策略用最近可读切片替代，因此少量异常/边界重复仍可能存在。

```text
全部序列的原始切片
  → 每个序列内部独立三张分组，G = ceil(N / 3)
  → 共享 DINOv2 Small：每组 384 维 CLS
  → Linear：每组 256 维 + 所属序列内的归一化物理深度编码
  → 每个序列独立的组 Transformer + masked attention pooling
  → 每个序列一个 256 维向量 + 对应的 11 维元数据投影
  → 同一 slot 的真实序列平均，得到 [B,6,256]
  → v1 slot Transformer + 12 个 label queries + 输出头
  → [B,12] logits
```

不设置切片数、组数或序列数上限，不再提供 `--num-slices`。图像采用紧凑布局 `[总有效组数,3,H,W]`，由 `group_counts`、`series_batch_indices`、`series_slot_indices` 保存序列和 Study 边界；只有组特征补齐到本 batch 最大组数，并用 mask 排除补齐位置。缺失 slot 用 `slot_mask` 排除。同 slot 内平均是在序列聚合之后进行，每个序列贡献一个向量。

位置编码由组内真实切片的物理中心归一化到 `[-1,1]` 后经过 MLP 得到；尾部复制不参与中心计算。不同序列各自计算深度，单张序列深度为 0。它替代 v1 固定长度的 `slice_position`。

## 训练

依赖与 v1 相同：PyTorch、transformers、pydicom、numpy、pandas、scikit-learn、matplotlib。训练脚本沿用 v1 的 PyTorch 2.9 精度设置接口。默认权重目录已修正为项目根目录的 `dinov2-pytorch-small-v1/`，默认标签为 `Baseline_v5/label.csv`，该文件复制自 v1。

从项目根目录运行：

```bash
python Baseline_v5/train.py --data-root data --backbone-mode frozen --amp bf16
```

Linux 四卡微调最后四层：

```bash
torchrun --standalone --nproc_per_node=4 Baseline_v5/train.py --data-root data --backbone-mode last4 --amp fp16
```

默认输出 `Baseline_v5/outputs/`。`--encoder-chunk-size` 仅控制一次 backbone 编码多少个组，不会采样或丢弃图片。`--no-metadata` 在训练与推理中均有效。

v5 的输入和位置编码改变，**v1–v4 检查点不能直接用 `--resume` 或 `--init-checkpoint` 加载**。DINOv2 预训练权重正常复用；v5 的 `best.pt`/`last.pt` 可用于后续 v5 微调或续训，检查点带有架构标识。

## Kaggle 推理

使用本目录 `Kaggle_Inference.ipynb`，附加比赛数据、DINOv2 Small 和 v5 的 `best.pt`。最后一格的 `CHECKPOINT_PATH` 默认自动查找唯一的 `best.pt`，有多个时填写确切路径。支持一张或两张 GPU，数据处理、模型和 collate 与训练源码一致；图像尺寸、crop 和元数据开关从检查点读取。

