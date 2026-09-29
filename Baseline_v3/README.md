# Baseline_v3：A2 槽位/序列间 Transformer 消融

本目录由当前 `baseline_v1` 独立复制而来。实验 A2 **仅移除槽位/序列间 Transformer 自注意力模块**，用于判断不同 MRI 序列在标签查询前相互交互是否有帮助。

保留的模块包括：

- DINOv2 切片特征提取；
- 切片 Transformer；
- 原始切片注意力池化；
- 序列元数据投影；
- 可学习槽位嵌入；
- 标签查询交叉注意力与逐标签输出头。

A2 中，切片聚合后的槽位特征先加入元数据投影和槽位嵌入，然后直接交给标签查询交叉注意力。缺失槽位仍通过标签注意力的 `key_padding_mask` 排除。

训练示例：

```bash
python Baseline_v3/train.py \
  --data-root data \
  --dinov2-model-dir dinov2-pytorch-small-v1 \
  --backbone-mode last6 \
  --epochs 10 \
  --batch-size 2 \
  --output-dir Baseline_v3/outputs/A2
```

为了与基准公平比较，请保持数据划分、随机种子、GPU 数、梯度累积、预处理和评估方式一致。A2 是相对 v1 的单项消融，不叠加 A1 的平均池化改动。

由于 A2 删除了 `slot_encoder` 参数，不能用 v1 或 A1 checkpoint 通过严格加载继续训练；应从相同 DINOv2 预训练权重重新训练。这里尚未记录 A2 分数。
