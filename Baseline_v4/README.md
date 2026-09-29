# Baseline_v4：A3 标签查询交叉注意力消融

本目录由当前 `baseline_v1` 独立复制而来。实验 A3 **仅移除标签查询交叉注意力**，用于判断 12 个标签使用独立 query 从不同 MRI 序列读取信息是否有帮助。

保留的模块包括：

- DINOv2 切片特征提取；
- 切片 Transformer 与原始切片注意力池化；
- 序列元数据投影；
- 槽位嵌入与槽位/序列间 Transformer；
- 逐标签输出权重和偏置。

A3 对槽位 Transformer 输出的有效槽位做掩码平均，得到一个共享 Study 特征；随后使用 12 组逐标签权重分别产生 logits。缺失槽位不会进入平均值。

训练示例：

```bash
python Baseline_v4/train.py \
  --data-root data \
  --dinov2-model-dir dinov2-pytorch-small-v1 \
  --backbone-mode last6 \
  --epochs 15 \
  --batch-size 2 \
  --accum-steps 2 \
  --amp bf16 \
  --output-dir Baseline_v4/outputs/A3
```

为了与基准公平比较，请保持数据划分、随机种子、GPU 数、梯度累积、预处理和评估方式一致。A3 是相对 v1 的单项消融，不叠加 A1 或 A2。

由于 A3 删除了 `label_queries` 和 `label_attention` 参数，不能用 v1、A1 或 A2 checkpoint 通过严格加载继续训练；应从相同 DINOv2 预训练权重重新训练。这里尚未记录 A3 分数。
