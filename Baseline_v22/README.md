# Baseline_v22：DINOv2 + 逐标签 attention-MIL

基于 v14 新增独立版本。保留 v14 的本地 Hugging Face DINOv2 Small backbone、CLS 输出、ImageNet 归一化、header 质量排序、五 slot 候选构建、140 mm physical crop、336 图像尺寸、训练随机 24 窗口、验证/推理全部有效候选（最多 80）、训练划分、loss、优化器、学习率调度与 EMA。

## 模型

后续流程按提供的 `train_knee.py` 中 `RaptorClassifier` 替换：

```text
有效 2.5D 窗口 → DINOv2 CLS [N,384]
→ 按 Study 汇为一个 bag（跨全部有效 slot）
→ LayerNorm(384)
→ Linear(384,256) → Tanh → Dropout(0.2) → Linear(256,12)
→ 每个标签在完整 Study 窗口维上 softmax
→ 逐标签加权求和 [B,12,384]
→ clsW 点积 + clsb → logits [B,12]
```

移除 v14 的 CLS 降维投影、物理位置编码、slice/slot Transformer、metadata 投影、slot embedding 和 label query 交叉注意力。slot 仍用于数据选择与输入一致性检查，所有有效窗口直接参加 Study 级 MIL。变长 batch 仅补齐特征，补齐位置不参与 softmax。DINOv2 分块编码后先拼接全部特征，再调用一次 MIL head，不对各块概率单独平均。

只替换模型 backbone 后的结构；参考文件的 CoAtNet、缓存 corpus、ROI、DDP/训练参数和 SWA 不引入本版。`--no-metadata` 保留作 v14 命令兼容参数，v22 始终不使用 metadata。

## 训练

在 `Baseline_v22` 目录运行：

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --data-root /root/RSNA/rsna-knee-abnormality-detection \
  --output-dir ./outputs/last6_attention_mil \
  --cache-dir ../Baseline_v14/input_cache \
  --crop-mm 140 --image-size 336 --train-windows 24 \
  --span-lo 0.02 --span-hi 0.98 \
  --batch-size 2 --accum-steps 2 --backbone-mode last6 \
  --epochs 15 --head-lr 3e-4 --backbone-lr 1e-5 \
  --ema-decay 0.999 --seed 42 --coverage-quantile 0.2 \
  --series-quality-workers 8 --amp bf16
```

`label.csv` 从 v14 原样继承；默认 DINOv2 目录为上级 `dinov2-pytorch-small-v1`。已有 v14 输入缓存可复用。模型从 DINOv2 预训练权重和新 MIL head 开始训练；`--init-checkpoint` 和 `--resume` 只接受 v22 架构，旧 v14 head 权重不兼容。架构标记为 `baseline_v22_dinov2_attention_mil_ema`。`best.pt` 的 `model` 对应实际 EMA 验证/推理权重。

## Kaggle 推理

使用本目录 `Kaggle_Inference.ipynb`，挂载 v22 `best.pt`、v14 同款 DINOv2 和比赛数据。只有一份权重时自动寻找 `best.pt`，否则填写 `CHECKPOINT_PATH`。保持 v14 单模型、最多双 GPU 分配 Study 的推理方式。

接入 v20 的 I/O 加速：每个 GPU 的 I/O 线程池并行读取 DICOM；同一 Study 的原始字节与 header 供质量排序、物理排序、像素解码复用；完成预处理后释放字节；有界线程预取 Study，按原顺序消费，完成的 CPU tensor 使用 pin memory 与 non-blocking 传输。默认 `DICOM_THREADS=16`、`PREFETCH_STUDIES=2`、`BATCH_SIZE=1`、`ENCODER_CHUNK_SIZE=16`。未引入 v20 的五折训练和五模型集成。

## 验证

```bash
python -m unittest discover -s Baseline_v22 -p 'test_*.py' -v
python tools/check_line_endings.py
```

测试覆盖原始 MIL 公式与梯度等价性、变长/乱序 Study bag 隔离、补齐屏蔽、编码分块、旧权重拒绝、notebook 模型同步，以及真实本地 DINOv2 小图的 last6 反向、优化器更新、EMA 与权重恢复。合成 DICOM 检查加速前后输入逐项一致、每文件单次读取、异常释放和预取顺序。完整 336 四卡训练、Kaggle GPU 加速比及比赛分数仍需实际运行确认。
