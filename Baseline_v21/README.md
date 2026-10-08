# Baseline_v21：RadImageNet pretrained ResNet50

基于 v14，将 DINOv2 Small CLS backbone 替换为 **RadImageNet pretrained ResNet50**。保留 header 质量排序、五个 slot、相邻三切片 2.5D 输入、训练随机 24 个窗口、验证/推理全部候选（最多 80）、物理位置编码、窗口/slot/label 聚合、损失、58 个 gold Study 验证划分和 EMA。v14 的数据代码与标签原样复制，详细序列规则见 [v14 README](../Baseline_v14/README.md)。

## 模型与预训练权重

- 采用 torchvision ResNet50，移除原分类层，使用全局平均池化的 **2048 维**特征，经 `LayerNorm + Linear(2048, 256)` 接入原聚合网络。
- Dataset 仍输出 `[0,1]` 的相邻三切片，模型中使用 `2*x-1` 归一化到 `[-1,1]`，对应 RadImageNet 官方 PyTorch 示例。通道仍按原来的切片顺序，不做 RGB/BGR 交换；这里三通道表达相邻位置。
- 保留 336×336 输入和 140 mm physical crop；ResNet50 的自适应全局池化支持该尺寸，不强制改为官方示例的 224×224。
- 所有训练模式均固定 BatchNorm 的 running mean/variance 和计数器，避免小 Study batch、窗口相关性和编码分块影响统计。解冻 stage 内的 BN affine 参数仍可训练。每次 `model.train()` 后也会重新固定统计。
- **训练必须加载本地预训练文件**，不在线下载，不使用 ImageNet 默认权重，不允许缺失文件后随机初始化。官方的 `backbone.0/1/4/5/6/7.*` 参数名显式映射为 torchvision 命名；所有参数、buffer 和形状必须完整匹配。已转换的 torchvision encoder state dict 也可加载；参数名匹配本身不证明来源，仍须使用 RadImageNet 文件。

官方来源：[BMEII-AI/RadImageNet](https://github.com/BMEII-AI/RadImageNet)，[PyTorch 示例](https://github.com/BMEII-AI/RadImageNet/blob/main/pytorch_example.ipynb)，[官方 PyTorch 权重压缩包](https://drive.google.com/file/d/1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR/view?usp=sharing)。解压后使用 `RadImageNet_pytorch/ResNet50.pt`，默认读取 `Baseline_v21` 上级目录的 `radimagenet/ResNet50.pt`，即服务器上的 `/root/RSNA/radimagenet/ResNet50.pt`；其他路径通过 `--radimagenet-weights` 指定。

此前下载的本地文件位于仓库根目录的 `weights/radimagenet/ResNet50.pt`，已完成严格加载验证。将其复制到服务器上的上述默认路径即可。文件 SHA256 为：

```text
08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734
```

权重位于 Git 忽略的 `weights/` 中，不随代码提交；训练服务器需要另行复制。启动日志、hyperparameters 和 checkpoint args 保存所用文件的 SHA256；续训要求与原训练一致。

## 四卡训练

需要 PyTorch、与之匹配的 torchvision、numpy、pandas、pydicom、scikit-learn、matplotlib，以及比赛 DICOM 使用的解码依赖；不再需要 transformers 或 DINOv2 文件。沿用支持 `torch.amp` 与 `fp32_precision` 的 v14 训练环境。

从 `Baseline_v21` 目录运行：

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --output-dir ./outputs/layer34_backbone2e-3_imsz224_layer34 \
  --cache-dir ../Baseline_v20/input_cache \
  --image-size 224 \
  --backbone-mode layer3+layer4 \
  --epochs 20 \
  --backbone-lr 2e-3 \
  --ema-decay 0.9995 \
  --swa
```

命令中省略的参数使用以下默认值：数据目录 `/root/RSNA/rsna-knee-abnormality-detection`、权重 `../radimagenet/ResNet50.pt`、`crop-mm=140`、`train-windows=24`、`batch-size=2`、`accum-steps=2`、`head-lr=3e-4`、`span-lo=0.02`、`span-hi=0.98`、`seed=42`、`coverage-quantile=0.2`、`series-quality-workers=8`、`amp=bf16`，以及关闭 metadata。`--no-metadata` 仍兼容旧命令；需要启用 metadata 时显式传入 `--metadata`。默认参数调整仅作用于 v21，checkpoint 仍保存实际解析后的完整参数，推理继续读取其 `no_metadata` 值。

`--backbone-mode` 支持 `frozen`（默认）、`layer4`、`layer3+layer4`、`full`。上面的命令使用当前实验的 224×224、20 epochs、backbone 学习率 2e-3 和 EMA 0.9995；其余默认值在上文列出。`layer3+layer4` 不能与 DINOv2 `last6` 视为等量解冻。编码 chunk 默认 24，可用 `--encoder-chunk-size` 调整；完整训练显存和分数需在服务器上测量。

原预处理缓存存储未归一化到 backbone 的 `[0,1]` 输入，因此可以复用 v14/v10 输入缓存；选中不同序列时，原缓存 key 会隔离。质量排序诊断与 coverage 阈值拟合规则也完全沿用 v14。

两个缓存统一由 `--cache-dir` 管理：图像 `.pt` 直接保存在该目录，序列质量 JSON 保存在其 `series_quality/` 子目录。上面的命令对应 `/root/RSNA/Baseline_v20/input_cache` 及其 `series_quality/` 子目录；不再提供 `--series-quality-cache-dir` 参数。已确认 v20 与 v21 的输入预处理、缓存 schema/key、序列质量缓存规则一致；相同数据路径、文件信息、预处理参数及所选序列时可以复用现有缓存。改变 `image-size` 会生成新的图像缓存，序列质量缓存仍可复用。若旧质量 JSON 位于独立目录，需将其复制到 `--cache-dir/series_quality/` 才能复用。`--no-cache` 同时关闭两种缓存。Kaggle notebook 的推理仍不写磁盘缓存，无需修改。

## Checkpoint 与 Kaggle 推理

架构标记为 `baseline_v21_radimagenet_resnet50_quality_selection_ema`，checkpoint 另存 `backbone_spec`，明确预训练来源、特征维度、归一化和 BN 策略。

- `best.pt` 的 `model` 是实际验证/推理权重，包含完整 ResNet50 和聚合头；原始训练权重和 EMA 状态分别保存。
- `--init-checkpoint` 只接受匹配的 v21 模型权重，可用于冻结训练后解冻微调。`--resume` 还恢复优化器、调度器、scaler、EMA，校验训练配置、预训练文件 SHA256 和序列选择阈值。
- v14 的 DINOv2 checkpoint 不能加载或续训为 v21。应从 RadImageNet 预训练权重开始训练。
- 使用本目录 `Kaggle_Inference.ipynb`，挂载比赛数据和 v21 `best.pt`。无需再挂载 DINOv2 或 RadImageNet 原始预训练文件；推理构造空结构后立即严格恢复完整 checkpoint。
- 多份 `best.pt` 时明确填写 `CHECKPOINT_PATH`。notebook 校验 v21 架构、backbone spec、slot、候选预算和预处理参数，使用训练保存的 coverage 阈值；支持一张或两张 Kaggle GPU。

## 推理 I/O 加速（沿用 v20）

notebook 新增 I/O 单元格，默认 `FAST_IO = True`。每张 GPU 使用 `DICOM_THREADS = 16` 个文件读取线程、`PREFETCH_STUDIES = 2` 个 Study 预处理线程。每个 Study 的 DICOM 文件只读取一次，同一份字节和 header 用于质量评估、切片排序和原来的 pydicom 像素解码；预处理结束后在 `finally` 中释放原始字节。最多预取两个待消费的 Study，按原顺序送入 GPU，重叠 CPU 预处理与 GPU 推理。

只改变 I/O 和 CPU 调度。保留 v21 单模型、全部候选、序列选择阈值、相邻切片、物理裁剪、分位数归一化、双线性缩放、metadata、float32 输入和权重、`BATCH_SIZE = 1`、`ENCODER_CHUNK_SIZE = 16`、CUDA float16 autocast 以及 float32 sigmoid。仍从 checkpoint 读取 `image_size`、`crop_mm` 和 span，预测存储与提交顺序也保持一致。无需重新训练或修改权重。

设置 `FAST_IO = False` 可恢复直接读取 DICOM、提前扫描质量和原来的 DataLoader，便于对照。比较提交结果时保持 batch 和 encoder chunk 不变。实际加速幅度取决于 Kaggle 存储和 GPU 利用率，本地没有测量倍数。

新增六项 CPU 回归检查：224×224、完整 80 个候选及缺失 slot 的全部输入张量精确相等；质量评估和序列选择一致；覆盖 MONOCHROME1/rescale、RLE 压缩、16-bit 标签配字节载荷的回退以及损坏像素的近邻切片回退。另检查每文件只读取一次、预处理乱序完成时仍按序消费、异常释放、关闭加速时恢复原始读取，以及 32×32 CPU 模型的 logits/概率在 metadata 开/关时逐位相等。两条完整 inference-worker 路径也在 CPU 上对照通过，其中 CUDA 传输、autocast 和锁页操作仅在测试中替换为 CPU 操作；生产 GPU 数值计算块另用 AST 确认与原 notebook 一致。v21 全部 17 项测试通过。本机没有 CUDA，尚未对比 Kaggle GPU 上的完整提交结果或测量推理耗时。

## 可选 SWA：Top3 EMA 权重平均

`--swa` 开启，默认关闭；省略开关或使用 `--no-swa` 可关闭。`--swa` 与 `--no-ema` 不可同时使用。

这里的 SWA 按当前实验需求定义：在全部已验证的 epoch 中，根据 **EMA 模型的验证 Macro AUC** 选取分数最高的三个不同 epoch，对这三份 **EMA 后的模型参数** 做等权平均，各占 `1/3`。不是平均三个 epoch 的 AUC 或预测概率，也不会改动学习率调度。同分时优先保留较早的 epoch，非有限 AUC 不进入候选。少于三个有效候选时不生成新融合权重。

只有 rank 0 保留三份独立的 CPU EMA 快照，不额外占用 GPU 模型显存。浮点参数和 buffer 做算术平均，整数 buffer（如 BN 计数器）使用最佳 EMA epoch 的值。v21 的 BN 运行统计保持固定，不重新拟合。

输出文件：

- `best.pt`：单个最佳 EMA epoch 的完整训练 checkpoint，行为沿用原版。
- `last.pt`：最后一个 epoch 的完整训练 checkpoint。
- `swa.pt`：Top3 EMA 融合的推理 checkpoint，`weights_type=swa_ema_top3`，保存完整模型、预处理 args、序列选择配置和三个来源 epoch/AUC/权重。
- `swa_validation.json`：训练完成后对融合模型重新验证所得的 Macro AUC 和逐标签 AUC。

Top3 候选变化且已经凑齐三份时，就更新 `swa.pt`，以便训练中断后仍有可用融合权重；这时融合文件及 JSON 的 `val_macro_auc` 为 `null`。训练完成后，将融合参数广播到各 rank，重新验证并写入实际分数。单模型 `best.pt` 的验证分数与融合分数分别保留。

开启 SWA 后，`last.pt` 和 `best.pt` 内嵌 `swa_top3` 候选，文件会更大。`--resume` 必须使用相同 SWA 开关并恢复候选，建议从 `last.pt` 续训。旧版未保存 Top3 候选的 checkpoint 不能直接以 `--resume --swa` 还原历史 Top3；可用 `--init-checkpoint` 开始新的实验。`swa.pt` 是推理文件，不含优化器/原始训练状态，不能用于 `--resume`。

在 Kaggle 使用融合模型时，上传 `swa.pt`，在 notebook 中明确设置 `CHECKPOINT_PATH` 为它的挂载路径；或者将上传的融合文件命名为 `best.pt` 使用原自动寻找逻辑。推理代码已兼容融合 checkpoint，无需上传三份候选或原始 RadImageNet 权重。

## 验证

从仓库根目录检查行尾：

```bash
python tools/check_line_endings.py
```

本目录的 `test_*.py` 测试文件已移除，以下保留此前的验证记录。

此前回归检查覆盖官方示例参数布局与特征一致性（含 336×336）、缺失/损坏/含分类头的权重拒绝、四种解冻范围及梯度、BN 统计固定、单窗口尾部 chunk、缺失 slot、分块一致性、优化器与 EMA 更新、checkpoint 保存/恢复、v14 checkpoint 拒绝，以及 notebook 与模型/数据代码的一致性。通过字节比较确认数据代码与 v14 完全相同，AST 比较确认窗口聚合和完整 forward 的后续运算保持一致。

此前使用 PyTorch 2.9.0 CPU / torchvision 0.24.0 完成全部 11 项回归测试，新增覆盖 Top3 EMA 排序、等权平均、EMA/原始权重区分、独立快照、续训恢复、融合导出、广播流程和 notebook 中的融合模型加载及前向。多卡广播使用 mock 检查流程，尚未实测 NCCL。另用下载的真实官方权重验证 318 个参数/buffer 的完整加载和 336×336 特征前向，并以默认 256 维聚合头、64×64 合成窗口完成 `layer3+layer4 + no-metadata` 的 BCE 反向、有限梯度检查、优化器更新、EMA 和完整 checkpoint 恢复；这不代表已测量 336×336 训练显存。

尚未完成全量真实 DICOM 训练、四卡运行、336×336 训练显存测量和线上评估；本版没有验证分数，不能据此判断优于 v14。
