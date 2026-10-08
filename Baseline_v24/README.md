# Baseline_v24：Qwen3.5-2B 检查级分类

沿用 v14 的序列质量排序与相邻切片三通道构造，替换 DINOv2 和原有聚合网络：视觉编码器（含 merger/projector）全量微调，语言模型基础参数冻结、各 decoder 层的 Linear 加 LoRA，最后一个有效 prompt 位置的 hidden state 接 LayerNorm、Dropout、Linear，输出 12 个 logits。训练使用带缺失掩码和软标签置信度权重的 BCE，推理使用 sigmoid，不生成报告或概率文本。

这是一套可训练、可恢复、可离线推理的实现，不是讨论区作者代码的复现，也没有验证其 0.950 分数或 3.5 小时耗时。

## 数据处理与验证口径

- `rsna_data.py` 与 v14 **逐字节一致**：五个 slot、候选预算 `[23,17,15,10,15]`、不重复 Series UID、header 质量排序、coverage 阈值、140 mm physical crop、前景 0.5/99.5 百分位归一化、相邻 `[-1,0,+1]` 切片堆成三通道均保持原样。
- 默认 `image_size=384`，不改变复制文件中的 v14 默认常量。Qwen 的 patch_size=16、merge_size=2，输入边长必须是 32 的倍数，因此不能直接沿用 v14 的 336。窗口是 float `[0,1]`；Qwen processor 关闭 resize 和 rescale，仅执行预训练归一化与 patch packing，避免再次除以 255。
- 训练保留 v14 随 epoch/UID 可复现的随机 24 窗口，每个有效 slot 至少一个，按 slot 和物理位置排列。384 输入的缓存 key 与旧尺寸隔离；可共用缓存目录，但旧的 336 缓存不会被当成 384 使用。
- v24 默认验证使用 **全部有效候选、最多 80 个窗口**（`--eval-windows 0`），沿用 v14 的验证覆盖范围；推理读取 checkpoint 保存的设置。可显式指定正数（至少 5）限制窗口数：按候选数量比例分配给有效 slot，每个有效 slot 至少一个，slot 内取等间隔分箱中点。限制窗口数会改变验证口径。
- prompt 明确区分五个 slot、缺失 slot 和窗口归一化位置。默认与用户的 v14 命令一致，关闭额外 metadata（`--no-metadata`）；仍保留 slot 和位置。`--metadata` 可加入 v14 返回的 fluid-sensitive/fat-suppression 两个标志，保留 v14 对未知标志的处理；不把启发式 TE/TR 分类当成准确的 PD/T2 名称。
- `label.csv` 从 v14 原样复制。官方完整 12 标签的 58 个 study 全部保留为验证集，即使不在弱标签 CSV 中；其余有非零置信度监督的 study 训练。coverage 阈值和可选正类权重仅由训练集拟合。报告不进入模型输入。
- 默认开启正类权重，沿用 v14 按训练软标签置信度估计的负/正比例并限制在 `[1,10]`；`--no-pos-weight` 可关闭，原有 `--pos-weight` 参数仍可使用。默认开启 EMA，decay 上限 0.9995，沿用 v14 的更新次数 warmup，仅在成功的 optimizer step 后更新。EMA 只维护视觉/merger、LoRA 和分类头的影子参数；冻结 LM 共用。验证和推理使用 EMA，训练使用原始参数；`--no-ema` 可关闭。`validation_history.json`、逐标签 AUC 和验证预测 CSV 用于比较实验；单一类别的标签 AUC 记为 null，宏平均跳过这些标签。

## 环境与基础模型

推荐独立 Linux CUDA 环境：

```bash
python -m pip install -r requirements.txt
```

已在 CPU 上验证 PyTorch 2.9.0、Transformers 5.18.0、PEFT 0.21.2、Accelerate 1.15.0。使用支持 Qwen3.5 的指定版本，不要直接沿用旧 DINOv2 环境的 Transformers。torch 与 torchvision 应安装匹配的 CUDA 版本。

GPU 训练建议安装 `causal-conv1d` 和 `flash-linear-attention` 的兼容版本，以启用 DeltaNet 优化 kernel。基础实现能在缺少这些包时使用 PyTorch fallback，速度与显存可能明显变差；这些可选 CUDA 扩展不在通用 requirements 中强制安装。[Transformers Qwen3.5 文档](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_5.md)

若服务器为 PPU-ZW810E，应使用 PPU 平台适配的 PyTorch、编译器和扩展，不能仅凭 `torch.version.cuda` 与包版本号选择 NVIDIA 的预编译 CUDA wheel。保留已可训练的 PPU 环境；可选 kernel 的适配需由平台镜像/SDK确认。[阿里云 PPU 编程指南](https://help.aliyun.com/zh/document_detail/2871803.html)

本地预训练模型已位于仓库根目录 `Qwen3.5-2B/`，与 `Baseline_v24/` 同级。训练和命令行推理默认自动使用这个目录，路径相对于源码解析，不依赖启动目录。当前机器对应 `F:/Kaggle/RSNA/Qwen3.5-2B`；将整个项目上传到训练服务器后，保持两个目录同级即可。可用 `--qwen-model-dir` 覆盖默认路径。

已检查配置、processor 和权重索引：唯一分片为 `model.safetensors-00001-of-00001.safetensors`，与索引一致，无需改名。下载目录包含 tokenizer、chat template、图像及视频 processor 配置。预训练目录已加入仓库 `.gitignore`，与 DINOv2 权重一样保留本地原文件。

训练和推理均 `local_files_only=True`，不自动联网下载，不使用 4-bit 量化。推理必须挂载**与训练相同的基础模型 snapshot**；checkpoint 只包含更新过的参数，配置校验不能证明两个不同 snapshot 的基础权重相同。

LoRA 仅插入 language model decoder 层的 Linear，包括普通 attention、DeltaNet 的 `in_proj_*`/`out_proj` 和 MLP，默认 rank 16、alpha 32、dropout 0.05。视觉参数、LoRA、分类头均保持 FP32 优化，bf16 模式下冻结 LM 使用 bf16 加载。fp16 训练使用 autocast/GradScaler、基础参数保持 FP32。梯度 checkpointing 默认开启且采用 non-reentrant；不在冻结 LM forward 外使用 `no_grad()`。

## 四卡训练

从本目录执行：

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --data-root /root/RSNA/rsna-knee-abnormality-detection \
  --labels-csv ./label.csv \
  --qwen-model-dir ../Qwen3.5-2B \
  --output-dir ./outputs/qwen35_2b_384_fa2_ema \
  --cache-dir ../Baseline_v14/input_cache \
  --image-size 384 --crop-mm 140 \
  --train-windows 24 --eval-windows 0 \
  --batch-size 2 --eval-batch-size 2 --accum-steps 2 \
  --epochs 20 --vision-lr 1e-5 --lora-lr 1e-4 --head-lr 3e-4 \
  --lora-rank 16 --lora-alpha 32 \
  --amp bf16 --attn-implementation flash_attention_2 \
  --eval-attn-implementation sdpa --eval-sdpa-backend math \
  --no-metadata --ema-decay 0.9995 --pos-weight \
  --span-lo 0.02 --span-hi 0.98 \
  --coverage-quantile 0.2 --series-quality-workers 8 \
  --num-workers 4 --cpu-threads 4 --seed 42
```

默认训练参数为 20 epochs、训练/验证每卡 batch size 2、累积 2 步、bf16、关闭额外 metadata、开启 EMA 和正类权重。`last6` 是 v14 的 DINOv2 参数，v24 保留视觉全量微调和语言 LoRA。上面的 FlashAttention 命令要求训练环境已安装兼容的 `flash-attn`；CLI 的注意力默认值仍为 `sdpa`，可显式选择。FlashAttention 不替代 DeltaNet 的可选加速库。

训练与验证独立选择注意力后端。验证默认使用 `--eval-attn-implementation sdpa --eval-sdpa-backend math`，明确禁用 fused SDPA kernel，避开长序列验证中外部 FlashAttention 的非法内存访问路径；全部候选窗口、batch size、EMA 和标签权重不变。math 模式可能更慢且使用更多显存，需在实际硬件验证。确认平台支持后可设置 `--eval-sdpa-backend auto` 允许 PyTorch 选择 fused kernel，或单独设置验证后端。

默认先用当前验证配置、EMA 权重运行完整的 58 study 验证检查，再开始训练；初始 AUC 只打印，不参与 best checkpoint 选择。`--validation-only` 只执行该检查，`--skip-validation-preflight` 可跳过。每轮验证结束后恢复训练注意力后端与原始参数；启动日志和每个验证 batch 打印后端、窗口数及 token 长度。已发生 GPU 非法内存访问的进程需要重启，不在受损的 device context 中自动重试。

正式训练前可先在同一四卡命令上加 `--smoke-test`，并使用独立的 `--output-dir ./outputs/v24_smoke`。此模式执行一次成功的 optimizer/scheduler/EMA 更新（`--accum-steps 2` 时包含 2 个训练 micro-batch），每个 rank 使用当前 eval batch size 和全部配置窗口验证一个 batch，再保存/恢复原始训练参数、EMA、optimizer、scheduler、scaler 和各 rank RNG。rank 0 随后通过正式推理加载器重新加载 checkpoint，核对预测与验证结果一致，并检查提交 CSV 的标签列、UID、有限概率和读写一致性。前四阶段均通过才打印 `SMOKE TEST PASSED`，写入 `smoke_test/result.json` 的 passed 状态并退出，不启动正式训练，也不先跑 58 study 的完整验证检查。

smoke 输出均在独立的 `smoke_test/` 子目录，包括 `best.pt`、`last.pt`、验证预测、仅含本次 rank 0 验证 batch 的 `submission.csv` 和结果 JSON。这些预测不代表完整验证 AUC或比赛提交；smoke checkpoint 含不足一个 epoch 的训练更新，禁止用于正式训练的 `--resume`。测试通过后移除 `--smoke-test`，改用正式输出目录，从原始预训练模型开始正常训练。一批样本的通过也不能保证所有后续输入都正常。

单卡用 `python train.py` 及同样参数。以上四卡命令的有效 batch 为 16 个 study；最后不足一个累积组时按实际组长缩放损失。`--amp fp16` 用于不支持 bf16 的 GPU；若显存不足，先降低 `--train-windows` 与 `--eval-windows`（至少 5），保持验证/推理口径一致。

384×384 每窗口通常有 144 个合并后视觉 tokens；24 窗口约 3456 个视觉 tokens，另加文本。代码核对 processor 网格、逐 study 图像 token 数及 padding，保留 processor 的 M-RoPE 输入。`--max-tokens 16384` 是报错上限，不会截断图像或悄悄删窗口。80 窗口的完整候选模式需实测显存和耗时。

## 保存和恢复

- `best.pt`：EMA 的视觉/merger、LoRA、分类头以及数据/模型配置（关闭 EMA 时保存原始参数）；不保存冻结 LM、优化器和 RNG，适合提交推理。
- `last.pt`：相同推理参数，加原始训练参数、EMA 影子参数与更新计数、优化器、scheduler、GradScaler、各 rank 的 RNG 与历史，用于恢复。恢复时训练参数与 EMA 分别加载，避免用 EMA 参数配原始优化器状态。checkpoint 的 `weights_type` 标记验证/推理使用的参数类型。
- `processor/`：训练所用 processor 备份；标准推理仍从同一基础 snapshot 加载 processor，并校验 tokenizer、template、归一化、patch 参数的签名。
- `hyperparameters.json`、`series_quality.csv`、`series_selection.csv`、`series_selection_summary.json`、`validation_history.json` 和 `valid_predictions_{last,best}.csv`：实验与选择诊断。

恢复时保留原命令的模型、训练与分布式配置，并追加：

```bash
--resume ./outputs/qwen35_2b_384_fa2_ema/last.pt
```

检查 LoRA target、标签顺序、slot、基础模型配置、processor 签名、coverage 阈值以及 batch/累积/总 epoch/学习率/EMA/正类权重等关键设置。恢复支持 epoch 边界，不恢复半个 epoch。`best.pt` 不能用于 `--resume`；`--init-checkpoint best.pt` 可作为新实验的初始化并重置优化器/历史。此前未带 EMA 的 v24 checkpoint 仍可推理或作为新实验初始化，不能直接恢复到这套新默认配置。v14 的 DINOv2 checkpoint 不能加载到 v24。

验证后端和 SDPA kernel 选择可在恢复时调整；训练后端仍要求一致。当前在验证成功后保存 epoch checkpoint，若第一轮验证就崩溃，则该轮还未生成可恢复的 `last.pt`。

## 离线推理与 Kaggle

```bash
python inference.py \
  --data-root /kaggle/input/rsna-knee-abnormality-detection \
  --checkpoint /kaggle/input/rsna-v24-weights/best.pt \
  --qwen-model-dir /kaggle/input/你的模型数据集/Qwen3.5-2B \
  --output /kaggle/working/submission.csv --amp auto \
  --attn-implementation sdpa --sdpa-backend math
```

多 GPU 可用 `torchrun --standalone --nproc_per_node=2 inference.py ...`，各 rank 分配不同 study、扫描自身 header、独立预测后合并。测试阶段使用保存的训练 coverage 阈值，不重新拟合；不写大型输入缓存。推理核对 UID 完整性、顺序、12 个概率列和有限值。

`Kaggle_Inference.ipynb` 内嵌与命令行相同的四个推理模块，默认在 `/kaggle/input` 下自动识别唯一的 Qwen3.5-2B snapshot，不绑定上传数据集名称；发现多个匹配目录时需要在配置 cell 指定 `QWEN_MODEL_DIR`。将本地 `Qwen3.5-2B/` 整体上传并挂载，修改比赛数据与 v24 checkpoint 路径即可。默认使用全部可见 GPU，自动选择 bf16/fp16。

命令行推理和 notebook 默认使用 SDPA math；notebook 的 `SDPA_BACKEND` 对应 CLI 的 `--sdpa-backend`，可显式改成 `auto` 以允许 fused kernel。推理后端可以与训练不同，仍保留相同图像、窗口与 prompt。

互联网关闭时必须提前准备匹配 Kaggle Linux/Python/CUDA 的依赖 wheels 并挂载。可在同版本联网环境执行 `python -m pip download --only-binary=:all: -d wheels -r requirements.txt`，再将 notebook 的 `WHEEL_DIR` 指向上传目录。torch/torchvision 的 CUDA 构建需与目标环境匹配；已有兼容版本时可单独准备其余包。需要 DeltaNet 扩展时也须提前准备兼容 wheel。notebook 不从网络安装包，也不隐式降级依赖。

## 目录与验证

只保留运行所需文件：`train.py`、`inference.py`、`rsna_model.py`、`rsna_data.py`、`vlm_data.py`、`Kaggle_Inference.ipynb`、`label.csv`、`requirements.txt`、README 和 `.gitignore`。开发期的测试脚本和 notebook 生成脚本已清理；notebook 已同步最终源码。

开发期 8 项离线集成测试已通过，使用真实 Qwen3.5 架构的小型随机模型与实际 processor，覆盖像素数值保真、图像 token 对齐、不同长度 batch/padding、vision/LoRA 梯度、冻结参数、gradient checkpointing、官方 ConditionalGeneration snapshot 的基础模型加载、更新参数恢复、gold 标签保留、窗口采样、训练循环及合成 DICOM 到提交文件的完整推理。下载的实际模型 processor 也已在 384 输入上验证。

EMA 更新后另通过 6 项 CPU 检查，覆盖默认参数与关闭开关、v14 的 EMA warmup、验证替换及异常后的参数恢复、末尾累积组、跳过 optimizer step 时不更新 EMA、EMA/原始参数分别保存与恢复后下一步更新一致、正类权重统计，以及 notebook 内嵌源码与 v14 数据文件的一致性。

验证后端修复已在真实小型 Qwen3.5 CPU 模型上检查 SDPA math 前向、padding、视觉/文本后端切换与异常后的恢复，并确认 math context 禁用 fused SDPA kernel；EMA 的 6 项检查仍通过。尚未在用户的 PPU 环境复现或验证 kernel 修复效果。

一步全流程 smoke 已使用真实小型 Qwen3.5、PEFT LoRA 和实际下载模型的 processor 离线跑通，覆盖上述训练/验证/保存/恢复/正式推理加载/提交 CSV 路径，并检查只执行一次 optimizer/EMA 更新、一个验证 batch、拒绝正式恢复部分 epoch checkpoint。该 CPU 测试使用小模型和 32 输入；实际 2B/384/四卡 PPU 路径需在服务器执行同一 smoke 模式验证。

尚未在 GPU 上完成完整 Qwen3.5-2B 训练、DDP/NCCL 或 Kaggle 隐藏测试运行，也没有实测 384 输入的显存、训练时间与 AUC。当前仓库的数据只包含少量 DICOM，完整训练需使用训练服务器的全量数据。
