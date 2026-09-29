# BaselineV2: Kaggle DINOv2 权重

`train.py` 默认读取项目根目录的 `dinov2-pytorch-small-v1/`，其中需要有 `config.json` 和 `pytorch_model.bin`。这是 Hugging Face `Dinov2Model` 格式，加载时只读取本地文件，不访问网络。训练标签默认读取本目录的 `label.csv`。

训练环境需要安装 PyTorch 和 `transformers`，以及本项目原有的数据处理依赖。示例：

```bash
python baseline_v2/train.py --data-root /path/to/rsna-knee-abnormality-detection
```

如果模型目录不在默认位置，添加 `--dinov2-model-dir /path/to/dinov2-pytorch-small-v1`。模型仍使用 BaselineV2 的 336 像素、140 mm 裁剪和 32 张切片默认设置；在 Kaggle 推理时要使用与训练相同的预处理和 `--num-slices`、`--image-size`、`--crop-mm` 参数。

微调最后六个 DINOv2 编码层时，添加 `--backbone-mode last6`；也可选择 `frozen`、`last2`、`last4` 或 `full`。

在 Linux 单机四卡上，从项目根目录运行最后四层微调：

```bash
torchrun --standalone --nproc_per_node=4 baseline_v2/train.py --data-root data --backbone-mode last4 --amp fp16
```

不需要 `dinov2/` 源码目录或 `dinov2_pretrain_weights/*.pth`。模型实现由 `transformers` 提供。新的模型参数名与旧版 `torch.hub` 实现不同，旧版训练检查点不能直接用 `--init-checkpoint` 或 `--resume` 加载。Kaggle notebook 已经使用同一种 Hugging Face DINOv2 模型加载方式。
