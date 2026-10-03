# Baseline_v13：沿物理深度分层抽取 24 个训练窗口

从 Baseline_v10 新增实验 2，仅修改训练窗口的选择方式。模型保留 v10 的 DINOv2 Small CLS（384 → 256 维），不加入 v12 的 patch mean。EMA、损失、逐标签窗口池化、slice/slot Transformer、label query 和 4349/58 固定划分沿用 v10。尚未完成完整训练和线上评估。

## 训练采样

每个 Study 默认抽取 24 个窗口。五个 slot 都齐全且真实 anchor 足够时，分配如下：

| slot | 既有候选窗口数 | 训练窗口数 |
| --- | ---: | ---: |
| Sagittal fluid | 23 | 7 |
| Sagittal second | 17 | 5 |
| Coronal fluid | 15 | 5 |
| Coronal second | 10 | 3 |
| Axial | 15 | 4 |
| 合计 | 80 | 24 |

1. 在每个有效 slot 内，以缓存窗口的中心像素索引识别并去重真实 anchor。不同候选索引指向同一原始 anchor 时，训练只将其视为一个可选位置。
2. 按 `7/5/5/3/4` 的权重分配整数预算，每个有效 slot 至少一个窗口。缺失 slot 的预算按剩余 slot 权重分配；短序列达到容量上限后，余量继续分配给其他 slot。
3. 将每个 slot 的实际候选物理深度范围划为与其配额相同数量的等宽区间，每个非空区间随机取一个真实 anchor。空区间用距区间中心最近的未选 anchor 补取；物理深度全相同时使用等数量分区。
4. Study 的不同真实 anchor 总数不少于 24 时不重复抽取；少于 24 时先覆盖全部真实 anchor，再按 slot 比例缺额补抽到 24。故候选数量达到 80 也不代表短序列一定有 80 个不同真实 anchor。
5. 抽样继续由 seed、epoch 和 Study UID 决定，支持 persistent workers 与相同配置续训。最终仍按 slot/原始物理位置排序，三通道始终对应原始 `[i-1,i,i+1]`。

`--train-windows` 默认 24，仍可指定其他不少于 5 的预算；其他预算按同一 slot 权重分配。

## 验证、推理与缓存

验证和推理不调用分层采样，保留 v10 全部有效候选，最多 80 个。140 mm 裁剪、336 尺寸、2%-98% anchor 范围、归一化、序列选择、候选配额与缓存内容/key 均沿用 v10，可以直接复用 `Baseline_v10/input_cache`。训练去重发生在读取缓存之后，不重建候选缓存，也不对验证/推理候选去重。

## 四卡训练

从 `Baseline_v13` 目录运行以下命令，与 v10 最佳配置作独立对照：

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --data-root /root/RSNA/rsna-knee-abnormality-detection \
  --output-dir ./outputs/last6_3e-4_depth_stratified24 \
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

默认 DINOv2 使用上级目录 `dinov2-pytorch-small-v1`，标签使用本目录 `label.csv`。缓存路径不同请调整 `--cache-dir`；没有既有缓存时可指定 `./input_cache`。

建议从同一份 DINOv2 预训练权重重新训练，比较本地逐标签 AUC 与线上得分。本实验同时改变 slot 配额、深度分层和重复 anchor 处理，应将结果解释为整套训练采样策略的效果。

## Checkpoint 与 Kaggle

- 架构标记为 `baseline_v13_80_candidates_depth_stratified24_ema`，checkpoint 记录训练 slot 权重。
- `--resume` 只接受匹配的 v13 checkpoint，避免将 v10 随机采样训练错误续接为同一实验。
- 模型参数形状与 v10 一致，`--init-checkpoint` 可严格加载匹配 slot 的 v8/v9/v10/v13 权重；这种继续微调应单独记录，不与从 DINO 预训练开始的实验混为同一对照。v12 的 768 维投影不兼容。
- EMA 沿用 v10，`best.pt` 的 `model` 已是实际验证/推理权重；保留原始训练权重与 EMA 状态供续训。
- 使用本目录 `Kaggle_Inference.ipynb`，挂载 v13 `best.pt`、比赛数据和 DINOv2。`CHECKPOINT_PATH` 留空时寻找唯一 `best.pt`；多份权重时填写实际路径。
- Notebook 严格检查 v13 版本，模型使用 CLS 特征，读取全部候选并按 checkpoint 选择 metadata 开关及预处理参数，不新增 TTA 或 SWA。

目录包含六个文件，不复制 v10 的权重或输入缓存。

## 本地验证

已检查五 slot 的 7/5/5/3/4 配额与各物理深度区间覆盖、全部 31 种非空 slot 组合、150 组随机短序列/重复 anchor/退化深度场景、非法输入、相同 seed/epoch 复现及跨 epoch 变化。Dataset 集成检查确认训练固定输出 24 个窗口、验证读取全部 80 个候选。

标签文件与 v10 逐字节一致；缓存构造/key 和候选预处理函数未变；模型除版本标记外与 v10 定义相同；notebook 数据与模型定义和训练模块一致。已用真实 DINOv2 Small 权重通过 CPU 前向和 BCE 反向、默认训练预算及 v13 续训检查。

尚未运行服务器四卡完整训练或 Kaggle 线上评分，分层采样的收益待实验确认。
