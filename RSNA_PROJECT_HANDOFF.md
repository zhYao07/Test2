# RSNA Knee Abnormality Detection 项目交接文档

> 用途：将本会话的赛题理解、EDA 结论、数据处理、模型设计、训练方案、已修复问题和后续计划一次性交给下一个会话。  
> 当前本地工作目录：`F:\RSNA`  
> 服务器工作目录：`/root/RSNA/workspace_yzh/baseline_v1`  
> 服务器数据目录：`/root/RSNA/rsna-knee-abnormality-detection`

## 1. 项目目标

Kaggle 竞赛：RSNA Knee Abnormality Detection。

- 比赛主页：<https://www.kaggle.com/competitions/rsna-knee-abnormality-detection>
- 数据页：<https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data>
- 参考 EDA：<https://www.kaggle.com/code/gchauhan/eda-rsna-knee-what-the-data-tells-us>

任务是以一个膝关节 MRI Study（检查）为单位，对 12 种异常进行多标签预测：

1. `ACL`：前交叉韧带异常
2. `MCL`：内侧副韧带异常
3. `Medial Meniscus`：内侧半月板异常
4. `Lateral Meniscus`：外侧半月板异常
5. `Medial OA`：内侧间室骨关节炎
6. `Lateral OA`：外侧间室骨关节炎
7. `PF OA`：髌股关节骨关节炎
8. `Effusion`：关节积液
9. `Synovitis`：滑膜炎
10. `Baker's`：腘窝囊肿（Baker 囊肿）
11. `Contusion`：骨挫伤
12. `Fracture`：骨折

每个 Study 包含若干 MRI Series（序列）；每个 Series 又包含多张 DICOM 切片。因此，数据层级是：

```text
Study（一次膝关节检查）
└── Series（某一扫描平面与成像参数组合）
    └── DICOM slices（沿空间方向排列的二维切片）
```

已有详细中文资料：

- `RSNA_Knee_Competition_Overview_ZH.md`：赛题信息汇总
- `EDA_Report_gchauhan_ZH.md`：公开 EDA 的中文汇总，英文/医学术语已补充中文解释

## 2. 关键医学和成像术语

- `Axial`：轴位/横断位，从上到下看膝关节。
- `Coronal`：冠状位，从正面看膝关节。
- `Sagittal`：矢状位，从侧面看膝关节。
- `FS / Fat Suppression`：脂肪抑制，使脂肪信号变暗，有利于观察水肿、积液和炎症。
- `nonFS`：非脂肪抑制。
- `T1`：T1 加权成像，解剖结构和脂肪显示较好。
- `PD`：质子密度加权成像，膝关节韧带、半月板和软骨评估常用。
- `T2`：T2 加权成像，液体和水肿通常呈高信号。
- `TR / Repetition Time`：重复时间，相邻射频激发之间的时间。
- `TE / Echo Time`：回波时间，从激发到采集回波信号的时间。
- `Fluid Sensitive`：液体敏感序列，对积液、水肿、炎症等更敏感。
- `Series`：一次固定扫描方向、脉冲序列和参数下采集的一组连续切片，不是单张图片。

当前代码依据 EDA 中的启发式规则，结合 `TE`、`TR` 和脂肪抑制信息粗略生成 `T1 / PD / T2 / unknown` 元数据标记；它不是严格的医学序列分类器。

## 3. 数据文件与标签现状

服务器完整数据结构：

```text
/root/RSNA/rsna-knee-abnormality-detection/
├── train.csv
├── train_series.csv
├── train_series/
├── test.csv
├── test_series.csv
├── test_series/
└── sample_submission.csv
```

公开融合软标签文件：

```text
./llm_labels_v4_blend.csv
```

已核对的数量：

- `llm_labels_v4_blend.csv`：4407 个 Study，12 个标签全部存在，无重复 Study UID。
- 官方 `train.csv`：4407 个 Study，其中恰好 58 个 Study 有完整的 12 标签真值。
- 当前固定划分：4349 个弱标签 Study 用作训练；58 个官方真值 Study 用作验证。
- 不做五折交叉验证。
- 58 个验证 Study 完全排除在训练集之外，避免泄漏。

这里曾讨论过“五折多标签分层”，其含义是五折交叉验证，并使每一折中多个标签的阳性比例尽量相近。但当前明确决定先不做五折，而采用固定的 4349/58 划分。

## 4. 当前代码文件

### 4.1 `rsna_data.py`

负责元数据加载、DICOM 排序与解码、序列预处理以及 `Dataset`。

### 4.2 `rsna_model.py`

定义基于 DINOv2 的 2.5D 层次化多序列、多标签模型。

### 4.3 `train.py`

负责固定训练/验证划分、单卡或八卡 DDP、损失函数、优化器、学习率调度、验证和 checkpoint。

### 4.4 本地 DINOv2

服务器不能访问 GitHub，因此源码和权重均从本地加载：

```text
./dinov2/
./dinov2_pretrain_weights/dinov2_vits14_pretrain.pth
```

训练脚本中明确使用：

```python
source="local"
pretrained=False
```

不会在线下载源码或权重。

## 5. 数据预处理完整流程

### 5.1 Series 槽位

每个 Study 最终组织为 6 个固定槽位：

```python
SLOTS = [
    "Axial_FS",
    "Axial_nonFS",
    "Coronal_FS",
    "Coronal_nonFS",
    "Sagittal_FS",
    "Sagittal_nonFS",
]
```

当前最终实现是**每个槽位最多选择一个 Series**，不是每槽位两个 Series。

如果某槽位有多个候选 Series，优先选择：

1. `Fluid_Sensitive` 更高的序列；
2. 切片数更多的序列。

缺失槽位使用全零张量，并由 `slot_mask=0` 标记；存在的槽位为 `slot_mask=1`。

因此单个 Study 的图像形状为：

```text
[6, S, 1, 224, 224]
```

训练 DataLoader 加上 batch 维后为：

```text
[B, 6, S, 1, 224, 224]
```

目前 `S=32`。早期测试输出 `[6, 24, 1, 224, 224]` 也可以训练，只是切片覆盖更少。

### 5.2 DICOM 空间排序

不使用文件名或 `InstanceNumber` 简单排序，而是：

1. 读取 `ImageOrientationPatient`；
2. 由两个方向向量叉乘得到切片法向量；
3. 读取每张图的 `ImagePositionPatient`；
4. 将位置投影到法向量上；
5. 按真实患者空间坐标排序；
6. 将主要方向统一为患者坐标正方向。

这样比按文件名排序更可靠，也能避免不同 Series 的方向相反。

### 5.3 切片采样

在整个物理覆盖范围内均匀生成 `S` 个目标位置，然后选择距离每个目标位置最近的原始切片。

优点：

- 不依赖原始切片编号；
- 不同切片间距和切片数量都能映射到固定长度；
- 保留从序列一端到另一端的整体解剖覆盖。

若原始序列少于 `S` 张，会重复采样部分切片。这也是不建议盲目将 `S` 提高到 60 以上的原因。

### 5.4 DICOM 像素处理

每张图执行：

1. `pydicom.dcmread` 解码像素；
2. 应用 `RescaleSlope` 和 `RescaleIntercept`；
3. 若为 `MONOCHROME1`，翻转灰度方向；
4. 在整个 Series 的非零有限值上计算 0.5% 和 99.5% 分位数；
5. 截断极端值并归一化到 `[0,1]`；
6. 按长边补零成正方形，避免直接拉伸改变解剖比例；
7. 双线性缩放到 `224×224`。

### 5.5 异常 DICOM 容错

训练曾在约第 100 step 因异常 DICOM 崩溃：

```text
The number of bytes of pixel data is less than expected
(173056 vs 346112 bytes)
```

该文件为 `416×416`，实际 PixelData 正好是 8 位大小，但头信息声明得像 16 位。当前处理为：

- 对未压缩、单帧、单通道且实际字节数等于像素数的情况，按真实 8 位数据恢复；
- 如果某张切片仍确实无法解码，则用该 Series 中空间位置最近的正常切片替代；
- 如果整个 Series 都无法解码，才抛出明确错误。

这样不会因为一张损坏切片使八卡训练整体终止。

### 5.6 Series 元数据特征

每个槽位还输出 11 维特征：

```python
SERIES_FEATURES = [
    "fluid_sensitive",
    "fat_suppression",
    "t1",
    "pd",
    "t2",
    "contrast_unknown",
    "te",
    "tr",
    "pixel_spacing",
    "slice_spacing",
    "coverage",
]
```

连续值进行了简单缩放：

- `TE / 100`
- `TR / 5000`
- `slice_spacing / 10`
- `coverage / 200`

### 5.7 标签张量

单个 Study 输出：

```text
targets       [12]
label_mask    [12]
label_weight  [12]
```

- `targets`：标签值，支持 0～1 的软标签。
- `label_mask`：标签是否存在。
- `label_weight = 2 * abs(label - 0.5)`：软标签置信度。
  - 标签为 0 或 1 时权重为 1；
  - 标签越接近 0.5，权重越低；
  - 正好为 0.5 时不对损失产生贡献。

## 6. 为什么选择 32 张切片

讨论结论：

- 24：可训练，计算最省，但可能跳过更多局部信息。
- **32：当前首选，覆盖与计算量平衡较好。**
- 40：建议后续作为主要对照实验。
- 48：合理的较高上限。
- 56～60：只有大多数原序列都达到该长度时才值得。
- 超过 60：通常不推荐，因为原序列平均约 30～50 张，会造成大量重复采样。

24 增至 32 时，DINOv2 图像编码计算量约增加 33%。当前 `train.py` 默认值已设为 32。

注意：`num_slices` 决定可学习位置编码 `slice_position` 的长度，因此 24 切片 checkpoint 不能直接严格加载到 32 切片模型，除非另外实现位置编码插值或忽略该参数。

## 7. 模型结构

当前类：`RSNADINOv2`。

整体数据流：

```text
DICOM Study [B,6,S,1,224,224]
    ↓
2.5D 三通道构造（前一张、当前张、后一张）
    ↓
DINOv2 对每张切片编码
    ↓
Slice Transformer + attention pooling
    ↓
每个槽位得到一个特征向量
    ↓
加入 Series 元数据特征和槽位位置嵌入
    ↓
Slot Transformer 融合六类序列
    ↓
12 个 label queries 对槽位做注意力读取
    ↓
12 个分类 logit
```

### 7.1 2.5D 输入

MRI 是单通道，但 DINOv2 预训练模型需要三通道。当前没有简单复制灰度图三次，而是构造：

```text
R = 前一张切片
G = 当前切片
B = 后一张切片
```

边界位置重复第一张或最后一张。这样三通道同时携带少量空间上下文。

随后使用 ImageNet 均值和标准差归一化，以匹配 DINOv2 的输入习惯。

### 7.2 DINOv2 Backbone

当前 backbone：

```text
dinov2_vits14
```

DINOv2 对所有有效槽位的切片编码；缺失槽位不会送入 backbone。为控制瞬时显存，切片按 `encoder_chunk_size=24` 分块编码。

### 7.3 切片级聚合

- Backbone 特征经过 `LayerNorm + Linear` 投影到 `hidden_dim=256`；
- 加入可学习切片位置编码；
- 使用两层 Slice Transformer 建模同一 Series 内不同切片关系；
- 用可学习 attention pooling 聚合为一个槽位特征。

### 7.4 槽位级融合

- 11 维 Series 元数据经过 MLP 投影到 256 维；
- 与图像槽位特征相加；
- 加入六类槽位的可学习嵌入；
- 一层 Slot Transformer 融合轴位、冠状位、矢状位以及 FS/nonFS 信息；
- `slot_mask` 屏蔽缺失槽位。

### 7.5 标签级输出

- 12 个可学习 `label_queries`，每个标签一个查询向量；
- Multi-head Attention 让每个标签从六个槽位中读取不同信息；
- 每个标签使用独立权重和偏置输出一个 logit。

这种设计符合医学直觉：不同疾病依赖的扫描平面和序列不同，例如韧带、半月板、积液和骨折不一定依赖相同槽位。

### 7.6 Backbone 冻结模式

训练参数支持：

| 模式 | 训练范围 |
|---|---|
| `frozen` | DINOv2 全部冻结，只训练后续任务网络 |
| `last2` | 解冻 DINOv2 最后 2 个 Block 和最终 Norm |
| `last4` | 解冻最后 4 个 Block 和最终 Norm |
| `full` | 整个 DINOv2 参与训练 |

当前第一阶段使用 `frozen`。被冻结时 backbone 在 `torch.no_grad()` 中运行，不保存反向图，显存明显降低。

原计划是两阶段训练：

1. 先冻结 backbone 训练任务头；
2. 从第一阶段 `best.pt` 加载，用 `last2` 和较低 backbone 学习率微调。

## 8. 训练配置

### 8.1 固定划分

`prepare_studies` 将官方 `train.csv` 和 `llm_labels_v4_blend.csv` 按 `StudyInstanceUID` 合并。

- 具有完整官方真值的 58 个 Study：验证集，验证标签强制使用官方真值；
- 其余 4349 个 Study：训练集，使用公开融合软标签。

代码断言验证集必须恰好为 58，否则直接报错，防止因标签文件错误而静默改变划分。

输出目录会写入：

```text
split.csv
```

用于记录每个 Study 属于训练集还是验证集。

### 8.2 损失函数

基础损失为：

```text
BCEWithLogitsLoss（逐标签，不先做 sigmoid）
```

最终权重同时考虑：

- `label_mask`：标签是否存在；
- `label_weight`：软标签置信度；
- `pos_weight`：训练集每个标签的类别不平衡。

`pos_weight` 根据带置信度的软标签估计，并截断在 `[1,10]`。可用 `--no-pos-weight` 关闭。

### 8.3 优化与学习率

- 优化器：AdamW
- 任务头学习率：`3e-4`
- Backbone 学习率：`1e-5`（只有解冻时才创建该参数组）
- Weight decay：`0.05`
- Warmup：总更新步数的 10%
- Warmup 后使用 cosine decay
- 梯度裁剪：`1.0`
- 默认训练轮数：8
- 默认随机种子：42；每个 DDP rank 会加上 rank

### 8.4 混合精度

默认：

```text
bf16
```

服务器 GPU 资源为 8 卡，每卡约 96 GB，适合 bf16。代码也支持 `fp16` 和 `none`。只有 fp16 会启用 `GradScaler`；bf16 不需要梯度缩放。

### 8.5 DDP 与有效 batch size

当前建议：

```text
单卡 batch size = 1
GPU 数 = 8
梯度累积 = 2
有效 batch size = 1 × 8 × 2 = 16
```

训练和验证均使用 `drop_last=False`，保证所有 Study 被覆盖。由于 4349 不能被 8 整除，`DistributedSampler` 会补齐极少数重复样本，这是 PyTorch DDP 的正常行为。

元数据只由 rank 0 从磁盘构造一次，再广播给其他 rank，避免 8 个进程同时重复扫描所有 DICOM 文件数量。

### 8.6 验证指标与 checkpoint

每轮训练结束后在 58 个官方真值 Study 上计算：

- 每个标签的 ROC AUC；
- 所有可计算标签的 Macro AUC。

若某标签在这 58 个样本中只有一个类别，则该标签 AUC 为 `NaN`，Macro AUC 忽略该项。

输出：

```text
last.pt   # 每轮覆盖，保存最新状态
best.pt   # 验证 Macro AUC 最优状态
split.csv # 固定划分记录
```

checkpoint 包含模型、优化器、调度器、scaler、epoch、best AUC 和命令行参数。

## 9. 当前推荐启动命令

在服务器工作目录执行：

```bash
cd /root/RSNA/workspace_yzh/baseline_v1

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 train.py \
  --data-root /root/RSNA/rsna-knee-abnormality-detection \
  --labels-csv ./llm_labels_v4_blend.csv \
  --output-dir ./outputs/dinov2_vits14_s32 \
  --num-slices 32 \
  --batch-size 1 \
  --accum-steps 2 \
  --num-workers 4 \
  --backbone dinov2_vits14 \
  --backbone-mode frozen \
  --amp bf16
```

本地 DINOv2 路径和权重路径已是默认值，因此不需要重复填写：

```text
--local-dinov2-repo ./dinov2
--backbone-weights ./dinov2_pretrain_weights/dinov2_vits14_pretrain.pth
```

启动时预期看到：

```text
labels=.../llm_labels_v4_blend.csv
DINOv2 code=.../dinov2
DINOv2 weights=.../dinov2_pretrain_weights/dinov2_vits14_pretrain.pth
split: weak-label train=4349 | gold-label valid=58 | total=4407
trainable parameters: 2818317
```

## 10. 第二阶段微调建议

第一阶段 `frozen` 完成并得到稳定的 `best.pt` 后，可进行 `last2` 微调：

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 train.py \
  --data-root /root/RSNA/rsna-knee-abnormality-detection \
  --labels-csv ./llm_labels_v4_blend.csv \
  --output-dir ./outputs/dinov2_vits14_s32_last2 \
  --num-slices 32 \
  --batch-size 1 \
  --accum-steps 2 \
  --num-workers 4 \
  --backbone dinov2_vits14 \
  --backbone-mode last2 \
  --init-checkpoint ./outputs/dinov2_vits14_s32/best.pt \
  --head-lr 1e-4 \
  --backbone-lr 1e-5 \
  --epochs 5 \
  --amp bf16
```

注意：`--init-checkpoint` 只加载模型参数，适合进入新的微调阶段；`--resume` 会同时恢复优化器、调度器和 epoch，只适合同一配置训练中断后续跑。

## 11. 已遇到并修复的问题

### 11.1 `rsna_data.py` 循环导入

服务器曾报：

```text
ImportError: cannot import name 'LABELS' from partially initialized module 'rsna_data'
```

原因是服务器上的 `rsna_data.py` 被错误内容覆盖，其中出现：

```python
from rsna_data import LABELS, SERIES_FEATURES, SLOTS
```

这行属于模型文件的逻辑，不应存在于数据文件。已通过重新上传正确文件解决。

上传后可先做快速导入检查：

```bash
python -c "from rsna_data import DEFAULT_ROOT, LABELS, KneeDataset, load_metadata; print('rsna_data OK', len(LABELS))"
python -c "from rsna_model import RSNADINOv2; print('rsna_model OK')"
```

### 11.2 异常 DICOM PixelData

详见第 5.5 节。已加入 8 位恢复和邻近切片替代。

### 11.3 warning 清理

已从根源处理：

- `torch.cuda.amp.GradScaler` 旧接口改为 `torch.amp.GradScaler`；
- Transformer 显式 `enable_nested_tensor=False`；
- NCCL 初始化显式传入当前 CUDA device；
- TF32 改用 PyTorch 新接口；
- DINOv2 的 `xFormers is available` 第三方提示按精确消息过滤；
- `label_queries` 从 `[1,12,256]` 参数改为 `[12,256]`，前向时再扩展 batch，解决 DDP gradient stride warning；
- `OMP_NUM_THREADS=1` 应写在启动命令前，torchrun 才不会输出默认设置提示；
- 异常退出和正常退出都会清理 distributed process group。

没有使用全局 `warnings.filterwarnings("ignore")`，以免掩盖真正问题。

### 11.4 NumPy 环境

目标版本是：

```text
numpy==1.26.4
```

曾执行不带版本约束的 `pip install matplotlib`，导致 pip 将 NumPy 升级为 2.5.3，并与 `pylibjpeg`、`numba`、`scipy`、`scikit-learn` 等冲突。环境中必须保持 NumPy 1.26.4。

如再次发生，可恢复：

```bash
pip install --force-reinstall "numpy==1.26.4"
```

安装其他包时应同时固定 NumPy，或使用 `--no-deps`（前提是依赖已满足），避免自动升级到 NumPy 2.x。

### 11.5 本地 Codex 补丁工具 ACL 问题

这不是训练项目问题，但本会话中修复过。根因是：

```text
C:\Users\86188\.codex\.sandbox\deny_read_acl_state.json
```

被写成 22 个 NUL 字节，导致 elevated sandbox 无法解析 JSON。损坏文件已备份，Codex 自动重建为：

```json
{
  "principals": {}
}
```

随后已用真实补丁工具创建/删除探针文件验证正常。

## 12. 当前状态与尚未确认的事项

### 已完成

- 赛题中文汇总；
- 公开 EDA 中文报告和术语注释；
- DICOM 空间排序、采样和预处理；
- 六槽位 Study Dataset；
- DINOv2 2.5D 层次模型；
- 4349 弱标签训练 / 58 真值验证固定划分；
- 8 卡 DDP 训练脚本；
- 本地 DINOv2 源码和权重加载；
- 异常 DICOM 容错；
- 已知 warning 的代码级修复；
- checkpoint 和验证 Macro AUC。

### 尚未完成或尚未验证

1. **最新版本尚未确认完整跑完一个 epoch。** 之前训练曾到 epoch 1 的 step 100 左右，因异常 DICOM 中断；修复后又处理了 warning，但没有在本会话中拿到完整 epoch 的最终日志。
2. **推理脚本尚未编写。** 当前只有数据、模型和训练流程。
3. **尚未正式比较 24/32/40/48 切片。** 当前只是基于数据特征选择 32 作为第一版。
4. **尚未执行第二阶段 `last2` 微调。** 这只是下一步建议。
5. **尚未做五折或模型集成。** 当前明确先不做。
6. **58 个验证样本很少。** 单次 Macro AUC 方差可能较大，适合快速迭代，但最终可靠评估仍可考虑交叉验证。
7. **尚未对异常切片替代数量做统计。** 当前处理保证不中断，但以后最好记录每个 Series 的替代次数，确认数据质量影响。

## 13. 下一会话建议先做什么

建议依次执行：

1. 将本地最新的 `rsna_data.py`、`rsna_model.py`、`train.py` 同步到服务器；
2. 运行两条 import smoke test；
3. 使用第 9 节命令启动 8 卡冻结训练；
4. 观察是否无 warning、无 DICOM 崩溃，并至少跑完一个 epoch；
5. 保存并记录每个标签 AUC、Macro AUC、训练耗时和 GPU 利用率；
6. 若第一阶段稳定，运行 `last2` 微调；
7. 补写推理脚本，读取 `best.pt` 并生成 submission；
8. 再比较 `num_slices=40`，不要一开始直接提高到 60；
9. 后续资源充足时再测试更大 DINOv2 backbone、更多解冻层和交叉验证。

## 14. 给下一会话的简短提示词

可以在新会话中直接发送：

```text
请先完整阅读 F:\RSNA\RSNA_PROJECT_HANDOFF.md，并检查 F:\RSNA 下的
rsna_data.py、rsna_model.py、train.py。不要重新设计已有流程。
当前目标是先让 8 卡 frozen DINOv2 训练完整跑完一个 epoch，检查最新日志；
若稳定，再继续第二阶段 last2 微调和推理脚本。
```

