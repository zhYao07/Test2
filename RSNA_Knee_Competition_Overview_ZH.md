# RSNA Knee Abnormality Detection 赛题速览

> 整理日期：2026-09-18（Asia/Shanghai）  
> 官方链接：[比赛主页](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) · [数据页](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data) · [评估说明](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation) · [规则](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)

## 1. 一句话理解

给定一次膝关节 MRI 检查中的多个 DICOM 序列和切片，输出该检查存在 **12 种膝关节异常**的概率。这是一个**检查级、多标签、弱监督的 3D/2.5D 医学影像分类任务**。

训练集还提供原始放射学报告，但正式测试时不提供报告。因此报告主要用于训练阶段挖掘标签、蒸馏知识或学习图文表征，最终推理必须依赖影像及随附的序列元数据。

## 2. 最重要的赛题事实

| 项目 | 内容 |
|---|---|
| 主办方 | Radiological Society of North America（RSNA） |
| 赛制 | Kaggle Research Code Competition |
| 输入 | 一个检查下的多序列膝关节 MRI，DICOM 格式；序列可来自轴位、冠状位和矢状位 |
| 输出 | 每个 `StudyInstanceUID` 对 12 个异常分别给出一个概率 |
| 主指标 | 12 个标签各自 ROC AUC 的宏平均（Macro-average AUC） |
| 训练规模 | 4,407 个 study、24,371 个 series；完整数据约 569.76 GB、819,640 个文件 |
| 金标准标签 | 只有 58 / 4,407 个训练 study 具有完整的 12 项二元标签 |
| 测试规模 | 约 1,300 个 study；当前公开的 3 个 study 只是示例，正式评分时会被替换 |
| 提交方式 | 必须通过 Kaggle Notebook；生成名为 `submission.csv` 的文件 |
| Notebook 限制 | CPU 或 GPU 运行时间不超过 9 小时；推理时关闭互联网 |
| 外部数据 | 官方页面注明允许自由、公开可用的外部数据和预训练模型，但仍须遵守比赛规则及许可要求 |
| 总奖金 | 77,000 美元：主榜 59,000 美元，效率赛道 18,000 美元 |

官方数据说明见 [Kaggle Data 页面](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)，评分与 Notebook 约束见 [Evaluation 页面](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation)。

## 3. 任务单位与数据层级

数据是三层结构：

```text
Study（一次膝关节检查；最终预测单位）
└── Series（一次特定序列/方位的采集）
    └── Slice（单张 DICOM 图像）
```

- `StudyInstanceUID`：一次完整检查的唯一 ID，也是提交表的一行。
- `SeriesInstanceUID`：检查内部某个 MRI 序列的唯一 ID。
- `SOPInstanceUID`：单张 DICOM 切片的唯一 ID。
- 一个 study 通常包含多个序列，不能把单张切片当作独立样本随机切分，否则会发生同一患者检查进入不同折的严重泄漏。
- 官方说明中，一个 series 通常有 20–45 张切片，中位数 30，长尾可达数百张。

## 4. 十二个预测目标

每个标签都是检查级 0/1 金标准；提交时输出 0–1 之间的置信度。官方临床说明指出，模棱两可或临界病例按阴性处理，以偏向特异性。更详细的判定标准见主办方发布的 [Knee Abnormality Detection AI Challenge Overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)。

| 提交列名 | 中文含义 | 官方判定要点（简化） |
|---|---|---|
| `ACL` | 前交叉韧带撕裂 | 高级别部分或全层撕裂；轻微信号改变、退变或无断裂的增厚不算阳性 |
| `MCL` | 内侧副韧带撕裂 | 高级别部分或完全急性撕裂，伴纤维中断及韧带内外水肿；低级别扭伤或陈旧改变不算阳性 |
| `Medial Meniscus` | 内侧半月板撕裂 | 异常信号明确到达半月板表面且至少见于两幅图像，或存在截断、缩小、移位碎片等形态异常 |
| `Lateral Meniscus` | 外侧半月板撕裂 | 与内侧半月板相同的标准，作用于外侧半月板 |
| `Medial OA` | 内侧胫股间室骨关节炎 | 约 1 cm 或更大范围、超过 50% 软骨厚度的高级别软骨缺损，可伴软骨下骨髓改变 |
| `Lateral OA` | 外侧胫股间室骨关节炎 | 同上，位于外侧间室 |
| `PF OA` | 髌股关节骨关节炎 | 同上，位于髌股间室 |
| `Effusion` | 关节积液 | 中量或大量液体使关节腔扩张 |
| `Synovitis` | 滑膜炎 | 滑膜内衬炎症与增厚 |
| `Baker's` | Baker / 腘窝囊肿 | 膝后典型位置的中量或大量液性集合 |
| `Contusion` | 骨挫伤 | 撞击导致的骨髓水肿样信号，但没有明确骨折线 |
| `Fracture` | 急性骨折 | 急性皮质中断或骨折线 |

标注流程：每个参考集 study 由两名肌骨放射科医师独立标注，分歧由第三名医师裁决，形成检查级共识标签。

## 5. 文件说明

### `train.csv`

每行对应一个训练 study：

- `StudyInstanceUID`：与 `train_series/<StudyInstanceUID>/` 目录对应。
- `Report`：原始自由文本放射学报告，可能是多种语言之一。
- 后续 12 列：二元标签；绝大多数 study 的这些字段为空，只有 58 行有完整标签。

这意味着：4,407 不是“4,407 个完全监督样本”。真正的强标签样本极少，其余 4,349 个样本主要通过报告提供弱监督信息。

### `train_series.csv`

每行对应一个训练 series：

- `StudyInstanceUID`
- `SeriesInstanceUID`
- `Fluid_Sensitive`：是否为液体敏感序列，如 T2、PD、STIR 等。
- `Fat_Suppression`：是否应用脂肪抑制。它常与液体敏感相关，但两者并不等价。
- `Anatomical_Plane`：`Sagittal`、`Coronal` 或 `Axial`。

### `train_series/`

目录格式：

```text
train_series/
└── <StudyInstanceUID>/
    └── <SeriesInstanceUID>/
        └── <SOPInstanceUID>.dcm
```

### 测试文件

- `test.csv`：只含 `StudyInstanceUID`，不含 `Report`。
- `test_series.csv`：与训练 series 表字段相同。
- `test_series/`：与训练 DICOM 目录层级相同。
- 本地和公开数据中的 3 个测试 study 只是占位示例。正式 Notebook 评分时，Kaggle 会替换为约 1,300 个真实测试 study 及其 DICOM。
- 因此代码不能硬编码测试行数、UID、series 数量或目录内容。

### `sample_submission.csv`

一行一个测试 study，共 13 列：UID 加 12 个概率列。官方样例把所有概率设为 `0.5`。

```csv
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
<uid>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
```

## 6. 评分指标

对每个标签单独计算 ROC AUC，再对 12 个 AUC 等权平均：

$$
\text{Final Score}=\frac{1}{12}\sum_{i=1}^{12}\operatorname{AUC}_i
$$

理解这个指标时要注意：

- 这是 **macro average**：稀有标签和常见标签权重完全相同。
- AUC 衡量正样本排序高于负样本的能力，不要求选择分类阈值。
- 概率校准不是主指标的直接要求；只要单标签内排序不变，AUC 就不变。
- 任何一个弱标签（例如 MCL）表现很差，都会以 `1/12` 的权重直接拖累总分。
- 本地验证也应报告每个标签 AUC 和宏平均，不能只看总 loss 或 micro AUC。

## 7. 这道题真正难在哪里

### 7.1 金标准极少，弱标签是核心

只有 58 个完整标注 study，无法支撑常规的纯监督 12 标签训练。其余报告是多语言自由文本，且报告措辞和比赛的严格阳性定义并非一一对应。例如轻度积液、低级别扭伤、软骨病和比赛定义的中重度异常可能不同。

可行的总体思路通常是：

1. 用 58 个金标准病例理解标签定义并校验流程。
2. 从全部多语种报告抽取 12 项弱标签及置信度。
3. 对不确定、否定、术后或历史性描述做专门处理，而不是简单关键词匹配。
4. 用弱标签训练影像模型，再用金标准做验证、重加权、校准或少量微调。
5. 保留“报告标签置信度”，对噪声更大的样本降权。

### 7.2 一个 study 是多序列、多平面、多切片

不同病变依赖不同平面和序列：ACL、半月板常依赖矢状位/冠状位，髌股软骨更依赖轴位；水肿、积液和撕裂在液体敏感及脂肪抑制序列上更明显。因此需要完成：

```text
DICOM 解码与排序
→ 序列内切片采样/2.5D 编码
→ 同一平面内聚合
→ 多平面/多序列融合
→ 12 个 study-level 概率
```

### 7.3 DICOM 异质性大

官方明确提醒：强度、方向和分辨率会随 study/series 变化；传输语法混合了未压缩 Explicit VR Little Endian、JPEG Lossless、JPEG 2000 和 Implicit VR Little Endian。常见风险包括：

- DICOM 解码器不支持某些压缩格式。
- 仅按文件名而非空间位置排序，导致切片顺序错误。
- 左右膝、方向、像素间距和矩阵大小不一致。
- 每个 study 的 series 数、每个 series 的切片数均不固定。
- 直接逐 slice 推理会超过 9 小时或显存限制。

### 7.4 训练与测试的输入不对称

训练时有报告，测试时无报告。不能构造依赖测试报告的模型路径；文本更适合充当标签教师、训练辅助模态或图文预训练信号。

### 7.5 分布漂移

数据来自多个国家、机构、设备和协议。官方还特别说明，训练集、Public Leaderboard 和最终测试集的异常患病率不保证一致。应减少对单一机构风格和先验阳性率的依赖。

## 8. 本地已下载数据核对结果

当前 `F:\RSNA\data` 是一个约 11.5 MB 的极小子集，但 CSV 元数据是完整训练表：

| 本地项目 | 核对结果 |
|---|---:|
| `train.csv` | 4,407 个 study，14 列 |
| 完整 12 标签的 study | 58 |
| `train_series.csv` | 24,371 个 series |
| 每个训练 study 的 series 数 | 最少 3；中位数 5；最多 14 |
| 平面分布 | Axial 5,898；Coronal 8,609；Sagittal 9,864 |
| `test.csv` | 3 个示例 study |
| `test_series.csv` | 15 个示例 series |
| 本地 DICOM | 训练 1 张、测试 1 张，仅足以验证路径和基本读取代码 |

58 个金标准样本中的阳性分布如下。这个表只能描述小型金标准集，不应被当作整个训练集或测试集的真实患病率：

| 标签 | 阳性数 / 58 | 阳性率 |
|---|---:|---:|
| ACL | 24 | 41.4% |
| MCL | 9 | 15.5% |
| Medial Meniscus | 26 | 44.8% |
| Lateral Meniscus | 23 | 39.7% |
| Medial OA | 15 | 25.9% |
| Lateral OA | 11 | 19.0% |
| PF OA | 21 | 36.2% |
| Effusion | 35 | 60.3% |
| Synovitis | 27 | 46.6% |
| Baker's | 12 | 20.7% |
| Contusion | 19 | 32.8% |
| Fracture | 18 | 31.0% |

本地样本的 `Report` 中已经能看到西班牙语、荷兰语等内容，印证了官方所说的多语言特征。另需注意，报告内可能含换行，不能用简单的“文件行数减一”计算样本数，应使用正规 CSV 解析器。

## 9. 提交与效率赛道

主榜提交要求：

- 只能通过 Kaggle Notebook 提交。
- CPU 或 GPU Notebook 都必须在 9 小时内完成。
- 推理时必须关闭互联网。
- 输出文件必须叫 `submission.csv`。
- 列名、列顺序和测试 UID 应以运行时的 `sample_submission.csv` 为准。
- 每个概率应为有限数值，建议最终检查 `NaN`、无穷值、重复 UID、缺失 UID 和 `[0,1]` 范围。

另有独立的效率赛道。候选提交需满足主榜提交选择条件，并在 Private Leaderboard 上高于 `sample_submission.csv` 基准。官方效率分数为：

$$
\text{Efficiency}=\frac{\text{AUC}}{\text{Benchmark}-\max\text{AUC}}+\frac{\text{RuntimeSeconds}}{32400}
$$

目标是最小化该分数。因为分母通常为负数，更高 AUC 会使第一项更小；运行时间越长，第二项越大。实际实现和名次以 Kaggle 官方效率榜为准。

奖金分配：

- 主榜前十名：$9,000 / $7,000 / $6,500 / $6,000 / $5,500 / 第 6–10 名各 $5,000。
- 效率赛道前三名：$7,000 / $6,000 / $5,000。

## 10. 时间线

官方截止时间均为当天 **23:59 UTC**：

| 事件 | UTC 时间 | 北京时间（UTC+8） |
|---|---|---|
| 比赛开始 | 2026-07-30 | — |
| 报名/接受规则截止 | 2026-10-15 23:59 | 2026-10-16 07:59 |
| 队伍合并截止 | 2026-10-15 23:59 | 2026-10-16 07:59 |
| 最终提交截止 | 2026-10-22 23:59 | 2026-10-23 07:59 |
| 获奖者材料截止 | 2026-11-05 23:59 | 2026-11-06 07:59 |

截至本文整理日，距离最终提交约 34 天。主办方保留调整赛程的权利，临近截止前应再次核对官方页面。

## 11. 建议的起步路线

### 第一阶段：把数据管线做对

1. 正确读取所有官方 CSV，确定 study → series → slice 映射。
2. 验证各类 DICOM 传输语法均能解码。
3. 根据 DICOM 空间信息排序切片，并统一方向、窗宽/归一化和尺寸。
4. 按 `StudyInstanceUID` 划分训练/验证，绝不按 slice 或 series 随机划分。
5. 做一个能在 3 个示例 test study 上动态生成合规 `submission.csv` 的端到端 Notebook。

### 第二阶段：建立可信的弱标签

1. 先对报告做语言识别、否定识别和术语标准化。
2. 根据官方 12 项定义提取三态结果：阳性 / 阴性 / 不确定，而不是强行二分类。
3. 用 58 个金标准病例评估报告抽取质量，并按标签设置规则或置信权重。
4. 保存标签来源与置信度，便于后续排查噪声。

### 第三阶段：建立影像 baseline

1. 从每个序列均匀抽取相邻切片，形成 2.5D 输入。
2. 按轴/冠/矢状位分别编码，再用 mean/max/attention MIL 聚合到 study。
3. 使用预训练 2D 编码器起步，先冻结大部分 backbone 验证数据流。
4. 用 12 个 sigmoid 输出和带掩码/样本权重的二元交叉熵训练。
5. 验证时同时记录每标签 AUC、宏平均 AUC、推理耗时和峰值显存。

### 第四阶段：可靠验证与提升

- 58 个强标签太少，单次 hold-out 的方差会非常大；优先采用 study-level 多折验证并报告折间波动。
- 将强标签验证和弱标签验证分开看，避免被伪标签自洽性误导。
- 做 plane-aware、sequence-aware 采样和融合。
- 逐标签分析：不同病变的最佳平面、序列和 pooling 方式可能不同。
- 最后再尝试高分辨率、多模型集成和 TTA，并持续检查 9 小时推理预算。

## 12. 常见误区

- **误区：4,407 条训练记录都有标签。** 实际只有 58 条有完整金标准标签。
- **误区：比赛是单标签 12 分类。** 实际是 12 个独立二元目标，一个 study 可同时有多个异常。
- **误区：测试时也有报告。** 正式测试不提供 `Report`。
- **误区：公开的 3 个 test study 就是测试集。** 它们只是占位示例，正式约 1,300 个。
- **误区：`Fluid_Sensitive == Fat_Suppression`。** 二者相关但不等价。
- **误区：按 DICOM 文件名排序即可。** 应优先使用可靠的空间位置/方向信息。
- **误区：准确率高就代表比赛分高。** 官方用的是 12 标签宏平均 ROC AUC。
- **误区：逐切片随机切分。** 这会产生同一 study 泄漏，验证分数失真。
- **误区：只追求最重的集成。** Notebook 有 9 小时限制，而且另有独立效率奖金。

## 13. 最简心智模型

```text
训练阶段：
多语种报告 ──→ 弱标签/教师信号 ─┐
                                ├─→ 多序列 MRI 模型
58 例专家金标 ─→ 验证与纠偏 ────┘

测试阶段：
DICOM + series 元数据
        └─→ 多平面/多切片聚合 ─→ 每个 study 的 12 个概率 ─→ submission.csv
```

如果只记住三件事：

1. **预测单位是 study，不是 slice。**
2. **只有 58 个金标准病例，报告弱监督是赛题核心。**
3. **测试无报告且必须在离线 Kaggle Notebook 的 9 小时内完成推理。**

## 14. 来源与口径说明

- [官方比赛主页与评估说明](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation)：任务、Macro AUC、提交格式、时间线、奖金、Notebook 与效率赛道要求。
- [官方数据页](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)：字段、目录结构、数据规模、测试替换机制、DICOM 传输语法及分布漂移提示。
- [主办方临床标签说明](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)：12 项异常的医学判定标准和标注流程。
- “4,407 个 study、24,371 个 series、58 个完整标签、标签阳性数、平面分布及本地 DICOM 数量”由本文对 `F:\RSNA\data` 中 CSV 与文件树直接核对得到。

> 注意：比赛页面、规则和日程可能更新。任何影响参赛资格、外部数据许可或最终提交的决定，应以截止日前的 Kaggle 官方页面为准。
