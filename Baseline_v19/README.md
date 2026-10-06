# Baseline_v19：多 series 层次融合实验

基于 v14，保持 DINOv2 Small CLS（384 → 256）、原始相邻三切片窗口、physical crop、EMA、损失函数及 slot Transformer / label query / output head。每个 slot 最多两条 series，分别编码后做 label-specific series attention。尚未进行完整训练或线上评估。

## 选择与质量条件

先完整执行 v14 的五 slot top1 选择，保留平面、Fluid Sensitive 偏好、几何 tuple 排序、同分 CSV 顺序与 Series UID 不重复规则。所有 top1 先保留，再按同样偏好与排序补充未使用的第二条，避免前面 slot 的 top2 抢走后面 slot 的 top1。

v14 的 coverage/FOV 是排序项，并未直接排除不达标序列。v19 新增以下 hard gate：**top1 与新增 series 都必须 `quality_usable=1`、coverage ≥ 该平面的训练阈值、最小 in-plane FOV ≥ `crop_mm`，才使用两条**。其余几何指标仍参与 v14 排序。若 top1 不达标，保留 v14 的单 series 回退，不添加 top2，避免新增缺失 slot。对比类别偏好先确定，再排除已使用 UID；不会为了凑第二条而跨类别补齐。

coverage 阈值仍由弱标签训练 Study 的可用几何统计拟合，默认第 20 百分位；gold 验证与测试 Study 不参与。某平面阈值为 0 时沿用 v14 不比较 coverage 的行为，geometry 和 FOV gate 仍然有效。这些 header 条件不检测运动伪影或像素解码质量。

## 候选预算与训练采样

| slot | 单 series | 两条 series（top1 + top2） |
| --- | ---: | ---: |
| Sagittal_fluid | 23 | 23 + 12 |
| Sagittal_second | 17 | 17 + 8 |
| Coronal_fluid | 15 | 15 + 8 |
| Coronal_second | 10 | 10 + 5 |
| Axial | 15 | 15 + 7 |
| 合计 | 80 | 120 |

top1 保留 v14 的完整 80 候选预算，仅有合格 top2 的 slot 增加额外候选，最多增加 40 个，因此每个 Study 最多 120 candidates。只有一条时仍为原预算；缺失 slot 不向其他 slot 转移预算。每条 series 分别采样和归一化，三通道相邻切片不跨 acquisition。

这是在 v19 原 80 候选方案上的直接修改：保留全部 top1 候选以评估新增 acquisition 的价值。训练随机抽样仍只取 24 个，因此不保证每次训练都看到全部 top1 候选；验证和推理才使用完整候选 bank。

训练默认仍抽 24 windows：先从每条实际入选 series 随机抽一个，再从全局剩余候选不放回随机抽取；只有候选不足时补重复窗口。不设固定 slot 配额。随机种子由 seed / epoch / Study UID 决定，抽完后按 series 内物理位置排序。CLI 最少允许 10 windows，以保证最多十条 series 都有窗口。验证和推理使用全部候选。

## 模型

```text
各 series 的 windows → DINO CLS → slice Transformer → label-specific window pooling
    [B,5,2,12,256]
    → 可选的各 series metadata projection
    → LayerNorm(256) + Linear(256,1,bias=False)
    → 在 series 维 dim=2 做 masked softmax / weighted sum
    [B,5,12,256]
    → 原有 slot embedding → slot Transformer → label query → 12 logits
```

slice Transformer 按 `(Study, slot, series)` 分组，不同 acquisition 不交换窗口信息。series scorer 共享参数，但输入已经是逐标签特征，因此每个标签有自己的 series 权重。单 series 权重为 1；缺失 series 权重为 0；全缺失 slot 返回零特征，随后由原有 slot mask 排除，避免全 mask softmax 产生 NaN。启用 metadata 时先将 `[B,5,2,11]` 元数据加到各 series 特征，再融合；`--no-metadata` 仍关闭并冻结该模块。

只新增 `series_attention` 的 768 个参数，不新增 Transformer。训练每 Study 编码图像数仍为 24；验证和推理的图像数上限从 80 增至 120，因此全量候选下的 DINO 工作量最多增加 50%。更多候选也会增加缓存占用和预处理开销，真实时间和显存需要训练服务器测量。

## 训练与推理

从 `Baseline_v19` 目录运行，与 v14 使用相同训练参数作对照；以下沿用 v14 README 的示例配置：

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --data-root /root/RSNA/rsna-knee-abnormality-detection \
  --output-dir ./outputs/last6_3e-4_multi_series \
  --cache-dir ./input_cache \
  --crop-mm 140 --image-size 336 --train-windows 24 \
  --span-lo 0.02 --span-hi 0.98 \
  --batch-size 2 --accum-steps 2 --backbone-mode last6 \
  --epochs 15 --head-lr 3e-4 --backbone-lr 1e-5 \
  --ema-decay 0.999 --seed 42 --coverage-quantile 0.2 \
  --series-quality-workers 8 --amp bf16 --no-metadata
```

`--no-pos-weight` 的使用应与对照 v14 保持一致。默认从同一份 DINO 预训练权重重新训练；`--init-checkpoint` 也支持 v14 或 v19，加载 v14 时只允许缺少新增的三个 series attention 参数，其他参数必须匹配。续训 `--resume` 仅接受同配置 v19。

架构标记为 `baseline_v19_120_candidates_multi_series_ema`。checkpoint 保存 `multi_series` 规则、top1/top2 预算、v14 coverage 阈值、slot 最大预算 `[35,25,23,15,22]` 和预处理参数；续训及本目录自包含的 `Kaggle_Inference.ipynb` 都检查这些配置。旧的 80 候选 v19 checkpoint 不作为本版续训或推理权重。推理使用 checkpoint 的训练阈值，不在测试集重新拟合。notebook 的数据与模型实现和 `.py` 文件保持同步，使用当前 v19 的 `best.pt`。

输入缓存使用独立 schema 20，key 包含每 slot 入选序列的路径、顺序、预算、DICOM 文件状态及原预处理参数。沿用原 `--cache-dir` 时会生成新的缓存文件，保留旧文件，不复用 v14 或旧 v19 的输入 bank。header 质量规则未变，可共用原有 `series_quality` 缓存。

输出 `series_selection.csv` 记录原 v14 top1、v19 top1/top2、各 series 预算与质量指标；`series_selection_summary.json` 增加双 series slot 数和总入选 series 数。`changed_slots` 比较 v14/v19 top1，正常应为 0。

## 本地验证

```bash
python test_multi_series.py -v
python ../tools/check_line_endings.py
```

11 项 CPU 回归测试覆盖质量 gate、保留 top1 全部候选、防止 UID 重复、120/24 预算、只有合格 top2 才增加预算、每 series 至少一个窗口、缓存隔离、batch/series 隔离、缺失 mask、逐标签权重、前向与反向、单 series 与 v14 输出一致、v14 权重初始化、EMA/checkpoint 恢复及 notebook 源码一致性。

原 80 候选版已使用本地真实 DINOv2 Small 权重完成 CPU 的 28×28、24 windows、10 series、last6 + no-metadata 前向、BCE 反向、优化器和 EMA 更新；此次候选扩容不改变网络结构。真实四卡训练、336×336 显存、全量选择分布和线上分数尚未验证。
