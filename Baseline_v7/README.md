# Baseline_v7：64 个原始相邻候选，训练随机12窗口

将原 v7 输入替换为 v8 已验证的五 slot 候选窗口输入。DINOv2 Small CLS（384维）、256维投影、两层序列内 Transformer、一层 slot Transformer、MRI元数据、逐标签 query及分类头、软标签置信度加权 BCE、类别权重、58例人工标签验证、AdamW与warmup/cosine均沿用v1。保留 v7 无 bias 逐标签窗口池化，位置编码改为原始anchor物理深度的MLP，以支持随机及可变窗口数。

## 五个slot与候选

| slot | 序列选择偏好 | 候选窗口数 |
|---|---|---:|
| Sagittal fluid | 优先Fluid Sensitive=1 | 18 |
| Sagittal second | 优先Fluid Sensitive=0 | 14 |
| Coronal fluid | 优先Fluid Sensitive=1 | 12 |
| Coronal second | 优先Fluid Sensitive=0 | 8 |
| Axial | 无Fluid Sensitive偏好 | 12 |

每个slot只选一个序列，按CSV候选顺序选择，已有偏好则优先满足；不重复选同一SeriesInstanceUID，无合适序列则缺失，不重分配预算。这里按Fluid Sensitive选择，不是v1的FS/nonFS六slot。

最多64是**候选anchor/window数**，不是缓存64张灰度图后再拼邻居。每个序列按原始排序后约2%-98%的索引范围选anchor，范围内沿用v1按物理位置均匀采样。每个anchor取原始 `[i-1,i,i+1]`，首尾复制边界，不跨序列。短序列允许重复anchor，与v1固定预算采样一致。损坏切片沿用v1最近可读切片恢复，因此异常文件处可能不再真正连续。

140mm物理中心裁剪（行列间距分开，FOV不足补零）、336尺寸、0.5/99.5裁剪后前景分位数及float32精度保留。统计基于该序列全部候选anchor，包含重复anchor原有权重，与当次随机选中的12个窗口无关。与公开代码的uint8缓存、2/98强度统计和跨slot窗口不同。

## 训练、验证和聚合

- 每个Study每次训练抽12个窗口，五个有效slot各至少抽一个，其余从剩余候选随机抽取。候选数不少于12时不重复候选索引；候选不足12时才补抽。短序列的不同候选索引仍可能对应同一个原始anchor。
- 抽样由seed、epoch和Study UID确定，不依赖worker数/访问顺序。persistent_workers通过共享epoch更新抽样；每轮重新选择，续训沿用同一seed和epoch序列。
- 验证和推理不随机抽样，使用全部有效候选，最多64个。缺失slot时总数减少，每个Study至少一个有效slot。
- 概念训练输入 `[B,12,3,336,336]`；实现按有效窗口拼接为 `[总窗口数,3,H,W]`，并携带Study、slot和原始位置索引。只补齐特征，不补齐图像，不编码不存在的窗口。
- 序列内先加物理位置编码，再做Transformer和 `LayerNorm(256) → Linear(256,12,bias=False)` 窗口池化。Transformer和窗口softmax均屏蔽特征补齐位置，空slot不进入slice Transformer。
- 得到 `[B,5,12,256]` 后加入元数据/slot embedding，按标签使用共享slot Transformer；每个label query只读取自己的五个slot特征，输出 `[B,12]` logits。

## 缓存

默认在本目录 `input_cache` 首次访问Study时生成缓存；训练、验证可复用。缓存保存**float32唯一裁剪/resize切片＋窗口索引**，重叠窗口不重复存图，读取使用PyTorch mmap，训练只取所选窗口对应像素。完整缓存磁盘占用取决于唯一切片数，可能较大；这保留v1精度，不是公开方案约32GB的uint8缓存。

`--cache-dir` 可指向已有共享缓存；`--no-cache` 禁用磁盘缓存。key包含预处理参数、选中序列、源DICOM文件名/大小/修改时间，改变配置或文件会建立新缓存，不复用旧输入。并发写入采用临时文件和原子替换，缓存不包含标签。首次epoch含解码/建缓存开销，后续epoch更能反映训练速度。改变配置留下的旧缓存需自行按空间需要清理。

Kaggle推理不写磁盘缓存，每个Study直接处理一次，避免生成大型缓存输出。

## 四卡训练

从 `Baseline_v7` 目录运行，DINOv2默认读取上级目录 `dinov2-pytorch-small-v1`，其他位置指定 `--dinov2-model-dir`。

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --data-root /root/RSNA/rsna-knee-abnormality-detection \
  --output-dir ./outputs/last6_3e-4 \
  --cache-dir ./input_cache \
  --crop-mm 140 \
  --image-size 336 \
  --train-windows 12 \
  --span-lo 0.02 \
  --span-hi 0.98 \
  --batch-size 2 \
  --accum-steps 2 \
  --backbone-mode last6 \
  --epochs 15 \
  --amp bf16
```

新版v7不使用 `--num-slices`；五个slot候选配额固定为18/14/12/8/12。可将 `--train-windows` 设为24、32或64作预算消融，至少5；若训练数量超过该Study候选数，会重复补抽，验证仍只看全部候选一次。第一版保持336，384可用 `--image-size 384` 单独比较。

日志 `windows` 是该rank当前batch实际送入DINO的窗口总数（batch=2、train-windows=12时是24）；`candidates` 是该batch的候选窗口总数，最多128，并非本次编码数。

## 权重与Kaggle推理

- checkpoint保存新版v7架构标记、五slot顺序、候选预算和全部训练参数。
- `--resume` 仅接受新版v7相同输入/微调/元数据/seed配置，并恢复训练状态。
- `--init-checkpoint` 仅接受新版v7模型权重。v1/旧版v7的slot身份及位置编码不同，不自动迁移；DINO仍从同一预训练目录初始化。
- `Kaggle_Inference.ipynb` 独立嵌入数据和模型实现，根据checkpoint读取尺寸、裁剪和覆盖范围，使用全部候选，支持最多两张GPU。
- 上传新版v7 `best.pt`并挂载DINOv2和比赛数据。notebook的 `CHECKPOINT_PATH` 留空会自动定位唯一best.pt，多份权重时填写实际路径。输出sigmoid概率，不做窗口TTA或排名转换。

目录与v1一样只包含六个文件，不附加测试文件或旧训练权重。
