# API 报告软标签提取

当前版本 `3-soft-consistent` 使用 LLM 直接估计连续软分数，不将硬标签简单替换为 0.05/0.95。旧硬标签结果保持原样，不能当作软标签缓存。Python 3.10+。

调用沿用 api_test.py：OpenAI SDK、Responses API、https://api.catbeeai.com/v1、gpt-6.1-sol，密钥从 generate_label 环境变量读取。

## 重新提取同一批 20 份报告

在 F:\Kaggle\RSNA\label_extract 中运行：

```powershell
python -m pip install -r requirements.txt
python extract.py --limit 20 --workers 3 --output output/pilot20_soft_v3
```

默认 seed=42，与首次硬标签试跑使用同一批 study 和同样 6 份金标报告示例，方便比较。命令会产生新的 API 用量。默认输出目录为 output/pilot20_soft_v3，不覆盖旧 output/pilot20 和 output/pilot20_soft。v3 统一目标与证据口径，旧 20 份不能直接充当 v3 缓存；需要重新执行才会生成新分数。

仅准备输入：`python extract.py --limit 20 --prepare-only`，不会发出 API 请求。

若终端未设置密钥，先设置 `$env:generate_label = "你的API密钥"`。代码不打印或写入密钥。运行提取会将报告及示例发送到 api_test.py 指定的服务。

## 分数与训练权重

| 字段 | 含义 |
|---|---|
| label | 连续软目标：报告支持病例符合比赛阳性定义的可能程度，不是提取置信度 |
| report_status | PRESENT / ABSENT / UNCERTAIN / NOT_MENTIONED：报告描述状态 |
| evidence_strength | DIRECT / PARTIAL / CONFLICTING / INSUFFICIENT：证据支持程度 |
| evidence | 原始报告的精确原句 |
| score_reason | 分数的简短中文依据 |
| uncertainty_reason | 缺失条件、疑似、矛盾或无依据的原因 |

软分数允许 0.001～0.999 的连续值，拒绝硬 0/1。提示词给出定性参考区间，要求根据具体证据选值；不是固定的标签平滑公式。这些分数是未校准的 LLM 估计，不能当作经过验证的真实概率。

部分证据不再一律丢为未知：例如明确全层软骨缺失但没有尺寸，可以保留软估计，记录范围缺失。疑似和矛盾也可保留谨慎的软估计。未提及、无有用当前证据、证据无效或非膝报告仍为 null，mask=0，不用 0.5 自动制造监督。

训练权重独立于软分数：DIRECT=1.0、PARTIAL=0.35、CONFLICTING=0.1、INSUFFICIENT=0.0。这是待验证的工程起点，未做临床校准。明确正常可以有低软分数、高证据权重。

labels.csv 的 12 个目标列为软分数，各附 __mask 和 __weight。使用支持软目标的损失，并独立应用 mask/weight。空值不能填零后当有效监督。

## 输出

| 文件 | 内容 |
|---|---|
| labels.csv | UID、12 类软分数、mask/weight、提取状态 |
| labels_detail.csv | 分数、状态、证据强度、原句与偏移、评分依据、不确定原因 |
| results.jsonl | 成功结果检查点与验证警告 |
| attempts.jsonl / raw/ | 所有调用记录与原始响应，包含失败响应 |
| usage.csv | 每次请求耗时及 input/output/total/cached/reasoning token |
| summary.json | 完成数、有分数项/屏蔽项、各类分数分布、累计 token 和平均请求时间 |
| selected_studies.csv | 目标报告和 UID |
| examples.json / prompt.txt / schema.json | 实际示例、提示词和结构 |
| manifest.json | 模型参数、输入、提示词、提取版本、证据权重的指纹 |

缺失 usage 留空，不冒充零。缓存输入和推理输出已经包含在输入/输出 token 中，不能重复加总。重试和失败的已知 token 也计入用量。金额按该服务商计费，不套用官方价格。

金标示例保持真实二元影像标签，不伪造示例软分数；它们用于理解目标定义，不是报告提取的完美真值。目标请求只含 UID 和报告。待提取池排除有标签 study 及其规范化文本重复。

## 并发与续跑

默认 workers=1；--workers 3 最多同时进行三个请求，独立客户端、加锁写入。发现失败后停止提交新任务，已发请求完成后落盘。Ctrl+C 也等待已发请求保存。不要两个进程同时写同一目录。

同样命令续跑会跳过成功病例，可以只改变 workers。其他参数、数据、提示词或版本变化须新目录。硬标签目录不能续跑软提取，不会改写其结果。

SDK 内置重试关闭，--retries 1 表示最多一次显式重试。默认不传 reasoning，可给 --reasoning-effort low。--max-output-tokens 4096 包含推理预算；incomplete 响应不算成功。

若网关不支持 json_schema，可显式 --format json_object 或 --format prompt 并给新输出目录。本地验证相同，不自动切换模型或格式。

## 复核与验证

软提取完成后运行 `python make_review.py --run output/pilot20_soft_v3`，在浏览器打开目录内的 oa_effusion_review.html。支持建议软分数、证据强度、评分依据和笔记，导出 CSV，不自动覆盖结果。笔记仅尝试浏览器本地暂存，请及时导出。

旧结果仍可用 `python make_review.py --run output/pilot20` 查看，明确标为 LEGACY_HARD，不转换分数。

从仓库根目录运行离线测试：

```powershell
python -m unittest discover -s label_extract -p "test_*.py" -v
```

测试使用假客户端，不消耗 API token，覆盖软分数、未知屏蔽、非法数值、证据、用量、并发、续跑和旧结果隔离。真实质量和速度由新试跑确认。

接口参考：[官方 Responses 结构化输出文档](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)。目标定义和软估计原则见 target_definitions.md。

## 对照原有标签

已按 UID 对照 Baseline_v22/label.csv（与其他 Baseline 标签文件内容相同）和 labels_blend.csv。运行 `python compare_labels.py --run output/pilot20_soft`，仅本地读取，无 API 调用；在该 run 的 legacy_comparison 中保存全 240 项原句、理由、三套分数和逐类统计。可以用 --old 指定其他原有标签文件。0.5 仅用于定位方向分歧，不将一致率当准确率，不改写源标签。

v3 规则见 target_definitions.md 的跨病例统一证据口径，软分数不按旧文件机械映射。原有 label.csv 是软标签参照而非金标；未提及仍为 null/mask=0。DIRECT/PARTIAL/CONFLICTING/INSUFFICIENT 权重不变，阳性缺少必要条件时统一 PARTIAL，明确亚阈值可用 DIRECT 支持低分。
