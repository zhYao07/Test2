# L1 报告标签提取

本目录实现 `RSNA_Report_Label_Extraction_Plan_ZH.md` 的报告阶段：多语言全文提取、原文证据、独立标准映射、审计、180 份试跑门槛和全量导出。没有图像教师、共现补全、滑膜炎补全或 MRI 训练，也不修改 Baseline 的标签读取接口。

## 当前执行状态

截至 2026-10-05，提取脚本与 36 项离线测试已完成。当前实际模型试跑产物位于 `report_labels/L1_20261005_v5/`；早期准备记录保留于 `L1_20261005/` 和 `L1_20261005_v2/`。用户已指定 `gpt-6.1-sol` 并明确授权提取子代理。Codex CLI 0.160.0 已复用本机 ChatGPT 登录完成真实探针，配置 `allow_agent_execution=true`。V3 的 8 份真实响应中发现局部否定映射缺陷，已保留诊断缓存并停止。V4 再完成 57 份报告的独立诊断审计后发现骨髓候选/限定否定问题，已停止并保留 pilot_diagnostic 产物；V5 使用 prompt 1.3 重新进行 180 份真实试跑；全量提取仍须等待文本审计门槛通过。运行状态以当前产物中的 `pilot/run.json` 为准。

离线测试的假响应只用于临时目录的程序验证，不能视为真实模型的提取质量。准备阶段不会制造空白的“已完成标签”文件。

## 配置模型并运行

### 用户指定的 GPT‑6.1 Sol 路径

`model_config.gpt61sol.json` 使用已安装的官方 Codex CLI 和现有登录，不复制或导出登录 token。适配器按单份报告启动 ephemeral、read-only 调用，禁用 shell、其他代理、apps、plugins、浏览器和电脑操作，返回受 JSON Schema 约束的最终响应；原始事件和用量进入审计缓存。已完成离线适配器测试和真实模型探针。当前调用固定 HTTPS 传输，避免本机 WebSocket 超时重试，最多并发 3 份报告。

启动这种调用属于启动提取子代理，本次已经获得用户明确授权。运行：

```powershell
python report_label_extraction/l1.py preflight --config report_label_extraction/model_config.gpt61sol.json
python report_label_extraction/l1.py pilot --output report_labels/L1_20261005_v5 --config report_label_extraction/model_config.gpt61sol.json
```

试跑完成后执行下述同一套文本审计和 gate，再使用该配置运行 full。当前受限 shell 缺少 Codex 可识别的 home，`codex login status` 在正常用户环境中已验证登录；真正批处理须在能够正常读取现有登录的执行环境中运行，不重新设置 HOME 或导出密钥。

精确请求的模型 slug 为 `gpt-6.1-sol`，CLI 版本为 0.160.0。[官方非交互文档](https://learn.chatgpt.com/docs/non-interactive-mode)说明 CLI 复用现有认证并支持 output schema。CLI 响应未暴露不可变服务端 snapshot，所以当前只能记录模型 slug、CLI 版本、prompt/schema/代码指纹和请求时间，不能宣称已锁定不可变权重版本。调用使用固定 reasoning effort；CLI 不提供本适配器的 temperature 参数，输出也不能宣称完全确定。

### 独立 HTTP 服务路径

依赖是 Python 3.10+ 和 `jsonschema`；当前环境已具备。所用协议为兼容 `/v1/chat/completions` 的 HTTP JSON 接口，不依赖某一家 SDK。其他协议须编写适配器。

复制 `model_config.example.json` 到新配置文件，填入：

- `base_url`：服务根地址，通常以 `/v1` 结尾，脚本追加 `/chat/completions`。
- `model` 与 `model_version`：真实的固定模型 ID、供应商 snapshot 或本地 revision；不应使用会自动变化的版本作为可复现基线。
- `context_window_tokens`：该模型/服务实际允许的上下文长度；示例 65536 是占位预算，不代表选定了模型。
- `api_key_env`：已设置凭据的环境变量名；配置文件不接受明文 `api_key`。无认证的本地服务可设置 `auth_required=false`。
- `response_format`：优先 `json_schema`；仅支持 JSON mode 的兼容服务显式改为 `json_object`，仍执行相同的本地 schema 检查。

不要将密钥写进仓库或聊天。HTTP 后端不会读取 Codex 登录凭据。`preflight` 不访问网络；第一份试跑报告同时验证服务能否正常返回结构化结果。HTTP 鉴权、接口或配置失败会停止运行，避免生成全量错误结果。

在仓库根目录运行：

```powershell
python report_label_extraction/l1.py preflight --config report_label_extraction/model_config.local.json
python report_label_extraction/l1.py pilot --output report_labels/L1_20261005_v2 --config report_label_extraction/model_config.local.json
```

试跑后阅读 `pilot/text_audit.html`，对 `pilot/text_review.csv` 中的 2160 个“报告 × 类别”项记录独立文本复核。判断内容包括描述状态、证据、否定范围、解剖、程度、时效及标准映射。不可仅根据可定位的证据自动填写 PASS。

- 保留项：`disposition=KEEP`，七个审计维度均为 `PASS`，填写真实 `reviewer`。
- 排除项：`disposition=MASK`，审计维度填 `PASS/FAIL/NA`，填写 reviewer 和具体 notes。全量导出会屏蔽该项及完全相同报告的同类项，保留提取原文和审计记录。
- `PENDING`、缺行、重复行、UID/hash 不一致、保留的关键错误均不能通过门槛。解析成功率须达到预设的 99%。另要求至少 216 个保留监督项；膝关节已提及项七维文本错读率 ≤5%，每类已提及 ≥20 项时该类错读率 ≤10%。合法排除且无 FAIL 的项不计错读。门槛是预先声明的工程操作规则，不是经临床验证的质量阈值。

```powershell
python report_label_extraction/l1.py gate --output report_labels/L1_20261005_v2
python report_label_extraction/l1.py full --output report_labels/L1_20261005_v2 --config report_label_extraction/model_config.local.json
```

full 必须有通过的 gate，且模型配置、代码、prompt/schema、映射器、数据和已审计产物的指纹一致。若在 gold 检查后更改规则，应另建版本，后续 gold 指标应称为开发结果。中断后用同一命令续跑，已完成的完全相同文本按缓存复用；中断留下的不完整末行先保留字节备份再恢复，完整/中间损坏记录停止调查；缓存中的永久解析失败不自动反复请求，需新版本试跑修正。

需要重新准备新版本时：

```powershell
python report_label_extraction/l1.py prepare --train data/train.csv --output report_labels/L1_NEW_VERSION
```

prepare 拒绝覆盖已有准备版本。所有新 CSV/JSON/Markdown/HTML 均使用 UTF-8 和 LF。报告原文放在 JSONL 字符串中，内部换行被 JSON 转义，无损保存。

## 数据、映射与审计边界

完整 train UID 清单决定输出覆盖率；不会 inner join 丢病例。专家标签单独按 UID 保存，绝不进入模型输入，不被报告标签覆盖。180 份试跑来自非 gold 报告，并排除任何包含 gold UID 的规范化报告组；90 份按语言/长度轮流分层抽样，90 份按高风险表达抽样，均去重并补足。

语言识别采用透明的多语言词典提示和相对差值，不是可靠的概率估计；未识别报告保留 UNKNOWN，仍由多语言模型读取全文。原文 SHA256 用于完全重复文本缓存；NFC、大小写和空白规范化哈希用于分组。近似重复仅生成候选复核表，绝不合并或复用。候选搜索使用 SimHash 分桶与字符串相似度，不保证检出所有近重复；容量限制计数会写入 preparation.json。

模型先返回描述状态、证据、部位、程度、大小、时效和逐类事实，再提出 `llm_meets_criteria`。Python 映射器独立根据事实判断 `meets_criteria`，不采用模型的自报置信度，也不直接采用其 YES/NO。事实仍由模型解释原文，因此独立映射器不能替代语义复核。

prompt/schema 1.1 增加报告整体部位及原文证据；1.2 增加 class_absence_supported 独立事实，局部否定不足时 UNKNOWN 且 mask=0。独立来源审计发现试跑中有一份肩部报告，因此 `report_scope=OTHER/UNCLEAR` 或部位证据无效时保留 UID，并屏蔽全部 12 类膝关节目标。其他解剖部位出现在病史中不能据此排除整份报告。

临床定义来自仓库的比赛概览第 4 节和提取方案第 5 节，出处为 [主办方临床说明](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)。本次浏览主办方链接未返回可读取正文，当前实现未新增或声称重新核验官方细节；保守保留程度/范围/时效不足的 UNKNOWN。例如 OA 需同时支持 >50% 深度与约 ≥10 mm 范围；中量积液还需关节腔扩张证据；骨挫伤需创伤性及该病灶无骨折线的证据。

schema 键、枚举和分数范围必须正确；response hash 必须与输入一致。证据只接受原文字面或仅空白变化的匹配，并保存原文字符偏移。无法定位、事实无证据、关键矛盾、解剖错配等项 mask=0；关键词只提出否定、侧别和历史范围的复核，不改写报告描述状态。解析/格式和可修复证据异常最多重试两次，仍失败记 ERROR，绝不静默填阴性。

全文预算以 UTF-8 字节数作为保守 token 上界，加 schema/prompt、输出预算和模板余量；预检查不满足时整批停止。当前最长报告 4743 字符，示例 65536 预算可容纳。**本版没有分段汇总功能**；遇到更长报告须改用足够大的上下文或单独实现并审计分段功能，不能截掉末尾结论。服务输出因长度终止也视为失败。

训练映射保留方案的起始设置：明确 YES 为 0.95、NO 为 0.05，weight=1、mask=1；证据不足但有模型软分数时 weight=0.25；未提及、历史无当前评估、关键矛盾、失败及复核排除项为空值、weight=0、mask=0。软分数没有经过校准。`labels_training_hard.csv` 仅保留明确 YES/NO 的硬 0/1，未知软项不参与。

## 产物合同

准备目录包含完整 `reports_manifest.csv`、无 gold 标签的 `reports_source.jsonl`、按 UID 原样保存的 `gold_original.csv`、`pilot_sample.csv`、`near_duplicate_candidates.csv`、prompt/schema 和输入校验摘要。

pilot/full 提取阶段分别导出：

| 文件 | 内容 |
|---|---|
| `labels_raw.jsonl` | 每 UID 的 12 类描述、证据、事实、最终映射、原始响应、重试记录和版本元数据 |
| `labels_long.csv` | 每 UID × 类别一行，含 target、weight、mask、来源和复核原因 |
| `labels_training.csv` | UID、12 个目标列、12 个 `__weight` 和 12 个 `__mask` |
| `labels_training_hard.csv` | 已知项硬 0/1 对照，未知项留空并 mask |
| `labels_audit.csv` | 规则、证据、映射分歧、解析失败及人工 MASK 处理记录 |
| `labels_metrics.json` | 覆盖率、语言 × 类别状态、有效监督、总权重、软分数、失败/复核率、token 用量；full 附 gold 检查 |
| `quality_report.md` | 质量摘要和逐类统计，明确区分文本可定位性与影像目标一致性 |
| `extraction_cache.jsonl` / `run.json` | 按原文哈希的逐份响应缓存与运行状态，支持同版本续跑 |

全量 gold 检查记录逐类可用分数 AUC 和其 UID 集合、bootstrap 区间，以及完整 gold 固定 0.5 占位 AUC；分别计算 Macro AUC，注明有效类别数。记录与专家明确分歧但不覆盖原标签。不同版本比较应使用相同 UID 子集，不能把覆盖率变化当成指标提升。

## 验证

```powershell
python -m unittest discover -s report_label_extraction -p 'test_*.py' -v
python tools/check_line_endings.py
```

测试包括临床映射边界、schema/原文证据、失败与截断处理、完整报告请求、不泄漏专家标签、180 份 gate、缓存续用、全量覆盖、固定 gold 占位统计及审计产物篡改检查。使用的均为合成报告和临时文件，不消耗模型请求。

CLI 当前通过 [官方 model_instructions_file 配置](https://learn.chatgpt.com/docs/config-file/config-reference)使用冻结的专用提取指令替换通用编码指令，不修改全局配置、认证或工具隔离。prompt 1.3 明确骨髓异常候选记录、限定性否定、退变因果和数字尺寸不能自行转量级的规则。审计口径见 AUDIT_GUIDE_ZH.md。
