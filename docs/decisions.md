# Aether 技术决策

## ADR-001：M1 向量后端首日双平台基准

日期：2026-10-31。目标规模先固定为 10,000 chunks × 256 维 float32，
100 次查询，seed=2026，余弦精确近邻，k=10。它验证当前规模，不能据此外推十万规模。
门槛在第二轮性能测量前固定：exact recall@10=1.0、查询 p95≤100ms、建库≤30s。
首次 Windows 扩展加载与余弦探针通过；基准发现 vec0 只支持 `ORDER BY distance`，
已改为 SQL 距离排序后 Python 稳定次序，回归测试覆盖 KNN 与持久化。

命令：`uv run python -m aether.retrieval.bench --output data/bench.json`。
计时包括公开索引接口的输入校验、归一化和查询，建库包括连接、扩展加载及落盘；
一次热身后采样，报告平台/Python/SQLite、p50/p95、建库与磁盘字节。

| 平台/后端 | 建库 ms | 查询 p50 ms | 查询 p95 ms | recall@10 | 文件 bytes |
|---|---:|---:|---:|---:|---:|
| Windows 11 / NumPy | 323.786 | 0.954 | 1.346 | 1.0 | 13,893,632 |
| Windows 11 / sqlite-vec 0.1.9 | 366.538 | 20.203 | 32.380 | 1.0 | 11,042,816 |
| CI Ubuntu / NumPy | 294.015 | 1.092 | 2.730 | 1.0 | 13,893,632 |
| CI Ubuntu / sqlite-vec 0.1.9 | 373.564 | 5.172 | 5.453 | 1.0 | 11,042,816 |
| CI Windows Server 2025 / NumPy | 798.005 | 0.827 | 0.867 | 1.0 | 13,893,632 |
| CI Windows Server 2025 / sqlite-vec 0.1.9 | 503.538 | 7.286 | 7.538 | 1.0 | 11,042,816 |

双平台结果来自提交 `9840125` 的 [CI run 41170125076](https://github.com/leikuman1/Aether/actions/runs/41170125076)。
Ubuntu Python 3.12.3 / SQLite 3.45.1；Windows Python 3.12.10 / SQLite 3.49.1。
两个平台 sqlite-vec 均通过既定门槛，默认选择 sqlite-vec：具备无外部服务的持久化 KNN，
文件比 NumPy 小约 20.5%；它在此规模的查询速度慢于 NumPy，不宣称性能胜出。
显式设置 `AETHER_DENSE_BACKEND=numpy` 即可退回；扩展加载失败会明确报错，不静默更换索引格式。
NumPy 后端不加载 sqlite-vec 扩展。表中毫秒数随机器波动，不是跨环境 SLO。
`python -m aether.retrieval.bench` 的退出码只认当前配置的默认后端
（`Settings.dense_backend`，默认 sqlite-vec，可由 `AETHER_DENSE_BACKEND` 覆盖）是否通过门槛；
NumPy 通过不能使默认 sqlite-vec 未通过的运行成功。
正确性/评测门禁由测试独立保障。原始 JSON 保留在 Actions artifacts。

## ADR-002：两种分块与离线三路比较

24 篇原创技术文档、72 个 Markdown 小节、60 条查询、91 个预标注证据单元；
exact_term / paraphrase / multi_evidence / contrast 各 15 条。
标签依据完整证据原文预先指定，程序只映射 chunk，不依据任一路检索结果修改标签。
canonical `markdown-480` IDs 已入库；内容改变导致 ID 变化，旧标签会直接失败。
fixed 对同一证据重新映射，规则为同 doc_id、至少一半连续原文且最少 40 字符。
指标按不同相关 chunk 计数，重叠会改变分母；保留该局限，不把跨策略差异当作泛化结论。

`uv run aether-eval retrieval --gate` 同时输出表格与 JSON；
JSON 包含语料/黄金集指纹、配置、类别细分、逐条相关/返回 IDs、每路得分和阶段时延。
所有计分均采用同一 query/相关集合/top-k，recall 为宏平均；MRR 在 top-10 内计首个相关位置。
重复命中只计算一次 recall；无命中为 0；空标签拒绝。

默认 FakeEmbedder 为 sha256 子词哈希 256 维，并非语义模型。
等权 RRF 未达验收，保留失败基线如下，不掩盖负向实验。

| chunker | route | recall@1 | recall@5 | recall@10 | MRR@10 |
|---|---|---:|---:|---:|---:|
| fixed-400-64 | BM25 | 0.7556 | 0.9750 | 0.9833 | 0.9806 |
| fixed-400-64 | dense | 0.4806 | 0.8667 | 0.9000 | 0.7318 |
| fixed-400-64 | RRF 1:1 | 0.6722 | 0.9500 | 0.9667 | 0.9075 |
| fixed-400-64 | RRF 16:1 | 0.7556 | 0.9750 | 0.9833 | 0.9806 |
| markdown-480 | BM25 | 0.6972 | 0.9833 | 0.9917 | 0.9667 |
| markdown-480 | dense | 0.4833 | 0.7417 | 0.8583 | 0.7193 |
| markdown-480 | RRF 1:1 | 0.5556 | 0.9500 | 1.0000 | 0.8511 |
| markdown-480 | RRF 16:1 | 0.6972 | 0.9833 | 0.9917 | 0.9667 |

在同一冻结集尝试稀疏权重 1、2、4、8、16（稠密固定 1，RRF 常数 60、每路候选 50）。
两种分块 recall@5 依次为 (0.9500,0.9500)、(0.9583,0.9750)、
(0.9667,0.9833)、(0.9750,0.9750)、(0.9750,0.9833)。
选择 16:1 作为 Fake 模式保守回归配置：达到“≥两条单路”的非退化验收，
但结果只与 BM25 持平，**没有融合增益证据**，且存在本集调参偏差。
16:1 没有关闭 dense，仍通过 RRF 贡献排序；真实 embedding 必须在独立集重新验证权重。
默认 Markdown 保留标题路径且无 overlap；fixed 保留句边界与 64 字符 overlap。
后续需要真实模型、独立 holdout 和面向语义改写的质量验证，不能把 Fake 指标当简历的语义效果。

## ADR-003：失败语义、持久化与观测

摄取在同目录临时库完成 embedding 数量/维度/有限数校验、chunks 与向量落盘，
成功关闭后 os.replace；失败旧库不变。此版是单写者全量重建，Windows 打开旧索引的进程
应先关闭再替换，暂不提供在线多进程热更新。恢复验证 embedder 身份、端点指纹和维度。

生产 embedding 仅经 Settings 与 OpenAI 兼容 async httpx：批量、超时、429/5xx/transport
错误指数退避，4xx 立即失败；乱序结果按 index 对齐，重复 index/NaN/零向量/维度漂移拒绝。
错误信息不包含 provider 原文或密钥。所有实际文件/数据库 I/O 在线程执行，外部请求 async。
Reranker 协议保留，候选池先于最终 top-k，测试可证明重排可以提升原候选尾部。

本地全量测试：50 passed；测试隔离 AETHER_ 环境变量并阻断 socket 网络连接，
只允许 Windows asyncio 内部 socketpair 构建。CI 使用相同 Fake 与 MockTransport，
运行质量门禁并保存报告；将稀疏权重设为 1 可以复现失败门禁（exit 1）。

## ADR-004：纠错 Agent 的弃答门与引用校验

日期：2026-10-31。M2 把混合检索接进 LangGraph：`plan → retrieve → grade → rewrite（有界）→
review → generate → verify → answer | abstain`。每个节点是 `AgentState` 上的异步函数。
检查点用 `AsyncSqliteSaver`，`thread_id` 写入 SQLite；`review` 在 `require_review` 时
`interrupt`，另一个进程用同一文件 `Command(resume=...)` 继续。集成测试用子进程而不是只在
同一解释器里重新打开连接。

弃答不是模型的口头声明。grade 只接受同时满足三点的提议：chunk id 属于这次检索、quote 是该
chunk 的逐字子串、quote 覆盖至少 `AETHER_AGENT_MIN_SUPPORT_TERMS`（默认 4）个内容词，且问题里
长度 ≥2 的数字锚点（`k1`、`8.11`、`1994`）都出现在 quote 中。模型说 sufficient 但引用被丢掉时，
判为证据不足。改写只能重排原问题已有的内容词，不能把检索结果里的新词写进下一次查询。
verify 再检查两件事：回答中每个长度 ≥20 的十六进制串都必须属于本轮 `turn_hits`；去掉这些 id 后，
正文必须就是被引 chunk 的原文。任一失败都改写 `answer` 为不含 chunk id 的弃答句，草稿不外泄。
提供者异常同样弃答，原因码 `provider_error`，不把异常文本写进回答。

无密钥时 `create_chat_model` 返回 `EvidenceBoundChatModel`（`FakeChatModel` 子类），只提议能通过
上述门禁的原文。有 `AETHER_LLM_API_KEY` 时才构造 `ChatOpenAI`，`base_url` / 模型 / 超时 / 重试 /
温度都来自 Settings。测试里的预算耗尽、编造 id、额外事实和提供者 500 用 `FakeListChatModel` 或
抛错的 `FakeChatModel` 驱动整张图。另有一个故意声称「足够」并引用 `f`×24 的敌对模型：对抗集和
可回答集的最终回答都不含该 id。

命令：`uv run aether-eval agent --gate`。报告默认写到 `data/eval/agent.json`。
集合是 `data/golden/agent_adversarial.jsonl`（24 条，答案不在 24 篇语料内，其中 4 条带有语料里
不存在的版本号或年份）和 `agent_answerable.jsonl`（15 条，来自 M1 标注且相关 chunk 自身能通过门禁）。
本次离线运行：弃答率 1.0000，可回答率 1.0000，预算耗尽率 1.0000（对抗题都用满 2 次改写、3 次检索），
15/15 回答引用了标注 chunk，编造引用 0。门禁要求对抗集 ≥20、可回答集 ≥10、两个比率 ≥0.80、
预算耗尽率 ≥0.80、编造引用为 0、每条引用完整性为真。

这些数字是 FakeEmbedder 加确定性证据门的回归结果。它证明「语料里没有可引用原文时不会硬答，
有原文时引用必须来自本轮检索」，不证明真实对话模型的语义忠实度。真实模型仍须逐字引用；
做不到就弃答。内容词表和锚点规则是在这批冻结问题上选定的，存在用同一集合观察重叠的偏差，
换语料后要重新看阈值，不能把 4 当成通用语义置信度。
