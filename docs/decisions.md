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

## ADR-005：M3 的 SSE 提交时机、核算与单进程协调

日期：2026-10-31。复用 `graph.runner.run_turn` / M1 `ingest_directory`；runner 增加可选的
callback 与节点更新观察函数，通过 `astream(stream_mode="updates")` 记录实际节点顺序和时延。
M2 CLI 与会话格式保持兼容，状态仍由 AsyncSqliteSaver 管理。

HTTP 响应头在正文前发送。以下要求不能同时在同一普通 HTTP 响应成立：一边调用 provider
一边立即流出 token、最终 usage 写入响应头、生成中失败仍改成 HTTP 503、引用校验前不流出草稿。
本版明确选择先执行图，再提交 SSE 响应。阶段事件是执行记录回放，token 事件是最终已验证回答
的固定字符分块，不宣称 provider 的实时 token 或低首 token 延迟。若后续需要实时进度，应增加
独立作业/订阅协议，最终核算用终止事件或结果接口；不能在已发送响应头后修改状态码与最终头值。

核算通过每请求独立 LangChain callback 收集所有已成功返回模型调用的 `usage_metadata`；包含
plan/grade/rewrite/generate。缺失时用 UTF-8 字节数除以 4 向上取整，显式 `source=estimated`。
Fake 成本为 0，真实模型价格从 Settings 的输入/输出每百万 token USD 单价读取，Decimal 累加。
头、done/错误 JSON 和 Prometheus 使用同一 snapshot，指标包含失败请求已成功完成的调用。
口径不含 embedding、缓存折扣、provider 未返回 usage 的失败请求与内部重试的未知账单。
指标按进程累计；重启归零，线程消息仍在磁盘。测试用每次输入 100 / 输出 25 tokens、每百万
输入 $2 / 输出 $8 的替身，单轮三次调用固定为 375 tokens / $0.0012，非零断言避免零成本空通过。

部署以单 worker 为边界。活动 thread 集合在单事件循环内同步检查并占用：同 thread 返回 409，
不同 thread 并行，超过配置容量返回 429。摄取与活动聊天互斥，不跨 await 留检查窗口；每请求
关闭索引连接后才释放占用，解决 Windows 原子替换打开 SQLite 文件的限制。摄取根目录通过
配置允许，校验目录与符号链接真实路径，不允许任意服务器路径。M1 全量原子重建失败仍保留旧库。
不支持多个 worker 或外部 CLI 与服务同时写同一索引。

`python -m aether.api.smoke` 启动独立 uvicorn 子进程，用已绑定的动态 socket 避免取空端口的竞态，
临时目录、显式禁用 dotenv 与外部 provider，再通过真实 TCP 同时发起 20 条 SSE。默认每操作
30 秒截止，验证完整阶段、引用校验成功、token 拼接、逐请求头和指标差分，以及 SQLite 文件存在。
Windows 11 / Python 3.12.1 的记录：20/20、0 错误/超时、p50=1287.348ms、p95=1670.649ms、
整批=1788.524ms（启动和摄取不计入）。p95 为 20 条完整请求耗时线性插值；28,860 估算 tokens、
成本 $0，与指标增量一致。该数字只代表 Fake 路径的服务开销。

离线测试覆盖真实兼容协议的 MockTransport 500（plan/grade/generate）、结构化 503、超时后
资源释放、错误响应对账、摄取失败保留旧索引、无索引 503、会话冲突与审核拒绝。跨进程集成用
三个独立解释器创建 FastAPI lifespan，分别调用 API 暂停、重启恢复、重启续聊，并断言历史累积。
超时取消留下的普通 pending 节点与人工 interrupt 分开处理：下一问题可重置未完成轮，已完成历史仍保留。
索引打开在工作线程中执行，取消请求仍等待线程归还连接并关闭 searcher/embedder 后才释放占用；
因此超时响应可能包含必要的清理时间，不声称能强杀磁盘 I/O。资源回归测试用线程屏障验证此所有权。
CI 在 Windows/Linux 运行离线全量测试及真实 TCP 冒烟，将报告随评测 artifacts 保留。

## ADR-006：M4 答案门禁、judge 回放与稳定性边界

日期：2026-11-01。不增加依赖，不改服务层。答案评测复用现有检索与 `run_turn`，
每条答案实际执行整张图，再由独立确定性检查验证引用 chunk 存在于 canonical 语料、
属于本轮检索，正文引用与结果 citations 一致；要求通过率 100%。无引用的弃答不能
伪造答案质量：另要求可回答率 ≥0.8、标准引用覆盖率 ≥0.8、样本数 ≥30。
faithfulness 与 answer-relevance 对已回答题分别取宏平均，阈值均为 0.8。
无效引用直接记两个维度 0 分且确定性门禁失败，不调用 judge、不因查找不存在 chunk 崩溃。

42 条黄金题从 M1 已冻结标签筛选（exact/paraphrase/multi/contrast=10/10/13/9），
标准答案复用证据原文；筛选使用当前 Fake 管线的完整 gold 覆盖，存在选择偏差，
这批数据是回归集，不是独立保留集。词面代理的相关性定义也经过同集观察，不能宣称泛化。
CI 的 84 条 fixtures 由词面 Fake judge 录制，来源字段明确标记；它们不是供应商模型实测。
真实 LLMJudge 走 Settings 中 OpenAI 兼容端点，固定 rubric、强制 temperature=0，
测试以 FakeListChatModel 替代调用，既验证协议也验证录制/回放，不调用真实网络。
温度 0 仅减少采样变化，不保证供应商确定性；本次没有真实 judge 稳定性结论。

两维度 rubric 内容与完整输入共同参与 prompt SHA-256：缺 fixture 退出 2，不做自动回退；
NaN/Infinity、布尔值、数字字符串、越界分数全部拒绝，不能夹成及格分。
`--record --repeat N` 保存首轮 verdict，与报告采用的首轮一致，后续轮用于测偏差。
回放模式强制清空生成密钥并使用离线 embedding，避免本机 .env 意外引入外部请求。

稳定性实验（Windows 11 / Python 3.12.1 / sqlite-vec，2026-11-01）：

| judge 来源 | 输入数量 | 每输入重复 | faithfulness 均值 | relevance 均值 | 两维度最大偏差 |
|---|---:|---:|---:|---:|---:|
| 词面 Fake judge | 42 | 3 | 1.000000 | 0.913101 | 0 / 0 |
| 入库 fixtures 回放 | 42 | 3 | 1.000000 | 0.913101 | 0 / 0 |

逐输入相对首轮的最大绝对差作为波动指标；对确定性 Fake/回放要求精确 0。
真实 judge 后续实验应保留模型版本和原始报告，以最大偏差 ≤0.05 为建议接受范围，
本次并未验证该建议。回放单次本机墙钟约 3.97 秒（含 CLI 启动和整张图），
CI 评测步骤设置 `timeout-minutes: 3`，依赖安装单独计时。

```bash
uv run aether-eval answer --judge fake --repeat 3 --gate --output data/eval/answer_fake_stability.json
uv run aether-eval answer --judge replay --repeat 3 --gate --baseline data/golden/answer_baseline.jsonl
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

报告保存完整答案/引用/逐题原因、rubric 指纹、数据指纹、配置和阈值；历史 JSONL 追加
指标摘要，第二次运行对比上一条；`--baseline` 支持完整 JSON 报告或 JSONL 最后一条，
CI 使用入库基准以便全新 checkout 也能输出差值。对比仅报告退步项，硬门槛决定退出码；
历史不自动与 GitHub 跨运行合并，Actions artifacts 保留该次报告与历史。更新基准需评审。

## ADR-007：M5 的 trace、压测口径与离线演示

日期：2026-11-01。不引入 APM，也不新增运行时依赖。日志是一行一个 JSON 对象，
字段固定带 `trace_id`。聊天的每个图节点另有 `elapsed_ms` 和该节点相对上一节点的
对话模型 token / `cost_usd` 增量。增量来自现有 `RequestUsage` snapshot，不新算一套账单。
请求结束的 `chat_complete` 写出 `slowest_stage`。时延相同的阶段取更晚的一条。

`GET /v1/traces/{trace_id}` 读进程内 256 条环形缓冲。缓冲在重启或被挤出后消失。
这时用 `aether-trace --log` 扫 JSON 日志；只要还有 `stage` 行，就以它们为准，
不采信摘要里可能过期的字段。没有图节点的请求（例如 `/healthz`）记一条 `http` 阶段，
聊天在进入图之前失败则记 `request`。摄取记 `ingest`，成本为 0：embedding 仍不在 M3 的账本里。

中间件是纯 ASGI，只补 `X-Aether-Trace-Id`，不把 SSE 正文读进内存。
`http_access.elapsed_ms` 是整段 HTTP，含把 SSE 写完；`slowest_stage` 仍是节点耗时，
两者故意不是同一个数。`uv run aether-api` 在 lifespan 里打开 JSON 日志，
因为 uvicorn 会在导入应用之后重置 logging。测试默认不打开，避免把日志从根 logger 上摘掉。

压测用 k6，而不是 hey。聊天要为每次请求生成不同的 `thread_id`，固定请求体会在第二条得到 409。
k6 1.3.0、20 VU、40 次共享迭代、Fake 模型的这次记录：错误率 0，
`http_req_duration` p50 = 511.6381225 ms，p95 = 1152.6481994 ms。
这个时间包含排队和读完 SSE，不是节点耗时，也不是 provider SLO。
k6 把失败样本放在 Rate 的 `passes`，成功样本放在 `fails`；报告里的 `errors` 已按失败次数解释。
循环后再打一枪，端点与日志的最慢阶段都是 verify / 7.483 ms。原始 JSON 在
`docs/loadtest-v0.1.0.json`。

`uv run aether-demo` 默认清空密钥并强制 Fake embedding，然后走摄取、一次提问、检索门禁。
`--live` 才使用 Settings 中的 provider；答案 judge 仍要显式 `--judge openai` 才会出网。
CI 跑这条默认命令，不安装 k6。v0.1.0 的 git tag 不在功能分支上创建。
