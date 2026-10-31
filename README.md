# Aether

[![CI](https://github.com/leikuman1/Aether/actions/workflows/ci.yml/badge.svg)](https://github.com/leikuman1/Aether/actions/workflows/ci.yml)

可评测、可运维的 **Agentic RAG 引擎**：LangGraph 纠错式检索 Agent + BM25×稠密混合检索
+ CI 内置质量门禁 + 全异步 SSE 服务。不是另一个 RAG chatbot——**答错可检测、质量可度量、
行为可复现**。设计论证见 [PLAN.md](PLAN.md)（含"亮点与为什么不是玩具"对照表）。

## 亮点

- **纠错式 Agent 环**：证据分级 → 有界查询改写 → 引用约束生成 → 自检/弃答，而不是一次检索定生死
- **混合检索可度量**：BM25 + 稠密向量 RRF 融合，recall@k / MRR 在标注集上量化并进 CI 回归
- **评测即门禁**：黄金问答集 + LLM-as-judge + 引用真实性确定性检查，低于阈值 CI 直接红
- **服务化**：全异步 FastAPI，SSE 流式，会话跨进程恢复，逐请求 token/成本核算
- **CI 无密钥**：LLM 调用走 fixtures 回放，`uv run pytest` 离线可跑

## 快速开始

```bash
uv sync                                                  # 安装依赖（含 dev）
uv run pytest                                            # 全部测试：无网络、无密钥
uv run ruff check . && uv run ruff format --check .      # lint + 格式

cp .env.example .env                                     # 可选：填 AETHER_LLM_API_KEY 启用真实 LLM
uv run aether-api                                        # 启动服务 → http://127.0.0.1:8000/docs
```

## M1：可复现的检索演示

```bash
uv run aether-retrieval ingest data/golden/corpus --chunker markdown
uv run aether-retrieval query "RRF 的 k 和 top_k 有什么区别？"
uv run aether-retrieval info
uv run aether-eval retrieval --gate
uv run python -m aether.retrieval.bench --output data/bench.json
```

默认 FakeEmbedder，无网络、无密钥。生产 embedding 设置 `AETHER_EMBEDDING_PROVIDER=openai`，
并通过环境变量提供兼容端点、密钥、模型和输出维度；HTTP 客户端包含有界重试、超时、批量与响应校验。
索引完整持久化到单个 SQLite 文件，恢复时验证 embedding 身份与维度；摄取失败保留旧库。
查询 JSON 包含 chunk 来源、每路得分、阶段时延及候选数；Reranker 协议可替换。

24 篇文档、60 条查询、91 个预标注证据单元；两种分块与三路指标同时输出，
完整 JSON 默认保存到 `data/eval/retrieval.json`，包含逐条命中与类别细分。
默认保守 RRF 权重 16:1，在 FakeEmbedder 下与 BM25 的 recall@5 持平：
fixed 为 **0.9750**、Markdown 为 **0.9833**。等权 RRF 在本集反而下降，
不能把这些离线数字解读为真实 embedding 的语义质量或融合收益。
基准、分块对比、失败基线和配置选择见 [docs/decisions.md](docs/decisions.md)。

CI 在 Windows/Linux 上执行离线单测、检索门禁与 10,000×256 向量基准，
报告保留为 Actions artifacts。改回 `AETHER_RETRIEVAL_SPARSE_WEIGHT=1` 可演示门禁返回非零。

## M2：纠错 Agent

```bash
uv run aether-retrieval ingest data/golden/corpus --chunker markdown
uv run aether-eval agent --gate
uv run python -m aether.graph.session start \
  --thread-id demo --question "BM25 的 k1 和 b 分别控制什么，切块方式变化后要注意哪些统计量？" \
  --checkpoint data/sessions.sqlite --index data/index.db --require-review
uv run python -m aether.graph.session resume \
  --thread-id demo --checkpoint data/sessions.sqlite --index data/index.db --approve
uv run python -m aether.graph.session show \
  --thread-id demo --checkpoint data/sessions.sqlite --index data/index.db
```

图是 `plan → retrieve → grade →（证据不足则有界 rewrite）→ review → generate → verify`，
失败落到 `abstain`。verify 是确定性的：回答里每个 chunk id 必须来自本轮检索，正文必须是这些 chunk 的原文。
无密钥时对话模型是 `EvidenceBoundChatModel`（`FakeChatModel` 子类），只会抄录通过门禁的原文。

在冻结的 24 条对抗题（答案不在语料内）和 15 条可回答题上，`aether-eval agent --gate` 的离线结果是
弃答率 **1.0000**、可回答率 **1.0000**、改写预算耗尽率 **1.0000**、标注 chunk 命中率 **1.0000**、
编造引用 **0**。这证明的是证据门与引用校验，不是真实语义模型的答题质量。口径见
[docs/decisions.md](docs/decisions.md)。

## M3：异步 SSE 服务

`uv run aether-api` 启动单 worker 服务，`/docs` 提供交互式 OpenAPI。
`/healthz` 检查进程；`/readyz` 实际探测数据目录写入（未摄取时也可就绪）；
`/metrics` 暴露 Prometheus 进程累计请求数、模型输入/输出 tokens、USD 成本和阶段时延。
没有索引时聊天返回 `503 index_not_ready`，可先调用摄取接口：

```bash
# Bash 示例；Windows PowerShell 用 $env:AETHER_API_INGEST_ROOT = (Resolve-Path data/golden/corpus).Path
export AETHER_API_INGEST_ROOT="$(pwd)/data/golden/corpus"
uv run aether-api
# 在另一个终端：corpus_dir 使用服务器上的绝对路径，且须在配置的摄取根目录内。
curl -X POST http://127.0.0.1:8000/v1/ingest \
  -H 'Content-Type: application/json' \
  -d '{"corpus_dir":"/absolute/path/to/Aether/data/golden/corpus","chunker":"markdown"}'
curl -N -X POST http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"thread_id":"demo","question":"BM25 的 k1 和 b 分别控制什么，切块方式变化后要注意哪些统计量？"}'
```

SSE 顺序为 `stage`（`stage`、`elapsed_ms`）、`token`（`text`）、`done`（最终结果、
`usage`、`trace_id`）。**本版先完整执行图与引用校验，再发送 SSE**：HTTP 响应头必须在正文前
发出，而最终 usage 与 generate 阶段的失败要执行后才知道。此取舍保证最终核算头、provider 故障
HTTP 503，以及未验证草稿不流出；阶段是执行记录回放，token 是验证后回答的文本分块，
不代表模型原生 token，也不提供生成期间的实时首 token 延迟。取舍详见 ADR-005。

`thread_id` 可省略自动生成，返回 `X-Aether-Thread-Id`；同一 id 的下一轮自动恢复消息。
设置 `require_review:true` 会在生成前暂停，之后发送
`{"thread_id":"demo","resume":{"approved":true}}` 继续；`false` 则明确弃答。
`GET /v1/threads/demo` 可查看恢复状态。索引与检查点分别为
`AETHER_DATA_DIR/index.db`、`sessions.sqlite`，重启不丢失；可与 M2 session CLI 共用。
同 thread 并发返回 409，并发容量满返回 429；摄取时拒绝新聊天，聊天时拒绝摄取，避免在线替换仍打开的 SQLite。
这些协调在单服务进程内生效；请使用默认单 worker，不同时运行外部 CLI 重建同一个索引。

每个聊天响应包含 `X-Aether-Tokens`、`X-Aether-Cost-Usd`、`X-Aether-Usage-Source` 与
`X-Aether-Trace-Id`；成功 SSE 的 `done.usage` 与失败 JSON 的 `usage` 都与这些头一致。
模型返回 usage 时累计 plan/grade/rewrite/generate 的实际输入输出；缺失时明确标记 `estimated`，
采用 UTF-8 字节数 / 4 向上取整的演示口径。Fake 成本固定为 0。真实模型单价由
`AETHER_LLM_INPUT_COST_PER_MILLION` / `AETHER_LLM_OUTPUT_COST_PER_MILLION` 配置，默认 0，
使用 Decimal 核算 USD。范围只含已完成的对话模型调用，不含 embedding、缓存优惠或 provider 内部重试；
失败调用没有 usage 时无法推断其账单。指标随进程重启归零，会话持续保留。

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python -m aether.api.smoke --concurrency 20 --timeout 30 --output data/bench/m3-sse.json
```

冒烟会自行启动独立 uvicorn 进程、临时目录和真实 loopback HTTP，显式禁用 `.env`、密钥与外部 provider，
通过 API 摄取冻结黄金语料，同时释放 20 个请求；检查完整 SSE、引用门、响应头与 `/metrics` 增量。
Windows 11 / Python 3.12.1 / sqlite-vec / Fake 的本地记录：**20/20 成功、0 错误、0 超时，
p50 1287.348ms、p95 1670.649ms、整批 1788.524ms**；23,400 输入 + 5,460 输出 = 28,860 tokens，
成本 $0，指标完全对账。时延包含图、SQLite 检查点和 SSE 收取；不包含启动/摄取，
p95 使用线性插值。它度量离线服务开销，不代表真实 provider 延迟或跨机器 SLO。
CI 在 Windows/Linux 都运行该脚本，保存原始 JSON artifacts。

失败测试用 OpenAI 兼容 MockTransport 在 plan、grade、generate 分别注入真实 HTTP 500，
断言结构化 `503 provider_unavailable`、无异常原文/堆栈/草稿泄漏；另覆盖 504 超时、容量与会话冲突、
摄取失败保留旧库、非零价格及 usage 精确对账。重启集成测试用三个独立解释器经 API 完成暂停、恢复、续聊。

## 配置

环境变量前缀 `AETHER_`，完整见 [.env.example](.env.example)：

| 变量 | 默认 | 说明 |
|---|---|---|
| `AETHER_LLM_BASE_URL` | 智谱 OpenAI 兼容端点 | 任何 OpenAI 协议服务（GLM/OpenAI/DeepSeek/vLLM） |
| `AETHER_LLM_MODEL` | `glm-4.6` | 对话模型 |
| `AETHER_LLM_API_KEY` | 空 | 空 = 不发起外部调用（测试/演练模式） |
| `AETHER_AGENT_REWRITE_BUDGET` | `2` | 证据不足时的查询改写次数 |
| `AETHER_AGENT_MIN_SUPPORT_TERMS` | `4` | 引用原文至少要覆盖的问题内容词数 |
| `AETHER_DATA_DIR` | `./data` | 索引、会话、评测产物 |

## 路线图

M1 混合检索引擎 → M2 纠错 Agent 图 → M3 SSE 服务层 → M4 评测门禁 → M5 观测与发布。
M3 已落地：异步服务、校验后 SSE、跨进程会话恢复、usage 核算与 20 并发门禁。
里程碑 Issue（各带验收标准）见 [Issues](https://github.com/leikuman1/Aether/issues)。

## 结构

```
src/aether/         # api/ 服务层；graph/ Agent 状态机；retrieval/ 混合检索；eval/ 评测（M1+ 逐步落地）
tests/              # 无网络、无密钥
.github/workflows/  # CI：ruff + pytest
```
