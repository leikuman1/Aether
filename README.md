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
M2 已落地：状态机、弃答、SqliteSaver 会话与对抗集门禁。
里程碑 Issue（各带验收标准）见 [Issues](https://github.com/leikuman1/Aether/issues)。

## 结构

```
src/aether/         # api/ 服务层；graph/ Agent 状态机；retrieval/ 混合检索；eval/ 评测（M1+ 逐步落地）
tests/              # 无网络、无密钥
.github/workflows/  # CI：ruff + pytest
```
