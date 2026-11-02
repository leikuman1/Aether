# Aether

[![CI](https://github.com/leikuman1/Aether/actions/workflows/ci.yml/badge.svg)](https://github.com/leikuman1/Aether/actions/workflows/ci.yml)

## 项目介绍

Aether 是可评测、可运维的 **Agentic RAG 引擎**，基于 LangGraph、混合检索与异步 FastAPI，
将文档检索、证据检查、回答生成和质量回归串成一条可复现的流程。

## 解决什么问题

- 检索一次就回答容易遗漏证据：证据不足时有界改写查询，仍不足则弃答。
- 回答看似可信却难以核查：引用必须来自本轮检索，引用正文须与原文一致。
- 改动后质量是否退步不清楚：用黄金集、确定性引用检查与评分回放建立 CI 门禁。
- 服务出问题难以定位：按请求记录阶段时延、token 与成本，通过 trace ID 追踪。

## 怎么使用（快速开始）

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)，在仓库根目录执行：

```bash
uv sync
uv run aether-demo
```

演示默认强制使用离线 Fake 模型和 embedding，无需密钥，完成摄取、提问与检索门禁。
Fake / fixture 结果用于验证流程与回归，不代表真实模型的语义质量。

启动 API 前，先建立示例文档索引：

```bash
uv run aether-retrieval ingest data/golden/corpus --chunker markdown
uv run aether-api
```

打开 <http://127.0.0.1:8000/docs>，调用 `POST /v1/chat`，例如：

```json
{"thread_id":"demo","question":"RRF 的 k 和 top_k 有什么区别？"}
```

服务默认单 worker；SSE 在完整执行与引用校验后发送，阶段事件为执行记录回放。
配置统一使用 `AETHER_` 环境变量，参见 [.env.example](.env.example)。真实模型通过 OpenAI 兼容协议接入；
填写 `AETHER_LLM_*` 后可用 `uv run aether-demo --live`，真实 embedding 另需配置 provider、模型与维度。

开发检查：

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## 有哪些功能

- **混合检索**：BM25 + 稠密向量，通过 RRF 融合；支持 Markdown 分块与 SQLite 索引持久化。
- **纠错 Agent**：证据分级、有界查询改写、人工审核暂停/恢复、引用校验与弃答。
- **异步服务**：SSE 响应、持久化会话、并发限制及结构化错误。
- **质量门禁**：检索、Agent 与答案评测；离线 Fake / fixture 驱动测试，CI 无需密钥或外部模型调用。
- **运行观测**：JSON 阶段日志、trace 查询、Prometheus 指标及逐请求 token / 成本核算。

## 核心架构

```text
FastAPI async
├─ POST /v1/ingest → load → chunk → embed → SQLite（BM25 / sqlite-vec）
├─ POST /v1/chat → LangGraph → SSE（stage / token / done）
│  plan → retrieve → grade → review → generate → verify → answer | abstain
│                     └─ 证据不足 → 有界 rewrite → retrieve
└─ /healthz /readyz /metrics /v1/traces/{trace_id}
```

架构与验收见 [PLAN.md](PLAN.md)；证据门、SSE 与成本核算口径见 [技术决策](docs/decisions.md)。

## 评测与演示入口

```bash
uv run aether-eval retrieval --gate
uv run aether-eval agent --gate
uv run aether-eval answer --judge replay --gate --repeat 3 --baseline data/golden/answer_baseline.jsonl
uv run aether-demo --eval all
```

数据、标注与局限见 [黄金集说明](data/golden/README.md)；演示见 [终端实录](docs/demo-v0.1.0.txt)。
2026-11-01 Linux / Python 3.12.3 / k6 1.3.0 / Fake 模型压测：20 VU、40 请求、错误率 **0**，
p50 **511.638 ms**、p95 **1152.648 ms**，含排队与完整 SSE 收取；不代表真实 provider 延迟或跨机器 SLO。
复现与原始结果见 [压测说明](docs/loadtest.md) 和 [报告 JSON](docs/loadtest-v0.1.0.json)。

## 里程碑状态

- **M0–M1 已完成**：工程骨架、摄取、混合检索与检索评测。
- **M2–M3 已完成**：纠错 Agent、可恢复会话、异步 SSE 与用量核算。
- **M4–M5 已完成**：答案质量门禁、trace、压测与 v0.1.0。
- **M6 规划中**：独立挑战集、改写消融与失败归因，见 [M6 验收协议](docs/m6-plan.md)。

`v0.1.0` tag 已指向 `main@9e2d9d2`，尚未创建 GitHub Release；发布记录见 [CHANGELOG.md](CHANGELOG.md)。
