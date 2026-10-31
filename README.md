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

## 配置

环境变量前缀 `AETHER_`，完整见 [.env.example](.env.example)：

| 变量 | 默认 | 说明 |
|---|---|---|
| `AETHER_LLM_BASE_URL` | 智谱 OpenAI 兼容端点 | 任何 OpenAI 协议服务（GLM/OpenAI/DeepSeek/vLLM） |
| `AETHER_LLM_MODEL` | `glm-4.6` | 对话模型 |
| `AETHER_LLM_API_KEY` | 空 | 空 = 不发起外部调用（测试/演练模式） |
| `AETHER_DATA_DIR` | `./data` | 索引、会话、评测产物 |

## 路线图

M1 混合检索引擎 → M2 纠错 Agent 图 → M3 SSE 服务层 → M4 评测门禁 → M5 观测与发布。
里程碑 Issue（各带验收标准）见 [Issues](https://github.com/leikuman1/Aether/issues)。

## 结构

```
src/aether/         # api/ 服务层；graph/ Agent 状态机；retrieval/ 混合检索；eval/ 评测（M1+ 逐步落地）
tests/              # 无网络、无密钥
.github/workflows/  # CI：ruff + pytest
```
