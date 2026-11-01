# Aether · 项目计划

> 定位：可评测、可运维的 **Agentic RAG 引擎与服务**。Agent 为主、RAG 为辅——检索只是 Agent
> 可编排的能力之一。核心卖点不是"能回答"，而是**答错可检测、质量可度量、行为可复现**。

- 仓库：<https://github.com/leikuman1/Aether>
- 技术栈：Python 3.12 / uv / FastAPI / LangGraph / rank-bm25 + sqlite-vec
- 状态：M0–M5 已完成并发布 `v0.1.0`（`main@9e2d9d2`）；M6 规划中，见 [§5](#5-里程碑与验收)

## 1. 一句话故事

LangGraph 纠错式检索 Agent（证据分级 → 有界改写 → 引用约束生成 → 自检/弃答）
+ BM25×稠密向量 RRF 混合检索 + 跑在 CI 里的评测门禁 + 全异步 SSE 流式服务。

## 2. 亮点与为什么不是玩具（必读）

市面"又一个 RAG chatbot"的共同形态：query → embed → top-k → 拼 prompt → LLM → 回答。
一个脚本跑通，没有度量、没有失败语义、没有并发与观测。下表逐项对齐，并给出**验收方式**——
每个亮点都必须能被一条命令或一个测试证明，而不是自述。

| # | 维度 | 玩具缺什么 | Aether 补什么 | 怎么验收证明 |
|---|------|-----------|--------------|-------------|
| 1 | 检索质量 | 单路稠密检索，好坏全靠感觉 | BM25+稠密 RRF 融合、可切换分块与 rerank，标注集量化 recall@k / MRR | `uv run aether-eval retrieval` 输出三路对比指标表；hybrid recall@5 ≥ 任一单路，进 CI 回归门禁 |
| 2 | 答案有据 | 生成不受检索质量约束，幻觉无界 | 证据分级决定"继续/改写/弃答"，生成带引用约束，引用必须来自本轮真实检索到的 chunk | 对抗集（答案不在语料内）弃答率 ≥80%；引用真实性有确定性断言测试 |
| 3 | 评测体系 | "看起来对" | 黄金问答集 + LLM-as-judge（faithfulness/相关性）+ 确定性检查，评测进 CI | `uv run aether-eval answer` 出 JSON 报告；CI 无密钥跑 fixtures 回放，低于阈值即红 |
| 4 | 服务形态 | notebook / 同步脚本，单人单次 | 全异步 FastAPI、SSE 流式（token+阶段事件）、checkpointer 会话跨进程恢复、逐请求 token/成本核算 | 20 并发 SSE 冒烟；thread_id 重启续聊；X-Aether-\* 响应头与 /metrics 对账一致 |
| 5 | 可观测性 | 黑盒，坏了不知道坏在哪 | 每请求 trace_id，分阶段时延/成本的结构化日志与指标端点 | 演示：单请求定位最慢阶段；压测 p50/p95 存 README |
| 6 | 失败语义 | 无超时/重试/降级 | 有界改写预算、provider 故障结构化降级（503+错误码）、预算耗尽明确弃答 | 预算耗尽、provider 500 均有 mock 单测覆盖 |

**反玩具三问**（每个新功能必须至少命中一条，否则不做）：

1. 有没有数字（指标/阈值），而不是"更好了"？
2. 有没有对抗用例或失败路径的测试？
3. 运行行为是否可观测、可复现？

不满足 = 纯胶水 / 纯 UI，一律砍掉。

## 3. 架构

```
FastAPI（async，SSE）
 ├─ POST /v1/chat ── LangGraph 纠错环：plan → retrieve → grade
 │                    ↑ grade 不达标 → 查询改写重试（有界预算）↑
 │                    └→ generate（引用约束）→ verify → answer | abstain
 ├─ POST /v1/ingest ─ 摄取：load → chunk → embed → sqlite-vec / BM25 索引
 └─ /healthz /readyz /metrics ─ 观测：trace_id、分阶段时延、token 成本

评测（独立于服务，进 CI）：黄金集 + recall@k/MRR + LLM-judge + 引用真实性 → 质量门禁
```

LLM 走 OpenAI 兼容协议（默认智谱 GLM，环境变量可切 OpenAI / DeepSeek / 本地 vLLM）；
测试与 CI 用 FakeChatModel / FakeEmbedder，**无密钥、无网络**。

## 4. 技术选型

| 选型 | 理由 |
|---|---|
| uv + Python 3.12 | 锁文件可复现；CI `uv sync --frozen` |
| FastAPI + Uvicorn | ASGI 全异步，SSE 原生支持，OpenAPI 自带 |
| LangGraph | 显式状态机 + checkpointer：节点是可单测的纯函数式变换，支持中断/恢复/时间旅行调试 |
| rank-bm25 + sqlite-vec | 无外部服务的混合检索：当前规模（<10 万 chunk）够用且 CI 友好；M1 首日做基准验证，不达标退 numpy 内存索引（决策记入 docs/decisions.md） |
| 自研轻量评测 | RAGAS 类工具链过重；recall@k/MRR/faithfulness 自算可控，judge 用 rubric + 固定种子 |
| fixtures 回放 | LLM 响应录制/回放，CI 无密钥确定性跑评测 |

## 5. 里程碑与验收

对应 GitHub Issues（标题前缀 [M1]–[M6]）；M0–M5 已完成，M6 的实施清单见下方链接。

| 里程碑 | 内容 | 硬验收 |
|---|---|---|
| M0 | 骨架：uv 项目、FastAPI /healthz、测试、ruff、CI | CI 绿；`uv run pytest` 无网络可跑 |
| M1 | 摄取 + 混合检索 + 检索评测 | 标注集 ≥50 query；hybrid recall@5 ≥ 任一单路；指标表可复现 |
| M2 | 纠错 Agent 图 + 弃答语义 | 对抗集 ≥20 弃答率 ≥80%；预算耗尽有测试；会话可恢复 |
| M3 | 服务层：SSE + 会话 + 成本核算 | 20 并发冒烟；跨进程续聊；成本头与 /metrics 对账 |
| M4 | 评测体系 + CI 质量门禁 | 答案黄金集 ≥30；引用真实性 =100%；CI 阈值阻断 |
| M5 | 观测 + 压测 + 发布 v0.1.0 | 单请求定位最慢阶段；压测 p50/p95 入 README |
| M6（规划） | 独立挑战集与可归因失败评测 | 新查询 ≥60；rewrite 2/0 成对消融；质量/回归分开判定；逐题失败可复现 |

### M6：独立挑战集与可归因失败评测

实施 Issue：[#11](https://github.com/leikuman1/Aether/issues/11)。完整范围、命令、指标定义与风险见 [M6 验收协议](docs/m6-plan.md)。
M4 按 Fake 可回答性选择的 42 题仍作为回归集；M6 先冻结新题与阈值再跑，
补足“评测自身是否可信、纠错何时有用、失败在哪一层”的简历证据。

**范围**：复用现有检索、图、引用门和观测回调，比较默认图（rewrite budget=2）与同图
关闭改写（budget=0）。不改生产策略追分，不加 UI、新检索器、多 Agent 或实时 SSE。

**亮点与为什么不是玩具（M6 验收）**：

- [ ] **数字**：新查询 ≥60（可回答 ≥40、近似/部分证据但不可完整回答 ≥20），先标注冻结、与原题去重；
  两路报告总体/分组分母、完整证据作答覆盖率、不安全回答率、正确弃答率及 95% 区间。
  预注册质量目标分别为 ≥0.80、≤0.10、≥0.90，引用完整率 =100%、编造引用 =0；
  保留负向消融，不以“必须有增益”挑题。
- [ ] **失败路径**：全部弃答不能过质量门，provider 故障不算正确弃答；伪造/跨题引用、
  部分证据、预算耗尽、500/超时和协议错误都有 Fake/MockTransport 反例测试与退出码断言。
- [ ] **观测与复现**：逐题关联 run/case/variant/trace、attempts、失败分类、阶段时延和用量；
  冻结数据/配置/模型指纹；两次离线结果一致；Windows/Linux CI 回归门与 artifacts，≥3 条终端定位演示。
- [ ] **诚实边界**：quality 与 regression 分开显示；首轮质量失败要发布失败记录并链接后续修复 Issue，
  不刷新基准掩盖。live 显式启用；无实测标记 `not_run`，Fake 不充当真实语义质量结论。

验收命令（M6 待实现）：

```bash
uv run aether-eval challenge --mode offline --gate regression --baseline data/golden/challenge_baseline.json --output data/eval/challenge.json
```

另以 `--gate quality` 独立检查质量。新测试、全量 pytest/ruff、live 命令与退出码合同见完整协议。
残余风险：同语料/作者偏差、小样本宽区间、严格抽取式证据门导致过度弃答、真实 provider 非确定性；
M6 测量这些边界，不能据此宣称跨域泛化或 Agent 普遍优于单次 RAG。

## 6. 风险与对策

- **LLM-judge 不稳定**：rubric 固定、温度 0、多样本平均，阈值留缓冲。
- **sqlite-vec 平台差异**：M1 首日双平台验证；退路 numpy（该规模暴力计算足够）。
- **范围蔓延**：不满足反玩具三问的需求进"不做"清单。

## 7. 非目标（明确不做）

- 管理后台 / 花哨前端 / 多租户计费等 CRUD 工作
- 分布式向量库运维（Qdrant 集群等）——当前规模不构成合理工程决策
- 通用 Agent 平台化：只做"私有语料有据问答"这一件事
