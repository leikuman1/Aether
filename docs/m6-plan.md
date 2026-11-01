# M6：独立挑战集与可归因失败评测

状态：规划，尚未实现。基线：`main@9e2d9d2` / `v0.1.0`。实施 Issue：[M6 #11](https://github.com/leikuman1/Aether/issues/11)。

## 为什么做

M1 的 RRF 权重在原集上选定；M2 的可回答题受证据门筛选；M4 的 42 题按 Fake 管线完整覆盖筛选，84 条 judge fixtures 来自词面代理（ADR-002/004/006）。现有数字能证明回归约束，不能证明泛化或纠错收益。M6 把简历故事推进到“能设计不挑成功样本的实验，量化安全性与覆盖率取舍，定位失败，并公开负向结果”。

## 范围与实现边界

- 复用 `eval/agent.py`、`eval/answer.py` 的引用检查与报告约定、`graph.runner.run_turn` 的预算/阶段/callback 接口，以及 M3 的 `RequestUsage` 核算；新增评测编排，不复制生产 Agent。
- 在既有 24 篇冻结语料上新增查询级挑战集；用默认纠错图（rewrite budget=2）与同图关闭改写（budget=0）做成对消融。两路保留 plan/grade/verify，用相同索引、检索配置、模型与独立会话；这只测 rewrite 的边际贡献，不代表“Agent 胜过所有普通 RAG”。
- 默认离线运行真实检索和图，Fake 仅替代外部模型；显式 live 模式走 Settings/OpenAI 兼容协议。保留逐题失败轨迹、时延、用量和实验指纹，补一个成功/过度弃答/不安全回答或故障的终端演示。
- 不做：修改证据门、prompt、默认 RRF/改写策略来追分；新检索器/reranker、多 Agent、前端、实时 SSE、多 worker、APM 或评测平台。后续修复单开 Issue；本里程碑只建立可信实验闭环。

## 亮点与为什么不是玩具：验收清单

### 1. 数据先冻结，避免按结果选题

- [ ] 新增 `data/golden/challenge.jsonl`，至少 60 条新查询：可回答 ≥40（术语、语义改写、多证据、概念对比各 ≥10）；不可回答 ≥20（术语相似但事实/数字不存在 ≥10、问题仅部分可由语料支持 ≥10）。数量由预先指定的主类别计数，不重复凑数。
- [ ] 每条有唯一 ID、预期 answer/abstain、类别、人工标注理由；可回答题列出全部必要证据原文、doc/chunk ID，不可回答题说明缺失事实及检查语料的依据。部分可回答题要求对完整问题弃答，不把答一半算成功。
- [ ] 与 M1/M2/M4 查询做 Unicode NFKC + 大小写 + 空白归一化去重，并人工检查近义重述；不根据当前检索排名、词面门是否接受或模型输出筛选。记录标注者/复核方式；单人复核就如实说明，无外部专家一致性声明。
- [ ] 先独立提交数据、标注说明和冻结 manifest（语料/查询/标签 SHA-256、预注册配置/阈值），再首次运行。现有黄金集仅作开发/回归；挑战集曝光后只能称冻结回归集，调参需另建未来保留集。测试覆盖重复/空数据、缺类别、过期 chunk ID 和证据不在原文，错误退出 2。

### 2. 成对消融与不会被“全部弃答”骗过的指标

- [ ] 每题运行 budget=2/0 两路，独立 thread/checkpoint，验证除预算与运行标识外配置一致、无跨题历史。输出逐题决策、最终回答/引用、本轮检索及每次 attempt 的 IDs、改写次数、终止原因。
- [ ] 固定口径：`answerable_coverage` = 可回答题中“最终作答且包含全部 gold 引用并通过确定性引用校验”的比例；`unsafe_answer_rate` = 不可回答题中最终作答的比例；`correct_abstain_rate` = 不可回答题中证据不足/预算耗尽导致弃答的比例。provider 错误、超时、协议错误单列 operational failure，仍留在原分母，不算正确弃答；语义指标不能用 citation ID 真实性替代。
- [ ] 两路分别报告总体/类别 n、分子分母、上述三率、引用完整率、故障率、原因分布。主质量目标预注册为 coverage ≥0.80、unsafe ≤0.10、correct abstain ≥0.90、引用完整率 =1.00、编造引用数 =0；均为本集目标，不是生产 SLO。
- [ ] 报告两路 coverage/unsafe 的成对差值、覆盖成功的 gained/lost/tied 题 ID；按 query 成对 bootstrap（2,000 次、seed=2026）给差值 95% 区间，单路比例附 Wilson 95% 区间。小样本区间不能证明跨域收益；无增益或退步照实保留，不要求“纠错必胜”。
- [ ] `--gate quality` 只按绝对质量阈值判断，不能因基准也差而放行；`--gate regression` 验证冻结离线基准的逐题决策、引用和失败分类，无新增不安全回答/编造引用，报告自身质量 PASS/FAIL。故意降低覆盖、增加不安全回答、全部弃答和全部 provider 故障均有反例测试；全部弃答不能通过 quality。

### 3. 可归因、可复现且失败不被吞掉

- [ ] 每条以 `run_id + case_id + variant + trace_id` 关联 JSON 报告与 JSONL 阶段日志；记录 stage 顺序/时延、usage source、token/成本与最慢阶段。复用现有 callback/on_stage；阶段成本之和与已完成对话调用核算一致，非零 Fake usage/价格替身验证对账。
- [ ] 用互斥、固定优先级分类：operational failure → 引用校验失败 → 证据未全检出 → 已检出仍弃答 → 已答但 gold 未全覆盖 → 不可回答题错误作答 → 成功；明确证据“检出”按本轮所有 attempts 的并集，部分可回答负例不套正例检索漏失规则。不足以断言根因时称观测分类，不把它包装成因果解释；保留原始 reason/attempts 供复核。
- [ ] 每个 variant 输出整轮 p50/p95（含图/检查点，不含建索引和启动）、已完成对话调用 token/USD、额外改写开销；embedding 与失败调用未知账单沿用 M3 排除口径。Fake 成本为 0 且时延不代表真实 provider SLO。
- [ ] manifest 包含 git SHA、Python/依赖锁指纹、语料/数据/配置/prompt/rubric 指纹、模式、生成/embedding/judge 身份（未使用 judge 则明示）。两次离线运行除时间、随机 ID、时延等观测字段外，决策/引用/分类/指标一致；不要求真实模型逐字确定。
- [ ] Fake/MockTransport 覆盖 provider 500/超时、畸形输出、伪造/跨题引用、部分证据缺失、无检索结果、预算耗尽；故障保留逐题记录与已完成调用用量，不把异常原文、密钥、未验证草稿写入最终回答/公共 artifacts。损坏数据/不兼容基准/缺失回放 fixture 退出 2；已完成但门禁未过退出 1；通过退出 0。部分失败的报告明确 incomplete，不能成功退出。

### 4. 命令、CI 与演示

以下为 **M6 待实现的 CLI 合同**，当前 v0.1.0 尚无 `challenge` 子命令或这些测试文件。

```bash
# 数据/指标公式/成对隔离/门禁反例/异常/日志对账/CLI 退出码
uv run pytest -q tests/test_challenge_eval.py tests/test_challenge_cli.py

# 默认无网络、无密钥；即使本机 .env 有 provider 也强制 Fake
uv run aether-eval challenge --mode offline --gate regression --baseline data/golden/challenge_baseline.json --output data/eval/challenge.json
# 独立质量判定；失败必须写报告并退出 1
uv run aether-eval challenge --mode offline --gate quality --output data/eval/challenge_quality.json

uv run pytest
uv run ruff check .
uv run ruff format --check .
```

- [ ] `offline` 为默认模式，忽略本机密钥和外部 provider；socket 拦截测试保证不出网。CI 在 Windows/Linux 跑新测试与 regression 命令，保留完整报告/manifest/阶段日志 artifacts；时延不作跨机器门禁。原 M1–M5 门禁继续全绿。
- [ ] 入库首次离线基准及生成来源；不自动刷新 baseline/阈值/数据以修复红灯。若首轮 challenge quality 未达标，M6 可交付评测能力，但必须发布 FAIL 与逐项后续修复 Issue，README 不得宣称质量已过关；CI 的 regression PASS 与 quality FAIL 并列可见。
- [ ] README 给两路指标、区间、故障率、成本边界和 ≥3 条 case 的定位命令/终端记录（成功、错误弃答、故障或不安全回答）；必要时用明确标注的注入故障展示，不能冒充自然观测。更新 ADR，解释原集选择偏差、消融边界和首轮负向结果。

### 5. 真实模型证据边界

- [ ] 提供显式命令 `uv run aether-eval challenge --mode live --gate quality --output data/eval/challenge_live.json`，模型/端点/密钥只经 `AETHER_` Settings。缺少真实生成或真实 embedding 配置时退出 2，禁止静默回退 Fake；测试只用 MockTransport，CI 不调用真实服务。
- [ ] 有凭据/预算时，按冻结协议在全挑战集上跑两路，记录模型版本、时间、配置、失败和实际用量；若追加 LLM-judge，固定 rubric，注明同模型评审偏差，不能用它覆盖确定性失败。若无条件实测，README/报告明确 `live_validation=not_run`，不伪造 fixtures、不把 Fake 数字写成语义质量或真实 judge 稳定性。真实模型实测不是离线评测工程交付的隐含前提。

## 残余风险与完成定义

同语料新查询仍有主题与作者偏差，且规划者看过既有实现；它是查询级挑战，不是独立领域/盲测。词面门和严格抽取式 verify 可能导致真实模型大量弃答，M6 只量化、不放松它们。成对差值仍受 provider 非确定性影响，温度 0 不消除波动；单次 live 结果不作显著性结论。小类别样本的区间会很宽，完整引用覆盖也不等于语义完整。

完成 M6 要求数据协议、离线评测/反例测试、可复现报告、CI 与演示全部交付；绝对质量与真实模型验证状态分别报告。若质量失败，后续实现修复另立里程碑；此计划 PR 仅添加计划并链接实施 Issue，不关闭它。
