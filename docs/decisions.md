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

Linux 与 CI Windows 数字由同一提交的 Actions 双平台矩阵补入。若任一平台扩展不可用或
达不到门槛，统一默认 NumPy；两者都通过才将默认改为 sqlite-vec。
NumPy 后端不加载 sqlite-vec 扩展，保持可移植退路。性能样本不作 CI 稳定性承诺，
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
