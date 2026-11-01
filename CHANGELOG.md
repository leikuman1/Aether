# Changelog

## [0.1.0] - 2026-11-01

第一个可运行、可评测、可观测的版本。M0–M5。

### Added

- 项目骨架：uv、FastAPI `/healthz`、ruff、无网络测试与 CI。
- 混合检索：BM25 与稠密向量 RRF、两种分块、sqlite-vec，以及检索 recall@k / MRR 门禁。
- 纠错 Agent：证据分级、有界改写、引用约束、弃答，以及可恢复的 SQLite 会话。
- 异步 SSE 服务：阶段与 token 事件、用量响应头、`/metrics`、单进程并发控制。
- 答案评测：引用真实性、离线 judge 回放，以及 CI 质量门禁。
- 结构化日志：每个请求一个 `trace_id`，阶段行带时延和对话模型成本；`GET /v1/traces/{trace_id}` 标出最慢阶段。日志是重启后的依据，进程内只留 256 条。
- `uv run aether-demo`：默认无密钥跑通摄取、提问和检索评测。`--live` 才使用已配置的模型。
- k6 压测入口 `python -m aether.api.loadtest`。v0.1.0 的离线记录是 20 并发、40 请求、错误率 0、p50 511.638 ms、p95 1152.648 ms。原始数字在 `docs/loadtest-v0.1.0.json`。

### Notes

- 测试和默认演示不访问网络，也不读取密钥。
- 压测延迟包含排队和完整 SSE 正文，不是 provider SLO，也不是单个图节点的耗时。
- 成本只统计已完成的对话模型调用。Fake 路径成本为 0。
- 发布 tag 不在功能分支上创建。合并到 `main` 之后执行：

```bash
git fetch origin main
git tag -a v0.1.0 origin/main -m "Aether v0.1.0"
git push origin v0.1.0
```

这条命令必须在合并完成、且 `origin/main` 仍指向该发布提交时执行。
