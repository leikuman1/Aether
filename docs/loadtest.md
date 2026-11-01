# M5 压测记录

命令（需要本机有 k6；仓库不下载、不提交 k6 二进制）：

```bash
uv run python -m aether.api.loadtest \
  --concurrency 20 \
  --iterations 40 \
  --timeout 60 \
  --output docs/loadtest-v0.1.0.json
```

脚本是 [scripts/k6/chat.js](../scripts/k6/chat.js)。每个 iteration 使用 `load-${VU}-${ITER}` 作为
`thread_id`。hey 的请求体是固定的，第二条会碰到同一个 thread 并返回 409，所以不用 hey。
没有 k6 时把 `--driver builtin` 换成内置 httpx 窗口；已提交的数字不是那条路径。

服务由压测进程自己拉起：临时目录、空密钥、Fake embedding、不读 `.env`。
先摄取 `data/golden/corpus`，再以 20 个 VU 共享 40 次迭代调用 `POST /v1/chat`。
问题与 README 的演示问题相同。

## 2026-11-01 结果

环境：Linux 6.12.94+、Python 3.12.3、k6 1.3.0（`commit/5870e99ae8`）、Fake 对话模型、sqlite-vec。

| 项 | 值 |
|---|---:|
| 并发 | 20 |
| 请求 | 40 |
| 错误率 | 0 |
| 错误数 | 0 |
| p50 | 511.6381225 ms |
| p95 | 1152.6481994 ms |
| 平均 | 555.837035725 ms |
| 最小 | 28.602263 ms |
| 最大 | 1161.828798 ms |

`latency_ms` 是 k6 的 `http_req_duration`，单位毫秒，包含排队和把 SSE 正文读完。
它不是图节点耗时，也不是真实 provider 的延迟。换机器要重跑，不要把这组数当成 SLO。

k6 把 Rate 指标的失败样本记在 `passes`，成功样本记在 `fails`。
因此 JSON 里 `http_req_failed.fails = 40` 表示 40 次成功。
规范化之后的 `errors` 和 `error_rate` 使用失败次数，这次是 0 和 0.0。

循环结束之后的样本请求 `1e7745f953074546a9421cf9200054fa`：
`GET /v1/traces/{trace_id}` 与服务器 JSON 日志都给出最慢阶段 `verify`、7.483 ms。
`log_matches_endpoint` 为 true。样本不在被计时的 40 次里面，所以它不解释 p50。

原始报告：[loadtest-v0.1.0.json](loadtest-v0.1.0.json)。其中的 `base_url` 是当时的临时端口，已经关掉。
