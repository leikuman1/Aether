# AGENTS.md — 在本仓库干活的约定（人与 AI 适用）

## 命令

```bash
uv sync                            # 安装（含 dev）；改依赖后 uv lock 并提交锁文件
uv run pytest                      # 测试：必须无网络、无密钥全绿
uv run ruff check .                # lint
uv run ruff format .               # 格式化（提交前跑）
uv run aether-api                  # 本地起服务 → http://127.0.0.1:8000/docs
```

## 结构

- `src/aether/api/` FastAPI 服务层；`graph/`（M2）Agent 状态机；`retrieval/`（M1）混合检索；`eval/`（M4）评测
- `tests/` 单测+集成；LLM 一律 Fake/Fixture 驱动，禁止真实网络调用
- `data/` 运行产物（gitignore）；黄金集放 `data/golden/` 入库；技术决策记 `docs/decisions.md`

## 硬规矩

1. Python 3.12、src 布局、pydantic v2；I/O 一律 async
2. 外部 LLM/embedding 只走 OpenAI 兼容协议 + `aether.config.Settings`；禁止散落 hardcode
3. 新功能必须带测试与可量化指标（见 PLAN.md「反玩具三问」），否则不合并
4. 配置只经环境变量（`AETHER_` 前缀）；密钥与 `.env` 不进库
5. 提交信息用 Conventional Commits；合并前 CI 必须绿
