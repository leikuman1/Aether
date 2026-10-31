"""请求级模型 usage 与进程级 Prometheus 计数。缺失 usage 时显式标记估算。"""

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from aether.config import Settings


def estimate_tokens(text: str) -> int:
    # 离线代理口径，不声称是任何模型的 tokenizer。
    return math.ceil(len(text.encode("utf-8")) / 4)


class RequestUsage(BaseCallbackHandler):
    run_inline = True

    def __init__(self, settings: Settings):
        self.settings = settings
        self.input_tokens = 0
        self.output_tokens = 0
        self.estimated = False
        self.calls: dict[UUID, int] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id: UUID, **kwargs: Any):
        self.calls[run_id] = estimate_tokens("".join(m.text for batch in messages for m in batch))

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any):
        prompt_estimate = self.calls.pop(run_id, 0)
        message = response.generations[0][0].message
        usage = getattr(message, "usage_metadata", None)
        if usage is not None:
            self.input_tokens += usage["input_tokens"]
            self.output_tokens += usage["output_tokens"]
        else:
            self.estimated = True
            self.input_tokens += prompt_estimate
            self.output_tokens += estimate_tokens(message.text)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any):
        # 失败调用没有 provider usage，无法承诺账单；只统计已返回的调用。
        self.calls.pop(run_id, None)

    def snapshot(self) -> dict:
        offline = not self.settings.llm_api_key
        cost = Decimal(0)
        if not offline:
            cost = (
                Decimal(self.input_tokens) * Decimal(str(self.settings.llm_input_cost_per_million))
                + Decimal(self.output_tokens)
                * Decimal(str(self.settings.llm_output_cost_per_million))
            ) / Decimal(1_000_000)
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "cost_usd": format(cost, "f"),
            "source": "estimated" if self.estimated else "provider",
        }


@dataclass
class Metrics:
    success: int = 0
    error: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: Decimal = Decimal(0)
    stages: dict[str, tuple[int, float]] = field(default_factory=dict)

    def record(self, usage: dict, *, success: bool) -> None:
        self.success += int(success)
        self.error += int(not success)
        self.input_tokens += usage["input_tokens"]
        self.output_tokens += usage["output_tokens"]
        self.cost += Decimal(usage["cost_usd"])

    def stage(self, name: str, elapsed_ms: float) -> None:
        count, seconds = self.stages.get(name, (0, 0.0))
        self.stages[name] = count + 1, seconds + elapsed_ms / 1000

    def render(self) -> str:
        lines = [
            "# TYPE aether_chat_requests_total counter",
            f'aether_chat_requests_total{{status="success"}} {self.success}',
            f'aether_chat_requests_total{{status="error"}} {self.error}',
            "# TYPE aether_tokens_total counter",
            f'aether_tokens_total{{direction="input"}} {self.input_tokens}',
            f'aether_tokens_total{{direction="output"}} {self.output_tokens}',
            "# TYPE aether_cost_usd_total counter",
            f"aether_cost_usd_total {self.cost:f}",
            "# TYPE aether_stage_duration_seconds summary",
        ]
        for name, (count, seconds) in sorted(self.stages.items()):
            lines.extend(
                [
                    f'aether_stage_duration_seconds_count{{stage="{name}"}} {count}',
                    f'aether_stage_duration_seconds_sum{{stage="{name}"}} {seconds:.9f}',
                ]
            )
        return "\n".join(lines) + "\n"
