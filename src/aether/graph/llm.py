"""对话模型：有密钥时走 OpenAI 兼容接口，否则用证据约束的 FakeChatModel。

离线模型只能复述检索原文里已经通过门禁的句子，不能凭空写引用。
测试里也可以换上 FakeListChatModel 或会抛错的 FakeChatModel，图的节点不关心具体子类。
"""

import json
from typing import Any, override

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from aether.config import Settings
from aether.graph.grading import focused_query, propose_support

_INSTRUCTIONS = {
    "plan": (
        "把用户问题整理成一条检索查询。只能使用问题中已有的词，不要补充外部实体。"
        '只返回 JSON：{"query": "..."}。'
    ),
    "grade": (
        "判断检索片段能否逐字支持回答。sufficient 为真时，supporting.quote 必须是"
        "对应 chunk 文本的连续原文，chunk_id 必须来自给定命中。"
        '只返回 JSON：{"sufficient": true, "supporting": [{"chunk_id": "...", "quote": "..."}],'
        ' "reason": "..."}。证据不够时 sufficient 为 false 且 supporting 为空。'
    ),
    "rewrite": (
        "在不引入新实体、新数字和新术语的前提下重写检索查询，避开已经试过的查询。"
        '只返回 JSON：{"query": "..."}。'
    ),
    "generate": (
        "只允许把 supporting 里的 quote 原文抄进回答，并在每段原文后附上 [chunk_id]。"
        "不要改写、不要补充 quote 之外的事实。没有 supporting 时回答必须为空。"
        '只返回 JSON：{"answer": "..."}。'
    ),
}


class LLMUnavailable(Exception):
    """模型调用失败。节点捕获后弃答，不把异常文本写进回答。"""


def render_prompt(task: str, payload: dict) -> str:
    if task not in _INSTRUCTIONS:
        raise ValueError(f"unknown llm task: {task}")
    body = json.dumps(payload, ensure_ascii=False)
    return f"{_INSTRUCTIONS[task]}\n\nTASK: {task}\nCONTEXT_JSON:\n{body}"


def parse_prompt(text: str) -> tuple[str, dict]:
    task = ""
    for line in text.splitlines():
        if line.startswith("TASK: "):
            task = line.removeprefix("TASK: ").strip()
    marker = "CONTEXT_JSON:\n"
    index = text.find(marker)
    if index < 0:
        raise ValueError("prompt is missing CONTEXT_JSON")
    payload = json.loads(text[index + len(marker) :])
    if not isinstance(payload, dict):
        raise ValueError("prompt context must be a JSON object")
    return task, payload


def parse_json_object(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model output is not a JSON object")
    data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("model output is not a JSON object")
    return data


def message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


async def chat_json(llm: BaseChatModel, task: str, payload: dict) -> dict | None:
    """调用模型并解析 JSON。协议错误返回 None；传输/运行时失败抛 LLMUnavailable。"""
    try:
        message = await llm.ainvoke([HumanMessage(content=render_prompt(task, payload))])
    except Exception as exc:
        raise LLMUnavailable("llm_unavailable") from exc
    try:
        return parse_json_object(message_text(message))
    except (ValueError, json.JSONDecodeError):
        return None


class EvidenceBoundChatModel(FakeChatModel):
    """无密钥时的对话模型：提议和回答都从检索原文拷贝，不能发明引用。"""

    min_support_terms: int = 4
    min_quote_chars: int = 12
    calls: list[str] = Field(default_factory=list)

    @property
    @override
    def _llm_type(self) -> str:
        return "aether-evidence-bound-fake"

    def _respond(self, messages: list[BaseMessage]) -> str:
        task, payload = parse_prompt(message_text(messages[-1]))
        self.calls.append(task)
        if task == "plan":
            return json.dumps({"query": payload.get("question", "")}, ensure_ascii=False)
        if task == "grade":
            supports = propose_support(
                str(payload.get("question", "")),
                list(payload.get("hits") or []),
                min_terms=self.min_support_terms,
                min_quote_chars=self.min_quote_chars,
            )
            return json.dumps(
                {
                    "sufficient": bool(supports),
                    "supporting": [
                        {"chunk_id": item["chunk_id"], "quote": item["quote"]} for item in supports
                    ],
                    "reason": "verbatim-overlap" if supports else "no-supporting-span",
                },
                ensure_ascii=False,
            )
        if task == "rewrite":
            return json.dumps(
                {"query": focused_query(str(payload.get("question", "")))},
                ensure_ascii=False,
            )
        if task == "generate":
            parts = [
                f"{item['quote']} [{item['chunk_id']}]"
                for item in list(payload.get("supporting") or [])
                if item.get("quote") and item.get("chunk_id")
            ]
            return json.dumps({"answer": "\n".join(parts)}, ensure_ascii=False)
        raise ValueError(f"unknown llm task: {task}")

    @override
    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        return self._respond(messages)

    @override
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(content=self._respond(messages))
        return ChatResult(generations=[ChatGeneration(message=message)])


def create_chat_model(settings: Settings) -> BaseChatModel:
    """密钥为空时不创建外部客户端。"""
    if not settings.llm_api_key:
        return EvidenceBoundChatModel(
            min_support_terms=settings.agent_min_support_terms,
            min_quote_chars=settings.agent_min_quote_chars,
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model_name=settings.llm_model,
        openai_api_key=settings.llm_api_key,
        openai_api_base=settings.llm_base_url,
        request_timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
        temperature=settings.llm_temperature,
    )
