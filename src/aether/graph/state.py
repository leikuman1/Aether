"""Agent 显式状态。节点只读这个 TypedDict，并返回要更新的字段。"""

from typing import Annotated, TypedDict


def add_messages(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    """跨轮累加会话消息；空更新不覆盖已有历史。"""
    return [*(left or []), *(right or [])]


class Hit(TypedDict):
    chunk_id: str
    doc_id: str
    source: str
    heading: str
    text: str
    score: float


class Support(TypedDict):
    chunk_id: str
    quote: str
    overlap: int


class Attempt(TypedDict):
    query: str
    hit_ids: list[str]
    grade: str
    reason: str


class AgentState(TypedDict, total=False):
    """一轮问答的可检查点状态。messages 跨轮累加，其余字段每轮由入口覆盖。"""

    question: str
    plan: str
    query: str
    previous_queries: list[str]
    rewrite_count: int
    rewrite_budget: int
    min_support_terms: int
    min_quote_chars: int
    require_review: bool
    human_decision: str
    hits: list[Hit]
    turn_hits: list[Hit]
    attempts: list[Attempt]
    supporting: list[Support]
    grade: str
    grade_reason: str
    draft: str
    citations: list[str]
    citation_violations: list[str]
    verified: bool
    verify_reason: str
    decision: str
    abstain_reason: str
    answer: str
    error_code: str
    messages: Annotated[list[dict], add_messages]


class TurnResult(TypedDict):
    thread_id: str
    interrupted: bool
    interrupt: dict | None
    next: list[str]
    decision: str
    answer: str
    abstain_reason: str
    citations: list[str]
    citation_violations: list[str]
    retrieved_ids: list[str]
    rewrite_count: int
    attempts: list[dict]
    messages: list[dict]
    grade: str
    error_code: str
    verified: bool
