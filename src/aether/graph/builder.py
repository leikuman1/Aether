"""把节点接成 plan → retrieve → grade →（改写 | 审阅）→ generate → verify | abstain。"""

import asyncio
import time
from collections.abc import Awaitable, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph

from aether.graph.nodes import (
    Searcher,
    abstain,
    generate,
    grade,
    plan,
    retrieve,
    review,
    rewrite,
    route_after_grade,
    route_after_plan,
    route_after_review,
    route_after_rewrite,
    route_after_verify,
    verify,
)
from aether.graph.state import AgentState

NODE_NAMES = (
    "plan",
    "retrieve",
    "grade",
    "rewrite",
    "review",
    "generate",
    "verify",
    "abstain",
)


def build_graph(
    llm: BaseChatModel,
    searcher: Searcher,
    *,
    retrieval_k: int,
    on_stage_error: Callable[[str, float], Awaitable[None]] | None = None,
) -> StateGraph:
    if retrieval_k <= 0:
        raise ValueError("retrieval_k must be positive")

    async def _plan(state: AgentState) -> dict:
        return await plan(state, llm)

    async def _retrieve(state: AgentState) -> dict:
        return await retrieve(state, searcher, k=retrieval_k)

    async def _grade(state: AgentState) -> dict:
        return await grade(state, llm)

    async def _rewrite(state: AgentState) -> dict:
        return await rewrite(state, llm)

    async def _generate(state: AgentState) -> dict:
        return await generate(state, llm)

    def observed(name, node):
        async def invoke(state: AgentState) -> dict:
            started = time.perf_counter()
            try:
                return await node(state)
            except GraphBubbleUp:
                # 人工审阅 interrupt 是正常控制流，不是失败阶段。
                raise
            except (Exception, asyncio.CancelledError):
                if on_stage_error is not None:
                    await on_stage_error(name, (time.perf_counter() - started) * 1000)
                raise

        return invoke if on_stage_error is not None else node

    graph = StateGraph(AgentState)
    graph.add_node("plan", observed("plan", _plan))
    graph.add_node("retrieve", observed("retrieve", _retrieve))
    graph.add_node("grade", observed("grade", _grade))
    graph.add_node("rewrite", observed("rewrite", _rewrite))
    graph.add_node("review", observed("review", review))
    graph.add_node("generate", observed("generate", _generate))
    graph.add_node("verify", observed("verify", verify))
    graph.add_node("abstain", observed("abstain", abstain))
    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan", route_after_plan, {"retrieve": "retrieve", "abstain": "abstain"}
    )
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"review": "review", "rewrite": "rewrite", "abstain": "abstain"},
    )
    graph.add_conditional_edges(
        "rewrite", route_after_rewrite, {"retrieve": "retrieve", "abstain": "abstain"}
    )
    graph.add_conditional_edges(
        "review", route_after_review, {"generate": "generate", "abstain": "abstain"}
    )
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", route_after_verify, {"end": END, "abstain": "abstain"})
    graph.add_edge("abstain", END)
    return graph
