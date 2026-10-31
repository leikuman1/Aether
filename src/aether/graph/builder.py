"""把节点接成 plan → retrieve → grade →（改写 | 审阅）→ generate → verify | abstain。"""

from langchain_core.language_models.chat_models import BaseChatModel
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


def build_graph(llm: BaseChatModel, searcher: Searcher, *, retrieval_k: int) -> StateGraph:
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

    graph = StateGraph(AgentState)
    graph.add_node("plan", _plan)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("grade", _grade)
    graph.add_node("rewrite", _rewrite)
    graph.add_node("review", review)
    graph.add_node("generate", _generate)
    graph.add_node("verify", verify)
    graph.add_node("abstain", abstain)
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
