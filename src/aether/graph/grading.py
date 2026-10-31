"""证据是否够格：词项重叠 + 数字锚点必须原样出现在引用片段中。

这是弃答门，不依赖模型自报「足够」。模型只能提议引用；提议里的 chunk id
必须来自本轮检索，quote 必须是该 chunk 的逐字子串，并且覆盖问题中的数字锚点
与足够的内容词。改写不得引入原问题里没有的内容词，避免把语料词注入查询后伪造支持。
"""

from aether.retrieval.tokenizer import tokenize_words

# 功能词不参与重叠计数。领域词（bm25、k1、efsearch 等）故意不在表内。
_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "to",
        "and",
        "or",
        "in",
        "on",
        "for",
        "with",
        "without",
        "from",
        "by",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "what",
        "how",
        "why",
        "when",
        "where",
        "which",
        "who",
        "not",
        "no",
        "into",
        "over",
        "under",
        "than",
        "then",
        "so",
        "if",
        "at",
        "does",
        "do",
        "did",
        "can",
        "should",
        "would",
        "could",
        "about",
        "between",
        "after",
        "before",
        "during",
        "within",
        "using",
        "use",
        "used",
        "via",
        "vs",
        "each",
        "other",
        "的",
        "了",
        "和",
        "与",
        "或",
        "在",
        "是",
        "不",
        "也",
        "就",
        "都",
        "而",
        "及",
        "被",
        "把",
        "对",
        "从",
        "为",
        "以",
        "其",
        "这",
        "那",
        "什么",
        "如何",
        "怎么",
        "哪些",
        "是否",
        "可以",
        "需要",
        "应该",
        "以及",
        "如果",
        "因为",
        "所以",
        "但是",
        "一个",
        "没有",
        "不是",
        "我们",
        "它们",
        "他们",
        "进行",
        "使用",
        "通过",
        "时候",
        "之后",
        "之前",
        "其中",
        "这个",
        "那个",
        "分别",
        "注意",
        "处理",
        "影响",
        "作用",
        "区别",
        "不同",
        "相同",
        "问题",
        "方式",
        "对于",
        "不能",
        "不会",
        "不要",
        "已经",
        "可能",
        "或者",
        "还是",
        "因此",
        "同时",
        "然后",
        "这样",
        "那种",
        "直接",
        "当前",
        "主要",
    }
)


def _cjk(token: str) -> bool:
    return all("\u3400" <= char <= "\u9fff" for char in token)


def content_terms(text: str) -> set[str]:
    """内容词：拉丁词长度≥3，或含数字的标识（如 k1）；中文只计二元词。"""
    terms: set[str] = set()
    for token in tokenize_words(text.lower()):
        if token in _STOP or token.isdigit():
            continue
        if _cjk(token):
            if len(token) == 2:
                terms.add(token)
        elif len(token) >= 3 or any(char.isdigit() for char in token):
            terms.add(token)
    return terms


def anchors(text: str) -> set[str]:
    """长度≥2 且含数字的词。问题里的版本号、年份、k1 必须出现在引用原文中。"""
    found = set()
    for token in tokenize_words(text.lower()):
        if len(token) >= 2 and any(char.isdigit() for char in token):
            found.add(token)
    return found


def quote_overlap(question: str, quote: str) -> int:
    return len(content_terms(question) & content_terms(quote))


def quote_supports(
    question: str,
    quote: str,
    *,
    min_terms: int,
    min_quote_chars: int,
) -> int | None:
    """返回重叠词数；引用不够长、缺锚点或重叠不足时返回 None。"""
    if min_terms < 1 or min_quote_chars < 1:
        raise ValueError("min_terms and min_quote_chars must be positive")
    if len(quote) < min_quote_chars:
        return None
    lowered = quote.lower()
    if any(anchor not in lowered for anchor in anchors(question)):
        return None
    overlap = quote_overlap(question, quote)
    if overlap < min_terms:
        return None
    return overlap


def propose_support(
    question: str,
    hits: list[dict],
    *,
    min_terms: int,
    min_quote_chars: int,
) -> list[dict]:
    """从检索命中里挑出能逐字引用、且通过门禁的至多一条证据。"""
    accepted = []
    for hit in hits:
        quote = str(hit.get("text", ""))
        overlap = quote_supports(
            question,
            quote,
            min_terms=min_terms,
            min_quote_chars=min_quote_chars,
        )
        if overlap is None:
            continue
        accepted.append(
            {"chunk_id": str(hit["chunk_id"]).lower(), "quote": quote, "overlap": overlap}
        )
    accepted.sort(key=lambda item: (-item["overlap"], item["chunk_id"]))
    return accepted[:1]


def accept_support(
    question: str,
    hits: list[dict],
    proposal: dict | None,
    *,
    min_terms: int,
    min_quote_chars: int,
) -> list[dict]:
    """丢掉编造的 id、非逐字 quote，以及过不了锚点/重叠门禁的提议。"""
    if not isinstance(proposal, dict):
        return []
    items = proposal.get("supporting")
    if not isinstance(items, list):
        return []
    by_id = {str(hit["chunk_id"]).lower(): hit for hit in hits}
    accepted = []
    for item in items:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", "")).lower()
        quote = item.get("quote", "")
        hit = by_id.get(chunk_id)
        if hit is None or not isinstance(quote, str) or quote not in str(hit.get("text", "")):
            continue
        overlap = quote_supports(
            question,
            quote,
            min_terms=min_terms,
            min_quote_chars=min_quote_chars,
        )
        if overlap is None:
            continue
        accepted.append({"chunk_id": chunk_id, "quote": quote, "overlap": overlap})
    accepted.sort(key=lambda item: (-item["overlap"], item["chunk_id"]))
    deduped = []
    seen: set[str] = set()
    for item in accepted:
        if item["chunk_id"] in seen:
            continue
        seen.add(item["chunk_id"])
        deduped.append(item)
    return deduped[:1]


def constrain_query(question: str, proposed: str, previous: list[str]) -> str:
    """改写只能重排原问题里已有的内容词，不能从检索结果里借词。"""
    allowed = content_terms(question)
    candidate = proposed.strip()
    if candidate and candidate not in previous and content_terms(candidate) <= allowed:
        return candidate
    focused = " ".join(sorted(allowed))
    if focused and focused not in previous:
        return focused
    original = question.strip()
    return original


def focused_query(question: str) -> str:
    return " ".join(sorted(content_terms(question)))
