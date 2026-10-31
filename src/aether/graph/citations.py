"""引用完整性：回答里的 chunk id 必须属于本轮检索，正文必须是这些 chunk 的原文。"""

import re

# 本库 chunk id 是 sha256[:20]。更长的十六进制串也视为引用，避免只截断前 20 位后漏检。
CHUNK_ID_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{20,}(?![0-9a-f])")


def extract_chunk_ids(text: str) -> list[str]:
    """按出现顺序去重。连续十六进制长度不足 20 的词不算 chunk id。"""
    found: list[str] = []
    seen: set[str] = set()
    for match in CHUNK_ID_RE.finditer(text.lower()):
        chunk_id = match.group(0)
        if chunk_id not in seen:
            seen.add(chunk_id)
            found.append(chunk_id)
    return found


def fabricated_citation_ids(answer: str, allowed_ids: set[str]) -> list[str]:
    """出现在回答中、但不在本轮真实检索结果里的 chunk id。"""
    allowed = {item.lower() for item in allowed_ids}
    return [chunk_id for chunk_id in extract_chunk_ids(answer) if chunk_id not in allowed]


def _normalize(text: str) -> str:
    return " ".join(text.split())


def extractive_remainder(answer: str, quotes: list[str]) -> str:
    """删掉引用标记和已允许的原文后还剩下的文字。空字符串表示回答是抽取式的。"""
    body = CHUNK_ID_RE.sub(" ", answer.lower())
    body = body.replace("[", " ").replace("]", " ")
    remaining = _normalize(body)
    for quote in sorted(quotes, key=len, reverse=True):
        remaining = remaining.replace(_normalize(quote.lower()), " ")
        remaining = _normalize(remaining)
    return remaining.strip(" \t.;；。,，、：:！!？?（）()\"'`")


def is_extractive(answer: str, quotes: list[str]) -> bool:
    if not quotes:
        return False
    if not extract_chunk_ids(answer):
        return False
    normalized_answer = _normalize(CHUNK_ID_RE.sub(" ", answer.lower()))
    if any(_normalize(quote.lower()) not in normalized_answer for quote in quotes):
        return False
    return extractive_remainder(answer, quotes) == ""
