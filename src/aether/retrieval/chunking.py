"""固定窗口与 Markdown 结构分块；ID 包含策略参数及正文指纹。"""

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol

from aether.retrieval.types import Chunk, Document

_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def make_chunk_id(doc_id: str, strategy: str, ordinal: int, text: str = "") -> str:
    return hashlib.sha256(f"{doc_id}\0{strategy}\0{ordinal}\0{text}".encode()).hexdigest()[:20]


class Chunker(Protocol):
    @property
    def name(self) -> str: ...

    def split(self, doc: Document) -> list[Chunk]: ...


def _chunk(doc: Document, strategy: str, ordinal: int, heading: str, text: str) -> Chunk:
    return Chunk(
        make_chunk_id(doc.doc_id, strategy, ordinal, text),
        doc.doc_id,
        doc.source,
        ordinal,
        heading,
        text,
    )


@dataclass(frozen=True)
class FixedWindowChunker:
    size: int = 400
    overlap: int = 64

    def __post_init__(self) -> None:
        if self.size <= 0 or not 0 <= self.overlap < self.size:
            raise ValueError("require size > 0 and 0 <= overlap < size")

    @property
    def name(self) -> str:
        return f"fixed-{self.size}-{self.overlap}"

    def split(self, doc: Document) -> list[Chunk]:
        text = doc.text.replace("\r\n", "\n").strip()
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.size, len(text))
            if end < len(text):
                window = text[start:end]
                cut = max(window.rfind(e) for e in "。！？；;.!?\n")
                if cut >= self.size // 2:
                    end = start + cut + 1
            body = text[start:end].strip()
            if body:
                chunks.append(_chunk(doc, self.name, len(chunks), doc.doc_id, body))
            if end == len(text):
                break
            start = max(start + 1, end - self.overlap)
        return chunks


def _sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = [("", [])]
    stack: list[tuple[int, str]] = []
    fence: str | None = None
    for line in text.splitlines():
        marker = _FENCE.match(line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif (
                token[0] == fence[0]
                and len(token) >= len(fence)
                and not line.lstrip()[len(token) :].strip()
            ):
                fence = None
            sections[-1][1].append(line)
            continue
        heading = _HEADING.match(line) if fence is None else None
        if heading:
            level, title = len(heading[1]), heading[2].strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            sections.append((" > ".join(t for _, t in stack), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(lines).strip()) for h, lines in sections if "\n".join(lines).strip()]


@dataclass(frozen=True)
class MarkdownChunker:
    max_size: int = 480

    def __post_init__(self) -> None:
        if self.max_size < 100:
            raise ValueError("max_size must be >= 100")

    @property
    def name(self) -> str:
        return f"markdown-{self.max_size}"

    def split(self, doc: Document) -> list[Chunk]:
        chunks = []
        for heading, body in _sections(doc.text):
            # 标题保留为检索上下文；超长正文确定性硬切，尺寸预算包括标题。
            prefix = f"{heading}\n" if heading else ""
            if len(prefix) >= self.max_size:
                prefix = prefix[: self.max_size // 2] + "\n"
            budget = self.max_size - len(prefix)
            for start in range(0, len(body), budget):
                text = prefix + body[start : start + budget]
                chunks.append(_chunk(doc, self.name, len(chunks), heading, text))
        return chunks


def get_chunker(name: str) -> Chunker:
    if name == "fixed":
        return FixedWindowChunker()
    if name == "markdown":
        return MarkdownChunker()
    match = re.fullmatch(r"fixed-(\d+)-(\d+)", name)
    if match:
        return FixedWindowChunker(int(match[1]), int(match[2]))
    match = re.fullmatch(r"markdown-(\d+)", name)
    if match:
        return MarkdownChunker(int(match[1]))
    raise ValueError(f"unknown chunker: {name}")
