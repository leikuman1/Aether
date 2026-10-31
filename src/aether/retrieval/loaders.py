"""UTF-8 Markdown/txt 异步加载，路径和顺序跨平台稳定。"""

import asyncio
from pathlib import Path

from aether.retrieval.types import Document

SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt")


class LoaderError(ValueError):
    """语料类型、路径或编码无效。"""


def doc_id_for(path: Path, root: Path) -> str:
    return path.relative_to(root).with_suffix("").as_posix()


async def load_file(path: Path, root: Path) -> Document:
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise LoaderError(f"unsupported file type: {path.suffix}")
    try:
        text = await asyncio.to_thread(path.read_text, encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise LoaderError(f"cannot read UTF-8 document: {path}") from exc
    return Document(doc_id_for(path, root), path.relative_to(root).as_posix(), text)


def _paths(root: Path) -> list[Path]:
    if not root.is_dir():
        raise LoaderError(f"corpus directory missing: {root}")
    paths = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    if not paths:
        raise LoaderError("no supported documents found")
    if any(p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) for p in paths):
        raise LoaderError("corpus symlinks cannot escape the corpus root")
    return paths


async def load_directory(root: Path) -> list[Document]:
    return [await load_file(p, root) for p in await asyncio.to_thread(_paths, root)]
