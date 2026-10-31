"""Unicode 词项与 CJK 字/bigram；标点隔断 bigram，跨平台确定。"""

import re

_CJK = "\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af"
_PARTS = re.compile(rf"[{_CJK}]+|[^\W_{_CJK}]+", re.UNICODE)


def _is_cjk(ch: str) -> bool:
    return any(
        lo <= ord(ch) <= hi
        for lo, hi in (
            (0x3400, 0x9FFF),
            (0x3040, 0x30FF),
            (0xAC00, 0xD7AF),
        )
    )


def tokenize_words(text: str) -> list[str]:
    tokens = []
    for match in _PARTS.finditer(text.lower()):
        word = match[0]
        if all(_is_cjk(c) for c in word):
            tokens.extend(word)
            tokens.extend(word[i : i + 2] for i in range(len(word) - 1))
        else:
            tokens.append(word)
    return tokens


def tokenize_subwords(text: str) -> list[str]:
    tokens = tokenize_words(text)
    out = list(tokens)
    for word in tokens:
        if not all(_is_cjk(c) for c in word):
            padded = f"<{word}>"
            out.extend(padded[i : i + 3] for i in range(len(padded) - 2))
    return out


# 兼容索引调用，不保留全量语料的额外全局缓存。
cached_tokenize_words = tokenize_words
