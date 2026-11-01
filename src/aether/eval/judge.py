"""LLM-as-judge：固定 rubric、温度 0。离线词面代理 + fixtures 录制/回放。

judge 只给已经产出的回答打分，不参与生成。三个后端共用同一 prompt 协议
（rubric + TASK + CONTEXT_JSON），因此录制与回放可以在后端之间互换：

- ``LexicalJudge``：无密钥的确定性词面代理。faithfulness=回答逐行是否为被引
  chunk 原文；relevance=回答覆盖标准答案内容词的比例。它是回归下限，不是语义评审。
- ``LLMJudge``：OpenAI 兼容真实模型，固定 rubric、temperature=0（judge 温度
  不读 Settings，强制 0，减少采样变化）。
- ``ReplayingJudge``：按 prompt 指纹回放录制的响应。CI 用它无密钥确定性跑；
  缺 fixture 即报错，管线任何漂移都会迫使重新录制，不做静默回退。

录制：``--judge fake|openai`` 配 ``--record``，把每次 judge 响应按行写入
fixtures 文件（覆盖旧文件），提交入库后 CI 回放。
"""

import hashlib
import json
import math
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from aether.graph.citations import CHUNK_ID_RE
from aether.graph.grading import content_terms
from aether.graph.llm import message_text, parse_json_object

RUBRICS: dict[str, str] = {
    "faithfulness": (
        "你是答案忠实度评审。给定问题、系统回答和被引用 chunk 的原文，判断回答中的"
        "每个事实主张是否都被引用原文逐字或明确支持。评分 0.0-1.0：1.0=全部主张有据，"
        "0.5=约一半有据，0.0=基本无据或与原文矛盾。回答简短或只覆盖部分问题不扣分，"
        '那由相关性评审负责。只返回 JSON：{"score": <0.0-1.0>, "reason": "..."}'
    ),
    "relevance": (
        "你是答案相关性评审。给定问题、系统回答和标准参考答案，判断回答是否回应了"
        "问题的信息需求。评分 0.0-1.0：1.0=完整覆盖参考答案要点，0.5=覆盖约一半，"
        "0.0=答非所问。回答包含参考答案之外的额外正确背景不扣分。"
        '只返回 JSON：{"score": <0.0-1.0>, "reason": "..."}'
    ),
}

LEXICAL_MODEL = "aether-lexical-judge"
JUDGE_TEMPERATURE = 0.0


class JudgeError(Exception):
    """judge 协议或 fixtures 错误。CLI 转成退出码 2，不产生报告。"""


def rubric_fingerprint() -> str:
    """rubric 固定的证据：内容变化会改变指纹，报告与 fixtures 都记录它。"""
    digest = hashlib.sha256()
    for task in sorted(RUBRICS):
        digest.update(task.encode())
        digest.update(RUBRICS[task].encode())
    return digest.hexdigest()


def render_judge_prompt(task: str, payload: dict) -> str:
    if task not in RUBRICS:
        raise JudgeError(f"unknown judge task: {task}")
    body = json.dumps(payload, ensure_ascii=False)
    return f"{RUBRICS[task]}\n\nTASK: judge:{task}\nCONTEXT_JSON:\n{body}"


def prompt_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def _validate_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JudgeError("judge score must be a finite number in [0, 1]")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise JudgeError("judge score must be a finite number in [0, 1]")
    return float(value)


def parse_verdict(raw: str) -> tuple[float, str]:
    """解析 judge 输出 JSON；格式错误是协议失败，不猜分。"""
    try:
        data = parse_json_object(raw)
        score = _validate_score(data.get("score"))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise JudgeError(f"judge output is not {{score, reason}}: {raw[:120]!r}") from exc
    reason = str(data.get("reason", ""))
    return _validate_score(score), reason


class BaseJudge(ABC):
    """score(task, payload) -> (score, reason)。所有后端共享 prompt 与解析。"""

    name: str

    async def score(self, task: str, payload: dict) -> tuple[float, str]:
        prompt = render_judge_prompt(task, payload)
        score, reason = await self._respond(task, prompt, payload)
        return _validate_score(score), reason

    @abstractmethod
    async def _respond(self, task: str, prompt: str, payload: dict) -> tuple[float, str]:
        raise NotImplementedError


class LexicalJudge(BaseJudge):
    """确定性词面代理：忠实度=逐行原文校验，相关性=标准答案内容词覆盖。

    只用于无密钥回归。它验证「回答是否仍由被引原文与标准答案构成」，
    不具备真实模型的语义判断力；数字不能外推为语义质量。
    """

    name = LEXICAL_MODEL

    async def _respond(self, task: str, prompt: str, payload: dict) -> tuple[float, str]:
        if task == "faithfulness":
            return self._faithfulness(payload)
        return self._relevance(payload)

    def _faithfulness(self, payload: dict) -> tuple[float, str]:
        answer = str(payload.get("answer", ""))
        cited = [str(item.get("text", "")) for item in payload.get("citations", [])]
        body = CHUNK_ID_RE.sub(" ", answer.lower())
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        if not lines or not cited:
            return 0.0, "answer or citations empty"
        texts = [_norm(text) for text in cited]
        supported = sum(any(_norm(line) in text for text in texts) for line in lines)
        ratio = supported / len(lines)
        reason = f"{supported}/{len(lines)} answer lines are verbatim spans of cited chunks"
        return ratio, reason

    def _relevance(self, payload: dict) -> tuple[float, str]:
        question = str(payload.get("question", ""))
        answer = str(payload.get("answer", ""))
        reference = str(payload.get("reference_answer", ""))
        target = content_terms(reference) | content_terms(question)
        if not target:
            return 0.0, "no content terms in reference or question"
        covered = content_terms(answer) & target
        ratio = len(covered) / len(target)
        reason = f"{len(covered)}/{len(target)} reference content terms covered by answer"
        return ratio, reason


class LLMJudge(BaseJudge):
    """真实模型评审：固定 rubric，温度由构造方强制为 0。"""

    def __init__(self, llm: BaseChatModel, name: str) -> None:
        self.llm = llm
        self.name = name

    async def _respond(self, task: str, prompt: str, payload: dict) -> tuple[float, str]:
        try:
            message = await self.llm.ainvoke([HumanMessage(content=prompt)])
        except JudgeError:
            raise
        except Exception as exc:
            raise JudgeError("judge invocation failed") from exc
        return parse_verdict(message_text(message))


class RecordingJudge(BaseJudge):
    """包装 fake/llm 后端，按调用顺序录制 prompt 指纹与原始响应。"""

    def __init__(self, inner: BaseJudge) -> None:
        self.inner = inner
        self.name = inner.name
        self.recordings: dict[str, str] = {}

    async def _respond(self, task: str, prompt: str, payload: dict) -> tuple[float, str]:
        score, reason = await self.inner.score(task, payload)
        self.recordings.setdefault(
            prompt_key(prompt), json.dumps({"score": score, "reason": reason}, ensure_ascii=False)
        )
        return score, reason

    def fixture_lines(self) -> list[str]:
        recorded_at = datetime.now(UTC).isoformat(timespec="seconds")
        lines = []
        for key in sorted(self.recordings):
            lines.append(
                json.dumps(
                    {
                        "key": key,
                        "model": self.name,
                        "temperature": JUDGE_TEMPERATURE if self.name != LEXICAL_MODEL else None,
                        "recorded_at": recorded_at,
                        "response": self.recordings[key],
                    },
                    ensure_ascii=False,
                )
            )
        return lines


class ReplayingJudge(BaseJudge):
    """回放 fixtures：prompt 指纹必须逐条命中，缺项即失败。"""

    name = "fixtures-replay"

    def __init__(self, fixtures_path: Path) -> None:
        self.fixtures_path = fixtures_path
        self.responses = self._load(fixtures_path)

    @staticmethod
    def _load(path: Path) -> dict[str, str]:
        if not path.is_file():
            raise JudgeError(f"judge fixtures not found: {path}")
        responses: dict[str, str] = {}
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JudgeError(f"judge fixtures line {number} is not JSON") from exc
            key = str(row.get("key", ""))
            response = str(row.get("response", ""))
            if not key or not response:
                raise JudgeError(f"judge fixtures line {number} lacks key/response")
            if key in responses and responses[key] != response:
                raise JudgeError(f"conflicting judge fixtures for key {key[:12]}")
            responses[key] = response
        if not responses:
            raise JudgeError(f"judge fixtures are empty: {path}")
        return responses

    async def _respond(self, task: str, prompt: str, payload: dict) -> tuple[float, str]:
        key = prompt_key(prompt)
        raw = self.responses.get(key)
        if raw is None:
            raise JudgeError(
                f"missing judge fixture for {task} (key {key[:12]}); "
                f"pipeline changed - re-record with --record"
            )
        return parse_verdict(raw)


def create_judge(settings, mode: str, *, record: bool = False) -> BaseJudge:
    """按 CLI 模式构造 judge 后端（fake/openai）；回放用 ReplayingJudge。

    openai 模式必须有密钥；judge 温度不读 Settings，强制 0 以减少采样变化。
    """
    if mode not in {"fake", "openai"}:
        raise JudgeError(f"unknown judge backend: {mode}")
    if mode == "fake":
        judge: BaseJudge = LexicalJudge()
    else:
        if not settings.llm_api_key:
            raise JudgeError("--judge openai requires AETHER_LLM_API_KEY")
        from langchain_openai import ChatOpenAI

        judge = LLMJudge(
            ChatOpenAI(
                model_name=settings.llm_model,
                openai_api_key=settings.llm_api_key,
                openai_api_base=settings.llm_base_url,
                request_timeout=settings.llm_timeout,
                max_retries=settings.llm_max_retries,
                temperature=JUDGE_TEMPERATURE,
            ),
            settings.llm_model,
        )
    return RecordingJudge(judge) if record else judge
