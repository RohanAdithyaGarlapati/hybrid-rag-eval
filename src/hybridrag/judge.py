"""LLM as judge scoring, provider agnostic.

Scores an answer for faithfulness (is every claim supported by the context) and
answer relevance (does it address the question) against a versioned prompt, using
whichever backend is configured (Anthropic, or the free Groq tier). Reads keys from
the environment and skips gracefully (returning ``None``) when no backend exists.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ._llm import LLMClient

PROMPT_VERSION = "faithfulness-relevance-v1"

_JUDGE_PROMPT = """You are a strict evaluation judge. Prompt version: {version}.
Given a QUESTION, the retrieved CONTEXT, and an ANSWER, rate on a 0.0 to 1.0 scale:
1. faithfulness: is every claim in the ANSWER supported by the CONTEXT (1.0) or does
   it introduce unsupported claims (lower)?
2. answer_relevance: does the ANSWER directly address the QUESTION (1.0) or drift (lower)?
Respond with ONLY a JSON object: {{"faithfulness": <float>, "answer_relevance": <float>}}.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
"""


@dataclass
class JudgeScores:
    faithfulness: float
    answer_relevance: float
    provider: str
    model: str
    prompt_version: str


class AnthropicJudge:
    """Kept for name stability; scores via any configured backend, not only Anthropic."""

    def __init__(self, prompt_version: str = PROMPT_VERSION, *, max_tokens: int = 200) -> None:
        self.prompt_version = prompt_version
        self.max_tokens = max_tokens
        self._llm = LLMClient()

    @property
    def available(self) -> bool:
        return self._llm.available

    @property
    def reason_unavailable(self) -> str | None:
        return self._llm.reason_unavailable

    @property
    def provider(self) -> str | None:
        return self._llm.provider

    @property
    def model(self) -> str | None:
        return self._llm.model

    def score(self, question: str, context: str, answer: str) -> JudgeScores | None:
        """Return scores, or ``None`` when no backend is available (graceful skip)."""
        if not self.available:
            return None
        prompt = _JUDGE_PROMPT.format(
            version=self.prompt_version, question=question, context=context, answer=answer
        )
        text = self._llm.complete(prompt, max_tokens=self.max_tokens)
        return self._parse(text)

    def _parse(self, text: str) -> JudgeScores | None:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return JudgeScores(
            faithfulness=float(data.get("faithfulness", 0.0)),
            answer_relevance=float(data.get("answer_relevance", 0.0)),
            provider=self._llm.provider,
            model=self._llm.model,
            prompt_version=self.prompt_version,
        )
