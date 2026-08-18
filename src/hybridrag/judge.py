"""LLM as judge scoring via the Anthropic API.

The judge scores an answer for faithfulness (is every claim supported by the
context) and answer relevance (does it address the question) against a pinned model
and a versioned prompt. It reads ``ANTHROPIC_API_KEY`` from the environment and
reports itself unavailable, skipping gracefully, when the key or the SDK is absent.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

# Pinned model and prompt version so scores are reproducible and auditable.
JUDGE_MODEL = "claude-3-5-sonnet-20241022"
PROMPT_VERSION = "faithfulness-relevance-v1"

_FAITHFULNESS_PROMPT = """You are a strict evaluation judge. Prompt version: {version}.
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
    model: str
    prompt_version: str


class AnthropicJudge:
    def __init__(self, model: str = JUDGE_MODEL, prompt_version: str = PROMPT_VERSION) -> None:
        self.model = model
        self.prompt_version = prompt_version
        self._client = None
        self._reason_unavailable: str | None = None
        self._init_client()

    def _init_client(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            self._reason_unavailable = "ANTHROPIC_API_KEY is not set"
            return
        try:
            import anthropic  # deferred import
        except Exception as exc:  # noqa: BLE001
            self._reason_unavailable = f"anthropic SDK not importable: {exc}"
            return
        try:
            self._client = anthropic.Anthropic(api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            self._reason_unavailable = f"failed to construct client: {exc}"

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def reason_unavailable(self) -> str | None:
        return self._reason_unavailable

    def score(self, question: str, context: str, answer: str) -> JudgeScores | None:
        """Return scores, or ``None`` when the judge is unavailable (graceful skip)."""
        if not self.available:
            return None
        prompt = _FAITHFULNESS_PROMPT.format(
            version=self.prompt_version, question=question, context=context, answer=answer
        )
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")
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
            model=self.model,
            prompt_version=self.prompt_version,
        )
