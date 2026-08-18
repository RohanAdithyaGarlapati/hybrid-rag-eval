"""Grounded answer generation via the Anthropic API.

The generator turns a question plus retrieved context into an answer that is
instructed to stay grounded in the context and to say it does not know when the
context is insufficient. Like the judge, it uses a pinned model and a versioned
prompt, reads ``ANTHROPIC_API_KEY`` from the environment, and skips gracefully
(returning ``None``) when the key or SDK is absent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

GENERATOR_MODEL = "claude-3-5-sonnet-20241022"
GENERATOR_PROMPT_VERSION = "grounded-answer-v1"

_ANSWER_PROMPT = """You are a careful assistant. Prompt version: {version}.
Answer the QUESTION using ONLY the CONTEXT below. Do not use outside knowledge.
If the CONTEXT does not contain enough information to answer, reply exactly:
"I do not know based on the provided context."
Keep the answer to two or three sentences.

QUESTION:
{question}

CONTEXT:
{context}
"""


@dataclass
class GeneratedAnswer:
    text: str
    model: str
    prompt_version: str


class AnswerGenerator:
    def __init__(
        self,
        model: str = GENERATOR_MODEL,
        prompt_version: str = GENERATOR_PROMPT_VERSION,
        *,
        max_tokens: int = 300,
    ) -> None:
        self.model = model
        self.prompt_version = prompt_version
        self.max_tokens = max_tokens
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

    def generate(self, question: str, context: str) -> GeneratedAnswer | None:
        """Return a grounded answer, or ``None`` when the generator is unavailable."""
        if not self.available:
            return None
        prompt = _ANSWER_PROMPT.format(
            version=self.prompt_version, question=question, context=context
        )
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in msg.content if getattr(block, "type", "") == "text"
        ).strip()
        return GeneratedAnswer(text=text, model=self.model, prompt_version=self.prompt_version)
