"""Grounded answer generation, provider agnostic.

Turns a question plus retrieved context into an answer that must stay grounded in
the context and say it does not know when the context is insufficient. It uses a
versioned prompt and whichever backend is configured (Anthropic, or the free Groq
tier as a fallback), reading keys from the environment and skipping gracefully
(returning ``None``) when no backend is available.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._llm import LLMClient

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
    provider: str
    model: str
    prompt_version: str


class AnswerGenerator:
    def __init__(self, prompt_version: str = GENERATOR_PROMPT_VERSION, *, max_tokens: int = 300) -> None:
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

    def generate(self, question: str, context: str) -> GeneratedAnswer | None:
        """Return a grounded answer, or ``None`` when no backend is available."""
        if not self.available:
            return None
        prompt = _ANSWER_PROMPT.format(
            version=self.prompt_version, question=question, context=context
        )
        text = self._llm.complete(prompt, max_tokens=self.max_tokens)
        return GeneratedAnswer(
            text=text,
            provider=self._llm.provider,
            model=self._llm.model,
            prompt_version=self.prompt_version,
        )
