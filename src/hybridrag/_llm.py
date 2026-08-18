"""Provider agnostic chat client used by the generator and the judge.

Selects a backend from environment variables, preferring Anthropic and falling
back to the free Groq tier:

* ``ANTHROPIC_API_KEY``  -> Anthropic SDK, model from ``ANTHROPIC_MODEL``.
* ``GROQ_API_KEY``       -> Groq via the OpenAI compatible SDK, model from ``GROQ_MODEL``.

If neither key is present, the client is unavailable and callers skip gracefully.
Both SDKs are imported lazily so neither is a hard dependency. The active provider
and model are exposed so every generation can be stamped with what produced it.
"""

from __future__ import annotations

import os

ANTHROPIC_DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LLMClient:
    def __init__(self) -> None:
        self.provider: str | None = None
        self.client = None
        self.model: str | None = None
        self.reason_unavailable: str | None = None
        self._init()

    def _init(self) -> None:
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        groq_key = os.environ.get("GROQ_API_KEY")

        if anthropic_key:
            try:
                import anthropic  # deferred import
                self.client = anthropic.Anthropic(api_key=anthropic_key)
                self.provider = "anthropic"
                self.model = os.environ.get("ANTHROPIC_MODEL", ANTHROPIC_DEFAULT_MODEL)
                return
            except Exception as exc:  # noqa: BLE001
                self.reason_unavailable = f"anthropic backend failed: {exc}"

        if groq_key:
            try:
                from openai import OpenAI  # deferred import; Groq is OpenAI compatible
                self.client = OpenAI(base_url=GROQ_BASE_URL, api_key=groq_key)
                self.provider = "groq"
                self.model = os.environ.get("GROQ_MODEL", GROQ_DEFAULT_MODEL)
                return
            except Exception as exc:  # noqa: BLE001
                self.reason_unavailable = f"groq backend failed: {exc}"

        if self.reason_unavailable is None:
            self.reason_unavailable = "no ANTHROPIC_API_KEY or GROQ_API_KEY set"

    @property
    def available(self) -> bool:
        return self.client is not None

    def complete(self, prompt: str, *, max_tokens: int = 300) -> str:
        """Send a single user prompt and return the model's text reply."""
        if not self.available:
            raise RuntimeError("LLM client is unavailable")
        if self.provider == "anthropic":
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                block.text for block in msg.content if getattr(block, "type", "") == "text"
            ).strip()
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()
