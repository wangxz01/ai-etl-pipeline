"""Base agent providing retry logic, token tracking, and result callbacks."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from openai import OpenAI
from pydantic import BaseModel, Field

from src.config import get_settings

logger = logging.getLogger(__name__)


class TokenUsage(BaseModel):
    """Tracks cumulative token consumption."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion


class AgentResult(BaseModel):
    """Generic result container produced by every agent."""

    agent_name: str
    success: bool = True
    payload: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    elapsed_seconds: float = 0.0


# Type alias for progress callbacks
ProgressCallback = Callable[[str, float], None]


class BaseAgent:
    """Abstract base for all ETL agents.

    Provides:
    - OpenAI client construction from shared settings.
    - Retry-with-backoff for LLM calls.
    - Token usage tracking.
    - Optional progress callback support.
    """

    name: str = "base_agent"

    def __init__(
        self,
        max_retries: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        settings = get_settings()
        self._client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
        self._model = settings.deepseek_coder_model
        self._max_tokens = settings.deepseek_max_tokens
        self._temperature = settings.deepseek_temperature
        self._max_retries = max_retries or settings.max_retry_attempts
        self._progress_callback = progress_callback
        self.token_usage = TokenUsage()

    # -------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> AgentResult:
        """Run the agent. Subclasses must override :meth:`_run`."""
        start = time.monotonic()
        try:
            result = self._run(**kwargs)
            result.elapsed_seconds = time.monotonic() - start
            result.token_usage = self.token_usage.model_copy()
            return result
        except Exception as exc:
            logger.exception("%s failed", self.name)
            return AgentResult(
                agent_name=self.name,
                success=False,
                errors=[str(exc)],
                elapsed_seconds=time.monotonic() - start,
            )

    # -------------------------------------------------------------------
    # Override in subclasses
    # -------------------------------------------------------------------

    def _run(self, **kwargs: Any) -> AgentResult:
        raise NotImplementedError

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _call_llm(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Call the LLM with retry-with-exponential-backoff.

        Args:
            messages: OpenAI-style chat messages.
            model: Override the default model.
            temperature: Override the default temperature.

        Returns:
            The assistant's response content.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        attempt = 0
        last_error: Exception | None = None

        while attempt < self._max_retries:
            attempt += 1
            try:
                completion = self._client.chat.completions.create(
                    model=model or self._model,
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=temperature if temperature is not None else self._temperature,
                )
                # Track tokens
                if completion.usage:
                    self.token_usage.add(
                        prompt=completion.usage.prompt_tokens,
                        completion=completion.usage.completion_tokens,
                    )

                content = completion.choices[0].message.content
                if content is None:
                    raise ValueError("LLM returned an empty response")
                return content

            except Exception as exc:
                last_error = exc
                backoff = min(2**attempt, 30)
                logger.warning(
                    "%s: attempt %d/%d failed (%s). Retrying in %ds…",
                    self.name,
                    attempt,
                    self._max_retries,
                    exc,
                    backoff,
                )
                time.sleep(backoff)

        raise RuntimeError(
            f"{self.name}: exhausted {self._max_retries} retries. Last error: {last_error}"
        )

    def _emit_progress(self, message: str, fraction: float) -> None:
        """Emit a progress update via the callback, if one is registered."""
        if self._progress_callback:
            self._progress_callback(f"[{self.name}] {message}", fraction)
