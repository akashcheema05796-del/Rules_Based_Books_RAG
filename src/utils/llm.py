"""
LLM client wrapper — supports Anthropic Claude and OpenAI GPT.

Provider is selected by the `provider` constructor arg (or auto-detected from
the model name). Both expose the same `generate()` interface so the rest of
the codebase is provider-agnostic.

Spec §2, §10: seed=42, temp=0, backoff, cost tracking.
"""

import logging
from pathlib import Path
from typing import Optional

from src.utils.backoff import with_backoff
from src.utils.cache import get_cache
from src.utils.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


def _detect_provider(model: str) -> str:
    """Infer provider from model name if not specified explicitly."""
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"):
        return "openai"
    return "anthropic"


class LLMClient:
    """Provider-agnostic LLM client with caching, cost tracking, and backoff.

    Supports:
      - Anthropic  : claude-sonnet-4-5-20241022, claude-haiku-*, etc.
      - OpenAI     : gpt-4o, gpt-4o-mini, o1-*, etc.

    Prompt caching (cache_system=True) is honoured only for Anthropic;
    OpenAI calls silently ignore the flag.
    """

    def __init__(self, model="gpt-4o", temperature=0, max_tokens=4096,
                 seed=42, cache_dir=".cache", cost_tracker=None,
                 provider: str | None = None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.cache = get_cache("llm", cache_dir)
        self.cost_tracker = cost_tracker
        self.provider = provider or _detect_provider(model)

        if self.provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic()
        else:
            from openai import OpenAI
            self._client = OpenAI()

        logger.debug(f"LLMClient: provider={self.provider} model={self.model}")

    # ------------------------------------------------------------------ #
    # Internal API calls                                                   #
    # ------------------------------------------------------------------ #

    @with_backoff(max_retries=6, base_wait=1.0, max_wait=60.0)
    def _call_anthropic(self, system_prompt, user_prompt, max_tokens,
                        cache_system=False):
        msgs = [{"role": "user", "content": user_prompt}]
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "messages": msgs,
        }
        if system_prompt:
            if cache_system:
                kwargs["system"] = [{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
                kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)

        if self.cost_tracker:
            self.cost_tracker.record(
                self.model,
                response.usage.input_tokens,
                response.usage.output_tokens,
                "llm call",
                cached_input=cache_system,
            )
        return response.content[0].text

    @with_backoff(max_retries=6, base_wait=1.0, max_wait=60.0)
    def _call_openai(self, system_prompt, user_prompt, max_tokens,
                     cache_system=False):  # cache_system ignored for OpenAI
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "messages": messages,
            "seed": self.seed,
        }

        response = self._client.chat.completions.create(**kwargs)

        usage = response.usage
        if self.cost_tracker and usage:
            self.cost_tracker.record(
                self.model,
                usage.prompt_tokens,
                usage.completion_tokens,
                "llm call",
            )
        return response.choices[0].message.content

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: int | None = None,
                 cache_system: bool = False,
                 cache_key: str | None = None) -> str:
        """Generate a response, with optional disk-cache keying.

        Args:
            system_prompt: System/instruction prompt.
            user_prompt:   User message.
            max_tokens:    Override instance max_tokens for this call.
            cache_system:  Anthropic prompt-caching hint (ignored for OpenAI).
            cache_key:     If set, results are cached on disk under this key.

        Returns:
            Model response text.
        """
        if cache_key:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        mt = max_tokens or self.max_tokens
        if self.provider == "anthropic":
            result = self._call_anthropic(system_prompt, user_prompt, mt,
                                          cache_system)
        else:
            result = self._call_openai(system_prompt, user_prompt, mt,
                                       cache_system)

        if cache_key:
            self.cache.set(cache_key, result)
        return result

    def close(self):
        self.cache.close()
