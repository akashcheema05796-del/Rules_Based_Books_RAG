"""
Anthropic Claude client wrapper.
claude-sonnet-4-5, temp=0, seed=42 with backoff and cost tracking (spec §2, §10).
"""

import logging
from pathlib import Path
from typing import Optional

from anthropic import Anthropic

from src.utils.backoff import with_backoff
from src.utils.cache import get_cache
from src.utils.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class LLMClient:
    """Anthropic Claude client with caching, cost tracking, and backoff."""

    def __init__(self, model="claude-sonnet-4-5-20241022", temperature=0,
                 max_tokens=4096, seed=42, cache_dir=".cache",
                 cost_tracker=None):
        self.client = Anthropic()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.cache = get_cache("llm", cache_dir)
        self.cost_tracker = cost_tracker

    @with_backoff(max_retries=6, base_wait=1.0, max_wait=60.0)
    def _call_api(self, system_prompt, user_prompt, max_tokens=None,
                  cache_system=False):
        """Call Claude API with optional prompt caching."""
        msgs = [{"role": "user", "content": user_prompt}]
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
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

        response = self.client.messages.create(**kwargs)

        if self.cost_tracker:
            self.cost_tracker.record(
                self.model,
                response.usage.input_tokens,
                response.usage.output_tokens,
                f"llm call",
                cached_input=cache_system,
            )
        return response.content[0].text

    def generate(self, system_prompt, user_prompt, max_tokens=None,
                 cache_system=False, cache_key=None):
        """Generate a response with optional caching."""
        if cache_key:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        result = self._call_api(system_prompt, user_prompt, max_tokens,
                                cache_system)
        if cache_key:
            self.cache.set(cache_key, result)
        return result

    def close(self):
        self.cache.close()
