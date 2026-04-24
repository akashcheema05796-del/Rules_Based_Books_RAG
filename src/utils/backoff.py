"""
API call retry/backoff using tenacity.

Wraps API calls with exponential backoff: base=1s, max=60s, 6 retries (spec §10).
"""

import logging
from functools import wraps
from typing import Callable, Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError,
)

logger = logging.getLogger(__name__)

# Common API error types
RETRIABLE_EXCEPTIONS = (
    Exception,  # Broad catch — narrowed below for specific providers
)


def with_backoff(
    max_retries: int = 6,
    base_wait: float = 1.0,
    max_wait: float = 60.0,
    retriable_exceptions: tuple = None,
) -> Callable:
    """Decorator for API calls with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_wait: Base wait time in seconds.
        max_wait: Maximum wait time in seconds.
        retriable_exceptions: Tuple of exception types to retry on.

    Returns:
        Decorated function with retry logic.
    """
    if retriable_exceptions is None:
        # Default to common API rate limit / transient errors
        try:
            from openai import RateLimitError, APITimeoutError, APIConnectionError
            from anthropic import RateLimitError as AnthropicRateLimit
            from anthropic import APITimeoutError as AnthropicTimeout
            from anthropic import APIConnectionError as AnthropicConnection
            retriable_exceptions = (
                RateLimitError, APITimeoutError, APIConnectionError,
                AnthropicRateLimit, AnthropicTimeout, AnthropicConnection,
                ConnectionError, TimeoutError,
            )
        except ImportError:
            retriable_exceptions = (ConnectionError, TimeoutError)

    def decorator(func: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=base_wait, max=max_wait),
            retry=retry_if_exception_type(retriable_exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return func(*args, **kwargs)

        return wrapper

    return decorator


def api_call_with_backoff(func: Callable, *args, **kwargs) -> Any:
    """Execute a function with backoff retry logic.

    Args:
        func: The function to call.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Function result.

    Raises:
        RetryError: If all retries are exhausted.
    """
    @with_backoff()
    def _wrapped():
        return func(*args, **kwargs)

    return _wrapped()
