"""
Tokenizer utilities using tiktoken (cl100k_base).

Provides token counting used by all chunkers and metrics.
"""

import logging
from functools import lru_cache

import tiktoken

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_tokenizer(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Get a cached tiktoken encoding.

    Args:
        encoding_name: Name of the encoding (default: cl100k_base per spec §2).

    Returns:
        tiktoken.Encoding instance.
    """
    enc = tiktoken.get_encoding(encoding_name)
    logger.debug(f"Loaded tokenizer: {encoding_name}")
    return enc


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using cl100k_base encoding.

    Args:
        text: The text to tokenize.
        encoding_name: Tokenizer encoding name.

    Returns:
        Number of tokens.
    """
    enc = get_tokenizer(encoding_name)
    return len(enc.encode(text))


def truncate_to_tokens(text: str, max_tokens: int, encoding_name: str = "cl100k_base") -> str:
    """Truncate text to a maximum number of tokens.

    Args:
        text: Text to truncate.
        max_tokens: Maximum token count.
        encoding_name: Tokenizer encoding name.

    Returns:
        Truncated text.
    """
    enc = get_tokenizer(encoding_name)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def token_length_function(text: str) -> int:
    """Token length function for use with LangChain text splitters."""
    return count_tokens(text)
