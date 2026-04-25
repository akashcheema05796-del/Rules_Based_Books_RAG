"""
Answer generation for the generation-metrics phase (spec §8.2).

A thin generator that takes top-k retrieved chunks and produces an answer plus
an abstention signal. Distractor handling: if the model abstains, the answer is
the literal token ABSTAIN_TOKEN — `abstention_rate` counts these correctly.
"""

from __future__ import annotations

import logging

from src.corpus.models import GoldEntry, RetrievedChunk
from src.utils.llm import LLMClient

logger = logging.getLogger(__name__)

ABSTAIN_TOKEN = "I_DO_NOT_KNOW"

ANSWER_PROMPT = """You are a precise assistant answering questions about the AD&D 2nd
Edition rulebooks. Use ONLY the provided CONTEXTS. If the contexts do not contain
enough information to answer the question, respond with exactly:

  {abstain_token}

Do not speculate, do not rely on general D&D knowledge, do not use other editions.

CONTEXTS:
---
{contexts}
---

QUESTION: {query}

Answer (concise, 1-2 sentences, or {abstain_token}):"""


def generate_answer(
    query: str, retrieved: list[RetrievedChunk], llm: LLMClient,
    gold: GoldEntry | None = None, top_k: int = 5,
) -> str:
    contexts = "\n---\n".join(c.content for c in retrieved[:top_k])
    cache_key = f"answer:{gold.id}:{top_k}" if gold else None
    try:
        return llm.generate(
            "You are a precise, grounded QA assistant.",
            ANSWER_PROMPT.format(abstain_token=ABSTAIN_TOKEN,
                                 contexts=contexts, query=query),
            max_tokens=256,
            cache_key=cache_key,
        ).strip()
    except Exception as e:
        logger.warning(f"answer generation failed: {e}")
        return ABSTAIN_TOKEN


def is_abstention(answer: str) -> bool:
    return ABSTAIN_TOKEN in (answer or "").upper()


def abstention_rate(answers_by_type: dict[str, list[str]]) -> dict:
    """Abstention rate per query type. For distractors, higher is better."""
    out = {}
    for qt, answers in answers_by_type.items():
        if not answers:
            continue
        n_abstain = sum(1 for a in answers if is_abstention(a))
        out[qt] = {
            "n": len(answers),
            "n_abstain": n_abstain,
            "abstention_rate": round(n_abstain / len(answers), 4),
        }
    return out
