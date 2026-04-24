"""
Generation metrics via RAGAS (spec §8.2).

Faithfulness, Answer Correctness, Context Precision.
Judge = claude-sonnet-4-5, temperature=0, 3 trials.
"""

import logging
from typing import Optional

from src.corpus.models import GoldEntry, RetrievedChunk
from src.utils.llm import LLMClient

logger = logging.getLogger(__name__)

FAITHFULNESS_PROMPT = """Given the context and the answer, evaluate whether the answer
is faithful to (grounded in) the provided context. Score from 0.0 to 1.0.

Context:
{context}

Answer:
{answer}

Return ONLY a number between 0.0 and 1.0."""

CORRECTNESS_PROMPT = """Compare the generated answer with the reference answer.
Score from 0.0 to 1.0 based on correctness.

Reference answer: {reference}
Generated answer: {answer}

Return ONLY a number between 0.0 and 1.0."""

PRECISION_PROMPT = """Evaluate what fraction of the retrieved contexts are actually
relevant to answering the question. Score from 0.0 to 1.0.

Question: {query}
Retrieved contexts:
{contexts}

Return ONLY a number between 0.0 and 1.0."""


def compute_generation_metrics(
    query: str,
    answer: str,
    retrieved: list[RetrievedChunk],
    gold: GoldEntry,
    llm: LLMClient,
    n_trials: int = 3,
) -> dict:
    """Compute RAGAS-style generation metrics.

    Args:
        query: The input query.
        answer: Generated answer.
        retrieved: Retrieved chunks used for generation.
        gold: Gold standard entry.
        llm: LLM client for judging.
        n_trials: Number of trials for variance estimation.

    Returns:
        Dict with faithfulness, answer_correctness, context_precision
        (mean and std over trials).
    """
    context_text = "\n---\n".join([c.content for c in retrieved[:5]])
    contexts_list = "\n---\n".join([f"[{i+1}] {c.content[:500]}" for i, c in enumerate(retrieved[:5])])

    metrics = {"faithfulness": [], "answer_correctness": [], "context_precision": []}

    for trial in range(n_trials):
        # Faithfulness
        try:
            score = _judge_score(llm, FAITHFULNESS_PROMPT.format(
                context=context_text, answer=answer),
                cache_key=f"faith:{gold.id}:{trial}")
            metrics["faithfulness"].append(score)
        except Exception:
            metrics["faithfulness"].append(0.0)

        # Answer correctness
        try:
            score = _judge_score(llm, CORRECTNESS_PROMPT.format(
                reference=gold.reference_answer, answer=answer),
                cache_key=f"correct:{gold.id}:{trial}")
            metrics["answer_correctness"].append(score)
        except Exception:
            metrics["answer_correctness"].append(0.0)

        # Context precision
        try:
            score = _judge_score(llm, PRECISION_PROMPT.format(
                query=query, contexts=contexts_list),
                cache_key=f"prec:{gold.id}:{trial}")
            metrics["context_precision"].append(score)
        except Exception:
            metrics["context_precision"].append(0.0)

    # Compute mean and std
    import numpy as np
    result = {}
    for key in metrics:
        vals = metrics[key]
        result[f"{key}_mean"] = float(np.mean(vals))
        result[f"{key}_std"] = float(np.std(vals))

    return result


def _judge_score(llm, prompt, cache_key=None):
    """Get a numeric score from the LLM judge."""
    resp = llm.generate("You are a precise evaluation judge. Return only a number.",
                         prompt, max_tokens=10, cache_key=cache_key)
    try:
        score = float(resp.strip())
        return max(0.0, min(1.0, score))
    except ValueError:
        return 0.0
