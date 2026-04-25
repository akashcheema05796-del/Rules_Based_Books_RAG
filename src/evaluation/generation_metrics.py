"""
Generation metrics (spec §8.2).

RAGAS-style faithfulness, answer correctness, context precision. Judge uses a
calibrated 5-point rubric mapped to [0, 1] with anchors for each level, plus a
brief CoT step to stabilize scores at temperature=0. Trials are only useful
when the upstream pipeline is stochastic (HyDE, decomposition) or when the
judge itself is probed at temperature > 0.
"""

import logging
import re

import numpy as np

from src.corpus.models import GoldEntry, RetrievedChunk
from src.utils.llm import LLMClient

logger = logging.getLogger(__name__)

FAITHFULNESS_PROMPT = """You are a precise evaluation judge. Score how faithfully the
generated ANSWER is grounded in the provided CONTEXT. Grounded = every factual claim
in the answer is directly supported by the context.

CONTEXT:
---
{context}
---

ANSWER:
{answer}

Rubric (choose one):
  1.0 — every claim in the answer is directly supported by the context
  0.75 — all major claims supported; minor unsupported detail
  0.5 — mix of supported and unsupported claims
  0.25 — most claims contradicted by or absent from context
  0.0 — answer fabricates facts / contradicts context

Respond in two lines, exactly:
REASON: <one short sentence>
SCORE: <number in {{0.0, 0.25, 0.5, 0.75, 1.0}}>"""

CORRECTNESS_PROMPT = """You are a precise evaluation judge. Compare the GENERATED answer
to the REFERENCE answer. Semantic equivalence counts — wording need not match exactly.

REFERENCE answer: {reference}
GENERATED answer: {answer}

Rubric (choose one):
  1.0 — semantically equivalent; all key facts correct
  0.75 — mostly correct; minor wording or precision issue
  0.5 — partially correct; some key facts missing or wrong
  0.25 — mostly incorrect; a single detail salvageable
  0.0 — wrong, unrelated, or "I don't know" when reference is known

Respond in two lines, exactly:
REASON: <one short sentence>
SCORE: <number in {{0.0, 0.25, 0.5, 0.75, 1.0}}>"""

PRECISION_PROMPT = """You are a precise evaluation judge. For each retrieved context
below, decide if it is relevant to answering the QUESTION. Report the fraction that
are relevant as a score.

QUESTION: {query}

RETRIEVED CONTEXTS:
{contexts}

Rubric:
  fraction_relevant = (# contexts directly useful for answering) / (# contexts shown)
  Round to the nearest of {{0.0, 0.25, 0.5, 0.75, 1.0}}.

Respond in two lines, exactly:
REASON: <one short sentence>
SCORE: <number in {{0.0, 0.25, 0.5, 0.75, 1.0}}>"""

_SCORE_RE = re.compile(r"SCORE\s*:\s*([0-9.]+)", re.IGNORECASE)


def compute_generation_metrics(
    query: str,
    answer: str,
    retrieved: list[RetrievedChunk],
    gold: GoldEntry,
    llm: LLMClient,
    n_trials: int = 1,
) -> dict:
    """Compute RAGAS-style metrics. Caller chooses `n_trials`; only > 1 if the
    judge is stochastic (temperature > 0) or the pipeline feeding `answer` is."""
    context_text = "\n---\n".join(c.content for c in retrieved[:5])
    contexts_list = "\n---\n".join(
        f"[{i+1}] {c.content[:500]}" for i, c in enumerate(retrieved[:5])
    )

    buckets = {"faithfulness": [], "answer_correctness": [], "context_precision": []}

    for trial in range(n_trials):
        buckets["faithfulness"].append(_judge_score(
            llm, FAITHFULNESS_PROMPT.format(context=context_text, answer=answer),
            cache_key=f"faith:{gold.id}:{trial}"))
        buckets["answer_correctness"].append(_judge_score(
            llm, CORRECTNESS_PROMPT.format(reference=gold.reference_answer, answer=answer),
            cache_key=f"correct:{gold.id}:{trial}"))
        buckets["context_precision"].append(_judge_score(
            llm, PRECISION_PROMPT.format(query=query, contexts=contexts_list),
            cache_key=f"prec:{gold.id}:{trial}"))

    result = {}
    for key, vals in buckets.items():
        result[f"{key}_mean"] = float(np.mean(vals))
        result[f"{key}_std"] = float(np.std(vals))
    return result


def _judge_score(llm: LLMClient, prompt: str, cache_key: str | None = None) -> float:
    """Parse a rubric-scored reply. Falls back to first float if format slips."""
    try:
        resp = llm.generate(
            "You are a precise evaluation judge. Follow the output format exactly.",
            prompt, max_tokens=80, cache_key=cache_key,
        )
    except Exception as e:
        logger.warning(f"judge call failed: {e}")
        return 0.0

    m = _SCORE_RE.search(resp)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass
    m = re.search(r"([0-9]*\.?[0-9]+)", resp)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass
    return 0.0
