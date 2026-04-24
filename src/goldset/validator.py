"""
Gold standard validation utilities (spec §7.3).

Verify reference contexts contain answers, content-hash JSONL,
and support inter-rater agreement calculation.
"""

import hashlib
import json
import logging
from pathlib import Path

from src.corpus.models import GoldEntry

logger = logging.getLogger(__name__)


def validate_gold_standard(gold_path: Path, corpus_path: Path = None) -> dict:
    """Validate a gold standard JSONL file.

    Args:
        gold_path: Path to gold_standard.jsonl.
        corpus_path: Optional path to raw corpus for context validation.

    Returns:
        Validation report dictionary.
    """
    entries = load_gold(gold_path)
    report = {
        "total_entries": len(entries),
        "by_type": {},
        "issues": [],
        "content_hash": "",
    }

    # Count by type
    for e in entries:
        t = e.query_type
        report["by_type"][t] = report["by_type"].get(t, 0) + 1

    # Validate each entry
    for e in entries:
        if not e.query.strip():
            report["issues"].append(f"{e.id}: empty query")
        if not e.reference_answer.strip():
            report["issues"].append(f"{e.id}: empty reference answer")
        if e.query_type not in {"lore", "mechanical", "tabular",
                                "cross_reference", "monster", "numeric", "distractor"}:
            report["issues"].append(f"{e.id}: unknown query_type '{e.query_type}'")

    # Content hash
    content = gold_path.read_bytes()
    report["content_hash"] = hashlib.sha256(content).hexdigest()[:16]

    # Validate reference contexts against corpus
    if corpus_path and corpus_path.exists():
        corpus_text = corpus_path.read_text(encoding="utf-8")
        for e in entries:
            for ctx in e.reference_contexts:
                start = ctx.get("char_start", 0)
                end = ctx.get("char_end", 0)
                if start > 0 and end > start:
                    snippet = corpus_text[start:end]
                    answer_lower = e.reference_answer.lower()
                    if answer_lower not in snippet.lower():
                        report["issues"].append(
                            f"{e.id}: answer not found in reference context "
                            f"[{start}:{end}]")

    logger.info(f"Validation: {len(entries)} entries, {len(report['issues'])} issues")
    return report


def load_gold(gold_path: Path) -> list[GoldEntry]:
    """Load gold standard entries from JSONL."""
    entries = []
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(GoldEntry(**json.loads(line)))
    return entries


def compute_cohen_kappa(ratings_a: list[int], ratings_b: list[int]) -> float:
    """Compute Cohen's kappa for inter-rater agreement.

    Args:
        ratings_a: Ratings from reviewer A (0 or 1).
        ratings_b: Ratings from reviewer B (0 or 1).

    Returns:
        Cohen's kappa coefficient.
    """
    assert len(ratings_a) == len(ratings_b)
    n = len(ratings_a)
    if n == 0:
        return 0.0

    # Observed agreement
    agree = sum(1 for a, b in zip(ratings_a, ratings_b) if a == b)
    po = agree / n

    # Expected agreement
    pa1 = sum(ratings_a) / n
    pb1 = sum(ratings_b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)
