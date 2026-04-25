"""
Gold standard validation utilities (spec §7.3).

Workflow:
  1. `emit_review_sheet(gold_path, out_csv)` — produce a CSV for two reviewers,
     one row per entry with blank `approve` columns.
  2. Reviewers fill `approve_A` / `approve_B` in {0, 1}.
  3. `ingest_reviews(csv, gold_path)` — merges: drops rows rejected by both,
     marks `validator="human"` on rows approved by both, computes Cohen's kappa
     on the overlap slice, and refuses to freeze if kappa < 0.7.

`validator="auto"` is left on rows that never went through human review — the
runner warns when it encounters these.
"""

import csv
import hashlib
import json
import logging
from pathlib import Path

from src.corpus.models import GoldEntry

logger = logging.getLogger(__name__)


def validate_gold_standard(gold_path: Path, corpus_path: Path = None) -> dict:
    """Static validation — schema, coverage, span containment."""
    entries = load_gold(gold_path)
    report = {
        "total_entries": len(entries),
        "by_type": {}, "by_validator": {},
        "issues": [], "content_hash": "",
    }

    for e in entries:
        report["by_type"][e.query_type] = report["by_type"].get(e.query_type, 0) + 1
        report["by_validator"][e.validator] = report["by_validator"].get(e.validator, 0) + 1

    valid_types = {"lore", "mechanical", "tabular", "cross_reference",
                   "monster", "numeric", "distractor"}
    for e in entries:
        if not e.query.strip():
            report["issues"].append(f"{e.id}: empty query")
        if not e.reference_answer.strip():
            report["issues"].append(f"{e.id}: empty reference answer")
        if e.query_type not in valid_types:
            report["issues"].append(f"{e.id}: unknown query_type '{e.query_type}'")
        if e.query_type != "distractor" and not e.reference_contexts:
            report["issues"].append(f"{e.id}: missing reference_contexts")

    report["content_hash"] = hashlib.sha256(gold_path.read_bytes()).hexdigest()[:16]

    if corpus_path and corpus_path.exists():
        corpus_text = corpus_path.read_text(encoding="utf-8")
        for e in entries:
            for ctx in e.reference_contexts:
                start = ctx.get("char_start", 0)
                end = ctx.get("char_end", 0)
                if start > 0 and end > start:
                    snippet = corpus_text[start:end]
                    if e.reference_answer.lower() not in snippet.lower():
                        report["issues"].append(
                            f"{e.id}: answer not found in ref span [{start}:{end}]")

    auto_count = report["by_validator"].get("auto", 0)
    if auto_count:
        report["issues"].append(
            f"{auto_count} entries still marked validator='auto' — run the "
            f"human-review workflow before freezing the gold set.")

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


def emit_review_sheet(gold_path: Path, out_csv: Path, corpus_path: Path) -> None:
    """Dump a CSV for human review. Two reviewers fill approve_A / approve_B."""
    entries = load_gold(gold_path)
    corpus_text = corpus_path.read_text(encoding="utf-8") if corpus_path.exists() else ""
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "query_type", "query", "reference_answer",
                    "ref_snippet", "approve_A", "approve_B", "notes"])
        for e in entries:
            snippet = ""
            if e.reference_contexts and corpus_text:
                c = e.reference_contexts[0]
                s, t = c.get("char_start", 0), c.get("char_end", 0)
                snippet = corpus_text[s:t].replace("\n", " ")[:500]
            w.writerow([e.id, e.query_type, e.query, e.reference_answer,
                        snippet, "", "", e.notes])
    logger.info(f"Review sheet: {out_csv} ({len(entries)} rows)")


def ingest_reviews(
    reviews_csv: Path, gold_path: Path, out_path: Path,
    kappa_threshold: float = 0.7,
) -> dict:
    """Merge reviewer decisions back into the gold set.

    Keeps entries where BOTH reviewers approved. Computes Cohen's kappa over the
    full set (not just a 20-question slice — all reviewed rows qualify as the
    overlap when there are two reviewers per row). Refuses to write a frozen
    file if kappa < threshold.
    """
    ratings_a, ratings_b = [], []
    decisions: dict[str, tuple[int, int]] = {}
    with open(reviews_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a = _parse01(row.get("approve_A"))
            b = _parse01(row.get("approve_B"))
            if a is None or b is None:
                continue
            decisions[row["id"]] = (a, b)
            ratings_a.append(a)
            ratings_b.append(b)

    kappa = compute_cohen_kappa(ratings_a, ratings_b)
    report = {"n_reviewed": len(decisions), "cohen_kappa": kappa,
              "kappa_threshold": kappa_threshold}

    if len(decisions) == 0:
        report["error"] = "no reviewer decisions parsed"
        return report

    if kappa < kappa_threshold:
        report["error"] = (f"Cohen's kappa {kappa:.3f} < threshold {kappa_threshold}. "
                           f"Gold set NOT frozen. Reconcile disagreements first.")
        logger.error(report["error"])
        return report

    entries = load_gold(gold_path)
    kept = []
    for e in entries:
        if e.query_type == "distractor":
            kept.append(e)
            continue
        d = decisions.get(e.id)
        if d is None:
            continue  # unreviewed — drop
        if d[0] == 1 and d[1] == 1:
            e.validator = "human"
            kept.append(e)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in kept:
            f.write(json.dumps(e.model_dump(), ensure_ascii=False) + "\n")

    content_hash = hashlib.sha256(out_path.read_bytes()).hexdigest()[:16]
    report.update({"entries_frozen": len(kept), "content_hash": content_hash,
                   "frozen_path": str(out_path)})
    logger.info(f"Gold set frozen: {len(kept)} entries, kappa={kappa:.3f}, hash={content_hash}")
    return report


def _parse01(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("1", "y", "yes", "true", "approve", "a"):
        return 1
    if s in ("0", "n", "no", "false", "reject", "r"):
        return 0
    return None


def compute_cohen_kappa(ratings_a: list[int], ratings_b: list[int]) -> float:
    """Cohen's kappa for binary ratings."""
    assert len(ratings_a) == len(ratings_b)
    n = len(ratings_a)
    if n == 0:
        return 0.0
    agree = sum(1 for a, b in zip(ratings_a, ratings_b) if a == b)
    po = agree / n
    pa1 = sum(ratings_a) / n
    pb1 = sum(ratings_b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)
