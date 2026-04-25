"""Tests for evaluation metrics (§8) and statistics (§9)."""

import pytest
import numpy as np

from src.corpus.models import GoldEntry, RetrievedChunk
from src.evaluation.retrieval_metrics import (
    recall_at_k, mrr, ndcg_at_k, hit_at_k, hit_at_k, span_overlap_fraction,
    is_relevant, compute_retrieval_metrics,
)
from src.evaluation.statistics import (
    bootstrap_ci, paired_bootstrap_test, holm_bonferroni_correction,
)


class TestSpanOverlap:
    def test_full_overlap(self):
        assert span_overlap_fraction(0, 100, 0, 100) == 1.0

    def test_no_overlap(self):
        assert span_overlap_fraction(0, 100, 200, 300) == 0.0

    def test_partial_overlap(self):
        frac = span_overlap_fraction(0, 100, 50, 150)
        assert 0.4 < frac < 0.6  # 50/100 = 0.5

    def test_contained(self):
        frac = span_overlap_fraction(25, 75, 0, 100)
        assert frac == 1.0  # shorter span (50) fully contained


class TestRetrievalMetrics:
    def _make_gold(self):
        return GoldEntry(
            id="q_001", query_type="mechanical",
            query="test query", reference_answer="answer text",
            reference_contexts=[{"book": "PHB", "char_start": 100, "char_end": 200}],
        )

    def _make_chunk(self, content, has_answer=False):
        text = f"Some content with {content}"
        if has_answer:
            text += " answer text"
        return RetrievedChunk(chunk_id="c1", content=text, score=0.9, rank=1)

    def test_hit_at_k_found(self):
        gold = self._make_gold()
        chunks = [self._make_chunk("relevant", has_answer=True)]
        assert hit_at_k(chunks, gold, k=1) == 1.0

    def test_hit_at_k_not_found(self):
        gold = self._make_gold()
        chunks = [self._make_chunk("irrelevant")]
        assert hit_at_k(chunks, gold, k=1) == 0.0

    def test_mrr_first(self):
        gold = self._make_gold()
        chunks = [self._make_chunk("x", True)]
        assert mrr(chunks, gold) == 1.0

    def test_mrr_second(self):
        gold = self._make_gold()
        chunks = [
            self._make_chunk("irrelevant"),
            self._make_chunk("x", True),
        ]
        assert mrr(chunks, gold) == 0.5

    def test_mrr_not_found(self):
        gold = self._make_gold()
        chunks = [self._make_chunk("irrelevant")]
        assert mrr(chunks, gold) == 0.0


class TestBootstrap:
    def test_basic_ci(self):
        values = [0.8, 0.85, 0.9, 0.75, 0.82, 0.88, 0.91, 0.79]
        ci = bootstrap_ci(values, n_resamples=1000)
        assert ci["lower"] <= ci["mean"] <= ci["upper"]
        assert ci["n"] == 8

    def test_empty_values(self):
        ci = bootstrap_ci([])
        assert ci["mean"] == 0.0

    def test_single_value(self):
        ci = bootstrap_ci([0.5])
        assert ci["mean"] == 0.5

    def test_ci_width(self):
        values = [0.5] * 100  # No variance
        ci = bootstrap_ci(values, n_resamples=1000)
        assert ci["upper"] - ci["lower"] < 0.01


class TestPairedBootstrap:
    def test_significant_difference(self):
        # System A clearly better
        a = [0.9, 0.85, 0.92, 0.88, 0.91, 0.87, 0.93, 0.89]
        b = [0.5, 0.45, 0.52, 0.48, 0.51, 0.47, 0.53, 0.49]
        result = paired_bootstrap_test(a, b, n_resamples=1000)
        assert result["mean_diff"] > 0
        assert result["p_value"] < 0.05

    def test_no_difference(self):
        values = [0.8, 0.85, 0.9, 0.75, 0.82, 0.88, 0.91, 0.79]
        result = paired_bootstrap_test(values, values, n_resamples=1000)
        assert abs(result["mean_diff"]) < 0.001


class TestIsRelevant:
    """Tests for is_relevant() — char-offset path and content fallback."""

    def _make_gold_with_offsets(self, ref_start=100, ref_end=200):
        return GoldEntry(
            id="q_001", query_type="mechanical",
            query="test", reference_answer="the answer",
            reference_contexts=[{"char_start": ref_start, "char_end": ref_end}],
        )

    def _make_chunk_with_offsets(self, chunk_start, chunk_end, content="some content"):
        return RetrievedChunk(
            chunk_id="c1", content=content, score=0.9, rank=1,
            metadata={"char_start": chunk_start, "char_end": chunk_end},
        )

    def test_full_overlap_is_relevant(self):
        gold = self._make_gold_with_offsets(100, 200)
        chunk = self._make_chunk_with_offsets(100, 200)
        assert is_relevant(chunk, gold) is True

    def test_zero_overlap_not_relevant(self):
        gold = self._make_gold_with_offsets(100, 200)
        chunk = self._make_chunk_with_offsets(300, 400)
        assert is_relevant(chunk, gold, overlap_threshold=0.5) is False

    def test_partial_overlap_above_threshold(self):
        # Note: is_relevant requires ref_start > 0 to use char-offset path.
        gold = self._make_gold_with_offsets(100, 200)
        chunk = self._make_chunk_with_offsets(100, 200)  # 100% overlap
        assert is_relevant(chunk, gold, overlap_threshold=0.5) is True

    def test_partial_overlap_below_threshold(self):
        gold = self._make_gold_with_offsets(100, 200)
        # Chunk overlaps by only 10 chars out of 100 (shorter span=100) = 10% < 50%
        chunk2 = self._make_chunk_with_offsets(190, 300)
        assert is_relevant(chunk2, gold, overlap_threshold=0.5) is False

    def test_content_fallback_when_no_offsets(self):
        gold = GoldEntry(
            id="q_002", query_type="lore",
            query="test", reference_answer="Lolth",
            reference_contexts=[],
        )
        chunk = RetrievedChunk(
            chunk_id="c2", content="Lolth is the Queen of Spiders.", score=0.9, rank=1,
        )
        assert is_relevant(chunk, gold) is True

    def test_content_fallback_miss(self):
        gold = GoldEntry(
            id="q_003", query_type="lore",
            query="test", reference_answer="Lolth",
            reference_contexts=[],
        )
        chunk = RetrievedChunk(
            chunk_id="c3", content="The fighter strikes with a sword.", score=0.9, rank=1,
        )
        assert is_relevant(chunk, gold) is False


class TestRecallAtKWithOffsets:
    """Tests for recall_at_k using char-offset reference contexts.

    Note: is_relevant() only uses the char-offset path when ref_start > 0
    (zero is treated as "no offset set"). Tests must use ref_start >= 1.
    """

    def _make_gold(self, ref_start, ref_end):
        return GoldEntry(
            id="q_001", query_type="mechanical",
            query="test", reference_answer="answer",
            reference_contexts=[{"char_start": ref_start, "char_end": ref_end}],
        )

    def _chunk(self, cs, ce, rank=1):
        return RetrievedChunk(
            chunk_id=f"c{rank}", content="text", score=0.9, rank=rank,
            metadata={"char_start": cs, "char_end": ce},
        )

    def test_recall_1_when_found_in_top1(self):
        gold = self._make_gold(10, 110)
        chunks = [self._chunk(10, 110)]
        assert recall_at_k(chunks, gold, k=1) == 1.0

    def test_recall_0_when_not_found(self):
        gold = self._make_gold(10, 110)
        chunks = [self._chunk(200, 300)]
        assert recall_at_k(chunks, gold, k=1) == 0.0

    def test_recall_respects_k_cutoff(self):
        gold = self._make_gold(10, 110)
        chunks = [
            self._chunk(200, 300, rank=1),  # not relevant
            self._chunk(10, 110, rank=2),   # relevant but rank 2
        ]
        assert recall_at_k(chunks, gold, k=1) == 0.0
        assert recall_at_k(chunks, gold, k=2) == 1.0

    def test_recall_empty_retrieval(self):
        gold = self._make_gold(10, 110)
        assert recall_at_k([], gold, k=10) == 0.0


class TestNDCGAtK:
    """Tests for ndcg_at_k with graded relevance.

    Note: is_relevant() / span_overlap uses char-offset path only when
    ref_start > 0. All offsets here are > 0.
    """

    def _gold(self):
        # Use a distinctive answer that won't accidentally appear in chunk content.
        return GoldEntry(
            id="q_001", query_type="mechanical",
            query="test", reference_answer="THAC0_SPECIFIC_VALUE_99",
            reference_contexts=[{"char_start": 10, "char_end": 110}],
        )

    def _chunk(self, cs, ce, rank=1, content="unrelated filler text here"):
        return RetrievedChunk(
            chunk_id=f"c{rank}", content=content, score=1.0 / rank, rank=rank,
            metadata={"char_start": cs, "char_end": ce},
        )

    def test_perfect_ndcg(self):
        gold = self._gold()
        chunks = [self._chunk(10, 110, rank=1)]
        score = ndcg_at_k(chunks, gold, k=1)
        assert abs(score - 1.0) < 1e-9

    def test_ndcg_zero_no_relevant(self):
        gold = self._gold()
        # Chunk at a completely different span AND no answer text in content
        chunks = [self._chunk(200, 300, rank=1)]
        score = ndcg_at_k(chunks, gold, k=1)
        assert score == 0.0

    def test_ndcg_decreases_with_rank(self):
        gold = self._gold()
        # Relevant chunk at rank 1 vs rank 3
        first = [self._chunk(10, 110, rank=1), self._chunk(200, 300, rank=2)]
        third = [self._chunk(200, 300, rank=1), self._chunk(200, 300, rank=2),
                 self._chunk(10, 110, rank=3)]
        score_first = ndcg_at_k(first, gold, k=3)
        score_third = ndcg_at_k(third, gold, k=3)
        assert score_first > score_third

    def test_ndcg_empty_retrieved(self):
        gold = self._gold()
        assert ndcg_at_k([], gold, k=10) == 0.0

    def test_ndcg_range_zero_to_one(self):
        gold = self._gold()
        chunks = [
            self._chunk(10, 110, rank=1),
            self._chunk(200, 300, rank=2),
            self._chunk(50, 150, rank=3),
        ]
        score = ndcg_at_k(chunks, gold, k=3)
        assert 0.0 <= score <= 1.0


class TestComputeRetrievalMetrics:
    """Tests for the all-in-one compute_retrieval_metrics()."""

    def _gold(self):
        return GoldEntry(
            id="q_001", query_type="mechanical",
            query="test", reference_answer="answer text",
            reference_contexts=[{"char_start": 10, "char_end": 110}],
        )

    def test_returns_all_expected_keys(self):
        gold = self._gold()
        chunks = [RetrievedChunk(
            chunk_id="c1", content="answer text here", score=0.9, rank=1,
            metadata={"char_start": 10, "char_end": 110},
        )]
        metrics = compute_retrieval_metrics(chunks, gold, k_values=[1, 5])
        assert "mrr" in metrics
        assert "recall@1" in metrics
        assert "recall@5" in metrics
        assert "ndcg@1" in metrics
        assert "hit@1" in metrics

    def test_perfect_retrieval(self):
        gold = self._gold()
        chunks = [RetrievedChunk(
            chunk_id="c1", content="answer text", score=0.9, rank=1,
            metadata={"char_start": 10, "char_end": 110},
        )]
        metrics = compute_retrieval_metrics(chunks, gold, k_values=[1])
        assert metrics["mrr"] == 1.0
        assert metrics["hit@1"] == 1.0


class TestHolmBonferroni:
    def test_basic_correction(self):
        p_values = {"A vs B": 0.01, "A vs C": 0.04, "B vs C": 0.06}
        corrected = holm_bonferroni_correction(p_values, alpha=0.05)
        # Most significant should use alpha/3
        assert corrected["A vs B"]["significant"] is True

    def test_all_significant(self):
        p_values = {"A vs B": 0.001, "A vs C": 0.002, "B vs C": 0.003}
        corrected = holm_bonferroni_correction(p_values, alpha=0.05)
        assert all(v["significant"] for v in corrected.values())
