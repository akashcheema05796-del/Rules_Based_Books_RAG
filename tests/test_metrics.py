"""Tests for evaluation metrics (§8) and statistics (§9)."""

import pytest
import numpy as np

from src.corpus.models import GoldEntry, RetrievedChunk
from src.evaluation.retrieval_metrics import (
    recall_at_k, mrr, ndcg_at_k, hit_at_k, span_overlap_fraction,
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
