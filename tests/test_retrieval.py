"""Tests for retrieval methods (§6)."""

import pytest
from src.retrieval.hybrid_rrf import reciprocal_rank_fusion
from src.corpus.models import RetrievedChunk


class TestRRF:
    def test_basic_fusion(self):
        list_a = [
            RetrievedChunk(chunk_id="a", content="A", score=1.0, rank=1),
            RetrievedChunk(chunk_id="b", content="B", score=0.8, rank=2),
            RetrievedChunk(chunk_id="c", content="C", score=0.6, rank=3),
        ]
        list_b = [
            RetrievedChunk(chunk_id="b", content="B", score=1.0, rank=1),
            RetrievedChunk(chunk_id="a", content="A", score=0.7, rank=2),
            RetrievedChunk(chunk_id="d", content="D", score=0.5, rank=3),
        ]
        fused = reciprocal_rank_fusion([list_a, list_b], k=60)
        assert len(fused) == 4  # a, b, c, d
        # b should rank high (rank 1 in list_b, rank 2 in list_a)
        ids = [c.chunk_id for c in fused]
        assert "b" in ids[:2]
        assert "a" in ids[:2]

    def test_rrf_scores_decrease(self):
        results = [
            RetrievedChunk(chunk_id=f"c{i}", content=f"C{i}", score=1.0, rank=i+1)
            for i in range(5)
        ]
        fused = reciprocal_rank_fusion([results], k=60)
        for i in range(len(fused) - 1):
            assert fused[i].score >= fused[i+1].score

    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([], k=60)
        assert len(fused) == 0

    def test_single_result(self):
        results = [RetrievedChunk(chunk_id="x", content="X", score=1.0, rank=1)]
        fused = reciprocal_rank_fusion([results], k=60)
        assert len(fused) == 1
        assert fused[0].chunk_id == "x"


class TestRetrievedChunk:
    def test_default_score(self):
        chunk = RetrievedChunk(chunk_id="test", content="hello")
        assert chunk.score == 0.0
        assert chunk.rank == 0
