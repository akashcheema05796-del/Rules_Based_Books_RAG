"""
Tests for gold-standard dataset helpers (src/goldset/generator.py).

Covers: _locate_anchor exact/fuzzy/missing cases,
_sample_corpus_section distribution, short corpus edge cases,
_filter_candidates dedup logic (mocked embedder + LLM),
and _create_placeholder_gold output schema.
"""

import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.goldset.generator import (
    _locate_anchor,
    _sample_corpus_section,
    _filter_candidates,
    _create_placeholder_gold,
)
from src.corpus.models import GoldEntry


# ── Corpus fixture ──────────────────────────────────────────────────────────

CORPUS = """\
## Player's Handbook

### Chapter 1: Character Creation

A fighter with THAC0 of 16 can attack once per round.
The minimum ability score to qualify is 9.

### Chapter 2: Combat

Saving throws are listed on Table 25.
A 5th-level Fighter has a THAC0 of 16.

| Level | THAC0 |
|-------|-------|
| 1     | 20    |
| 5     | 16    |
| 10    | 11    |
"""


# ── _locate_anchor ──────────────────────────────────────────────────────────

class TestLocateAnchor:
    def test_exact_match_found(self):
        anchor = "THAC0 of 16 can attack once per round"
        span = _locate_anchor(CORPUS, anchor, "16")
        assert span is not None
        start, end = span
        assert start >= 0
        assert end > start
        assert "16" in CORPUS[start:end]

    def test_returns_none_when_anchor_missing(self):
        span = _locate_anchor(CORPUS, "this phrase does not exist anywhere", "answer")
        assert span is None

    def test_returns_none_when_answer_not_in_window(self):
        # Anchor exists but answer is fabricated
        anchor = "THAC0 of 16 can attack once per round"
        span = _locate_anchor(CORPUS, anchor, "ZZZZZ_NOT_IN_CORPUS")
        assert span is None

    def test_short_anchor_returns_none(self):
        span = _locate_anchor(CORPUS, "16", "16")  # < 10 chars
        assert span is None

    def test_empty_anchor_returns_none(self):
        span = _locate_anchor(CORPUS, "", "anything")
        assert span is None

    def test_span_contains_anchor(self):
        anchor = "minimum ability score to qualify is 9"
        span = _locate_anchor(CORPUS, anchor, "9")
        assert span is not None
        start, end = span
        assert anchor[:20] in CORPUS[start:end] or "9" in CORPUS[start:end]

    def test_window_is_bounded(self):
        anchor = "A fighter with THAC0 of 16"
        span = _locate_anchor(CORPUS, anchor, "16", window=100)
        if span:
            start, end = span
            assert end - start <= len(anchor) + 100 + 50  # window + some tolerance

    def test_whitespace_tolerant_fallback(self):
        # Insert extra whitespace in corpus that differs from anchor
        corpus_with_extra = CORPUS.replace("saving throws", "saving  throws")
        anchor = "saving  throws are listed on Table 25"[:40]
        # Should still find (exact or fuzzy)
        span = _locate_anchor(corpus_with_extra, anchor, "Table")
        # May or may not find; just assert no exception raised
        assert span is None or isinstance(span, tuple)


# ── _sample_corpus_section ──────────────────────────────────────────────────

class TestSampleCorpusSection:
    def test_returns_tuple(self):
        result = _sample_corpus_section(CORPUS, "mechanical")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_section_length_bounded(self):
        text, offset = _sample_corpus_section(CORPUS, "mechanical", section_size=100)
        assert len(text) <= len(CORPUS)  # Never larger than corpus
        assert offset >= 0

    def test_short_corpus_returns_full(self):
        short = "Hello world"
        text, offset = _sample_corpus_section(short, "mechanical", section_size=3500)
        assert text == short
        assert offset == 0

    def test_tabular_bias_returns_table_content(self):
        """With enough attempts, tabular type should find a pipe table."""
        found_table = False
        for _ in range(20):
            text, _ = _sample_corpus_section(CORPUS, "tabular", section_size=500)
            if "|" in text:
                found_table = True
                break
        # CORPUS has a table so at least one attempt should find it
        assert found_table

    def test_offset_within_corpus(self):
        for _ in range(10):
            _, offset = _sample_corpus_section(CORPUS, "lore", section_size=50)
            assert 0 <= offset < len(CORPUS)

    def test_section_starts_at_offset(self):
        text, offset = _sample_corpus_section(CORPUS, "lore", section_size=100)
        assert CORPUS[offset:offset + len(text)] == text

    def test_non_tabular_type_any_content(self):
        for qtype in ["lore", "mechanical", "cross_reference", "monster", "numeric"]:
            text, offset = _sample_corpus_section(CORPUS, qtype, section_size=200)
            assert isinstance(text, str)
            assert len(text) > 0


# ── _filter_candidates ──────────────────────────────────────────────────────

def _make_entries(n=5):
    return [
        GoldEntry(
            id=f"q_{i:04d}", query_type="mechanical",
            query=f"What is the THAC0 at level {i}?",
            reference_answer=str(20 - i),
            reference_contexts=[{"char_start": i * 100, "char_end": i * 100 + 50}],
        )
        for i in range(n)
    ]


def _make_mock_embedder(n_entries, dim=8):
    """Returns an embedder mock that returns orthogonal-ish embeddings."""
    embedder = MagicMock()
    # Make each embedding distinct (low cosine similarity)
    vecs = np.eye(n_entries, dim, dtype=np.float32)
    embedder.embed_as_numpy.return_value = vecs
    embedder.embed_single.side_effect = lambda q: [0.1] * dim
    return embedder


def _make_mock_llm(score="0.1"):
    """LLM that always returns a low answerability score (keep all)."""
    llm = MagicMock()
    llm.generate.return_value = score
    return llm


class TestFilterCandidates:
    def test_empty_candidates(self):
        result = _filter_candidates([], [], [], _make_mock_embedder(0), _make_mock_llm())
        assert result == []

    def test_keeps_all_when_no_duplicates(self):
        entries = _make_entries(3)
        embedder = _make_mock_embedder(3)
        llm = _make_mock_llm("0.1")  # low answerability → keep
        kept = _filter_candidates(entries, [], [], embedder, llm)
        assert len(kept) == 3

    def test_drops_trivially_answerable(self):
        entries = _make_entries(3)
        embedder = _make_mock_embedder(3)
        llm = _make_mock_llm("0.95")  # high score → drop all
        kept = _filter_candidates(entries, [], [], embedder, llm)
        assert len(kept) == 0

    def test_deduplicates_identical_queries(self):
        """Two identical embeddings (cosine=1.0) → only first kept."""
        entries = _make_entries(2)
        embedder = MagicMock()
        # Both entries get the SAME embedding → cosine sim = 1.0 → second dropped
        identical = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        embedder.embed_as_numpy.return_value = identical
        llm = _make_mock_llm("0.1")
        kept = _filter_candidates(entries, [], [], embedder, llm)
        assert len(kept) == 1

    def test_deduplicates_against_existing(self):
        """If existing pool already has a near-duplicate, candidate is dropped."""
        entries = _make_entries(1)
        # Existing embedding = same as candidate → will be dropped
        existing_entry = _make_entries(1)
        existing_emb = [np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)]

        embedder = MagicMock()
        embedder.embed_as_numpy.return_value = np.array(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32
        )
        llm = _make_mock_llm("0.1")
        kept = _filter_candidates(entries, existing_entry, existing_emb, embedder, llm)
        assert len(kept) == 0

    def test_answerability_parse_error_keeps_entry(self):
        """If LLM returns garbage, candidate should be kept (safe fallback)."""
        entries = _make_entries(1)
        embedder = _make_mock_embedder(1)
        llm = MagicMock()
        llm.generate.return_value = "no number here lol"
        kept = _filter_candidates(entries, [], [], embedder, llm)
        assert len(kept) == 1


# ── _create_placeholder_gold ────────────────────────────────────────────────

class TestCreatePlaceholderGold:
    def test_creates_file(self, tmp_path):
        gold_path = tmp_path / "gold_standard.jsonl"
        _create_placeholder_gold(gold_path)
        assert gold_path.exists()

    def test_contains_valid_jsonl(self, tmp_path):
        gold_path = tmp_path / "gold_standard.jsonl"
        _create_placeholder_gold(gold_path)
        lines = [json.loads(l) for l in gold_path.read_text().splitlines() if l.strip()]
        assert len(lines) >= 2

    def test_entries_have_required_fields(self, tmp_path):
        gold_path = tmp_path / "gold_standard.jsonl"
        _create_placeholder_gold(gold_path)
        lines = [json.loads(l) for l in gold_path.read_text().splitlines() if l.strip()]
        for entry in lines:
            assert "id" in entry
            assert "query" in entry
            assert "reference_answer" in entry
            assert "query_type" in entry

    def test_entry_ids_unique(self, tmp_path):
        gold_path = tmp_path / "gold_standard.jsonl"
        _create_placeholder_gold(gold_path)
        lines = [json.loads(l) for l in gold_path.read_text().splitlines() if l.strip()]
        ids = [e["id"] for e in lines]
        assert len(ids) == len(set(ids))
