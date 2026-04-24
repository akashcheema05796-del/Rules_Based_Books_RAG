"""Tests for corpus preparation (§4)."""

import json
import tempfile
from pathlib import Path

import pytest

from src.corpus.parser import parse_corpus, _is_table_content, _parse_table
from src.corpus.models import ASTNode, ChunkMetadata, BookSpan, GoldEntry


# --- Test Data ---

SAMPLE_MARKDOWN = """## Player's Handbook

### Chapter 1: Character Creation

This is some introductory text about creating characters in AD&D.

#### Ability Scores

Each character has six ability scores: Strength, Dexterity, Constitution,
Intelligence, Wisdom, and Charisma.

| Ability Score | Modifier |
|---|---|
| 3 | -3 |
| 4-5 | -2 |
| 6-8 | -1 |
| 9-12 | 0 |
| 13-15 | +1 |
| 16-17 | +2 |
| 18 | +3 |

### Chapter 2: Races

```
Special Code Block Example
This should be kept atomic.
```

## Dungeon Master Guide

### Chapter 1: Running the Game

The DM controls the game world and adjudicates rules.
"""


class TestParser:
    """Tests for markdown AST parser."""

    def test_parse_produces_nodes(self, tmp_path):
        corpus_file = tmp_path / "test_corpus.md"
        corpus_file.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
        nodes = parse_corpus(corpus_file)
        assert len(nodes) > 0

    def test_headings_detected(self, tmp_path):
        corpus_file = tmp_path / "test_corpus.md"
        corpus_file.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
        nodes = parse_corpus(corpus_file)
        headings = [n for n in nodes if n.node_type == "heading"]
        assert len(headings) >= 4  # h2, h3, h4 headers

    def test_h2_headings(self, tmp_path):
        corpus_file = tmp_path / "test_corpus.md"
        corpus_file.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
        nodes = parse_corpus(corpus_file)
        h2s = [n for n in nodes if n.node_type == "heading" and n.level == 2]
        assert len(h2s) >= 2
        titles = [h.content for h in h2s]
        assert any("Player" in t for t in titles)

    def test_tables_detected(self, tmp_path):
        corpus_file = tmp_path / "test_corpus.md"
        corpus_file.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
        nodes = parse_corpus(corpus_file)
        tables = [n for n in nodes if n.node_type == "table"]
        assert len(tables) >= 1

    def test_table_structure(self, tmp_path):
        corpus_file = tmp_path / "test_corpus.md"
        corpus_file.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
        nodes = parse_corpus(corpus_file)
        tables = [n for n in nodes if n.node_type == "table"]
        if tables:
            table = tables[0]
            assert table.header_row is not None
            assert table.body_rows is not None
            assert len(table.body_rows) > 0

    def test_code_blocks_detected(self, tmp_path):
        corpus_file = tmp_path / "test_corpus.md"
        corpus_file.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
        nodes = parse_corpus(corpus_file)
        code_blocks = [n for n in nodes if n.node_type == "code_block"]
        assert len(code_blocks) >= 1


class TestIsTableContent:
    def test_table_text(self):
        assert _is_table_content("| a | b |\n|---|---|\n| 1 | 2 |")

    def test_non_table(self):
        assert not _is_table_content("Hello world\nNo pipes here")


class TestModels:
    def test_chunk_metadata_auto_id(self):
        chunk = ChunkMetadata(strategy="recursive", content="test content",
                              book_title="PHB", book_index=0)
        assert chunk.chunk_id != ""
        assert len(chunk.chunk_id) == 16

    def test_chunk_metadata_deterministic_id(self):
        chunk1 = ChunkMetadata(strategy="recursive", content="same",
                               book_title="PHB", book_index=0, char_start=100)
        chunk2 = ChunkMetadata(strategy="recursive", content="same",
                               book_title="PHB", book_index=0, char_start=100)
        assert chunk1.chunk_id == chunk2.chunk_id

    def test_gold_entry(self):
        entry = GoldEntry(id="q_0001", query_type="mechanical",
                          query="What is THAC0?", reference_answer="16")
        assert entry.is_multi_hop is False
        assert entry.validator == "auto"

    def test_book_span(self):
        span = BookSpan(book_title="PHB", book_index=0,
                        start_offset=0, end_offset=1000)
        assert span.book_title == "PHB"
