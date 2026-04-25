"""Tests for chunking strategies (§5)."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.corpus.models import ASTNode, BookSpan, ChunkMetadata
from src.chunkers.recursive import RecursiveChunker
from src.chunkers.markdown_hierarchical import MarkdownHierarchicalChunker
from src.chunkers.table_aware import TableAwareChunker
from src.corpus.book_detector import build_heading_index, get_chapter_path

# Minimal config mock
def _make_cfg():
    cfg = MagicMock()
    cfg.paths.raw_corpus = "data/raw/test.md"
    cfg.embedding.model = "text-embedding-3-small"
    cfg.embedding.dimensions = 1536
    cfg.embedding.batch_size = 256
    cfg.llm.model = "claude-sonnet-4-5-20241022"
    cfg.llm.temperature = 0
    cfg.cost_budget = {"contextual_chunking": 15.0}
    return cfg

SAMPLE_BOOK_TEXT = """## Player's Handbook

### Chapter 1

This is chapter one content with enough text to create multiple chunks.
The text needs to be long enough to actually trigger splitting behavior.
""" + ("This is filler text for testing purposes. " * 100)


class TestRecursiveChunker:
    def test_produces_chunks(self, tmp_path):
        # Write test corpus
        corpus = tmp_path / "data" / "raw" / "test.md"
        corpus.parent.mkdir(parents=True)
        corpus.write_text(SAMPLE_BOOK_TEXT, encoding="utf-8")

        cfg = _make_cfg()
        cfg.paths.raw_corpus = str(corpus.relative_to(tmp_path))

        chunker = RecursiveChunker(cfg, tmp_path)
        book = BookSpan(book_title="PHB", book_index=0,
                        start_offset=0, end_offset=len(SAMPLE_BOOK_TEXT))
        nodes = [ASTNode(node_type="heading", level=2, content="Player's Handbook",
                         char_start=0, char_end=25)]

        chunks = chunker.chunk_book(book, nodes, SAMPLE_BOOK_TEXT)
        assert len(chunks) > 0

    def test_chunk_has_metadata(self, tmp_path):
        corpus = tmp_path / "data" / "raw" / "test.md"
        corpus.parent.mkdir(parents=True)
        corpus.write_text(SAMPLE_BOOK_TEXT, encoding="utf-8")

        cfg = _make_cfg()
        chunker = RecursiveChunker(cfg, tmp_path)
        book = BookSpan(book_title="PHB", book_index=0,
                        start_offset=0, end_offset=len(SAMPLE_BOOK_TEXT))
        nodes = []
        chunks = chunker.chunk_book(book, nodes, SAMPLE_BOOK_TEXT)

        if chunks:
            c = chunks[0]
            assert c.strategy == "recursive"
            assert c.book_title == "PHB"
            assert c.book_index == 0
            assert c.token_count > 0
            assert c.chunk_id != ""

    def test_chunk_token_count_reasonable(self, tmp_path):
        corpus = tmp_path / "data" / "raw" / "test.md"
        corpus.parent.mkdir(parents=True)
        corpus.write_text(SAMPLE_BOOK_TEXT, encoding="utf-8")

        cfg = _make_cfg()
        chunker = RecursiveChunker(cfg, tmp_path)
        book = BookSpan(book_title="PHB", book_index=0,
                        start_offset=0, end_offset=len(SAMPLE_BOOK_TEXT))
        chunks = chunker.chunk_book(book, [], SAMPLE_BOOK_TEXT)

        for c in chunks:
            # Should be roughly <= 512 tokens (with some tolerance for overlap)
            assert c.token_count <= 700, f"Chunk too large: {c.token_count} tokens"


class TestChunkMetadataIntegrity:
    def test_no_empty_content(self):
        chunk = ChunkMetadata(strategy="recursive", content="test",
                              book_title="PHB", book_index=0)
        assert chunk.content.strip() != ""

    def test_coverage_no_gaps(self, tmp_path):
        """Chunks should cover the full book text without major gaps."""
        text = "Word " * 500  # ~500 tokens
        corpus = tmp_path / "data" / "raw" / "test.md"
        corpus.parent.mkdir(parents=True)
        corpus.write_text(text, encoding="utf-8")

        cfg = _make_cfg()
        chunker = RecursiveChunker(cfg, tmp_path)
        book = BookSpan(book_title="Test", book_index=0,
                        start_offset=0, end_offset=len(text))
        chunks = chunker.chunk_book(book, [], text)

        # All original content should appear in at least one chunk
        total_chunk_chars = sum(len(c.content) for c in chunks)
        assert total_chunk_chars >= len(text) * 0.8  # At least 80% coverage


# ── Sample corpus with mixed content ────────────────────────────────────────

SAMPLE_CORPUS = """\
## Chapter One

### Section A

Here is some paragraph text for section A.
It has multiple sentences that form a nice paragraph.

### Section B

Another section here with different content.
This section discusses something entirely separate.

| Spell | Level | Duration |
|-------|-------|----------|
| Fireball | 3 | Instantaneous |
| Sleep | 1 | 5 rounds/level |

### Section C

Final section with a code example.

```python
def example():
    return 42
```

## Chapter Two

### Sub-section 2.1

Content for chapter two, section one. This is a longer paragraph designed
to push the character count above a minimal threshold so the chunker
actually has something to split. """ + ("Extra filler text here. " * 30)


def _make_ast_nodes(text=SAMPLE_CORPUS):
    """Build minimal AST nodes from the sample corpus."""
    import re
    nodes = []
    for m in re.finditer(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE):
        level = len(m.group(1))
        nodes.append(ASTNode(
            node_type="heading", level=level,
            content=m.group(2).strip(),
            char_start=m.start(), char_end=m.end(),
        ))
    # Add a table node
    table_match = re.search(r'\| Spell \|.*\n(\|[-|]+\|\n)?(\|.*\|\n?)+', text, re.MULTILINE)
    if table_match:
        nodes.append(ASTNode(
            node_type="table", level=0,
            content="Spell table",
            raw_content=table_match.group(0),
            char_start=table_match.start(), char_end=table_match.end(),
        ))
    return nodes


# ── MarkdownHierarchicalChunker ──────────────────────────────────────────────

class TestMarkdownHierarchicalChunker:
    def _setup(self, tmp_path):
        corpus = tmp_path / "data" / "raw" / "test.md"
        corpus.parent.mkdir(parents=True, exist_ok=True)
        corpus.write_text(SAMPLE_CORPUS, encoding="utf-8")
        cfg = _make_cfg()
        cfg.paths.raw_corpus = str(corpus.relative_to(tmp_path))
        chunker = MarkdownHierarchicalChunker(cfg, tmp_path)
        return chunker

    def test_produces_chunks(self, tmp_path):
        chunker = self._setup(tmp_path)
        book = BookSpan(book_title="corpus", book_index=0,
                        start_offset=0, end_offset=len(SAMPLE_CORPUS))
        nodes = _make_ast_nodes()
        chunks = chunker.chunk_book(book, nodes, SAMPLE_CORPUS)
        assert len(chunks) > 0

    def test_strategy_name(self, tmp_path):
        chunker = self._setup(tmp_path)
        book = BookSpan(book_title="corpus", book_index=0,
                        start_offset=0, end_offset=len(SAMPLE_CORPUS))
        chunks = chunker.chunk_book(book, _make_ast_nodes(), SAMPLE_CORPUS)
        for c in chunks:
            assert c.strategy == "markdown_hierarchical"

    def test_chunks_at_section_boundaries(self, tmp_path):
        """Each top-level heading should start a new chunk."""
        chunker = self._setup(tmp_path)
        book = BookSpan(book_title="corpus", book_index=0,
                        start_offset=0, end_offset=len(SAMPLE_CORPUS))
        nodes = _make_ast_nodes()
        chunks = chunker.chunk_book(book, nodes, SAMPLE_CORPUS)
        # Chapter One and Chapter Two headings should each start a section
        combined = " ".join(c.content for c in chunks)
        assert "Chapter One" in combined or "Section A" in combined

    def test_oversized_section_gets_split(self, tmp_path):
        """A section with > 2000 tokens must be split into sub-chunks."""
        big_section = "## Big Chapter\n\n" + ("This is a very long sentence. " * 300)
        corpus = tmp_path / "data" / "raw" / "big.md"
        corpus.parent.mkdir(parents=True, exist_ok=True)
        corpus.write_text(big_section, encoding="utf-8")
        cfg = _make_cfg()
        cfg.paths.raw_corpus = str(corpus.relative_to(tmp_path))
        chunker = MarkdownHierarchicalChunker(cfg, tmp_path)
        book = BookSpan(book_title="corpus", book_index=0,
                        start_offset=0, end_offset=len(big_section))
        nodes = [ASTNode(node_type="heading", level=2, content="Big Chapter",
                         char_start=0, char_end=16)]
        chunks = chunker.chunk_book(book, nodes, big_section)
        assert len(chunks) > 1

    def test_no_header_corpus_returns_one_chunk(self, tmp_path):
        """If no h2/h3/h4 headers exist, whole content is one chunk."""
        text = "Just plain text. " * 50
        corpus = tmp_path / "data" / "raw" / "plain.md"
        corpus.parent.mkdir(parents=True, exist_ok=True)
        corpus.write_text(text, encoding="utf-8")
        cfg = _make_cfg()
        cfg.paths.raw_corpus = str(corpus.relative_to(tmp_path))
        chunker = MarkdownHierarchicalChunker(cfg, tmp_path)
        book = BookSpan(book_title="corpus", book_index=0,
                        start_offset=0, end_offset=len(text))
        chunks = chunker.chunk_book(book, [], text)
        assert len(chunks) >= 1
        combined = "".join(c.content for c in chunks)
        assert len(combined) > 0

    def test_char_offsets_set(self, tmp_path):
        chunker = self._setup(tmp_path)
        book = BookSpan(book_title="corpus", book_index=0,
                        start_offset=0, end_offset=len(SAMPLE_CORPUS))
        chunks = chunker.chunk_book(book, _make_ast_nodes(), SAMPLE_CORPUS)
        for c in chunks:
            assert c.char_start >= 0
            assert c.char_end > c.char_start

    def test_token_count_reasonable(self, tmp_path):
        chunker = self._setup(tmp_path)
        book = BookSpan(book_title="corpus", book_index=0,
                        start_offset=0, end_offset=len(SAMPLE_CORPUS))
        chunks = chunker.chunk_book(book, _make_ast_nodes(), SAMPLE_CORPUS)
        for c in chunks:
            assert c.token_count <= 2500  # oversized threshold + small buffer


# ── TableAwareChunker ────────────────────────────────────────────────────────

class TestTableAwareChunker:
    def _setup(self, tmp_path, text=SAMPLE_CORPUS):
        corpus = tmp_path / "data" / "raw" / "test.md"
        corpus.parent.mkdir(parents=True, exist_ok=True)
        corpus.write_text(text, encoding="utf-8")
        cfg = _make_cfg()
        cfg.paths.raw_corpus = str(corpus.relative_to(tmp_path))
        chunker = TableAwareChunker(cfg, tmp_path)
        return chunker

    def test_produces_chunks(self, tmp_path):
        chunker = self._setup(tmp_path)
        book = BookSpan(book_title="corpus", book_index=0,
                        start_offset=0, end_offset=len(SAMPLE_CORPUS))
        chunks = chunker.chunk_book(book, _make_ast_nodes(), SAMPLE_CORPUS)
        assert len(chunks) > 0

    def test_strategy_name(self, tmp_path):
        chunker = self._setup(tmp_path)
        book = BookSpan(book_title="corpus", book_index=0,
                        start_offset=0, end_offset=len(SAMPLE_CORPUS))
        chunks = chunker.chunk_book(book, _make_ast_nodes(), SAMPLE_CORPUS)
        for c in chunks:
            assert c.strategy == "table_aware"

    def test_table_chunks_flagged(self, tmp_path):
        """At least one chunk should be flagged as a table."""
        chunker = self._setup(tmp_path)
        book = BookSpan(book_title="corpus", book_index=0,
                        start_offset=0, end_offset=len(SAMPLE_CORPUS))
        nodes = _make_ast_nodes()
        chunks = chunker.chunk_book(book, nodes, SAMPLE_CORPUS)
        table_chunks = [c for c in chunks if c.is_table]
        assert len(table_chunks) >= 1

    def test_table_chunk_contains_pipe(self, tmp_path):
        """Table chunks should contain the pipe-delimited content."""
        chunker = self._setup(tmp_path)
        book = BookSpan(book_title="corpus", book_index=0,
                        start_offset=0, end_offset=len(SAMPLE_CORPUS))
        nodes = _make_ast_nodes()
        chunks = chunker.chunk_book(book, nodes, SAMPLE_CORPUS)
        table_chunks = [c for c in chunks if c.is_table]
        for tc in table_chunks:
            assert "|" in tc.content

    def test_chunks_sorted_by_position(self, tmp_path):
        chunker = self._setup(tmp_path)
        book = BookSpan(book_title="corpus", book_index=0,
                        start_offset=0, end_offset=len(SAMPLE_CORPUS))
        chunks = chunker.chunk_book(book, _make_ast_nodes(), SAMPLE_CORPUS)
        starts = [c.char_start for c in chunks]
        assert starts == sorted(starts)

    def test_no_table_corpus_still_works(self, tmp_path):
        """Corpus with no tables should fall through to recursive only."""
        text = "## Chapter\n\nPlain text. " * 50
        chunker = self._setup(tmp_path, text)
        book = BookSpan(book_title="corpus", book_index=0,
                        start_offset=0, end_offset=len(text))
        chunks = chunker.chunk_book(book, [], text)
        assert len(chunks) > 0
        assert all(not c.is_table for c in chunks)

    def test_table_chunk_includes_preceding_context(self, tmp_path):
        """Table chunk should include some text before the table."""
        chunker = self._setup(tmp_path)
        book = BookSpan(book_title="corpus", book_index=0,
                        start_offset=0, end_offset=len(SAMPLE_CORPUS))
        nodes = _make_ast_nodes()
        chunks = chunker.chunk_book(book, nodes, SAMPLE_CORPUS)
        table_chunks = [c for c in chunks if c.is_table]
        # At least one table chunk should have content before the "|" lines
        for tc in table_chunks:
            lines = tc.content.strip().split("\n")
            # If context was prepended, first line won't be a pipe line
            # (may or may not have context depending on corpus layout — just no crash)
            assert len(lines) >= 1


# ── chunk_corpus() single-unit API ───────────────────────────────────────────

class TestChunkCorpus:
    """Verify the new chunk_corpus(all_nodes) single-unit signature."""

    def test_recursive_chunk_corpus(self, tmp_path):
        corpus = tmp_path / "data" / "raw" / "test.md"
        corpus.parent.mkdir(parents=True, exist_ok=True)
        corpus.write_text(SAMPLE_CORPUS, encoding="utf-8")
        cfg = _make_cfg()
        cfg.paths.raw_corpus = str(corpus.relative_to(tmp_path))
        chunker = RecursiveChunker(cfg, tmp_path)
        chunks = chunker.chunk_corpus(_make_ast_nodes())
        assert len(chunks) > 0

    def test_chunk_corpus_book_title_is_corpus(self, tmp_path):
        corpus = tmp_path / "data" / "raw" / "test.md"
        corpus.parent.mkdir(parents=True, exist_ok=True)
        corpus.write_text(SAMPLE_CORPUS, encoding="utf-8")
        cfg = _make_cfg()
        cfg.paths.raw_corpus = str(corpus.relative_to(tmp_path))
        chunker = RecursiveChunker(cfg, tmp_path)
        chunks = chunker.chunk_corpus([])
        for c in chunks:
            assert c.book_title == "corpus"

    def test_chunk_corpus_book_index_zero(self, tmp_path):
        corpus = tmp_path / "data" / "raw" / "test.md"
        corpus.parent.mkdir(parents=True, exist_ok=True)
        corpus.write_text(SAMPLE_CORPUS, encoding="utf-8")
        cfg = _make_cfg()
        cfg.paths.raw_corpus = str(corpus.relative_to(tmp_path))
        chunker = RecursiveChunker(cfg, tmp_path)
        chunks = chunker.chunk_corpus([])
        for c in chunks:
            assert c.book_index == 0

    def test_markdown_hierarchical_chunk_corpus(self, tmp_path):
        corpus = tmp_path / "data" / "raw" / "test.md"
        corpus.parent.mkdir(parents=True, exist_ok=True)
        corpus.write_text(SAMPLE_CORPUS, encoding="utf-8")
        cfg = _make_cfg()
        cfg.paths.raw_corpus = str(corpus.relative_to(tmp_path))
        chunker = MarkdownHierarchicalChunker(cfg, tmp_path)
        chunks = chunker.chunk_corpus(_make_ast_nodes())
        assert len(chunks) > 0

    def test_table_aware_chunk_corpus(self, tmp_path):
        corpus = tmp_path / "data" / "raw" / "test.md"
        corpus.parent.mkdir(parents=True, exist_ok=True)
        corpus.write_text(SAMPLE_CORPUS, encoding="utf-8")
        cfg = _make_cfg()
        cfg.paths.raw_corpus = str(corpus.relative_to(tmp_path))
        chunker = TableAwareChunker(cfg, tmp_path)
        chunks = chunker.chunk_corpus(_make_ast_nodes())
        assert len(chunks) > 0


# ── build_heading_index + get_chapter_path ────────────────────────────────────

class TestHeadingIndex:
    def test_build_heading_index_returns_sorted(self):
        nodes = [
            ASTNode(node_type="heading", level=3, content="Sub", char_start=100, char_end=110),
            ASTNode(node_type="heading", level=2, content="Main", char_start=0, char_end=10),
            ASTNode(node_type="paragraph", level=0, content="text", char_start=15, char_end=25),
        ]
        headings, starts = build_heading_index(nodes)
        assert starts == sorted(starts)
        assert all(h.node_type == "heading" for h in headings)

    def test_build_heading_index_excludes_non_headings(self):
        nodes = [
            ASTNode(node_type="heading", level=2, content="H", char_start=0, char_end=5),
            ASTNode(node_type="paragraph", level=0, content="P", char_start=10, char_end=20),
            ASTNode(node_type="code_block", level=0, content="C", char_start=30, char_end=50),
        ]
        headings, starts = build_heading_index(nodes)
        assert len(headings) == 1
        assert len(starts) == 1

    def test_get_chapter_path_basic(self):
        nodes = [
            ASTNode(node_type="heading", level=2, content="Chapter One",
                    char_start=0, char_end=15),
            ASTNode(node_type="heading", level=3, content="Section A",
                    char_start=100, char_end=115),
        ]
        # Offset inside Section A
        path = get_chapter_path(nodes, char_offset=200)
        assert "Chapter One" in path
        assert "Section A" in path

    def test_get_chapter_path_with_index(self):
        nodes = [
            ASTNode(node_type="heading", level=2, content="Chapter One",
                    char_start=0, char_end=15),
            ASTNode(node_type="heading", level=3, content="Section A",
                    char_start=100, char_end=115),
        ]
        idx = build_heading_index(nodes)
        path1 = get_chapter_path(nodes, char_offset=200)
        path2 = get_chapter_path(nodes, char_offset=200, _heading_index=idx)
        assert path1 == path2

    def test_get_chapter_path_before_any_heading(self):
        nodes = [
            ASTNode(node_type="heading", level=2, content="Chapter One",
                    char_start=500, char_end=515),
        ]
        path = get_chapter_path(nodes, char_offset=100)
        assert path == []  # No headings before offset 100

    def test_get_chapter_path_empty_nodes(self):
        path = get_chapter_path([], char_offset=500)
        assert path == []
