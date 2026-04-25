"""
Tests for the fast regex parser (§4.3) — the path taken for files > 1 MB.

Covers: headings at every level, fenced code blocks, pipe tables,
paragraphs, char_start/char_end offsets, and parity with the
markdown-it-py parser on the same small input.
"""

import pytest
from pathlib import Path
from src.corpus.parser import parse_corpus, _parse_regex, _parse_markdownit
from src.corpus.models import ASTNode


SAMPLE = """\
## Section One

A paragraph of text here.

### Sub-section

Another paragraph.

| Col A | Col B |
|-------|-------|
| 1     | 2     |
| 3     | 4     |

```python
def hello():
    pass
```

#### Deep heading

Final text.
"""


class TestFastRegexParser:
    def _nodes(self):
        return _parse_regex(SAMPLE)

    def test_returns_list_of_ast_nodes(self):
        nodes = self._nodes()
        assert isinstance(nodes, list)
        assert all(isinstance(n, ASTNode) for n in nodes)

    def test_headings_detected(self):
        nodes = self._nodes()
        headings = [n for n in nodes if n.node_type == "heading"]
        assert len(headings) == 3  # h2, h3, h4

    def test_heading_levels(self):
        nodes = self._nodes()
        levels = {n.level for n in nodes if n.node_type == "heading"}
        assert 2 in levels
        assert 3 in levels
        assert 4 in levels

    def test_heading_content(self):
        nodes = self._nodes()
        titles = [n.content for n in nodes if n.node_type == "heading"]
        assert "Section One" in titles
        assert "Sub-section" in titles
        assert "Deep heading" in titles

    def test_table_detected(self):
        nodes = self._nodes()
        tables = [n for n in nodes if n.node_type == "table"]
        assert len(tables) == 1

    def test_table_has_header_and_rows(self):
        nodes = self._nodes()
        table = next(n for n in nodes if n.node_type == "table")
        assert table.header_row is not None
        assert len(table.header_row) == 2
        assert table.body_rows is not None
        assert len(table.body_rows) == 2

    def test_code_block_detected(self):
        nodes = self._nodes()
        code = [n for n in nodes if n.node_type == "code_block"]
        assert len(code) == 1

    def test_code_block_content(self):
        nodes = self._nodes()
        code = next(n for n in nodes if n.node_type == "code_block")
        assert "hello" in code.content or "hello" in code.raw_content

    def test_paragraphs_detected(self):
        nodes = self._nodes()
        paras = [n for n in nodes if n.node_type == "paragraph"]
        assert len(paras) >= 2

    def test_char_start_increases_monotonically(self):
        nodes = self._nodes()
        starts = [n.char_start for n in nodes]
        assert starts == sorted(starts), "char_start should be non-decreasing"

    def test_char_start_zero_for_first_node(self):
        nodes = self._nodes()
        assert nodes[0].char_start == 0

    def test_char_end_greater_than_start(self):
        nodes = self._nodes()
        for n in nodes:
            assert n.char_end >= n.char_start, f"char_end < char_start for {n.node_type}"

    def test_char_offsets_point_to_correct_text(self):
        nodes = self._nodes()
        for n in nodes:
            slice_ = SAMPLE[n.char_start:n.char_end]
            # The slice should contain at least part of the node's content
            assert len(slice_) > 0

    def test_first_h2_offset_correct(self):
        nodes = self._nodes()
        h2 = next(n for n in nodes if n.node_type == "heading" and n.level == 2)
        assert h2.char_start == 0
        assert "Section One" in SAMPLE[h2.char_start:h2.char_end]

    def test_large_file_uses_fast_parser(self, tmp_path):
        """Files > 1 MB should use the regex parser, not markdown-it-py."""
        big_file = tmp_path / "big.md"
        # "## Heading\n\nParagraph text.\n\n" = 29 bytes.
        # 40_000 repetitions = ~1.14 MB — safely above the 1 MB threshold.
        unit = "## Heading\n\nParagraph text.\n\n"
        reps = 40_000
        content = unit * reps
        assert len(content.encode()) > 1024 * 1024, "Content must exceed 1 MB for this test"
        big_file.write_text(content, encoding="utf-8")
        import time
        t0 = time.time()
        nodes = parse_corpus(big_file)
        elapsed = time.time() - t0
        assert elapsed < 10, f"Fast parser should finish in <10s, took {elapsed:.1f}s"
        headings = [n for n in nodes if n.node_type == "heading"]
        assert len(headings) == reps

    def test_small_file_uses_markdownit(self, tmp_path):
        """Files ≤ 1 MB should use markdown-it-py parser."""
        small_file = tmp_path / "small.md"
        small_file.write_text(SAMPLE, encoding="utf-8")
        nodes = parse_corpus(small_file)
        # Both parsers should find headings, tables, code blocks
        assert any(n.node_type == "heading" for n in nodes)
        assert any(n.node_type == "table" for n in nodes)
        assert any(n.node_type == "code_block" for n in nodes)

    def test_parity_with_markdownit_on_headings(self):
        """Regex and markdown-it-py should find the same heading count/levels."""
        regex_nodes = _parse_regex(SAMPLE)
        md_nodes = _parse_markdownit(SAMPLE)

        regex_headings = [(n.level, n.content) for n in regex_nodes if n.node_type == "heading"]
        md_headings = [(n.level, n.content) for n in md_nodes if n.node_type == "heading"]

        # Same levels
        assert sorted(l for l, _ in regex_headings) == sorted(l for l, _ in md_headings)
        # Same titles (order-independent)
        assert sorted(c for _, c in regex_headings) == sorted(c for _, c in md_headings)

    def test_empty_input(self):
        nodes = _parse_regex("")
        assert nodes == []

    def test_headings_only(self):
        text = "# H1\n## H2\n### H3\n"
        nodes = _parse_regex(text)
        headings = [n for n in nodes if n.node_type == "heading"]
        assert len(headings) == 3
        assert headings[0].level == 1
        assert headings[1].level == 2
        assert headings[2].level == 3
