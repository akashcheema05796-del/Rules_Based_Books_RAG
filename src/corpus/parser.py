"""
Markdown AST parser using markdown-it-py (spec §4.3).

Preserves tables as atomic nodes, code blocks as atomic,
lists with depth annotation, blockquotes atomic when short.
All chunkers consume AST nodes, not raw text.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from markdown_it import MarkdownIt

from src.corpus.models import ASTNode

logger = logging.getLogger(__name__)


def parse_corpus(raw_path: Path) -> list[ASTNode]:
    """Parse a markdown corpus file into AST nodes.

    Args:
        raw_path: Path to the raw markdown file.

    Returns:
        List of top-level ASTNode instances.
    """
    logger.info(f"Reading corpus from {raw_path}")
    text = raw_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    total_chars = len(text)
    logger.info(f"Corpus: {len(lines)} lines, {total_chars} chars")

    md = MarkdownIt("default", {"html": True})
    md.enable("table")
    tokens = md.parse(text)

    nodes = _tokens_to_nodes(tokens, text, lines)
    logger.info(f"Parsed {len(nodes)} top-level AST nodes")
    return nodes


def _tokens_to_nodes(tokens, full_text, lines) -> list[ASTNode]:
    """Convert markdown-it tokens to ASTNode tree."""
    nodes = []
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token.type == "heading_open":
            level = int(token.tag[1])  # h1 -> 1, h2 -> 2, etc.
            content_parts = []
            i += 1
            while i < len(tokens) and tokens[i].type != "heading_close":
                if tokens[i].content:
                    content_parts.append(tokens[i].content)
                if tokens[i].children:
                    for child in tokens[i].children:
                        if child.content:
                            content_parts.append(child.content)
                i += 1
            heading_text = " ".join(content_parts).strip()
            line_start = token.map[0] if token.map else 0
            line_end = token.map[1] if token.map else line_start + 1
            char_start = _line_to_char(lines, line_start)
            char_end = _line_to_char(lines, line_end)
            raw = "\n".join(lines[line_start:line_end])

            nodes.append(ASTNode(
                node_type="heading",
                level=level,
                content=heading_text,
                raw_content=raw,
                char_start=char_start,
                char_end=char_end,
            ))

        elif token.type == "paragraph_open":
            content_parts = []
            line_start = token.map[0] if token.map else 0
            i += 1
            while i < len(tokens) and tokens[i].type != "paragraph_close":
                if tokens[i].content:
                    content_parts.append(tokens[i].content)
                if tokens[i].children:
                    for child in tokens[i].children:
                        if child.content:
                            content_parts.append(child.content)
                i += 1
            line_end = tokens[i].map[1] if (i < len(tokens) and tokens[i].map) else (line_start + 1)
            para_text = " ".join(content_parts).strip()
            char_start = _line_to_char(lines, line_start)
            char_end = _line_to_char(lines, line_end)
            raw = "\n".join(lines[line_start:line_end])

            # Check if paragraph contains a table (pipe-delimited)
            is_table = _is_table_content(raw)
            if is_table:
                table_node = _parse_table(raw, char_start, char_end)
                if table_node:
                    nodes.append(table_node)
                else:
                    nodes.append(ASTNode(
                        node_type="paragraph", content=para_text,
                        raw_content=raw, char_start=char_start, char_end=char_end,
                    ))
            else:
                nodes.append(ASTNode(
                    node_type="paragraph", content=para_text,
                    raw_content=raw, char_start=char_start, char_end=char_end,
                ))

        elif token.type == "fence" or token.type == "code_block":
            line_start = token.map[0] if token.map else 0
            line_end = token.map[1] if token.map else line_start + 1
            char_start = _line_to_char(lines, line_start)
            char_end = _line_to_char(lines, line_end)
            raw = "\n".join(lines[line_start:line_end])

            nodes.append(ASTNode(
                node_type="code_block", content=token.content or "",
                raw_content=raw, char_start=char_start, char_end=char_end,
                meta={"info": token.info or ""},
            ))

        elif token.type == "blockquote_open":
            content_parts = []
            line_start = token.map[0] if token.map else 0
            depth = 0
            i += 1
            while i < len(tokens) and tokens[i].type != "blockquote_close":
                if tokens[i].content:
                    content_parts.append(tokens[i].content)
                if tokens[i].children:
                    for child in tokens[i].children:
                        if child.content:
                            content_parts.append(child.content)
                i += 1
            line_end = tokens[i].map[1] if (i < len(tokens) and tokens[i].map) else (line_start + 1)
            char_start = _line_to_char(lines, line_start)
            char_end = _line_to_char(lines, line_end)
            raw = "\n".join(lines[line_start:line_end])

            nodes.append(ASTNode(
                node_type="blockquote", content=" ".join(content_parts).strip(),
                raw_content=raw, char_start=char_start, char_end=char_end,
            ))

        elif token.type in ("bullet_list_open", "ordered_list_open"):
            list_type = "bullet" if token.type == "bullet_list_open" else "ordered"
            line_start = token.map[0] if token.map else 0
            close_type = token.type.replace("_open", "_close")
            list_items = []
            i += 1
            while i < len(tokens) and tokens[i].type != close_type:
                if tokens[i].type == "list_item_open":
                    item_parts = []
                    i += 1
                    while i < len(tokens) and tokens[i].type != "list_item_close":
                        if tokens[i].content:
                            item_parts.append(tokens[i].content)
                        if tokens[i].children:
                            for child in tokens[i].children:
                                if child.content:
                                    item_parts.append(child.content)
                        i += 1
                    list_items.append(" ".join(item_parts).strip())
                i += 1
            line_end = tokens[i].map[1] if (i < len(tokens) and tokens[i].map) else (line_start + 1)
            char_start = _line_to_char(lines, line_start)
            char_end = _line_to_char(lines, line_end)
            raw = "\n".join(lines[line_start:line_end])

            nodes.append(ASTNode(
                node_type="list", content="\n".join(list_items),
                raw_content=raw, char_start=char_start, char_end=char_end,
                meta={"list_type": list_type, "item_count": len(list_items)},
            ))

        elif token.type == "html_block":
            line_start = token.map[0] if token.map else 0
            line_end = token.map[1] if token.map else line_start + 1
            char_start = _line_to_char(lines, line_start)
            char_end = _line_to_char(lines, line_end)
            raw = "\n".join(lines[line_start:line_end])

            # Check if HTML block is actually a table
            if _is_table_content(raw):
                table_node = _parse_table(raw, char_start, char_end)
                if table_node:
                    nodes.append(table_node)
                    i += 1
                    continue

            nodes.append(ASTNode(
                node_type="html_block", content=token.content or "",
                raw_content=raw, char_start=char_start, char_end=char_end,
            ))

        # Also handle inline tables that markdown-it may not parse
        elif token.type == "table_open":
            line_start = token.map[0] if token.map else 0
            line_end = token.map[1] if token.map else (line_start + 1)
            i += 1
            # Skip to table_close
            while i < len(tokens) and tokens[i].type != "table_close":
                i += 1
            char_start = _line_to_char(lines, line_start)
            char_end = _line_to_char(lines, line_end)
            raw = "\n".join(lines[line_start:line_end])

            table_node = _parse_table(raw, char_start, char_end)
            if table_node:
                nodes.append(table_node)

        i += 1

    # Post-process: detect tables in raw lines that parser missed
    nodes = _detect_missed_tables(nodes, lines, full_text)
    return nodes


def _line_to_char(lines, line_num):
    """Convert line number to character offset."""
    offset = 0
    for i in range(min(line_num, len(lines))):
        offset += len(lines[i]) + 1  # +1 for newline
    return offset


def _is_table_content(text):
    """Check if text looks like a markdown table."""
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return False
    pipe_lines = sum(1 for line in lines if "|" in line)
    return pipe_lines >= 2 and pipe_lines / len(lines) > 0.5


def _parse_table(raw_text, char_start, char_end):
    """Parse a markdown table into an ASTNode with structured data."""
    lines = [l.strip() for l in raw_text.strip().split("\n") if l.strip()]
    table_lines = [l for l in lines if "|" in l]

    if len(table_lines) < 2:
        return None

    def parse_row(line):
        cells = [c.strip() for c in line.split("|")]
        # Remove empty first/last if line starts/ends with |
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        return cells

    header_row = parse_row(table_lines[0])

    # Check for alignment row (contains dashes and colons)
    align_row = None
    data_start = 1
    if len(table_lines) > 1:
        potential_align = table_lines[1]
        if re.match(r'^[\s|:\-]+$', potential_align):
            align_row = parse_row(potential_align)
            data_start = 2

    body_rows = [parse_row(line) for line in table_lines[data_start:]]

    return ASTNode(
        node_type="table",
        content=raw_text,
        raw_content=raw_text,
        char_start=char_start,
        char_end=char_end,
        header_row=header_row,
        align_row=align_row,
        body_rows=body_rows,
    )


def _detect_missed_tables(nodes, lines, full_text):
    """Scan for tables that the parser might have missed (in paragraph nodes)."""
    result = []
    for node in nodes:
        if node.node_type == "paragraph" and _is_table_content(node.raw_content):
            table = _parse_table(node.raw_content, node.char_start, node.char_end)
            if table:
                result.append(table)
                continue
        result.append(node)
    return result


def save_ast(nodes: list[ASTNode], output_path: Path) -> None:
    """Save AST nodes to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [node.model_dump() for node in nodes]
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"AST saved to {output_path} ({len(nodes)} nodes)")


def load_ast(input_path: Path) -> list[ASTNode]:
    """Load AST nodes from JSON."""
    data = json.loads(input_path.read_text(encoding="utf-8"))
    return [ASTNode(**d) for d in data]
