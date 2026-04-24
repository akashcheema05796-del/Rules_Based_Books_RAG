"""
Book boundary detection — manifest-driven (spec §4.2).

Uses h2 headers matched against books_manifest.yaml to detect 26 book boundaries.
"""

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

import yaml

from src.corpus.models import ASTNode, BookSpan

logger = logging.getLogger(__name__)


def _normalize_title(title: str) -> str:
    """Normalize a title for matching: lowercase, strip punctuation/trademarks."""
    title = title.lower().strip()
    title = title.replace("®", "").replace("™", "").replace("©", "")
    title = re.sub(r'[^\w\s]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title


def _load_manifest(manifest_path: Path) -> list[str]:
    """Load book titles from manifest YAML."""
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    return data.get("books", [])


def detect_books(
    ast_nodes: list[ASTNode],
    raw_path: Path,
    manifest_path: Path,
    interim_path: Path,
) -> list[BookSpan]:
    """Detect book boundaries using h2 headers matched against manifest.

    Args:
        ast_nodes: Parsed AST nodes.
        raw_path: Path to raw corpus (for total size).
        manifest_path: Path to books_manifest.yaml.
        interim_path: Path for interim outputs (unmatched headers).

    Returns:
        List of BookSpan objects.
    """
    manifest_titles = _load_manifest(manifest_path)
    normalized_manifest = {_normalize_title(t): t for t in manifest_titles}

    text = raw_path.read_text(encoding="utf-8")
    total_chars = len(text)

    # Find all h2 headings
    h2_nodes = [n for n in ast_nodes if n.node_type == "heading" and n.level == 2]
    logger.info(f"Found {len(h2_nodes)} h2 headers")

    # Match against manifest
    matched = []
    unmatched = []
    used_titles = set()

    for node in h2_nodes:
        normalized = _normalize_title(node.content)
        # Try exact match
        if normalized in normalized_manifest:
            original_title = normalized_manifest[normalized]
            if original_title not in used_titles:
                matched.append((node, original_title))
                used_titles.add(original_title)
                continue

        # Try fuzzy match (substring)
        found = False
        for norm_key, orig_title in normalized_manifest.items():
            if orig_title in used_titles:
                continue
            if norm_key in normalized or normalized in norm_key:
                matched.append((node, orig_title))
                used_titles.add(orig_title)
                found = True
                break

        if not found:
            unmatched.append(node)

    # Sort matched by position
    matched.sort(key=lambda x: x[0].char_start)

    # Create BookSpan objects
    books = []
    for i, (node, title) in enumerate(matched):
        start = node.char_start
        if i + 1 < len(matched):
            end = matched[i + 1][0].char_start
        else:
            end = total_chars

        books.append(BookSpan(
            book_title=title,
            book_index=i,
            start_offset=start,
            end_offset=end,
            h2_header_text=node.content,
        ))

    # Dump unmatched headers
    if unmatched:
        interim_path.mkdir(parents=True, exist_ok=True)
        unmatched_file = interim_path / "unmatched_headers.txt"
        with open(unmatched_file, "w", encoding="utf-8") as f:
            for node in unmatched:
                f.write(f"Line offset {node.char_start}: {node.content}\n")
        logger.warning(f"{len(unmatched)} unmatched h2 headers → {unmatched_file}")

    # Save book spans
    interim_path.mkdir(parents=True, exist_ok=True)
    spans_file = interim_path / "book_spans.json"
    spans_data = [b.model_dump() for b in books]
    spans_file.write_text(json.dumps(spans_data, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Detected {len(books)} books (target: {len(manifest_titles)})")
    if len(books) != len(manifest_titles):
        missing = set(manifest_titles) - used_titles
        logger.warning(f"Missing books: {missing}")

    return books


def get_book_for_offset(books: list[BookSpan], char_offset: int) -> Optional[BookSpan]:
    """Find which book a character offset belongs to."""
    for book in books:
        if book.start_offset <= char_offset < book.end_offset:
            return book
    return None


def get_chapter_path(
    ast_nodes: list[ASTNode],
    char_offset: int,
    max_depth: int = 4,
) -> list[str]:
    """Get the chapter path (heading hierarchy) for a character offset.

    Returns list like ["Chapter 3", "The Warrior", "Fighters"].
    """
    path = []
    heading_nodes = [n for n in ast_nodes if n.node_type == "heading"]

    for level in range(2, max_depth + 1):
        best = None
        for h in heading_nodes:
            if h.level == level and h.char_start <= char_offset:
                if best is None or h.char_start > best.char_start:
                    best = h
        if best:
            path.append(best.content)

    return path
