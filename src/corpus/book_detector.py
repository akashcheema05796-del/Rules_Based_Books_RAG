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
    """Normalize a title for matching.

    Handles: lowercase, NFKC, replacement char / trademarks / curly quotes,
    markdown bold+italic markers (``**``, ``_``), then strips all non-word
    non-space chars.
    """
    title = unicodedata.normalize("NFKC", title).lower().strip()
    # Drop encoding-replacement + trademark glyphs
    for ch in ("\uFFFD", "\u00AE", "\u2122", "\u00A9"):
        title = title.replace(ch, "")
    # Drop markdown emphasis markers so `**The Complete Book of Elves**` normalizes
    title = title.replace("**", " ").replace("__", " ").replace("*", " ").replace("_", " ")
    # Normalize curly quotes and dashes before the punctuation sweep
    title = title.replace("\u2019", "'").replace("\u2018", "'")
    title = title.replace("\u201C", '"').replace("\u201D", '"')
    title = title.replace("\u2013", "-").replace("\u2014", "-")
    title = re.sub(r"[^\w\s]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
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

    # Find all h2 headings from the AST
    h2_nodes = [n for n in ast_nodes if n.node_type == "heading" and n.level == 2]
    logger.info(f"AST produced {len(h2_nodes)} h2 headers")

    # Regex fallback: the AST sometimes loses h2 nodes. Rescan the raw text and
    # synthesize ASTNode-like records with correct char_start so downstream code
    # keeps working.
    if len(h2_nodes) < 50:
        logger.warning("AST h2 count < 50; scanning raw text with regex fallback.")
        h2_nodes = []
        for m in re.finditer(r"^##[ \t]+(.+?)[ \t]*$", text, flags=re.MULTILINE):
            node = ASTNode(
                node_type="heading", level=2,
                content=m.group(1).strip(),
                raw_content=m.group(0),
                char_start=m.start(), char_end=m.end(),
            )
            h2_nodes.append(node)
        logger.info(f"Regex fallback found {len(h2_nodes)} h2 headers")

    # Match against manifest. Strategy: for each manifest title, pick the
    # earliest unused h2 whose normalized text contains the manifest title as a
    # substring. This is more robust than greedy per-node matching, which mis-
    # assigns long/short title pairs.
    matched: list[tuple] = []
    used_nodes: set[int] = set()
    used_titles: set[str] = set()

    norm_nodes = [(i, _normalize_title(n.content)) for i, n in enumerate(h2_nodes)]
    # Process manifest titles longest-first so `Dungeon Master Option: High-Level
    # Campaigns` is matched before `Dungeon Master Guide` doesn't steal its node.
    ordered_manifest = sorted(
        normalized_manifest.items(), key=lambda kv: -len(kv[0])
    )
    for norm_key, orig_title in ordered_manifest:
        best_idx = None
        for i, norm_text in norm_nodes:
            if i in used_nodes:
                continue
            if norm_key in norm_text or norm_text == norm_key:
                best_idx = i
                break
        if best_idx is not None:
            matched.append((h2_nodes[best_idx], orig_title))
            used_nodes.add(best_idx)
            used_titles.add(orig_title)

    unmatched = [h2_nodes[i] for i, _ in norm_nodes if i not in used_nodes]

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


def build_heading_index(ast_nodes: list[ASTNode]) -> tuple[list, list[int]]:
    """Pre-compute sorted heading nodes + their char_starts for fast lookup.

    Call once per corpus/chunking pass and pass the result to
    `get_chapter_path_fast` for O(log n) per chunk instead of O(n).

    Returns:
        (sorted_heading_nodes, sorted_starts)
    """
    headings = sorted(
        (n for n in ast_nodes if n.node_type == "heading"),
        key=lambda n: n.char_start,
    )
    starts = [n.char_start for n in headings]
    return headings, starts


def get_chapter_path(
    ast_nodes: list[ASTNode],
    char_offset: int,
    max_depth: int = 4,
    _heading_index: tuple | None = None,
) -> list[str]:
    """Get the chapter path (heading hierarchy) for a character offset.

    Returns list like ["Chapter 3", "The Warrior", "Fighters"].

    Pass `_heading_index = build_heading_index(ast_nodes)` once per batch to
    avoid re-sorting on every call (critical for large corpora).
    """
    import bisect

    if _heading_index is not None:
        heading_nodes, starts = _heading_index
    else:
        heading_nodes = sorted(
            (n for n in ast_nodes if n.node_type == "heading"),
            key=lambda n: n.char_start,
        )
        starts = [n.char_start for n in heading_nodes]

    path = []
    for level in range(2, max_depth + 1):
        idx = bisect.bisect_right(starts, char_offset) - 1
        j = idx
        best = None
        while j >= 0:
            if heading_nodes[j].level == level:
                best = heading_nodes[j]
                break
            j -= 1
        if best:
            path.append(best.content)

    return path
