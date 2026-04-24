"""
Corpus profiling (spec §4.1).
Generates results/corpus_profile.json with corpus statistics.
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path

from src.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)


def profile_corpus(raw_path: Path, output_path: Path, cfg=None) -> dict:
    """Profile the corpus and write statistics.

    Args:
        raw_path: Path to raw markdown corpus.
        output_path: Path for output JSON.
        cfg: Configuration (optional).

    Returns:
        Profile dictionary.
    """
    text = raw_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    total_tokens = count_tokens(text)

    # Header distribution
    header_counts = Counter()
    header_samples = {}
    for i, line in enumerate(lines):
        match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if match:
            level = len(match.group(1))
            header_counts[f"h{level}"] += 1
            key = f"h{level}"
            if key not in header_samples:
                header_samples[key] = []
            if len(header_samples[key]) < 5:
                header_samples[key].append({
                    "line": i + 1,
                    "text": match.group(2).strip()[:100],
                })

    # Table detection
    table_lines = 0
    in_table = False
    table_count = 0
    for line in lines:
        if "|" in line and line.strip().startswith("|"):
            if not in_table:
                table_count += 1
                in_table = True
            table_lines += 1
        else:
            in_table = False

    # Code block detection
    code_blocks = 0
    code_lines = 0
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            if in_code:
                in_code = False
            else:
                in_code = True
                code_blocks += 1
        elif in_code:
            code_lines += 1

    # Bold text lines (potential book markers)
    bold_h2_count = 0
    for line in lines:
        if re.match(r'^##\s+\*\*', line):
            bold_h2_count += 1

    profile = {
        "file_path": str(raw_path),
        "file_size_bytes": raw_path.stat().st_size,
        "total_lines": len(lines),
        "total_characters": len(text),
        "total_tokens_cl100k": total_tokens,
        "empty_lines": sum(1 for l in lines if not l.strip()),
        "headers": {
            "distribution": dict(sorted(header_counts.items())),
            "total": sum(header_counts.values()),
            "samples": header_samples,
        },
        "tables": {
            "table_count": table_count,
            "table_lines": table_lines,
            "table_line_fraction": round(table_lines / len(lines), 4) if lines else 0,
        },
        "code_blocks": {
            "block_count": code_blocks,
            "code_lines": code_lines,
        },
        "bold_h2_headers": bold_h2_count,
        "avg_line_length": round(sum(len(l) for l in lines) / len(lines), 1) if lines else 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    logger.info(f"Corpus profile: {total_tokens} tokens, {len(lines)} lines, "
                f"{table_count} tables, {sum(header_counts.values())} headers")

    return profile
