"""
Mechanical metrics (spec §8.3).

Table Integrity Score and Citation Validity.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from src.corpus.models import ChunkMetadata

logger = logging.getLogger(__name__)


def table_integrity_score(chunks: list[ChunkMetadata]) -> dict:
    """Compute Table Integrity Score per strategy.

    Fraction of chunks that don't sever tables (orphan body rows,
    header-only chunks, split tables).

    Args:
        chunks: List of all chunks for a strategy.

    Returns:
        Dict with integrity metrics.
    """
    total_table_chunks = 0
    intact_tables = 0
    issues = []

    for chunk in chunks:
        if not chunk.contains_table and not chunk.is_table:
            continue

        total_table_chunks += 1
        content = chunk.content

        # Check for table integrity issues
        table_lines = [l for l in content.split("\n") if "|" in l]
        if not table_lines:
            continue

        has_header = False
        has_separator = False
        has_body = False

        for i, line in enumerate(table_lines):
            stripped = line.strip()
            if re.match(r'^[\s|:\-]+$', stripped):
                has_separator = True
            elif i == 0:
                has_header = True
            else:
                has_body = True

        if has_header and has_separator and has_body:
            intact_tables += 1
        elif has_body and not has_header:
            issues.append({
                "chunk_id": chunk.chunk_id,
                "issue": "orphan_body_rows",
                "description": "Table body rows without header",
            })
        elif has_header and not has_body:
            issues.append({
                "chunk_id": chunk.chunk_id,
                "issue": "header_only",
                "description": "Table header without body rows",
            })

    integrity = intact_tables / total_table_chunks if total_table_chunks > 0 else 1.0

    return {
        "total_table_chunks": total_table_chunks,
        "intact_tables": intact_tables,
        "integrity_score": round(integrity, 4),
        "issues_count": len(issues),
        "issues": issues[:10],  # Sample of issues
    }


def citation_validity(
    cited_chunk_ids: list[str],
    claims: list[str],
    chunks_by_id: dict[str, ChunkMetadata],
) -> dict:
    """Check that cited chunk IDs exist and contain cited claims.

    Args:
        cited_chunk_ids: List of chunk IDs cited in the answer.
        claims: List of claims made in the answer.
        chunks_by_id: Mapping of chunk ID to ChunkMetadata.

    Returns:
        Validation report.
    """
    total = len(cited_chunk_ids)
    valid_ids = 0
    claims_found = 0

    for cid in cited_chunk_ids:
        if cid in chunks_by_id:
            valid_ids += 1

    for claim in claims:
        claim_lower = claim.lower()
        for cid in cited_chunk_ids:
            if cid in chunks_by_id:
                if claim_lower in chunks_by_id[cid].content.lower():
                    claims_found += 1
                    break

    return {
        "total_citations": total,
        "valid_ids": valid_ids,
        "id_validity": valid_ids / total if total > 0 else 1.0,
        "claims_found": claims_found,
        "claim_validity": claims_found / len(claims) if claims else 1.0,
    }
