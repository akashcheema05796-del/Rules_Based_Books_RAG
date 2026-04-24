"""
Pydantic data models for the benchmark (spec §4.4, §7.2).
"""

import hashlib
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class ChunkMetadata(BaseModel):
    """Metadata for a single chunk (spec §4.4)."""

    chunk_id: str = ""
    strategy: str
    content: str
    book_title: str
    book_index: int = Field(ge=0, le=25)
    chapter_path: list[str] = Field(default_factory=list)
    char_start: int = 0
    char_end: int = 0
    token_count: int = 0
    is_table: bool = False
    contains_table: bool = False
    contains_code: bool = False
    parent_chunk_id: Optional[str] = None
    contextual_prefix: Optional[str] = None

    def model_post_init(self, __context) -> None:
        if not self.chunk_id:
            raw = f"{self.strategy}:{self.content}:{self.char_start}"
            self.chunk_id = hashlib.sha1(raw.encode()).hexdigest()[:16]


class ASTNode(BaseModel):
    """A parsed markdown AST node."""

    node_type: str  # heading, paragraph, table, code_block, list, blockquote, text, etc.
    level: int = 0  # heading level (1-6) or list depth
    content: str = ""
    raw_content: str = ""  # original markdown
    children: list["ASTNode"] = Field(default_factory=list)
    char_start: int = 0
    char_end: int = 0
    # Table-specific
    header_row: Optional[list[str]] = None
    align_row: Optional[list[str]] = None
    body_rows: Optional[list[list[str]]] = None
    # Metadata
    meta: dict = Field(default_factory=dict)


class BookSpan(BaseModel):
    """A detected book boundary."""

    book_title: str
    book_index: int
    start_offset: int
    end_offset: int
    start_line: int = 0
    end_line: int = 0
    h2_header_text: str = ""


class GoldEntry(BaseModel):
    """Gold standard question entry (spec §7.2)."""

    id: str
    query_type: str  # lore, mechanical, tabular, cross_reference, monster, numeric
    is_multi_hop: bool = False
    query: str
    reference_answer: str
    reference_contexts: list[dict] = Field(default_factory=list)
    validator: str = "auto"
    notes: str = ""


class RetrievedChunk(BaseModel):
    """A chunk returned by a retriever."""

    chunk_id: str
    content: str
    score: float = 0.0
    rank: int = 0
    metadata: dict = Field(default_factory=dict)


class BenchmarkResult(BaseModel):
    """A single benchmark result row."""

    chunking_strategy: str
    retrieval_method: str
    k: int
    trial: int = 1
    query_id: str = ""
    query_type: str = ""
    # Retrieval metrics
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    hit_at_k: float = 0.0
    # Generation metrics
    faithfulness: float = 0.0
    answer_correctness: float = 0.0
    context_precision: float = 0.0
    # Mechanical metrics
    table_integrity: float = 0.0
    citation_validity: float = 0.0
    # System metrics
    query_latency_ms: float = 0.0
    query_cost_usd: float = 0.0
