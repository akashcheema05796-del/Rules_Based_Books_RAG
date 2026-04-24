"""
Contextual Retrieval Chunking (spec §5.3).

First chunks with recursive, then calls Claude to generate a context blurb
for each chunk. Cache keyed by (book, chapter_path, content_hash).
Uses prompt caching on WHOLE_BOOK.
"""

import hashlib
import logging
from pathlib import Path

from omegaconf import DictConfig

from src.chunkers import BaseChunker
from src.chunkers.recursive import RecursiveChunker
from src.corpus.models import ASTNode, BookSpan, ChunkMetadata
from src.utils.llm import LLMClient
from src.utils.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class ContextualChunker(BaseChunker):
    """Contextual retrieval: recursive + LLM-generated context prefix."""

    strategy_name = "contextual"

    def __init__(self, cfg: DictConfig, project_root: Path):
        super().__init__(cfg, project_root)
        self.base_chunker = RecursiveChunker(cfg, project_root)
        self.base_chunker.strategy_name = "contextual"  # Override

        # Load prompt template
        prompt_path = project_root / "configs" / "prompts" / "contextual.md"
        self.prompt_template = prompt_path.read_text(encoding="utf-8")

        # Cost tracker for contextual calls
        budget = cfg.get("cost_budget", {}).get("contextual_chunking", 15.0)
        self.cost_tracker = CostTracker(budget_usd=budget, stage="contextual_chunking")
        self.llm = LLMClient(
            model=cfg.llm.model,
            temperature=cfg.llm.temperature,
            max_tokens=200,
            cache_dir=str(project_root / ".cache"),
            cost_tracker=self.cost_tracker,
        )

    def chunk_book(self, book, nodes, full_text):
        # First: recursive chunking
        base_chunks = self.base_chunker.chunk_book(book, nodes, full_text)

        # Get whole book text for prompt caching
        book_text = full_text[book.start_offset:book.end_offset]

        # Generate context for each chunk
        for i, chunk in enumerate(base_chunks):
            content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()[:12]
            cache_key = f"ctx:{book.book_title}:{'.'.join(chunk.chapter_path)}:{content_hash}"

            try:
                self.cost_tracker.check_budget()
            except Exception:
                logger.warning("Budget exceeded for contextual chunking, "
                               "remaining chunks will not have context")
                break

            prompt = self.prompt_template.replace("{WHOLE_BOOK}", book_text)
            prompt = prompt.replace("{CHUNK_CONTENT}", chunk.content)

            context = self.llm.generate(
                system_prompt="",
                user_prompt=prompt,
                max_tokens=200,
                cache_system=True,
                cache_key=cache_key,
            )

            chunk.contextual_prefix = context.strip()

            if (i + 1) % 50 == 0:
                logger.info(f"  Contextual: {i+1}/{len(base_chunks)} chunks processed")
                self.cost_tracker.log_summary()

        self.cost_tracker.log_summary()
        return base_chunks
