"""
Index builder (spec §6 preamble).

For each chunking strategy, builds:
- ChromaDB dense index (cosine, HNSW, text-embedding-3-small)
- BM25 sparse index (pickled BM25Okapi)
"""

import json
import logging
import pickle
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from omegaconf import DictConfig
from tqdm import tqdm

from src.corpus.models import ChunkMetadata
from src.utils.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)


def _load_chunks(chunks_path: Path) -> list[ChunkMetadata]:
    """Load chunks from JSONL file, deduplicating by chunk_id."""
    seen_ids: set[str] = set()
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunk = ChunkMetadata(**json.loads(line))
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    chunks.append(chunk)
    return chunks


def _tokenize_for_bm25(text: str) -> list[str]:
    """Simple tokenization for BM25."""
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def build_indices(
    strategy_name: str,
    chunks_path: Path,
    cfg: DictConfig,
    project_root: Path,
    force: bool = False,
) -> None:
    """Build dense (ChromaDB) and sparse (BM25) indices.

    Args:
        strategy_name: Name of the chunking strategy.
        chunks_path: Path to chunks.jsonl.
        cfg: Configuration.
        project_root: Project root.
        force: Rebuild even if index already exists.
    """
    dense_dir = project_root / cfg.paths.vector_store / strategy_name
    bm25_path = project_root / cfg.paths.bm25_index / strategy_name / "bm25_index.pkl"

    if not force and dense_dir.exists() and bm25_path.exists():
        logger.info(f"Indices already exist for {strategy_name}, skipping. Pass force=True to rebuild.")
        return

    chunks = _load_chunks(chunks_path)
    logger.info(f"Building indices for {strategy_name}: {len(chunks)} chunks")

    # --- Dense Index (ChromaDB) ---
    _build_dense_index(strategy_name, chunks, cfg, project_root)

    # --- Sparse Index (BM25) ---
    _build_bm25_index(strategy_name, chunks, cfg, project_root)


def _build_dense_index(strategy_name, chunks, cfg, project_root):
    """Build ChromaDB dense index."""
    import shutil
    persist_dir = project_root / cfg.paths.vector_store / strategy_name

    # Fully wipe and recreate the directory so there are no leftover partial indices
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))

    collection = client.create_collection(
        name=strategy_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare embedding client
    embed_client = EmbeddingClient(
        model=cfg.embedding.model,
        dimensions=cfg.embedding.dimensions,
        batch_size=cfg.embedding.batch_size,
        cache_dir=str(project_root / cfg.paths.cache),
    )

    # Batch add to ChromaDB
    batch_size = 100
    for i in tqdm(range(0, len(chunks), batch_size), desc=f"Dense index [{strategy_name}]"):
        batch = chunks[i:i + batch_size]
        texts = []
        for c in batch:
            # For contextual chunks, prepend the contextual prefix
            if c.contextual_prefix:
                texts.append(f"{c.contextual_prefix}\n\n{c.content}")
            else:
                texts.append(c.content)

        embeddings = embed_client.embed_batch(texts)

        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{
                "book_title": c.book_title,
                "book_index": c.book_index,
                "chapter_path": json.dumps(c.chapter_path),
                "is_table": c.is_table,
                "contains_table": c.contains_table,
                "contains_code": c.contains_code,
                "token_count": c.token_count,
                "strategy": c.strategy,
            } for c in batch],
        )

    embed_client.close()
    logger.info(f"Dense index built: {persist_dir} ({len(chunks)} chunks)")


def _build_bm25_index(strategy_name, chunks, cfg, project_root):
    """Build BM25 sparse index."""
    persist_dir = project_root / cfg.paths.bm25_index / strategy_name
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Tokenize corpus for BM25
    tokenized_corpus = []
    chunk_ids = []
    chunk_contents = []

    for chunk in tqdm(chunks, desc=f"BM25 index [{strategy_name}]"):
        text = chunk.content
        if chunk.contextual_prefix:
            text = f"{chunk.contextual_prefix} {text}"
        tokenized_corpus.append(_tokenize_for_bm25(text))
        chunk_ids.append(chunk.chunk_id)
        chunk_contents.append(text)

    # Build BM25 index
    bm25 = BM25Okapi(
        tokenized_corpus,
        k1=cfg.sparse_index.k1,
        b=cfg.sparse_index.b,
    )

    # Save index, chunk IDs, and contents
    index_data = {
        "bm25": bm25,
        "chunk_ids": chunk_ids,
        "chunk_contents": chunk_contents,
        "tokenized_corpus": tokenized_corpus,
    }
    index_path = persist_dir / "bm25_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump(index_data, f)

    logger.info(f"BM25 index built: {index_path} ({len(chunks)} chunks)")
