"""Smoke test for the entire RAG pipeline."""

import os
import json
import logging
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

from main import run_parse, run_chunk, run_index, run_eval_phase1

logger = logging.getLogger(__name__)

# Mock configurations
MOCK_CFG = OmegaConf.create({
    "seed": 42,
    "paths": {
        "raw_corpus": "data/raw/DnD_Second_edition__all_26_books.md",
        "interim": "data/interim",
        "processed": "data/processed",
        "gold": "data/gold",
        "vector_store": "data/vector_store",
        "bm25_index": "data/bm25_index",
        "results": "results",
        "figures": "results/figures",
        "chunk_samples": "results/chunk_samples",
        "cache": ".cache",
        "log_file": "process.log",
    },
    "corpus": {
        "books_manifest": "configs/books_manifest.yaml",
    },
    "embedding": {
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "batch_size": 256,
        "provider": "openai",
    },
    "llm": {
        "model": "claude",
        "temperature": 0,
        "max_tokens": 100,
    },
    "sparse_index": {"k1": 1.5, "b": 0.75},
    "fusion": {"rrf_k": 60},
    "reranker": {"model": "mock", "top_n_input": 10, "batch_size": 32, "use_fp16": False},
    "retrieval": {"k_values": [1, 3], "default_k": 3},
    "evaluation": {"n_trials": 1},
    "cost_budget": {"total": 100.0, "contextual_chunking": 10.0},
    "rate_limit": {"base_wait": 1.0, "max_wait": 1.0, "max_retries": 1},
})


@patch("src.utils.embeddings.EmbeddingClient")
def test_smoke_pipeline(mock_embed_client, tmp_path):
    """End-to-end smoke test of the pipeline with mocked APIs."""
    # Setup mock embedding
    mock_instance = MagicMock()
    mock_instance.embed_batch.return_value = [[0.1] * 1536 for _ in range(100)]
    mock_instance.embed_single.return_value = [0.1] * 1536
    mock_embed_client.return_value = mock_instance

    # Create dummy corpus
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "DnD_Second_edition__all_26_books.md").write_text(
        "## Players Handbook\n\n### Chapter 1\n\nTest content.\n",
        encoding="utf-8"
    )

    # Create dummy manifest
    manifest_dir = tmp_path / "configs"
    manifest_dir.mkdir()
    (manifest_dir / "books_manifest.yaml").write_text(
        "books:\n  - \"Players Handbook\"", encoding="utf-8"
    )

    # Create dummy gold set
    gold_dir = tmp_path / "data" / "gold"
    gold_dir.mkdir(parents=True)
    gold_path = gold_dir / "gold_standard.jsonl"
    gold_path.write_text(
        json.dumps({"id": "q1", "query_type": "lore", "query": "Test?", "reference_answer": "Test content", "is_multi_hop": False, "validator": "auto", "reference_contexts": []}) + "\n",
        encoding="utf-8"
    )

    # Patch config paths to be relative (main.py will prepend PROJECT_ROOT)
    cfg = MOCK_CFG.copy()
    cfg.paths.raw_corpus = "data/raw/DnD_Second_edition__all_26_books.md"
    cfg.corpus.books_manifest = "configs/books_manifest.yaml"

    # Run pipeline stages
    from main import PROJECT_ROOT
    with patch("main.PROJECT_ROOT", tmp_path):
        # Parse
        run_parse(cfg)
        assert (tmp_path / cfg.paths.interim / "book_00.md").exists()

        # Chunk
        run_chunk(cfg, ["recursive"])
        assert (tmp_path / cfg.paths.processed / "recursive" / "chunks.jsonl").exists()

        # Index
        run_index(cfg, ["recursive"])
        assert (tmp_path / cfg.paths.vector_store / "recursive").exists()
        assert (tmp_path / cfg.paths.bm25_index / "recursive" / "bm25_index.pkl").exists()

        # Eval Phase 1
        with patch("src.retrieval.dense.DenseRetriever.load_index"):
            # Mock Chromadb behavior for dense retriever
            with patch("chromadb.PersistentClient") as mock_chroma:
                mock_collection = MagicMock()
                mock_collection.query.return_value = {
                    "ids": [["chunk1"]],
                    "documents": [["Test content"]],
                    "distances": [[0.1]],
                    "metadatas": [[{"char_start": 0, "char_end": 10}]]
                }
                mock_chroma.return_value.get_collection.return_value = mock_collection

                run_eval_phase1(cfg, ["recursive"])

        assert (tmp_path / cfg.paths.results / "phase1_results.csv").exists()
