"""
RAG Chunking × Retrieval Benchmark v3 — Main Orchestrator

Hydra-driven CLI that dispatches to the appropriate pipeline stage.

Usage:
    python main.py stage=profile
    python main.py stage=parse
    python main.py stage=chunk chunking=recursive
    python main.py stage=index chunking=all
    python main.py stage=eval_phase1
    python main.py stage=eval_phase2 chunking=contextual retrieval=hybrid_rerank
    python main.py stage=eval_phase3
    python main.py stage=report
"""

import json
import os
import sys
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv, find_dotenv

# Load environment variables — find_dotenv() walks up from cwd so the .env
# in the project root is found even when running from a git worktree.
load_dotenv(find_dotenv(usecwd=True), override=True)

# Project root
PROJECT_ROOT = Path(__file__).parent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "process.log", mode="a"),
    ],
)
logger = logging.getLogger("rag_benchmark")

# All chunking strategies
ALL_CHUNKING = ["recursive", "markdown_hierarchical", "contextual", "table_aware", "adaptive"]

# All retrieval methods
ALL_RETRIEVAL = [
    "dense", "bm25", "hybrid_rrf",
    # "hybrid_rerank" excluded — CPU cross-encoder inference is ~1 min/query (no GPU)
    "metadata_filter", "small_to_big", "hyde", "query_decomposition",
]


def resolve_strategies(value: str, all_options: list[str]) -> list[str]:
    """Resolve 'all' or comma-separated strategy names."""
    if value == "all":
        return all_options
    return [s.strip() for s in value.split(",")]


def run_profile(cfg: DictConfig) -> None:
    """§4.1: Corpus profiling."""
    from src.corpus.profile import profile_corpus
    from src.utils.seeds import set_all_seeds

    set_all_seeds(cfg.seed)
    raw_path = PROJECT_ROOT / cfg.paths.raw_corpus
    output_path = PROJECT_ROOT / cfg.paths.results / "corpus_profile.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Profiling corpus: {raw_path}")
    profile_corpus(raw_path, output_path, cfg)
    logger.info(f"Profile saved to {output_path}")


def run_parse(cfg: DictConfig) -> None:
    """§4.3: AST parsing — corpus treated as one unit."""
    import json as _json
    from src.corpus.parser import parse_corpus
    from src.utils.seeds import set_all_seeds

    set_all_seeds(cfg.seed)
    raw_path = PROJECT_ROOT / cfg.paths.raw_corpus
    interim_path = PROJECT_ROOT / cfg.paths.interim
    interim_path.mkdir(parents=True, exist_ok=True)

    logger.info("Parsing corpus to AST...")
    ast_nodes = parse_corpus(raw_path)
    logger.info(f"Parsed {len(ast_nodes)} top-level AST nodes")

    # Save a lightweight parse summary (not the full AST — that's ~GB on disk).
    node_type_counts: dict[str, int] = {}
    for n in ast_nodes:
        node_type_counts[n.node_type] = node_type_counts.get(n.node_type, 0) + 1

    summary = {
        "total_nodes": len(ast_nodes),
        "node_types": node_type_counts,
        "corpus_chars": (PROJECT_ROOT / cfg.paths.raw_corpus).stat().st_size,
    }
    summary_path = interim_path / "parse_summary.json"
    summary_path.write_text(_json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Parse summary saved to {summary_path}")


def run_chunk(cfg: DictConfig, chunking_strategies: list[str]) -> None:
    """§5: Run chunking strategies — full corpus as one unit."""
    from src.corpus.parser import parse_corpus
    from src.chunkers import get_chunker
    from src.utils.seeds import set_all_seeds

    set_all_seeds(cfg.seed)
    raw_path = PROJECT_ROOT / cfg.paths.raw_corpus

    logger.info("Loading parsed corpus...")
    ast_nodes = parse_corpus(raw_path)

    for strategy_name in chunking_strategies:
        logger.info(f"Chunking with strategy: {strategy_name}")
        output_dir = PROJECT_ROOT / cfg.paths.processed / strategy_name
        output_dir.mkdir(parents=True, exist_ok=True)

        chunker = get_chunker(strategy_name, cfg, PROJECT_ROOT)
        all_chunks = chunker.chunk_corpus(ast_nodes)

        # Write chunks to JSONL
        import json
        output_file = output_dir / "chunks.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")

        logger.info(f"  -> {len(all_chunks)} chunks written to {output_file}")

        # Save chunk samples (50 random)
        import random
        samples_dir = PROJECT_ROOT / cfg.paths.chunk_samples
        samples_dir.mkdir(parents=True, exist_ok=True)
        sample_chunks = random.sample(all_chunks, min(50, len(all_chunks)))
        sample_file = samples_dir / f"{strategy_name}_samples.jsonl"
        with open(sample_file, "w", encoding="utf-8") as f:
            for chunk in sample_chunks:
                f.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")

        logger.info(f"  -> {len(sample_chunks)} samples saved to {sample_file}")


def run_index(cfg: DictConfig, chunking_strategies: list[str]) -> None:
    """Build dense + BM25 indices for each chunking strategy."""
    from src.retrieval.indexer import build_indices
    from src.utils.seeds import set_all_seeds

    set_all_seeds(cfg.seed)

    for strategy_name in chunking_strategies:
        logger.info(f"Building indices for: {strategy_name}")
        chunks_path = PROJECT_ROOT / cfg.paths.processed / strategy_name / "chunks.jsonl"
        if not chunks_path.exists():
            logger.warning(f"  Chunks not found at {chunks_path}, skipping.")
            continue

        build_indices(strategy_name, chunks_path, cfg, PROJECT_ROOT)
        logger.info(f"  -> Indices built for {strategy_name}")


def run_eval_phase1(cfg: DictConfig, chunking_strategies: list[str]) -> None:
    """Phase 1: Chunking isolation (dense-only retrieval)."""
    from src.evaluation.runner import run_phase1
    from src.utils.seeds import set_all_seeds

    set_all_seeds(cfg.seed)
    logger.info("=== Phase 1: Chunking Isolation ===")
    run_phase1(chunking_strategies, cfg, PROJECT_ROOT)


def run_eval_phase2(
    cfg: DictConfig,
    chunking_strategies: list[str],
    retrieval_methods: list[str],
) -> None:
    """Phase 2: Retrieval sweep."""
    from src.evaluation.runner import run_phase2
    from src.utils.seeds import set_all_seeds

    set_all_seeds(cfg.seed)
    logger.info("=== Phase 2: Retrieval Sweep ===")
    run_phase2(chunking_strategies, retrieval_methods, cfg, PROJECT_ROOT)


def run_eval_phase3(cfg: DictConfig) -> None:
    """Phase 3: Query-type routing."""
    from src.evaluation.runner import run_phase3
    from src.utils.seeds import set_all_seeds

    set_all_seeds(cfg.seed)
    logger.info("=== Phase 3: Query-Type Routing ===")
    run_phase3(cfg, PROJECT_ROOT)


def run_goldset(cfg: DictConfig) -> None:
    """§7: Generate gold standard dataset."""
    from src.goldset.generator import generate_gold_standard
    from src.utils.seeds import set_all_seeds

    set_all_seeds(cfg.seed)
    logger.info("Generating gold standard dataset...")
    gold_path = PROJECT_ROOT / cfg.paths.gold
    gold_path.mkdir(parents=True, exist_ok=True)
    generate_gold_standard(cfg, PROJECT_ROOT)
    logger.info(f"Gold standard saved to {gold_path}")


def run_validate_gold(cfg: DictConfig) -> None:
    """§7.3: Emit review sheet OR ingest reviewer decisions.

    Controlled by `cfg.validate.action`: 'emit' (default) or 'ingest'.
    """
    from src.goldset.validator import emit_review_sheet, ingest_reviews, validate_gold_standard

    gold_dir = PROJECT_ROOT / cfg.paths.gold
    gold_path = gold_dir / "gold_standard.jsonl"
    corpus_path = PROJECT_ROOT / cfg.paths.raw_corpus
    action = cfg.get("validate", {}).get("action", "static")

    if action == "emit":
        out_csv = gold_dir / "review_sheet.csv"
        emit_review_sheet(gold_path, out_csv, corpus_path)
        logger.info(f"Review sheet emitted: {out_csv}")
    elif action == "ingest":
        reviews_csv = gold_dir / "review_sheet.csv"
        frozen = gold_dir / "gold_standard.frozen.jsonl"
        report = ingest_reviews(reviews_csv, gold_path, frozen)
        logger.info(f"Review ingest report: {json.dumps(report, indent=2)}")
    else:
        report = validate_gold_standard(gold_path, corpus_path)
        logger.info(f"Static validation: {json.dumps(report, indent=2)}")


def run_report(cfg: DictConfig) -> None:
    """§9: Generate report with stats, plots, and writeup."""
    from src.evaluation.report import generate_report
    from src.utils.seeds import set_all_seeds

    set_all_seeds(cfg.seed)
    logger.info("Generating final report...")
    generate_report(cfg, PROJECT_ROOT)
    logger.info("Report generated.")


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig) -> None:
    """Main entry point — dispatches to pipeline stages."""

    # Hydra changes cwd; we need to track original
    original_cwd = hydra.utils.get_original_cwd()
    os.chdir(original_cwd)

    stage = cfg.get("stage", None)
    if stage is None:
        logger.error("No stage specified. Use: python main.py stage=<stage>")
        logger.error("Stages: profile, parse, goldset, validate_gold, chunk, index, eval_phase1, eval_phase2, eval_phase3, report")
        sys.exit(1)

    # Resolve strategy lists
    chunking_val = cfg.get("chunking_methods", "all")
    retrieval_val = cfg.get("retrieval_methods", "all")
    chunking_strategies = resolve_strategies(chunking_val, ALL_CHUNKING)
    retrieval_methods = resolve_strategies(retrieval_val, ALL_RETRIEVAL)

    logger.info(f"Stage: {stage}")
    logger.info(f"Chunking strategies: {chunking_strategies}")
    logger.info(f"Retrieval methods: {retrieval_methods}")

    dispatch = {
        "profile": lambda: run_profile(cfg),
        "parse": lambda: run_parse(cfg),
        "goldset": lambda: run_goldset(cfg),
        "chunk": lambda: run_chunk(cfg, chunking_strategies),
        "index": lambda: run_index(cfg, chunking_strategies),
        "eval_phase1": lambda: run_eval_phase1(cfg, chunking_strategies),
        "eval_phase2": lambda: run_eval_phase2(cfg, chunking_strategies, retrieval_methods),
        "eval_phase3": lambda: run_eval_phase3(cfg),
        "validate_gold": lambda: run_validate_gold(cfg),
        "report": lambda: run_report(cfg),
    }

    if stage not in dispatch:
        logger.error(f"Unknown stage: {stage}")
        logger.error(f"Valid stages: {list(dispatch.keys())}")
        sys.exit(1)

    dispatch[stage]()
    logger.info(f"Stage '{stage}' completed successfully.")


if __name__ == "__main__":
    main()
