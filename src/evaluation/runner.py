"""
Evaluation runner (spec §8, §11).

Orchestrates Phase 1 (chunking isolation), Phase 2 (retrieval sweep),
Phase 3 (query-type routing). Writes results/benchmark.csv.
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from src.corpus.models import GoldEntry, BenchmarkResult
from src.retrieval import get_retriever
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.evaluation.system_metrics import LatencyTracker
from src.goldset.validator import load_gold
from src.utils.logging import BenchmarkLogger, RunLog

logger = logging.getLogger(__name__)


def _load_gold_set(cfg, project_root):
    """Load gold standard dataset."""
    gold_path = project_root / cfg.paths.gold / "gold_standard.jsonl"
    if not gold_path.exists():
        logger.error(f"Gold set not found: {gold_path}. Run 'make goldset' first.")
        return []
    return load_gold(gold_path)


def _evaluate_single(retriever, query, gold, k_values):
    """Evaluate a single query against a retriever."""
    latency_start = time.perf_counter()
    retrieved = retriever.retrieve(query.query, k=max(k_values))
    latency_ms = (time.perf_counter() - latency_start) * 1000

    metrics = compute_retrieval_metrics(retrieved, query, k_values)
    metrics["query_latency_ms"] = latency_ms
    return metrics, retrieved


def run_phase1(chunking_strategies, cfg, project_root):
    """Phase 1: Chunking isolation — dense-only retrieval."""
    gold_set = _load_gold_set(cfg, project_root)
    if not gold_set:
        return

    k_values = list(cfg.retrieval.k_values)
    n_trials = cfg.evaluation.n_trials
    bench_logger = BenchmarkLogger(project_root / cfg.paths.log_file)
    all_results = []

    for strategy in chunking_strategies:
        logger.info(f"Phase 1: {strategy} × dense")

        for trial in range(1, n_trials + 1):
            try:
                retriever = get_retriever("dense", cfg, project_root, strategy)
            except Exception as e:
                logger.error(f"Failed to load retriever for {strategy}: {e}")
                continue

            with bench_logger.timed_run("phase1", strategy=strategy, method="dense") as run:
                for gold in tqdm(gold_set, desc=f"{strategy}/dense/t{trial}"):
                    metrics, _ = _evaluate_single(retriever, gold, gold, k_values)

                    for k in k_values:
                        result = BenchmarkResult(
                            chunking_strategy=strategy,
                            retrieval_method="dense",
                            k=k, trial=trial,
                            query_id=gold.id,
                            query_type=gold.query_type,
                            recall_at_k=metrics.get(f"recall@{k}", 0),
                            mrr=metrics.get("mrr", 0),
                            ndcg_at_k=metrics.get(f"ndcg@{k}", 0),
                            hit_at_k=metrics.get(f"hit@{k}", 0),
                            query_latency_ms=metrics.get("query_latency_ms", 0),
                        )
                        all_results.append(result)

                run.chunk_count = len(gold_set)

    # Write results
    _save_results(all_results, project_root / cfg.paths.results / "phase1_results.csv")
    _append_benchmark(all_results, project_root / cfg.paths.results / "benchmark.csv")


def run_phase2(chunking_strategies, retrieval_methods, cfg, project_root):
    """Phase 2: Full retrieval sweep."""
    gold_set = _load_gold_set(cfg, project_root)
    if not gold_set:
        return

    k_values = list(cfg.retrieval.k_values)
    n_trials = cfg.evaluation.n_trials
    bench_logger = BenchmarkLogger(project_root / cfg.paths.log_file)
    all_results = []

    for strategy in chunking_strategies:
        for method in retrieval_methods:
            logger.info(f"Phase 2: {strategy} × {method}")

            for trial in range(1, n_trials + 1):
                try:
                    retriever = get_retriever(method, cfg, project_root, strategy)
                except Exception as e:
                    logger.error(f"Failed: {strategy}/{method}: {e}")
                    continue

                with bench_logger.timed_run("phase2", strategy=strategy, method=method) as run:
                    for gold in tqdm(gold_set, desc=f"{strategy}/{method}/t{trial}"):
                        metrics, _ = _evaluate_single(retriever, gold, gold, k_values)

                        for k in k_values:
                            result = BenchmarkResult(
                                chunking_strategy=strategy,
                                retrieval_method=method,
                                k=k, trial=trial,
                                query_id=gold.id,
                                query_type=gold.query_type,
                                recall_at_k=metrics.get(f"recall@{k}", 0),
                                mrr=metrics.get("mrr", 0),
                                ndcg_at_k=metrics.get(f"ndcg@{k}", 0),
                                hit_at_k=metrics.get(f"hit@{k}", 0),
                                query_latency_ms=metrics.get("query_latency_ms", 0),
                            )
                            all_results.append(result)

                    run.chunk_count = len(gold_set)

    _save_results(all_results, project_root / cfg.paths.results / "phase2_results.csv")
    _append_benchmark(all_results, project_root / cfg.paths.results / "benchmark.csv")


def run_phase3(cfg, project_root):
    """Phase 3: Query-type routing analysis."""
    results_path = project_root / cfg.paths.results / "benchmark.csv"
    if not results_path.exists():
        logger.error("No benchmark results found. Run Phase 1 and 2 first.")
        return

    df = pd.read_csv(results_path)

    # Analyze best (chunking, retrieval) per query type
    query_types = df["query_type"].unique()
    routing_table = {}

    for qt in query_types:
        qt_df = df[df["query_type"] == qt]
        # Group by strategy+method, compute mean recall@10
        grouped = qt_df.groupby(["chunking_strategy", "retrieval_method"])
        mean_recall = grouped["recall_at_k"].mean()
        best = mean_recall.idxmax() if len(mean_recall) > 0 else ("unknown", "unknown")
        routing_table[qt] = {
            "best_strategy": best[0],
            "best_method": best[1],
            "mean_recall": float(mean_recall.max()) if len(mean_recall) > 0 else 0,
        }

    # Save routing table
    output = project_root / cfg.paths.results / "routing_table.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(routing_table, f, indent=2)

    logger.info(f"Phase 3 routing table: {output}")
    for qt, info in routing_table.items():
        logger.info(f"  {qt}: {info['best_strategy']}/{info['best_method']} "
                     f"(recall={info['mean_recall']:.3f})")


def _save_results(results, path):
    """Save results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.model_dump() for r in results]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info(f"Results saved: {path} ({len(rows)} rows)")


def _append_benchmark(results, path):
    """Append results to the main benchmark.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.model_dump() for r in results]
    df_new = pd.DataFrame(rows)

    if path.exists():
        df_existing = pd.read_csv(path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(path, index=False)
    logger.info(f"Benchmark updated: {path} ({len(df)} total rows)")
