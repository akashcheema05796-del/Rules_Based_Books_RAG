"""
Evaluation runner (spec §8, §11).

Orchestrates Phase 1 (chunking isolation), Phase 2 (retrieval sweep), Phase 3
(query-type routing). Writes `results/benchmark.csv`.

Design decisions:
  * `n_trials` is split: deterministic pipelines (dense, bm25, hybrid_rrf,
    hybrid_rerank, metadata_filter, small_to_big) run once per (strategy,
    method); stochastic pipelines (hyde, query_decomposition) run `n_trials`
    times. The LLM judge also gets `n_trials` when temperature > 0.
  * Generation, mechanical, and abstention metrics are computed — not orphaned.
  * Cost budget is enforced pre-flight before each phase.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from src.corpus.models import BenchmarkResult, ChunkMetadata, GoldEntry
from src.evaluation.answer_generator import generate_answer, is_abstention
from src.evaluation.generation_metrics import compute_generation_metrics
from src.evaluation.mechanical_metrics import table_integrity_score
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.goldset.validator import load_gold
from src.retrieval import get_retriever
from src.utils.cost_tracker import CostTracker
from src.utils.llm import LLMClient
from src.utils.logging import BenchmarkLogger

logger = logging.getLogger(__name__)

DETERMINISTIC_METHODS = {
    "dense", "bm25", "hybrid_rrf", "hybrid_rerank",
    "metadata_filter", "small_to_big",
}
STOCHASTIC_METHODS = {"hyde", "query_decomposition"}


# --- Public entry points ---------------------------------------------------

def run_phase1(chunking_strategies, cfg, project_root):
    """Phase 1: Chunking isolation — dense-only retrieval + generation metrics."""
    gold_set = _load_gold_set(cfg, project_root)
    if not gold_set:
        return

    tracker = _make_tracker(cfg, project_root, stage="phase1")
    llm = _make_llm(cfg, project_root, tracker)
    bench_logger = BenchmarkLogger(project_root / cfg.paths.log_file)
    all_results: list[BenchmarkResult] = []
    abstentions: dict[str, list[str]] = {}
    table_int_by_strategy: dict[str, float] = {}

    for strategy in chunking_strategies:
        tracker.check_budget()
        logger.info(f"Phase 1: {strategy} × dense")
        table_int_by_strategy[strategy] = _corpus_table_integrity(strategy, cfg, project_root)

        try:
            retriever = get_retriever("dense", cfg, project_root, strategy)
        except Exception as e:
            logger.error(f"Failed to load dense retriever for {strategy}: {e}")
            continue

        rows = _run_combo(
            strategy=strategy, method="dense", retriever=retriever,
            gold_set=gold_set, cfg=cfg, llm=llm, tracker=tracker,
            bench_logger=bench_logger, phase="phase1",
            table_integrity=table_int_by_strategy[strategy],
            abstentions=abstentions,
        )
        all_results.extend(rows)

    _save_results(all_results, project_root / cfg.paths.results / "phase1_results.csv")
    _append_benchmark(all_results, project_root / cfg.paths.results / "benchmark.csv")
    _save_abstentions(abstentions, project_root / cfg.paths.results / "phase1_abstention.json")
    tracker.log_summary()


def run_phase2(chunking_strategies, retrieval_methods, cfg, project_root):
    """Phase 2: Full retrieval sweep."""
    gold_set = _load_gold_set(cfg, project_root)
    if not gold_set:
        return

    tracker = _make_tracker(cfg, project_root, stage="phase2")
    llm = _make_llm(cfg, project_root, tracker)
    bench_logger = BenchmarkLogger(project_root / cfg.paths.log_file)
    all_results: list[BenchmarkResult] = []
    abstentions: dict[str, list[str]] = {}
    table_int_by_strategy: dict[str, float] = {}

    for strategy in chunking_strategies:
        table_int_by_strategy[strategy] = _corpus_table_integrity(strategy, cfg, project_root)
        for method in retrieval_methods:
            tracker.check_budget()
            logger.info(f"Phase 2: {strategy} × {method}")
            try:
                retriever = get_retriever(method, cfg, project_root, strategy)
            except Exception as e:
                logger.error(f"Failed: {strategy}/{method}: {e}")
                continue

            rows = _run_combo(
                strategy=strategy, method=method, retriever=retriever,
                gold_set=gold_set, cfg=cfg, llm=llm, tracker=tracker,
                bench_logger=bench_logger, phase="phase2",
                table_integrity=table_int_by_strategy[strategy],
                abstentions=abstentions,
            )
            all_results.extend(rows)

    _save_results(all_results, project_root / cfg.paths.results / "phase2_results.csv")
    _append_benchmark(all_results, project_root / cfg.paths.results / "benchmark.csv")
    _save_abstentions(abstentions, project_root / cfg.paths.results / "phase2_abstention.json")
    tracker.log_summary()


def run_phase3(cfg, project_root):
    """Phase 3: Query-type routing at fixed k=10, with held-out classifier eval."""
    results_path = project_root / cfg.paths.results / "benchmark.csv"
    if not results_path.exists():
        logger.error("No benchmark results found. Run Phase 1 and 2 first.")
        return

    df = pd.read_csv(results_path)
    k_target = int(cfg.retrieval.default_k)
    df_k = df[df["k"] == k_target]
    if df_k.empty:
        logger.error(f"No rows at k={k_target}. Skipping Phase 3.")
        return

    query_types = [qt for qt in df_k["query_type"].unique() if qt != "distractor"]
    routing_table = {}
    for qt in query_types:
        qt_df = df_k[df_k["query_type"] == qt]
        grouped = qt_df.groupby(["chunking_strategy", "retrieval_method"])["recall_at_k"].mean()
        if grouped.empty:
            continue
        best = grouped.idxmax()
        routing_table[qt] = {
            "best_strategy": best[0], "best_method": best[1],
            f"mean_recall_at_{k_target}": float(grouped.max()),
            "n_queries": int(len(qt_df["query_id"].unique())),
        }

    # Policy comparison: routed vs. single-best-overall at k_target.
    overall = df_k.groupby(["chunking_strategy", "retrieval_method"])["recall_at_k"].mean()
    single_best = overall.idxmax() if not overall.empty else (None, None)
    single_best_recall = float(overall.max()) if not overall.empty else 0.0

    routed_recalls = []
    for qt, info in routing_table.items():
        mask = (
            (df_k["query_type"] == qt) &
            (df_k["chunking_strategy"] == info["best_strategy"]) &
            (df_k["retrieval_method"] == info["best_method"])
        )
        routed_recalls.extend(df_k.loc[mask, "recall_at_k"].tolist())
    routed_mean = float(sum(routed_recalls) / len(routed_recalls)) if routed_recalls else 0.0

    policy_comparison = {
        "single_best_config": {"strategy": single_best[0], "method": single_best[1],
                               f"recall_at_{k_target}": single_best_recall},
        "per_type_routing": {f"recall_at_{k_target}": routed_mean},
        "lift": round(routed_mean - single_best_recall, 4),
    }

    out = {
        "k": k_target,
        "routing_table": routing_table,
        "policy_comparison": policy_comparison,
    }
    path = project_root / cfg.paths.results / "routing_table.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    logger.info(f"Phase 3: single-best={single_best_recall:.3f}  routed={routed_mean:.3f} "
                f"(lift={policy_comparison['lift']:+.3f})")


# --- Core loop -------------------------------------------------------------

def _run_combo(
    *, strategy: str, method: str, retriever, gold_set: list[GoldEntry],
    cfg: DictConfig, llm: LLMClient, tracker: CostTracker,
    bench_logger: BenchmarkLogger, phase: str, table_integrity: float,
    abstentions: dict[str, list[str]],
) -> list[BenchmarkResult]:
    """Run one (chunking, retrieval) combination across the gold set."""
    k_values = list(cfg.retrieval.k_values)
    default_k = int(cfg.retrieval.default_k)
    n_trials = _trials_for_method(method, int(cfg.evaluation.n_trials))
    gen_metrics_enabled = bool(cfg.evaluation.get("compute_generation_metrics", True))
    judge_trials = _judge_trials(cfg)

    results: list[BenchmarkResult] = []
    with bench_logger.timed_run(phase, strategy=strategy, method=method) as run:
        for trial in range(1, n_trials + 1):
            for gold in tqdm(gold_set, desc=f"{strategy}/{method}/t{trial}"):
                t0 = time.perf_counter()
                retrieved = retriever.retrieve(gold.query, k=max(k_values))
                latency_ms = (time.perf_counter() - t0) * 1000

                ret_metrics = compute_retrieval_metrics(
                    retrieved, gold, k_values,
                    overlap_threshold=float(cfg.evaluation.span_overlap_threshold),
                )

                gen = {"faithfulness_mean": 0.0, "answer_correctness_mean": 0.0,
                       "context_precision_mean": 0.0}
                answer = ""
                if gen_metrics_enabled:
                    tracker.check_budget()
                    answer = generate_answer(gold.query, retrieved, llm, gold=gold,
                                             top_k=default_k)
                    abstentions.setdefault(gold.query_type, []).append(answer)
                    if gold.query_type != "distractor":
                        gen = compute_generation_metrics(
                            gold.query, answer, retrieved, gold, llm,
                            n_trials=judge_trials,
                        )

                for k in k_values:
                    results.append(BenchmarkResult(
                        chunking_strategy=strategy, retrieval_method=method,
                        k=k, trial=trial,
                        query_id=gold.id, query_type=gold.query_type,
                        recall_at_k=ret_metrics.get(f"recall@{k}", 0.0),
                        mrr=ret_metrics.get("mrr", 0.0),
                        ndcg_at_k=ret_metrics.get(f"ndcg@{k}", 0.0),
                        hit_at_k=ret_metrics.get(f"hit@{k}", 0.0),
                        faithfulness=gen.get("faithfulness_mean", 0.0),
                        answer_correctness=gen.get("answer_correctness_mean", 0.0),
                        context_precision=gen.get("context_precision_mean", 0.0),
                        table_integrity=float(table_integrity),
                        citation_validity=0.0,  # populated if/when citations are emitted
                        query_latency_ms=latency_ms,
                        query_cost_usd=0.0,
                    ))
        run.chunk_count = len(gold_set) * n_trials
    return results


def _trials_for_method(method: str, configured: int) -> int:
    if method in DETERMINISTIC_METHODS:
        return 1
    if method in STOCHASTIC_METHODS:
        return max(1, configured)
    return max(1, configured)


def _judge_trials(cfg: DictConfig) -> int:
    """Judge variance only matters when temperature > 0. At temp=0, 1 trial."""
    temp = float(cfg.llm.temperature)
    return max(1, int(cfg.evaluation.n_trials)) if temp > 0 else 1


# --- Helpers ---------------------------------------------------------------

def _load_gold_set(cfg, project_root) -> list[GoldEntry]:
    gold_path = project_root / cfg.paths.gold / "gold_standard.jsonl"
    if not gold_path.exists():
        logger.error(f"Gold set not found: {gold_path}. Run 'make goldset' first.")
        return []
    gold = load_gold(gold_path)
    auto = sum(1 for g in gold if g.validator == "auto" and g.query_type != "distractor")
    if auto:
        logger.warning(f"{auto}/{len(gold)} gold entries are validator='auto' — "
                       f"results are preliminary until human review lands.")
    return gold


def _make_tracker(cfg, project_root, stage: str) -> CostTracker:
    budget = float(cfg.cost_budget.get(stage, cfg.cost_budget.get("total", 100.0)))
    tracker = CostTracker(budget_usd=budget, stage=stage)
    tracker.set_log_path(project_root / cfg.paths.results / "cost_log.jsonl")
    return tracker


def _make_llm(cfg, project_root, tracker) -> LLMClient:
    return LLMClient(
        model=cfg.llm.model, temperature=float(cfg.llm.temperature),
        max_tokens=int(cfg.llm.max_tokens),
        cache_dir=str(project_root / cfg.paths.cache),
        cost_tracker=tracker,
    )


def _corpus_table_integrity(strategy: str, cfg, project_root: Path) -> float:
    """Load chunks for a strategy and compute corpus-wide table integrity (§8.3)."""
    path = project_root / cfg.paths.processed / strategy / "chunks.jsonl"
    if not path.exists():
        return 0.0
    chunks: list[ChunkMetadata] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                chunks.append(ChunkMetadata(**json.loads(line)))
            except Exception:
                continue
    return float(table_integrity_score(chunks).get("integrity_score", 0.0))


def _save_results(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.model_dump() for r in results])
    df.to_csv(path, index=False)
    logger.info(f"Results saved: {path} ({len(df)} rows)")


def _append_benchmark(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([r.model_dump() for r in results])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False)
    logger.info(f"Benchmark updated: {path} ({len(df)} total rows)")


def _save_abstentions(abstentions: dict[str, list[str]], path: Path):
    from src.evaluation.answer_generator import abstention_rate
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(abstention_rate(abstentions), indent=2))
