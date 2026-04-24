"""
Report generation (spec §9, §12).

Generates results/report.md with findings, significance claims,
and deployment recommendation. Also generates figures.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.evaluation.statistics import bootstrap_ci, pairwise_significance

logger = logging.getLogger(__name__)


def generate_report(cfg: DictConfig, project_root: Path) -> None:
    """Generate the final benchmark report."""
    results_dir = project_root / cfg.paths.results
    figures_dir = project_root / cfg.paths.figures
    figures_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path = results_dir / "benchmark.csv"
    if not benchmark_path.exists():
        logger.error("No benchmark.csv found. Run evaluation phases first.")
        return

    df = pd.read_csv(benchmark_path)
    report_lines = ["# RAG Chunking × Retrieval Benchmark v3 — Results Report\n"]

    # --- Summary Statistics ---
    report_lines.append("## 1. Summary\n")
    report_lines.append(f"- Total evaluation rows: {len(df)}")
    report_lines.append(f"- Chunking strategies: {df['chunking_strategy'].nunique()}")
    report_lines.append(f"- Retrieval methods: {df['retrieval_method'].nunique()}")
    report_lines.append(f"- Query types: {df['query_type'].nunique()}")
    report_lines.append(f"- K values tested: {sorted(df['k'].unique())}")
    report_lines.append("")

    # --- Phase 1: Chunking Isolation ---
    report_lines.append("## 2. Phase 1: Chunking Isolation (Dense-Only)\n")
    phase1 = df[df["retrieval_method"] == "dense"]
    if len(phase1) > 0:
        k10 = phase1[phase1["k"] == 10]
        if len(k10) > 0:
            summary = k10.groupby("chunking_strategy").agg({
                "recall_at_k": ["mean", "std"],
                "mrr": ["mean", "std"],
                "ndcg_at_k": ["mean", "std"],
            }).round(4)
            report_lines.append("### Recall@10 by Chunking Strategy\n")
            report_lines.append("| Strategy | Recall@10 (mean ± std) | MRR | nDCG@10 |")
            report_lines.append("|---|---|---|---|")
            for strategy in summary.index:
                r = summary.loc[strategy]
                report_lines.append(
                    f"| {strategy} | "
                    f"{r[('recall_at_k', 'mean')]:.4f} ± {r[('recall_at_k', 'std')]:.4f} | "
                    f"{r[('mrr', 'mean')]:.4f} | "
                    f"{r[('ndcg_at_k', 'mean')]:.4f} |")
            report_lines.append("")

            # Bootstrap CIs
            report_lines.append("### Bootstrap 95% Confidence Intervals\n")
            for strategy in k10["chunking_strategy"].unique():
                vals = k10[k10["chunking_strategy"] == strategy]["recall_at_k"].values
                ci = bootstrap_ci(list(vals))
                report_lines.append(
                    f"- **{strategy}**: {ci['mean']:.4f} "
                    f"[{ci['lower']:.4f}, {ci['upper']:.4f}]")
            report_lines.append("")

    # --- Phase 2: Retrieval Sweep ---
    report_lines.append("## 3. Phase 2: Retrieval Sweep\n")
    phase2 = df[df["k"] == 10]
    if len(phase2) > 0:
        pivot = phase2.groupby(
            ["chunking_strategy", "retrieval_method"]
        )["recall_at_k"].mean().unstack()
        if len(pivot) > 0:
            report_lines.append("### Recall@10 Heatmap (Strategy × Method)\n")
            report_lines.append(pivot.round(4).to_markdown())
            report_lines.append("")

    # --- Phase 3: Query Type Analysis ---
    report_lines.append("## 4. Phase 3: Query-Type Routing\n")
    routing_path = results_dir / "routing_table.json"
    if routing_path.exists():
        routing = json.loads(routing_path.read_text())
        report_lines.append("| Query Type | Best Strategy | Best Method | Recall@10 |")
        report_lines.append("|---|---|---|---|")
        for qt, info in routing.items():
            report_lines.append(
                f"| {qt} | {info['best_strategy']} | "
                f"{info['best_method']} | {info['mean_recall']:.4f} |")
        report_lines.append("")

    # --- Headline Finding ---
    report_lines.append("## 5. Headline Finding\n")
    report_lines.append("*Analysis pending full evaluation data.*\n")
    report_lines.append("The headline question: **Does the choice of chunking strategy "
                        "still matter once a strong retrieval pipeline (hybrid + rerank) "
                        "is in place?**\n")

    # --- Deployment Recommendation ---
    report_lines.append("## 6. Deployment Recommendation\n")
    report_lines.append("*To be determined after full Phase 2 and Phase 3 analysis.*\n")

    # Generate plots
    _generate_plots(df, figures_dir)

    # Write report
    report_path = results_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info(f"Report generated: {report_path}")


def _generate_plots(df: pd.DataFrame, figures_dir: Path) -> None:
    """Generate benchmark visualization plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping plots")
        return

    # 1. Recall@k by chunking strategy (Phase 1)
    phase1 = df[df["retrieval_method"] == "dense"]
    if len(phase1) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        for strategy in phase1["chunking_strategy"].unique():
            s_data = phase1[phase1["chunking_strategy"] == strategy]
            means = s_data.groupby("k")["recall_at_k"].mean()
            ax.plot(means.index, means.values, marker="o", label=strategy)
        ax.set_xlabel("k")
        ax.set_ylabel("Recall@k")
        ax.set_title("Phase 1: Recall@k by Chunking Strategy (Dense-Only)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(figures_dir / "phase1_recall_by_strategy.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 2. Strategy × Method heatmap
    k10 = df[df["k"] == 10]
    if len(k10) > 0 and k10["retrieval_method"].nunique() > 1:
        pivot = k10.groupby(
            ["chunking_strategy", "retrieval_method"]
        )["recall_at_k"].mean().unstack()
        if len(pivot) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
            ax.set_title("Recall@10: Chunking × Retrieval Method")
            fig.savefig(figures_dir / "phase2_heatmap.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # 3. Per query-type breakdown
    if len(k10) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        query_types = k10["query_type"].unique()
        for idx, qt in enumerate(query_types[:6]):
            ax = axes[idx // 3][idx % 3]
            qt_data = k10[k10["query_type"] == qt]
            means = qt_data.groupby("chunking_strategy")["recall_at_k"].mean()
            means.plot(kind="bar", ax=ax, color=sns.color_palette("husl", len(means)))
            ax.set_title(f"Recall@10: {qt}")
            ax.set_ylabel("Recall")
            ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        fig.savefig(figures_dir / "query_type_breakdown.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"Plots saved to {figures_dir}")
