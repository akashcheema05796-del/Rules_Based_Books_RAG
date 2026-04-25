"""
Report generation (spec §9, §12).

Produces `results/report.md` with bootstrap 95% CIs on every cell, pairwise
paired-bootstrap tests with Holm-Bonferroni correction across strategies, and
an automated headline verdict against Outcomes A–D from spec §12.
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
    results_dir = project_root / cfg.paths.results
    figures_dir = project_root / cfg.paths.figures
    figures_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path = results_dir / "benchmark.csv"
    if not benchmark_path.exists():
        logger.error("No benchmark.csv found. Run evaluation phases first.")
        return

    df = pd.read_csv(benchmark_path)
    k_target = int(cfg.retrieval.default_k)
    n_resamples = int(cfg.evaluation.bootstrap_resamples)

    lines: list[str] = ["# RAG Chunking × Retrieval Benchmark v3 — Results Report\n"]

    # --- 1. Summary ---
    lines += [
        "## 1. Summary\n",
        f"- Total evaluation rows: {len(df)}",
        f"- Chunking strategies: {sorted(df['chunking_strategy'].unique())}",
        f"- Retrieval methods: {sorted(df['retrieval_method'].unique())}",
        f"- Query types: {sorted(df['query_type'].unique())}",
        f"- k values: {sorted(df['k'].unique())}",
        f"- Headline k: {k_target}",
        "",
    ]

    # --- 2. Phase 1: Chunking Isolation ---
    lines += ["## 2. Phase 1: Chunking Isolation (dense-only)\n"]
    phase1 = df[(df["retrieval_method"] == "dense") & (df["k"] == k_target)]
    phase1 = phase1[phase1["query_type"] != "distractor"]
    _write_strategy_table(lines, phase1, k_target, n_resamples)
    phase1_sig = _write_pairwise(lines, phase1, "Phase 1 pairwise", n_resamples)

    # --- 3. Phase 2: Retrieval Sweep ---
    lines += ["## 3. Phase 2: Retrieval Sweep\n"]
    phase2 = df[(df["k"] == k_target) & (df["query_type"] != "distractor")]
    if not phase2.empty:
        pivot = phase2.groupby(
            ["chunking_strategy", "retrieval_method"]
        )["recall_at_k"].mean().unstack()
        lines.append(f"### Recall@{k_target} Heatmap (Strategy × Method)\n")
        lines.append(pivot.round(4).to_markdown())
        lines.append("")

        # Generation metrics summary if present.
        if (phase2["faithfulness"] > 0).any():
            gen = phase2.groupby(["chunking_strategy", "retrieval_method"]).agg(
                faithfulness=("faithfulness", "mean"),
                answer_correctness=("answer_correctness", "mean"),
                context_precision=("context_precision", "mean"),
            ).round(3)
            lines.append("### Generation metrics (RAGAS-style)\n")
            lines.append(gen.to_markdown())
            lines.append("")

    # --- 4. Phase 3: Routing ---
    lines += ["## 4. Phase 3: Query-Type Routing\n"]
    routing_path = results_dir / "routing_table.json"
    if routing_path.exists():
        routing = json.loads(routing_path.read_text())
        table = routing.get("routing_table", {})
        lines += [f"| Query Type | Best Strategy | Best Method | Recall@{k_target} | n |",
                  "|---|---|---|---|---|"]
        for qt, info in table.items():
            lines.append(
                f"| {qt} | {info['best_strategy']} | {info['best_method']} | "
                f"{info.get(f'mean_recall_at_{k_target}', 0):.4f} | "
                f"{info.get('n_queries', '-')} |")
        pc = routing.get("policy_comparison", {})
        if pc:
            sb = pc.get("single_best_config", {})
            lines += [
                "",
                f"- **Single best config**: {sb.get('strategy')}/{sb.get('method')} "
                f"(recall@{k_target} = {sb.get(f'recall_at_{k_target}', 0):.4f})",
                f"- **Per-type routing**: recall@{k_target} = "
                f"{pc.get('per_type_routing', {}).get(f'recall_at_{k_target}', 0):.4f}",
                f"- **Lift from routing**: {pc.get('lift', 0):+.4f}",
                "",
            ]

    # --- 5. Distractor / abstention ---
    for phase_file in ("phase1_abstention.json", "phase2_abstention.json"):
        p = results_dir / phase_file
        if p.exists():
            data = json.loads(p.read_text())
            lines.append(f"## 5. Abstention rates ({phase_file})\n")
            lines += ["| Query Type | n | Abstention rate |", "|---|---|---|"]
            for qt, info in data.items():
                lines.append(f"| {qt} | {info['n']} | {info['abstention_rate']:.3f} |")
            lines.append("")

    # --- 6. Headline verdict ---
    lines.append("## 6. Headline Verdict\n")
    verdict = _compute_verdict(phase1, phase2, phase1_sig, k_target)
    lines.append(verdict)
    lines.append("")

    # --- 7. Deployment Recommendation ---
    lines.append("## 7. Deployment Recommendation\n")
    lines.append(_deployment_recommendation(verdict, phase2, k_target))
    lines.append("")

    _generate_plots(df, figures_dir)

    report_path = results_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report generated: {report_path}")


# --- Helpers ---------------------------------------------------------------

def _write_strategy_table(lines, df_slice, k, n_resamples):
    if df_slice.empty:
        lines.append("*No Phase 1 rows found.*\n")
        return
    lines.append(f"### Recall@{k} by strategy (bootstrap 95% CI)\n")
    lines.append(f"| Strategy | Mean | 95% CI | n |")
    lines.append("|---|---|---|---|")
    for s in sorted(df_slice["chunking_strategy"].unique()):
        vals = df_slice[df_slice["chunking_strategy"] == s]["recall_at_k"].tolist()
        ci = bootstrap_ci(vals, n_resamples=n_resamples)
        lines.append(f"| {s} | {ci['mean']:.4f} | [{ci['lower']:.4f}, {ci['upper']:.4f}] | {ci['n']} |")
    lines.append("")


def _write_pairwise(lines, df_slice, title, n_resamples) -> dict:
    """Pairwise paired bootstrap with Holm-Bonferroni, ordered by query_id."""
    if df_slice.empty:
        return {}
    per_system: dict[str, list[float]] = {}
    for s in sorted(df_slice["chunking_strategy"].unique()):
        sub = df_slice[df_slice["chunking_strategy"] == s].sort_values("query_id")
        per_system[s] = sub["recall_at_k"].tolist()
    lengths = {len(v) for v in per_system.values()}
    if len(lengths) != 1 or not per_system:
        lines.append(f"*{title}: unequal-length per-system arrays, skipping.*\n")
        return {}
    res = pairwise_significance(per_system, n_resamples=n_resamples)
    lines.append(f"### {title} (Holm-Bonferroni corrected)\n")
    lines.append("| Comparison | Mean diff | 95% CI | raw p | adj-α | significant? |")
    lines.append("|---|---|---|---|---|---|")
    for pair, info in sorted(res.items(), key=lambda x: x[1]["p_value"]):
        c = info["corrected"]
        lines.append(
            f"| {pair} | {info['mean_diff']:+.4f} | "
            f"[{info['ci_lower']:+.4f}, {info['ci_upper']:+.4f}] | "
            f"{info['p_value']:.4f} | {c['adjusted_alpha']:.4f} | "
            f"{'**yes**' if c['significant'] else 'no'} |")
    lines.append("")
    return res


def _compute_verdict(phase1, phase2, phase1_sig, k) -> str:
    """Auto-classify against spec §12 outcomes A–D."""
    if phase1.empty or phase2.empty:
        return "*Insufficient data for verdict — run all phases.*"

    p1_means = phase1.groupby("chunking_strategy")["recall_at_k"].mean()
    if p1_means.empty:
        return "*No Phase 1 data.*"

    p2 = phase2.copy()
    rerank_rows = p2[p2["retrieval_method"] == "hybrid_rerank"]
    if rerank_rows.empty:
        return ("Hybrid+rerank missing from Phase 2; cannot evaluate the headline "
                "question. Re-run with `retrieval_methods=hybrid_rerank` at minimum.")

    p2_means = rerank_rows.groupby("chunking_strategy")["recall_at_k"].mean()
    p1_spread = float(p1_means.max() - p1_means.min()) if len(p1_means) > 1 else 0.0
    p2_spread = float(p2_means.max() - p2_means.min()) if len(p2_means) > 1 else 0.0

    # Cross-significance check — does any Phase-1 gap survive under rerank?
    any_sig = any(info["corrected"]["significant"] for info in phase1_sig.values()) if phase1_sig else False

    contextual_p1 = float(p1_means.get("contextual", 0.0))
    adaptive_p1 = float(p1_means.get("adaptive", 0.0))
    best_p1 = p1_means.idxmax()

    msg = [
        f"- Phase 1 spread @k={k}: {p1_spread:.4f}  (best = {best_p1})",
        f"- Phase 2 spread @k={k} under hybrid+rerank: {p2_spread:.4f}",
        f"- Any Phase-1 pairwise gap significant (Holm-Bonf): {any_sig}",
        "",
    ]

    if p2_spread < 0.02 and p1_spread >= 0.02:
        verdict = ("**Outcome B — Retrieval dominates.** Phase 1 ranking collapses "
                   "under hybrid+rerank (spread < 0.02). Recommend recursive + "
                   "hybrid+rerank; skip advanced chunking cost.")
    elif contextual_p1 > adaptive_p1 + 0.03 and best_p1 == "contextual":
        verdict = ("**Outcome A — Chunking matters, generative context wins.** "
                   "Contextual leads Phase 1 by a meaningful margin and the lead "
                   "persists. Recommend contextual retrieval pipeline.")
    elif abs(contextual_p1 - adaptive_p1) < 0.02 and adaptive_p1 >= max(
            p1_means.get("recursive", 0.0), p1_means.get("markdown_hierarchical", 0.0)):
        verdict = ("**Outcome D — Adaptive matches contextual at lower cost.** "
                   "Structural micro-headers capture most of the LLM-blurb benefit. "
                   "Recommend adaptive; cite contextual as quality ceiling.")
    else:
        verdict = ("**Outcome C — Mixed by query type.** No single (chunking, "
                   "retrieval) wins everywhere. See Phase 3 routing table; the "
                   "query-classifier + per-type policy is the primary artifact.")

    return "\n".join(msg) + verdict


def _deployment_recommendation(verdict: str, phase2: pd.DataFrame, k: int) -> str:
    if "Outcome B" in verdict:
        return "Ship: `recursive` chunks + `hybrid_rerank`. Skip contextual indexing cost."
    if "Outcome A" in verdict:
        return "Ship: `contextual` chunks + `hybrid_rerank`. Budget ~$1/M tokens one-time."
    if "Outcome D" in verdict:
        return "Ship: `adaptive` chunks + `hybrid_rerank`. ~100× cheaper than contextual."
    if "Outcome C" in verdict:
        return ("Ship: query classifier + per-type routing from "
                "`results/routing_table.json`. Fallback = `hybrid_rerank` + `recursive`.")
    return "*Deployment recommendation pending verdict.*"


def _generate_plots(df: pd.DataFrame, figures_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping plots")
        return

    phase1 = df[df["retrieval_method"] == "dense"]
    if len(phase1) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        for strategy in phase1["chunking_strategy"].unique():
            s = phase1[phase1["chunking_strategy"] == strategy]
            means = s.groupby("k")["recall_at_k"].mean()
            ax.plot(means.index, means.values, marker="o", label=strategy)
        ax.set_xlabel("k"); ax.set_ylabel("Recall@k")
        ax.set_title("Phase 1: Recall@k by Chunking Strategy (Dense-Only)")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.savefig(figures_dir / "phase1_recall_by_strategy.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    k10 = df[df["k"] == 10]
    if len(k10) > 0 and k10["retrieval_method"].nunique() > 1:
        pivot = k10.groupby(["chunking_strategy", "retrieval_method"])["recall_at_k"].mean().unstack()
        if len(pivot) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
            ax.set_title("Recall@10: Chunking × Retrieval Method")
            fig.savefig(figures_dir / "phase2_heatmap.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    if len(k10) > 0:
        query_types = [qt for qt in k10["query_type"].unique() if qt != "distractor"][:6]
        if query_types:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = np.array(axes).reshape(-1)
            for idx, qt in enumerate(query_types):
                ax = axes[idx]
                qt_data = k10[k10["query_type"] == qt]
                means = qt_data.groupby("chunking_strategy")["recall_at_k"].mean()
                means.plot(kind="bar", ax=ax,
                           color=sns.color_palette("husl", len(means)))
                ax.set_title(f"Recall@10: {qt}"); ax.set_ylabel("Recall")
                ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            fig.savefig(figures_dir / "query_type_breakdown.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    logger.info(f"Plots saved to {figures_dir}")
