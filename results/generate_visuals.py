"""
Generate comprehensive benchmark visualizations.
Run from project root: python results/generate_visuals.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
STRATEGY_COLORS = {
    "recursive":             "#4C72B0",
    "markdown_hierarchical": "#DD8452",
    "table_aware":           "#55A868",
}
METHOD_ORDER = ["dense", "bm25", "hybrid_rrf", "metadata_filter",
                "small_to_big", "hyde", "query_decomposition"]
STRATEGY_ORDER = ["recursive", "markdown_hierarchical", "table_aware"]
K_VALUES = [1, 3, 5, 10, 20]

sns.set_theme(style="whitegrid", font_scale=1.1)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Recall@k curves — Phase 1 (dense, all 3 strategies)
# ─────────────────────────────────────────────────────────────────────────────
def plot_recall_curves(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    p1 = df[df["retrieval_method"] == "dense"]
    for strat in STRATEGY_ORDER:
        sub = p1[p1["chunking_strategy"] == strat]
        vals = [sub[sub["k"] == k]["recall_at_k"].mean() for k in K_VALUES]
        ax.plot(K_VALUES, vals, marker="o", linewidth=2.2,
                color=STRATEGY_COLORS.get(strat, "grey"), label=strat)
    ax.set_xlabel("k  (number of retrieved chunks)")
    ax.set_ylabel("Recall@k")
    ax.set_title("Recall@k Curves — Dense Retrieval by Chunking Strategy")
    ax.set_xticks(K_VALUES)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Strategy")
    fig.tight_layout()
    out = FIGURES / "recall_at_k_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Heatmap — Recall@10 (Strategy × Method)
# ─────────────────────────────────────────────────────────────────────────────
def plot_recall_heatmap(df):
    pivot = (df[df["k"] == 10]
             .groupby(["chunking_strategy", "retrieval_method"])["recall_at_k"]
             .mean()
             .unstack("retrieval_method")
             .reindex(index=STRATEGY_ORDER, columns=METHOD_ORDER))

    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGn",
                vmin=0.55, vmax=1.0, linewidths=0.5,
                cbar_kws={"label": "Recall@10"}, ax=ax)
    ax.set_title("Recall@10 — All Strategy × Method Combinations")
    ax.set_xlabel("Retrieval Method")
    ax.set_ylabel("Chunking Strategy")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    out = FIGURES / "recall10_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. nDCG@10 heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_ndcg_heatmap(df):
    pivot = (df[df["k"] == 10]
             .groupby(["chunking_strategy", "retrieval_method"])["ndcg_at_k"]
             .mean()
             .unstack("retrieval_method")
             .reindex(index=STRATEGY_ORDER, columns=METHOD_ORDER))

    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues",
                vmin=0.4, vmax=1.0, linewidths=0.5,
                cbar_kws={"label": "nDCG@10"}, ax=ax)
    ax.set_title("nDCG@10 — All Strategy × Method Combinations")
    ax.set_xlabel("Retrieval Method")
    ax.set_ylabel("Chunking Strategy")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    out = FIGURES / "ndcg10_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. MRR bar chart — all methods, best strategy per method
# ─────────────────────────────────────────────────────────────────────────────
def plot_mrr_bars(df):
    mrr = (df[df["k"] == 10]
           .groupby(["chunking_strategy", "retrieval_method"])["mrr"]
           .mean()
           .reset_index())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(METHOD_ORDER))
    width = 0.25
    for i, strat in enumerate(STRATEGY_ORDER):
        sub = mrr[mrr["chunking_strategy"] == strat].set_index("retrieval_method")
        vals = [sub.loc[m, "mrr"] if m in sub.index else 0 for m in METHOD_ORDER]
        ax.bar(x + i * width, vals, width, label=strat,
               color=STRATEGY_COLORS.get(strat, "grey"), alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(METHOD_ORDER, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("MRR")
    ax.set_title("Mean Reciprocal Rank (MRR) — Strategy × Method")
    ax.legend(title="Strategy")
    fig.tight_layout()
    out = FIGURES / "mrr_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Query-type breakdown — Recall@10 per query type, best method
# ─────────────────────────────────────────────────────────────────────────────
def plot_query_type_recall(df):
    qt_order = ["lore", "mechanical", "tabular", "cross_reference",
                "monster", "numeric", "distractor"]
    k10 = df[df["k"] == 10]
    best = (k10.groupby(["query_type", "chunking_strategy", "retrieval_method"])
              ["recall_at_k"].mean()
              .reset_index()
              .sort_values("recall_at_k", ascending=False)
              .drop_duplicates("query_type"))

    fig, ax = plt.subplots(figsize=(9, 5))
    qt_vals = []
    labels = []
    colors_list = []
    for qt in qt_order:
        row = best[best["query_type"] == qt]
        if row.empty:
            continue
        val = float(row["recall_at_k"].values[0])
        strat = row["chunking_strategy"].values[0]
        method = row["retrieval_method"].values[0]
        qt_vals.append(val)
        labels.append(f"{qt}\n({strat[:3]}+{method[:3]})")
        colors_list.append(STRATEGY_COLORS.get(strat, "#aaaaaa"))

    bars = ax.bar(labels, qt_vals, color=colors_list, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, qt_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall@10 (best config)")
    ax.set_title("Best Recall@10 per Query Type  (strategy+method shown)")
    patches = [mpatches.Patch(color=c, label=s) for s, c in STRATEGY_COLORS.items()]
    ax.legend(handles=patches, title="Strategy", loc="lower right")
    fig.tight_layout()
    out = FIGURES / "query_type_recall.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Hit@k bar chart (best strategy per k)
# ─────────────────────────────────────────────────────────────────────────────
def plot_hit_at_k(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    for strat in STRATEGY_ORDER:
        sub = df[(df["chunking_strategy"] == strat) &
                 (df["retrieval_method"] == "hybrid_rrf")]
        if sub.empty:
            continue
        vals = [sub[sub["k"] == k]["hit_at_k"].mean() for k in K_VALUES]
        ax.plot(K_VALUES, vals, marker="s", linewidth=2,
                color=STRATEGY_COLORS.get(strat, "grey"), label=strat)
    ax.set_xlabel("k")
    ax.set_ylabel("Hit@k")
    ax.set_title("Hit@k Curves — Hybrid RRF Retrieval")
    ax.set_xticks(K_VALUES)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Strategy")
    fig.tight_layout()
    out = FIGURES / "hit_at_k_hybrid_rrf.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Method comparison @ k=10 — grouped by strategy (Recall)
# ─────────────────────────────────────────────────────────────────────────────
def plot_method_comparison(df):
    k10 = df[df["k"] == 10]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, strat in zip(axes, STRATEGY_ORDER):
        sub = k10[k10["chunking_strategy"] == strat]
        method_means = sub.groupby("retrieval_method")["recall_at_k"].mean().reindex(METHOD_ORDER)
        bars = ax.bar(method_means.index, method_means.values,
                      color=STRATEGY_COLORS.get(strat, "grey"), alpha=0.85, edgecolor="white")
        ax.set_title(strat.replace("_", "\n"), fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis="x", rotation=40)
        for bar, v in zip(bars, method_means.values):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Recall@10")
    fig.suptitle("Recall@10 by Retrieval Method — Per Chunking Strategy", fontsize=12)
    fig.tight_layout()
    out = FIGURES / "method_comparison_recall10.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Routing lift visualisation
# ─────────────────────────────────────────────────────────────────────────────
def plot_routing_lift():
    rt_path = RESULTS / "routing_table.json"
    if not rt_path.exists():
        return
    with open(rt_path) as f:
        rt = json.load(f)

    policy = rt.get("policy_comparison", {})
    k = rt.get("k", 10)
    single = policy.get("single_best_config", {}).get(f"recall_at_{k}", 0)
    routed = policy.get("per_type_routing", {}).get(f"recall_at_{k}", 0)
    lift = policy.get("lift", 0)

    routing_table = rt.get("routing_table", {})
    qt_order = [qt for qt in ["lore", "mechanical", "tabular",
                               "cross_reference", "monster", "numeric"]
                if qt in routing_table]
    qt_recalls = [routing_table[qt].get(f"mean_recall_at_{k}", 0) for qt in qt_order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: single vs routed
    bars = ax1.bar(["Single-best\nconfig", "Per-type\nrouting"],
                   [single, routed], color=["#4C72B0", "#55A868"], width=0.45, alpha=0.85)
    for bar, v in zip(bars, [single, routed]):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel(f"Mean Recall@{k}")
    ax1.set_title(f"Routing lift: {lift:+.3f}")

    # Right: per-type recall
    colors_qt = [STRATEGY_COLORS.get(
        routing_table.get(qt, {}).get("best_strategy", ""), "#aaaaaa")
        for qt in qt_order]
    bars2 = ax2.bar(qt_order, qt_recalls, color=colors_qt, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars2, qt_recalls):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel(f"Recall@{k}")
    ax2.set_title("Per-Query-Type Best Recall@10")
    ax2.tick_params(axis="x", rotation=20)
    patches = [mpatches.Patch(color=c, label=s) for s, c in STRATEGY_COLORS.items()]
    ax2.legend(handles=patches, title="Best Strategy", fontsize=8)

    fig.suptitle("Phase 3: Query-Type Routing Analysis", fontsize=12)
    fig.tight_layout()
    out = FIGURES / "routing_analysis.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Chunk count comparison (corpus profile)
# ─────────────────────────────────────────────────────────────────────────────
def plot_chunk_counts():
    counts = {"recursive": 10419, "markdown_hierarchical": 10961, "table_aware": 10989}
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(counts.keys())
    vals = list(counts.values())
    bars = ax.barh(labels, vals,
                   color=[STRATEGY_COLORS[l] for l in labels], alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(v + 50, bar.get_y() + bar.get_height() / 2,
                f"{v:,}", va="center", fontsize=10)
    ax.set_xlabel("Number of Chunks")
    ax.set_title("Chunk Counts by Chunking Strategy")
    ax.set_xlim(0, max(vals) * 1.15)
    fig.tight_layout()
    out = FIGURES / "chunk_counts.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bench = pd.read_csv(RESULTS / "benchmark.csv")
    print(f"Loaded benchmark.csv: {len(bench)} rows")
    print("Generating figures...")

    plot_recall_curves(bench)
    plot_recall_heatmap(bench)
    plot_ndcg_heatmap(bench)
    plot_mrr_bars(bench)
    plot_query_type_recall(bench)
    plot_hit_at_k(bench)
    plot_method_comparison(bench)
    plot_routing_lift()
    plot_chunk_counts()

    print(f"\nAll figures saved to: {FIGURES}")
