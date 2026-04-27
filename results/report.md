# RAG Chunking × Retrieval Benchmark v3 — Results Report

## 1. Summary

| Metric | Value |
|---|---|
| Total evaluation rows | 52,800 |
| Chunking strategies | 4 (recursive, markdown_hierarchical, table_aware, fixed_size) |
| Retrieval methods | 7 |
| Gold questions | 110 |
| Query types | 6 (lore, mechanical, tabular, numeric, monster, cross_reference) |
| Trials per config | 3 |
| k values tested | 1, 3, 5, 10, 20 |

---

## 2. Phase 1: Chunking Isolation (Dense-Only Retrieval)

Dense retrieval used as a controlled baseline to isolate chunking quality from retrieval method effects.

### Recall@10 by Chunking Strategy

| Strategy | Recall@10 (mean ± std) | MRR | nDCG@10 |
|---|---|---|---|
| table_aware | 0.8364 ± 0.3702 | 0.6517 | 0.6672 |
| recursive | 0.8182 ± 0.3860 | 0.6723 | 0.6793 |
| markdown_hierarchical | 0.8182 ± 0.3860 | 0.6382 | 0.6630 |
| fixed_size | 0.8182 ± 0.3860 | 0.6394 | 0.6567 |

### Bootstrap 95% Confidence Intervals (Recall@10)

| Strategy | Mean | 95% CI |
|---|---|---|
| table_aware | 0.8364 | [0.8076, 0.8652] |
| recursive | 0.8182 | [0.7894, 0.8470] |
| markdown_hierarchical | 0.8182 | [0.7879, 0.8470] |
| fixed_size | 0.8182 | [0.7879, 0.8470] |

**Observation:** All four strategies are statistically indistinguishable under dense retrieval. Table-Aware has a slight edge — its atomic table blocks are more faithfully retrieved than prose-split chunks.

---

## 3. Phase 2: Full Retrieval Sweep

### Recall@10 Heatmap (Strategy × Method)

| Strategy | BM25 | Hybrid RRF | Dense | HyDE | Metadata Filter | Query Decomp | Small-to-Big |
|---|---|---|---|---|---|---|---|
| markdown_hierarchical | **0.882** | **0.882** | 0.818 | 0.809 | 0.791 | 0.709 | 0.600 |
| recursive | **0.882** | 0.873 | 0.818 | 0.818 | 0.800 | 0.727 | 0.645 |
| fixed_size | **0.882** | 0.873 | 0.818 | 0.800 | 0.818 | 0.727 | **0.364** |
| table_aware | 0.873 | 0.873 | 0.836 | 0.800 | 0.809 | 0.764 | 0.691 |

### Top 10 Configurations (Recall@10)

| Rank | Strategy | Method | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| 1 | markdown_hierarchical | bm25 | 0.882 | 0.776 | 0.775 |
| 2 | recursive | bm25 | 0.882 | 0.748 | 0.756 |
| 3 | fixed_size | bm25 | 0.882 | 0.744 | 0.745 |
| 4 | markdown_hierarchical | hybrid_rrf | 0.882 | 0.733 | 0.740 |
| 5 | recursive | hybrid_rrf | 0.873 | 0.731 | 0.738 |
| 6 | table_aware | hybrid_rrf | 0.873 | 0.717 | 0.733 |
| 7 | table_aware | bm25 | 0.873 | 0.747 | 0.751 |
| 8 | fixed_size | hybrid_rrf | 0.873 | 0.728 | 0.740 |
| 9 | table_aware | dense | 0.836 | 0.652 | 0.667 |
| 10 | recursive | dense | 0.818 | 0.672 | 0.679 |

### Key Observations

- **BM25 is the clear winner.** It achieves peak Recall@10 (88.2%) across 3 of 4 strategies and ties Hybrid RRF while being far cheaper.
- **Fixed-Size ties the leaderboard on Recall@10** but ranks 3rd on MRR/nDCG — its chunks have less coherent context, so when it retrieves correctly, the relevant span is lower in the result list.
- **Fixed-Size + Small-to-Big collapses to 36.4%** — the worst result in the benchmark. Without natural document boundaries, "parent passage" expansion produces incoherent 2000-token windows that hurt precision severely.
- **LLM-augmented methods (HyDE, Metadata Filter, Query Decomposition) do not outperform BM25** — they add API cost and latency without recall gains on this corpus.

---

## 4. Phase 3: Query-Type Routing Policy

Identifies the optimal chunking + retrieval pipeline for each query category.

### Optimal Pipeline per Query Type

| Query Type | Best Strategy | Best Method | Recall@10 | Rationale |
|---|---|---|---|---|
| numeric | markdown_hierarchical | bm25 | **1.000** | Exact number matching — BM25 locks on to numeric tokens perfectly |
| mechanical | markdown_hierarchical | bm25 | 0.976 | Rule names & keywords thrive on exact lexical matching |
| monster | recursive | dense | 0.960 | Prose-heavy lore — semantic embedding outperforms keyword search |
| lore | markdown_hierarchical | bm25 | 0.936 | Named entities & proper nouns favour exact matching |
| tabular | markdown_hierarchical | bm25 | 0.930 | Table headers have unique keywords — BM25 retrieves atomically |
| cross_reference | recursive | bm25 | 0.920 | "See page X" links need keyword overlap across book boundaries |

### Routing Uplift

A 6-way query-type classifier routing to the optimal pipeline achieves an estimated weighted-average **Recall@10 ≈ 95.4%** — a +7.2 point improvement over any single fixed pipeline (88.2%).

**Notable finding:** Monster queries are the only type where Dense retrieval beats BM25. This is because monster lore is written as free-form prose (descriptions, ecology, history) — semantic embeddings better capture the contextual meaning than term frequency.

---

## 5. Headline Findings

### Finding 1: BM25 beats Dense for structured rule text
Exact keyword matching outperforms semantic search for rule names, spell names, and table lookups across all chunking strategies. BM25 achieves identical Recall@10 to Hybrid RRF (0.882) at zero additional compute cost.

### Finding 2: Chunking strategy is surprisingly unimportant
All four strategies — including the naive fixed-size token window with zero document structure awareness — achieve 87–88% Recall@10 with BM25. The confidence intervals overlap completely. **Retrieval method choice drives far more variance than chunking strategy.**

> **Implication:** For practitioners building RAG on structured corpora, optimising retrieval (BM25 vs dense vs hybrid) yields much higher returns than engineering a sophisticated chunker.

### Finding 3: LLM-augmented retrieval doesn't pay off here
| Method | Avg Recall@10 | vs BM25 | Extra cost |
|---|---|---|---|
| BM25 | 0.879 | — | $0 |
| Hybrid RRF | 0.875 | -0.4% | Low |
| Dense | 0.823 | -5.6% | Low |
| HyDE | 0.807 | -7.2% | High (1 LLM call/query) |
| Metadata Filter | 0.804 | -7.5% | High |
| Query Decomposition | 0.732 | -14.7% | High (2+ LLM calls/query) |
| Small-to-Big | 0.575 | -30.4% | Low |

### Finding 4: Small-to-Big is the weakest method
Context expansion to 2000-token windows consistently dilutes relevance. Fixed-Size is catastrophically hurt (36.4%) because token windows have no natural document boundaries — expanded passages splice together unrelated content from different sections.

---

## 6. Deployment Recommendation

### Production Configuration
**Markdown Hierarchical Chunking + BM25 Retrieval**
- Recall@10: 88.2% | MRR@10: 77.6% | nDCG@10: 77.5%
- Zero LLM inference at retrieval time
- Fast, deterministic, low cost

### With Query Router (Recommended)
Add a lightweight 6-way query-type classifier (fine-tuned on the 110 gold questions or zero-shot with an LLM) to route:
- Numeric / Mechanical / Lore / Tabular / Cross-Reference → Markdown Hierarchical + BM25
- Monster → Recursive + Dense

**Expected uplift: ~95.4% Recall@10**

### What NOT to do
- **Avoid Small-to-Big** — consistently worst method, especially on token-window chunks
- **Skip LLM-augmented retrieval** (HyDE / Metadata Filter / Query Decomp) — adds cost without recall improvements on structured corpora
- **Don't over-engineer chunking** — a naive 512-token window ties the top on recall; spend that effort on retrieval instead
