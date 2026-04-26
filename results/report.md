# RAG Chunking × Retrieval Benchmark v3 — Results Report

## 1. Summary

- Total evaluation rows: 19800
- Chunking strategies: ['markdown_hierarchical', 'recursive', 'table_aware']
- Retrieval methods: ['bm25', 'dense', 'hybrid_rrf', 'hyde', 'metadata_filter', 'query_decomposition', 'small_to_big']
- Query types: ['cross_reference', 'distractor', 'lore', 'mechanical', 'monster', 'numeric', 'tabular']
- k values: [np.int64(1), np.int64(3), np.int64(5), np.int64(10), np.int64(20)]
- Headline k: 10

## 2. Phase 1: Chunking Isolation (dense-only)

### Recall@10 by strategy (bootstrap 95% CI)

| Strategy | Mean | 95% CI | n |
|---|---|---|---|
| markdown_hierarchical | 0.8900 | [0.8450, 0.9300] | 200 |
| recursive | 0.8900 | [0.8450, 0.9300] | 200 |
| table_aware | 0.9100 | [0.8700, 0.9450] | 200 |

### Phase 1 pairwise (Holm-Bonferroni corrected)

| Comparison | Mean diff | 95% CI | raw p | adj-α | significant? |
|---|---|---|---|---|---|
| markdown_hierarchical vs recursive | +0.0000 | [-0.0400, +0.0400] | 0.5533 | 0.0167 | no |
| markdown_hierarchical vs table_aware | -0.0200 | [-0.0600, +0.0200] | 0.8744 | 0.0250 | no |
| recursive vs table_aware | -0.0200 | [-0.0500, +0.0050] | 0.9490 | 0.0500 | no |

## 3. Phase 2: Retrieval Sweep

### Recall@10 Heatmap (Strategy × Method)

| chunking_strategy     |   bm25 |   dense |   hybrid_rrf |   hyde |   metadata_filter |   query_decomposition |   small_to_big |
|:----------------------|-------:|--------:|-------------:|-------:|------------------:|----------------------:|---------------:|
| markdown_hierarchical |   0.97 |    0.89 |         0.97 |   0.88 |              0.89 |                  0.81 |           0.65 |
| recursive             |   0.97 |    0.89 |         0.96 |   0.86 |              0.89 |                  0.84 |           0.7  |
| table_aware           |   0.96 |    0.91 |         0.96 |   0.86 |              0.91 |                  0.84 |           0.74 |

## 4. Phase 3: Query-Type Routing

| Query Type | Best Strategy | Best Method | Recall@10 | n |
|---|---|---|---|---|
| lore | markdown_hierarchical | bm25 | 1.0000 | 25 |
| mechanical | markdown_hierarchical | bm25 | 1.0000 | 25 |
| tabular | markdown_hierarchical | dense | 1.0000 | 20 |
| cross_reference | markdown_hierarchical | hybrid_rrf | 1.0000 | 15 |
| monster | recursive | dense | 1.0000 | 10 |
| numeric | markdown_hierarchical | bm25 | 1.0000 | 5 |

- **Single best config**: markdown_hierarchical/bm25 (recall@10 = 0.8818)
- **Per-type routing**: recall@10 = 1.0000
- **Lift from routing**: +0.1182

## 5. Abstention rates (phase1_abstention.json)

| Query Type | n | Abstention rate |
|---|---|---|

## 5. Abstention rates (phase2_abstention.json)

| Query Type | n | Abstention rate |
|---|---|---|

## 6. Headline Verdict

Hybrid+rerank missing from Phase 2; cannot evaluate the headline question. Re-run with `retrieval_methods=hybrid_rerank` at minimum.

## 7. Deployment Recommendation

*Deployment recommendation pending verdict.*
