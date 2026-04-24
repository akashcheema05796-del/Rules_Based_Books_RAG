# RAG Chunking × Retrieval Benchmark — Plan v3 (Final)

## 0. What Changed From v2

v2 was a pure chunking benchmark with 8 strategies. Literature review and corpus inspection led to:

| | v2 | v3 |
|---|---|---|
| Scope | Chunking only | Chunking × Retrieval (2 axes, phased) |
| Chunking strategies | 8 | **5** (evidence-pruned) |
| Retrieval methods | Implicit (dense-only) | **8** explicit methods |
| Cut strategies | — | Fixed, semantic, propositional, parent/child (moved), late chunking, whole-book oracle |
| Headline question | "Which chunker wins?" | "Does chunking still matter under strong retrieval?" |
| Recursive chunk size | 1000 / 100 overlap | **512 / 15% overlap** (FloTorch 2026) |
| Hybrid fusion | — | **RRF, k=60** (2026 default) |
| Three-stage pipeline | — | BM25 + Dense + Rerank (IBM Blended RAG consensus) |

---

## 1. Project Objective

Measure how chunking strategy and retrieval method jointly affect retrieval quality, generation quality, and cost on a 26-book AD&D 2e Markdown corpus (~15 MB, ~148k lines, ~20k table lines, ~9.7k h2 headers).

**The headline question:** *Does the choice of chunking strategy still matter once a strong retrieval pipeline (hybrid + rerank) is in place?* Either answer is useful. If yes → invest in contextual retrieval. If no → use recursive and save the chunking budget.

**Deliverables**

1. `results/benchmark.csv` — one row per (chunking, retrieval, k, trial) with all metrics.
2. `results/figures/` — per-query-type breakdowns, cost/quality Pareto, chunk-size distributions.
3. `results/chunk_samples/` — 50 sampled chunks per chunking strategy.
4. `results/report.md` — findings, significance claims, deployment recommendation.

---

## 2. Experimental Design — Three Phases

Chunking × retrieval = 5 × 8 = 40 combinations before k-sweeps and trials. A staged design avoids the combinatorial blowup.

### Phase 1: Chunking Isolation (dense-only retrieval)

Fixed retrieval (dense, top-10), sweep chunking. Produces a ranking of chunking strategies. ~5 strategies × 3 trials ≈ 15 runs.

### Phase 2: Retrieval Sweep

For each chunking strategy surviving Phase 1 (all 5 if costs allow, top 3 if tight), sweep all 8 retrieval methods. ~5 × 8 × 3 ≈ 120 runs. **This is where the headline question gets answered.**

### Phase 3: Query-Type Routing

Break down Phase 2 results by query type (lore / mechanical / tabular / cross-ref / monster / numeric). Test whether different (chunking, retrieval) configurations win different query types, and whether a query classifier + routing policy beats any single configuration.

### Controlled Variables (identical across every run)

- Embedding model: `text-embedding-3-small` (1536-dim). Pinned once, used everywhere.
- Vector store: ChromaDB, cosine distance, HNSW defaults.
- Sparse index: `rank_bm25` (Okapi BM25, k1=1.5, b=0.75).
- Fusion: RRF with k=60 (2026 default per Prem AI benchmark guide).
- Reranker: `bge-reranker-v2-m3` (open, fast, strong).
- LLM judge / generator: `claude-sonnet-4-5`, `temperature=0`, `seed=42`.
- Tokenizer for chunk sizing: `cl100k_base`.
- Random seed: `42` everywhere.
- Top-k for retrieval metrics: `k ∈ {1, 3, 5, 10, 20}`.

---

## 3. Workspace Setup

```
/project_root
├── configs/
│   ├── base.yaml                       # shared: embedding model, k, seeds, LLM
│   ├── chunking/
│   │   ├── recursive.yaml              # 512 tok / 15% overlap
│   │   ├── markdown_hierarchical.yaml
│   │   ├── contextual.yaml
│   │   ├── table_aware.yaml
│   │   └── adaptive.yaml               # similarity-threshold + micro-headers
│   ├── retrieval/
│   │   ├── dense.yaml
│   │   ├── bm25.yaml
│   │   ├── hybrid_rrf.yaml
│   │   ├── hybrid_rerank.yaml          # BM25 + Dense → RRF → cross-encoder
│   │   ├── metadata_filter.yaml        # self-querying on is_table, book_title
│   │   ├── small_to_big.yaml           # retrieve child chunks, return parent
│   │   ├── hyde.yaml                   # hypothetical doc expansion
│   │   └── query_decomposition.yaml    # for multi-hop
│   └── books_manifest.yaml             # 26 book titles for boundary detection
├── data/
│   ├── raw/                            # DnD_Second_edition__all_26_books.md
│   ├── interim/                        # AST dump, per-book splits
│   ├── processed/                      # {chunking}/{book}.jsonl chunk files
│   ├── gold/                           # gold_standard.jsonl (stratified, with reference contexts)
│   ├── vector_store/                   # {chunking}/ Chroma dirs (dense)
│   └── bm25_index/                     # {chunking}/ BM25 indices (sparse)
├── src/
│   ├── corpus/                         # §4: profile, parse, segment
│   ├── chunkers/                       # §5: 5 strategies
│   ├── retrieval/                      # §6: 8 retrieval methods
│   ├── evaluation/                     # §7: metrics + statistical analysis
│   ├── goldset/                        # §7.1: generation + validation
│   └── utils/                          # cache, cost tracking, logging, backoff
├── tests/
├── notebooks/
├── results/
├── main.py                             # Hydra-driven orchestrator
├── Makefile
├── .env.example
└── requirements.txt                    # pinned with hashes (pip-compile)
```

**Dependencies (pinned):** `langchain`, `langchain-text-splitters`, `chromadb`, `rank_bm25`, `sentence-transformers` (for reranker), `FlagEmbedding` (bge-reranker-v2), `ragas`, `markdown-it-py`, `tiktoken`, `anthropic`, `openai` (for embeddings), `hydra-core`, `pydantic`, `pandas`, `scipy` (bootstrap), `tenacity`, `diskcache`, `tqdm`, `pytest`.

---

## 4. Corpus Preparation

### 4.1 Profile (run once)

`src/corpus/profile.py` → `results/corpus_profile.json`: total tokens, header distribution per level, table count / line count / size stats, code-block stats. Drives parameter defaults.

### 4.2 Book Boundary Detection — Manifest-Driven

Critical finding from corpus inspection: `#` headers do **not** delimit the 26 books (only 3 exist corpus-wide, 2 are false positives). Books are at `##` level with bold titles.

1. Parse to AST (§4.3).
2. Normalize h2 titles (lowercase, strip punctuation, ®/™).
3. Match against `configs/books_manifest.yaml` (curated list of 26 AD&D 2e book titles).
4. Emit `(start_offset, end_offset, book_title)` spans.
5. Unit test: exactly 26 books, no overlaps, full byte coverage.
6. Fallback: unmatched candidate h2s dumped to `interim/unmatched_headers.txt` for manual curation.

### 4.3 Markdown Parsing — AST, Not Regex

Parse once with `markdown-it-py`. Preserve:

- **Tables** as atomic nodes (`header_row`, `align_row`, `body_rows`). Never split internally.
- **Code blocks** as atomic.
- **Lists** with depth annotation.
- **Blockquotes** atomic when short.

All chunkers consume nodes, not raw text.

### 4.4 Metadata Schema

```python
class ChunkMetadata(BaseModel):
    chunk_id: str                     # sha1(strategy + content + position)
    strategy: str
    book_title: str
    book_index: int                   # 0..25
    chapter_path: list[str]           # ["Chapter 3", "The Warrior", "Fighters"]
    char_start: int
    char_end: int
    token_count: int                  # cl100k_base
    is_table: bool                    # chunk is primarily a table
    contains_table: bool              # chunk includes a table among other content
    contains_code: bool
    parent_chunk_id: str | None       # for small-to-big retrieval
    contextual_prefix: str | None     # for contextual retrieval chunking
```

---

## 5. Chunking Strategies (5)

All strategies consume the AST from §4.3 and write to `data/processed/{strategy}/`. **Evidence basis** column cites the primary supporting source.

| # | Strategy | Parameters | Evidence basis |
|---|---|---|---|
| 5.1 | **Recursive** | 512 tokens, 15% overlap, markdown separators | FloTorch 2026 (69% on academic papers); chemistry RAG arXiv:2506.17277 (R100-0 wins 25-config sweep); Vectara NAACL 2025 |
| 5.2 | **Markdown Hierarchical** | Split on h2/h3/h4; oversized chunks (>2000 tok) post-split with §5.1 | Corpus has 9.7k h2 headers — structural free lunch; arXiv:2402.05131 (financial reports) |
| 5.3 | **Contextual Retrieval** | Prepend LLM-generated context blurb before embedding. Cache keyed by (book, chapter_path, content_hash) | Anthropic: 35% failure reduction alone, 49% with BM25, 67% with rerank. Independently replicated by Unstructured, Milvus, LlamaIndex. arXiv:2504.19754 beats late chunking |
| 5.4 | **Table-Aware** | Each table = one chunk with ~200 tok preceding context prepended; non-table falls through to §5.1 | Corpus-specific (20k table lines, 20% of gold set is tabular); no general benchmark but obvious from corpus profile |
| 5.5 | **Adaptive** | Sentence-level embeddings → merge adjacent sentences while cosine sim > τ (default 0.75) **and** token count < 512; prepend deterministic micro-header `[{book} › {chapter_path}]` to each chunk | Clinical RAG PMC12649634: adaptive won 87% accuracy vs 50% baseline on medical rule content (closest analogue to TTRPG rules); beat semantic and propositional in same head-to-head |

**Contextual retrieval prompt** (from Anthropic, versioned in `configs/prompts/contextual.md`):

```
<document>{WHOLE_BOOK}</document>
Here is the chunk we want to situate within the overall document:
<chunk>{CHUNK_CONTENT}</chunk>
Please give a short succinct context to situate this chunk within the overall document
for the purposes of improving search retrieval of the chunk. Answer only with the
succinct context and nothing else.
```

Use prompt caching on `{WHOLE_BOOK}` — this is what makes the strategy economically viable (~$1.02/M tokens with caching).

**Adaptive chunking implementation note.** Adaptive (§5.5) is deliberately a middle ground between semantic and contextual:

- **Boundaries** come from sentence-embedding cosine similarity (like semantic) — no LLM call per chunk.
- **Micro-headers** are deterministic structural prefixes derived from the AST (`[Player's Handbook › Chapter 9 › Combat]`) — no LLM call per chunk.
- **Cost**: one embedding per sentence at index time (reuses the pinned embedding model), zero LLM calls. An order of magnitude cheaper than contextual.
- **Risk**: the clinical paper that validated this method used medical content with strong topical continuity between sentences. TTRPG rules jump sharply between adjacent paragraphs (stat block → flavor text → table → rule). Tune τ on a held-out validation slice before committing — starting at 0.75 is a guess, not a result.
- **Phase 2 interpretation**: because adaptive and contextual address the same failure mode via different mechanisms (structural vs. generative context), their Phase 2 results are directly comparable — if adaptive matches contextual at a fraction of the cost, that's the deployment recommendation.

**Explicitly cut and why**

- **Fixed** — dominated by recursive in every benchmark.
- **Pure Semantic** (vanilla `SemanticChunker`) — Vectara NAACL 2025 and chemistry benchmark both show costs aren't justified by gains. Adaptive (§5.5) subsumes it by adding micro-headers, which is where the clinical paper's gains came from.
- **Propositional** — clinical RAG paper PMC12649634 places it below adaptive and semantic; atomizing TTRPG rules loses qualifiers ("...only if level ≥ 3").
- **Late chunking** — dominated by contextual retrieval in arXiv:2504.19754 head-to-head.
- **Parent/Child** — reclassified as a **retrieval method** (§6.6 small-to-big), not a chunking strategy. Composable with any chunker.
- **Whole-book oracle** — not useful once retrieval is the focus; books exceed practical context windows anyway.

---

## 6. Retrieval Methods (8)

Each method operates on the indices built per chunking strategy. For each chunking strategy, build **two indices**: a ChromaDB dense index and a BM25 sparse index.

| # | Method | Description | Evidence basis |
|---|---|---|---|
| 6.1 | **Dense-only** | Cosine similarity on `text-embedding-3-small`. Baseline reference. | — |
| 6.2 | **BM25-only** | Sparse keyword retrieval. Critical for proper nouns / exact terms ("THAC0", "Drow", "Vorpal"). | Prem AI 2026: BM25 essential for exact-match domains |
| 6.3 | **Hybrid (Dense + BM25, RRF k=60)** | Run both, fuse via Reciprocal Rank Fusion. | IBM Blended RAG (three-way best); Milvus Pass@5 = 84.7%; Prem AI default |
| 6.4 | **Hybrid + Rerank (three-stage)** | 6.3 retrieves top-50, `bge-reranker-v2-m3` rescores, return top-k. | Pinecone: 48% improvement over single-method; Anthropic: 67% failure reduction combined with contextual |
| 6.5 | **Metadata Filter + Dense** | LLM parses query into `(filter, semantic_query)`; filter on `is_table`, `book_title`, `chapter_path`; dense-retrieve within. | Near-free given v3 metadata schema; targets tabular queries specifically |
| 6.6 | **Small-to-Big** | Chunk small (200 tok), retrieve small, return parent section (2000 tok) for context. Replaces parent/child chunking. | Standard LangChain pattern; decouples retrieval precision from context breadth |
| 6.7 | **HyDE** | LLM generates hypothetical answer → embed answer → retrieve. Evaluated only on lore/narrative queries. | Mixed evidence; helps on semantic queries (+6 pts NDCG per one benchmark), hurts on keyword queries |
| 6.8 | **Query Decomposition** | LLM splits multi-hop query into sub-queries; retrieve per sub-query; merge. For cross-reference queries only. | Agentic RAG: 94.5% HotpotQA, 89.7% 2WikiMultiHop |

**Explicitly skipped**

- **SPLADE / learned sparse** — IBM three-way retrieval is compelling but adds deployment complexity beyond scope. Note in report as future work.
- **ColBERT late interaction** — strong but heavy. Cross-encoder reranker is the 80/20.
- **LLM reranker** — cross-encoder gets ~85% of the benefit at ~1% of the cost.
- **GraphRAG / RAPTOR** — whole separate projects.

---

## 7. Gold Standard Dataset

Target: **200 stratified questions** with reference contexts. Stored at `data/gold/gold_standard.jsonl`.

### 7.1 Stratification

| Query type | Count | Example | Expected winner |
|---|---|---|---|
| Lore / narrative | 50 | "What are Drow social hierarchies?" | Dense + HyDE |
| Rules / mechanical | 50 | "THAC0 of level-5 Fighter?" | Hybrid + rerank |
| Tabular lookup | 40 | "Longsword damage S-M / L?" | Metadata filter + table-aware chunking |
| Cross-reference (multi-hop) | 30 | "Spells a multiclass fighter/mage can cast in plate armor?" | Query decomposition |
| Monster stats | 20 | "Hit Dice of adult red dragon?" | BM25 or hybrid |
| Numeric aggregation | 10 | "3rd-level spells for level-9 Priest?" | Metadata filter + hybrid |

Expected-winner column is a hypothesis to test, not a constraint.

### 7.2 Gold Entry Schema

```json
{
  "id": "q_0001",
  "query_type": "mechanical",
  "is_multi_hop": false,
  "query": "What is the THAC0 of a 5th-level human Fighter?",
  "reference_answer": "16",
  "reference_contexts": [
    {"book": "Player's Handbook",
     "chapter_path": ["Chapter 9", "Combat"],
     "char_start": 482301, "char_end": 483120}
  ],
  "validator": "human",
  "notes": "PHB Table 38"
}
```

### 7.3 Generation & Validation

1. **Seed**: prompt Claude with sampled sections (stratified across books and query types) → ~400 candidates.
2. **Auto-filter**: embedding dedup (similarity > 0.92), discard questions answerable without retrieval, discard if answer not in corpus.
3. **Human validation**: 2 reviewers confirm reference contexts actually contain the answer. 20-question overlap for inter-rater agreement (Cohen's κ ≥ 0.7 required).
4. **Version**: content-hash the final JSONL; all strategies evaluated against one frozen version.
5. **Distractors**: 10 additional questions whose answers aren't in the corpus — test robustness (good system should return low confidence / "I don't know").

---

## 8. Evaluation Metrics

### 8.1 Retrieval Metrics (primary — headline)

Computed against `reference_contexts`. Overlap defined as ≥ 50% intersection of shorter span.

- **Recall@k** (headline) — reported at k ∈ {1, 3, 5, 10, 20}
- **MRR** — mean reciprocal rank of first relevant chunk
- **nDCG@k** — graded by overlap fraction
- **Hit@k** — binary recall

### 8.2 Generation Metrics (secondary)

Via RAGAS. Judge = `claude-sonnet-4-5`, `temperature=0`, 3 trials.

- **Faithfulness** — answer grounded in retrieved context
- **Answer Correctness** — against `reference_answer`
- **Context Precision** — fraction of retrieved contexts actually relevant

### 8.3 Mechanical Metrics

- **Table Integrity Score** — fraction of retrieved chunks that don't sever tables (orphan body rows, header-only chunks, split tables). Reported per strategy at index time (corpus-wide) and at query time (gold-set averaged).
- **Citation Validity** — cited chunk IDs exist and contain cited claims.

### 8.4 Systems Metrics (first-class)

- **Chunking time, chunking cost** (USD — relevant for §5.3 contextual)
- **Indexing time, index size on disk** (both dense and sparse)
- **Query latency** p50 and p95 per retrieval method
- **Query cost** (HyDE, decomposition, rerank each add cost)
- **Chunk count, token distribution** (mean, p50, p95, max, coefficient of variation)

---

## 9. Statistical Analysis

- **Bootstrap 95% CIs** on every metric (10k resamples over gold-set questions).
- **Pairwise significance**: paired bootstrap test on retrieval metrics. Holm–Bonferroni correction across all strategy pairs.
- **Judge variance**: 3 trials per RAGAS metric; report mean ± SD.
- Every "A beats B" claim in the report must cite corrected p-value and CI.

---

## 10. Reproducibility & Operations

- **Seeds**: `random`, `numpy`, `torch`, LLM `seed=42`.
- **Determinism**: `temperature=0` on every LLM call.
- **Pinned deps**: `pip-compile` → `requirements.txt` with hashes.
- **Caching**: `diskcache` for embeddings, contextual blurbs, HyDE outputs, reranker scores. Log cache hit rate.
- **Rate limiting**: `tenacity` exponential backoff (base=1s, max=60s, 6 retries).
- **Cost budget**: per-stage caps in `configs/base.yaml`; abort with projection before expensive stages (§5.3 contextual build, §6.7 HyDE at query time).
- **Checkpointing**: every §4/§5/§6 output resumable.
- **Logging**: JSONL to `process.log` — per run: strategy, method, chunk count, tokens, wall time, dollar cost.

---

## 11. Execution Pipeline

```bash
# One-time
make profile       # §4.1  corpus profile
make parse         # §4.2-4.3  AST + book spans
make goldset      # §7  generate + human validation

# Phase 1: chunking isolation
make chunk         # §5  all 5 strategies
make index         # build dense + BM25 indices for each chunking
make eval-phase1   # §8 retrieval metrics, dense-only retrieval

# Phase 2: retrieval sweep
make eval-phase2   # §8 × all retrieval methods × top chunking strategies

# Phase 3: query-type breakdown
make eval-phase3   # per-query-type analysis + routing policy

make report        # §9 stats + plots + writeup
```

Each phase is idempotent and checkpointed. Re-run a single (chunking, retrieval) combo via `CHUNKING=contextual RETRIEVAL=hybrid_rerank make eval-phase2`.

---

## 12. Expected Outcomes & How to Interpret Them

The headline question now has a sharper form with adaptive in the mix: *does the structural-context approach (adaptive) match the generative-context approach (contextual) at a fraction of the cost?*

**Outcome A — Chunking matters even with strong retrieval, generative context wins.** Contextual retrieval (§5.3) wins Phase 1 by ≥ 5 points recall@10, and that lead persists in Phase 2 under hybrid+rerank. Adaptive trails by a meaningful margin. Recommendation: invest in contextual retrieval pipeline; budget the ~$1/M-token one-time cost.

**Outcome B — Retrieval dominates.** Phase 1 ranking collapses in Phase 2 once hybrid+rerank is applied; all chunking strategies cluster within CI. Recommendation: use recursive + hybrid + rerank; skip the advanced chunking indexing cost.

**Outcome C (most likely) — Mixed by query type.** Contextual/adaptive wins lore, table-aware wins tabular, recursive+hybrid wins mechanical. Phase 3 produces a routing policy. Recommendation: query classifier + per-type retrieval config, reported as the primary artifact.

**Outcome D — Adaptive matches contextual at lower cost.** Adaptive's structural micro-headers capture most of what contextual's LLM blurbs provide, with ~100× lower indexing cost. This is the most *practically valuable* outcome because it means "context-aware chunking" is cheap. Recommendation: deploy adaptive; cite contextual as the quality ceiling. Watch for this in Phase 1 results before committing Phase 2 budget.

---

## 13. Open Questions Before Kickoff

1. Embedding model: `text-embedding-3-small` (API) vs `bge-small-en-v1.5` (local GPU). Defaults to the former in this spec — flip if GPU is available and cost is a concern.
2. Total cost budget for a full run? Sets whether Phase 2 sweeps 2 or 4 chunking strategies.
3. Who validates the gold set? (Minimum 2 reviewers required for κ.)
4. Public release of chunks? (AD&D 2e material is copyrighted — internal evaluation only unless cleared by WotC/Hasbro.)
5. Is SPLADE or ColBERT in scope for v2 of this benchmark, given the three-way retrieval literature?
