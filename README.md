# RAG Chunking × Retrieval Benchmark

A systematic, phased benchmark evaluating **4 chunking strategies** × **7 retrieval methods** on the AD&D 2nd Edition corpus — a 15.7M-character, structure-heavy knowledge base of 26 rulebooks.

## Key Results

| Chunking Strategy | Best Method | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| **Markdown Hierarchical** | BM25 | **88.2%** | **77.6%** | **77.5%** |
| Recursive | BM25 | 88.2% | 74.8% | 75.6% |
| Fixed-Size | BM25 | 88.2% | 74.4% | 74.5% |
| Table-Aware | BM25 | 87.3% | 74.7% | 75.1% |

> **TL;DR — BM25 dominates.** Lexical matching outperforms dense/hybrid/LLM-augmented retrieval for structured rule-based text. Chunking strategy has minimal impact on peak recall — all four strategies tie at 88.2% with BM25. The recommended production configuration is **Markdown Hierarchical + BM25**.

### Routing Policy (Phase 3)

A query-type classifier can push weighted-average Recall@10 from **88.2% → ~95.4%** by routing to the optimal pipeline per query category:

| Query Type | Best Strategy | Best Method | Recall@10 |
|---|---|---|---|
| Numeric | Markdown Hierarchical | BM25 | **100.0%** |
| Mechanical | Markdown Hierarchical | BM25 | 97.6% |
| Monster | Recursive | Dense | 96.0% |
| Lore | Markdown Hierarchical | BM25 | 93.6% |
| Tabular | Markdown Hierarchical | BM25 | 93.0% |
| Cross-Reference | Recursive | BM25 | 92.0% |

Monster is the only query type where Dense beats BM25 — semantic search wins for prose-heavy lore content.

---

## Corpus

- **26 AD&D 2nd Edition rulebooks** as a single Markdown file
- 15.7M characters · 3.94M tokens · 148K lines
- Content types: combat rules, spell tables, monster stat blocks, lore, cross-references

## Benchmark Scale

```
4 strategies × 7 methods × 110 gold questions × 3 trials × 5 k-values = 52,800 rows
```

---

## Chunking Strategies

| Strategy | Chunks | Description |
|---|---|---|
| `recursive` | ~10,400 | LangChain-style recursive splits on headers → double-newlines → newlines. 512 token max, 77 token overlap. |
| `markdown_hierarchical` | ~10,900 | AST-based — groups content under heading nodes, preserves chapter path in metadata. |
| `table_aware` | ~11,200 | Keeps markdown tables atomic (never splits mid-table), adds `is_table` metadata flag. |
| `fixed_size` | ~8,300 | Pure token-window via tiktoken. 512-token windows, 64-token overlap. No document structure awareness. |

## Retrieval Methods

| Method | Phase | Description |
|---|---|---|
| `dense` | 1 | OpenAI `text-embedding-3-small` (1536d) + FAISS `IndexFlatIP` cosine search |
| `bm25` | 1 | BM25Okapi sparse retrieval (`rank_bm25`, k1=1.5, b=0.75) |
| `hybrid_rrf` | 1 | Reciprocal Rank Fusion (k=60) merging Dense + BM25 result lists |
| `small_to_big` | 1 | Retrieve small chunks, expand to 2000-token parent passages |
| `metadata_filter` | 2 | GPT-4o-mini parses query → JSON filters (book_title, is_table) + dense search |
| `hyde` | 2 | Hypothetical Document Embeddings — LLM generates answer, embed & retrieve |
| `query_decomposition` | 2 | LLM decomposes multi-hop query into sub-queries, RRF merge |

---

## Setup

### 1. Install dependencies

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and add your API keys:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key   # optional — only needed for contextual chunking
```

### 3. Provide corpus

Place the master markdown file inside `data/raw/`:

```
data/raw/DnD_Second_edition__all_26_books.md
```

---

## Running the Pipeline

This project uses [Hydra](https://hydra.cc/) for configuration. Run stages sequentially:

```bash
# 1. Parse corpus and cache AST (run once — takes ~40 min first time)
python main.py +stage=parse

# 2. Generate gold-standard evaluation questions
python main.py +stage=goldset

# 3. Chunk corpus (specify strategy or use 'all')
python main.py +stage=chunk +chunking_methods=markdown_hierarchical
python main.py +stage=chunk +chunking_methods=all

# 4. Build FAISS + BM25 indices
python main.py +stage=index +chunking_methods=all

# 5. Phase 1 — Chunking isolation (dense, BM25, hybrid, small-to-big)
python main.py +stage=eval_phase1 +chunking_methods=all

# 6. Phase 2 — Full retrieval sweep (includes LLM-augmented methods)
python main.py +stage=eval_phase2 +chunking_methods=all

# 7. Phase 3 — Query-type routing analysis + final report
python main.py +stage=report
```

> **Note:** Set `KMP_DUPLICATE_LIB_OK=TRUE` on Windows if you see OpenMP conflicts.

---

## Directory Structure

```
├── configs/                   # Hydra YAML configurations
│   └── base.yaml
├── src/
│   ├── chunkers/              # Chunking strategies
│   │   ├── recursive.py
│   │   ├── markdown_hierarchical.py
│   │   ├── table_aware.py
│   │   ├── fixed_size.py
│   │   └── adaptive.py        # Experimental (high chunk count, not benchmarked)
│   ├── corpus/                # AST parsing, book boundary detection
│   ├── evaluation/            # Metrics, runner, report generation
│   ├── goldset/               # Gold-standard question generation
│   ├── retrieval/             # Indexer, retriever factory, all 7 methods
│   └── utils/                 # Embeddings (w/ token truncation), tokenizer, seeds
├── data/
│   ├── raw/                   # Source corpus (not tracked in git)
│   ├── interim/               # AST cache (ast_nodes.pkl)
│   ├── processed/             # Chunk JSONL files per strategy
│   └── indices/               # FAISS + BM25 index files
├── results/
│   ├── benchmark.csv          # 52,800-row full results table
│   ├── phase1_results.csv
│   ├── phase2_results.csv
│   ├── report.md              # Auto-generated report with routing policy
│   └── figures/               # Recall heatmap, Recall@k curves, query-type breakdown
├── tests/                     # Pytest suite
├── main.py                    # Hydra entry point
└── requirements.txt
```

---

## Findings Summary

1. **BM25 beats Dense** — Lexical matching outperforms semantic search for rule names, spell names, and table lookups. BM25 achieves the same Recall@10 as Hybrid RRF at a fraction of the compute cost.

2. **Chunking strategy barely matters** — All four strategies (including naive fixed-size token windows) achieve 87–88% Recall@10 with BM25. Investing in complex structure-aware chunking yields minimal recall gain.

3. **LLM methods don't pay off** — HyDE (80.0–81.8%), Metadata Filter (79.1–82.0%), and Query Decomposition (70.9–76.4%) all trail BM25. The LLM overhead is not justified for this corpus type.

4. **Small-to-Big is the weakest method** — Context expansion to 2000-token windows dilutes relevance signals. Fixed-Size is hit especially hard (36.4% vs 60–69% for structure-aware strategies) because it has no natural parent-boundary to expand to.

5. **Routing unlocks ~95.4% recall** — A 6-way query-type classifier routes to the optimal pipeline per category. Monster queries uniquely benefit from Dense retrieval; everything else uses BM25.

---

## Testing

```bash
pytest tests/ -v
```
