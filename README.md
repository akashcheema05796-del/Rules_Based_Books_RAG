# RAG Chunking × Retrieval Benchmark v3

A comprehensive, phased benchmark system designed to evaluate chunking strategies and retrieval methods for Retrieval-Augmented Generation (RAG) using the **AD&D 2nd Edition** corpus as a challenging, structure-heavy dataset.

## Features

- **Robust Corpus Parsing**: Uses an AST-based markdown parser to safely extract headers, code blocks, and complex tabular data without severing semantic meaning.
- **5 Custom Chunking Strategies**:
  - `recursive`: LangChain-style recursive character splitting (512 tokens).
  - `markdown_hierarchical`: Structural chunking strictly respecting header boundaries.
  - `contextual`: LLM-powered (Prompt Caching) document-level prefix injection for isolated chunks.
  - `table_aware`: Preserves entire markdown tables within single chunks.
  - `adaptive`: Dynamically merges sentences based on cosine similarity thresholding (Sentence Transformers).
- **8 Retrieval Pipelines**:
  - `dense`: Standard ChromaDB cosine similarity.
  - `bm25`: Sparse keyword search via `rank_bm25`.
  - `hybrid_rrf`: Dense + BM25 fusion using Reciprocal Rank Fusion.
  - `hybrid_rerank`: Hybrid RRF + `BAAI/bge-reranker-v2-m3` cross-encoder reranking.
  - `metadata_filter`: LLM-powered query intent parsing (e.g., table vs. text) to filter ChromaDB.
  - `small_to_big`: Retrieves micro-chunks (200 tokens) but expands context to parent sections (2000 tokens).
  - `hyde`: Hypothetical Document Embeddings via LLM generation.
  - `query_decomposition`: Multi-hop query breakdown into sub-queries.
- **Automated Evaluation Engine**:
  - Computes classic retrieval metrics (Recall@k, MRR, nDCG@k) using precise span-overlap calculations.
  - RAGAS-style Generation Metrics (Faithfulness, Correctness, Precision) using an LLM Judge.
  - Statistical analysis computing Bootstrap 95% CIs and Holm-Bonferroni corrected pairwise significance tests.
- **Cost Tracking**: Enforces hard budget limits per pipeline phase for OpenAI and Anthropic API usage.

## Setup

1. **Install Dependencies**:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   Copy `.env.example` to `.env` and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   ```

3. **Provide Corpus**:
   Place the master markdown file `DnD_Second_edition__all_26_books.md` inside `data/raw/`.

## Usage

This project uses `Hydra` for configuration management. You can run individual stages of the pipeline using `main.py` or the provided `Makefile`.

### Pipeline Stages

```bash
# 1. Profile the corpus and detect books
python main.py stage=parse

# 2. Generate the Gold Standard evaluation dataset
python main.py stage=goldset

# 3. Execute all chunking strategies
python main.py stage=chunk

# 4. Build Dense and BM25 indices
python main.py stage=index

# 5. Run Phase 1 Evaluation (Chunking Isolation - Dense Only)
python main.py stage=eval_phase1

# 6. Run Phase 2 Evaluation (Full Retrieval Sweep)
python main.py stage=eval_phase2

# 7. Generate final report and statistical plots
python main.py stage=report
```

Or using `make`:
```bash
make all        # Runs the complete pipeline end-to-end
make clean      # Cleans up the data/ and results/ directories
make test       # Runs the pytest suite
```

## Testing

The project includes a robust test suite covering corpus parsing, metrics math, and chunking integrity.
```bash
pytest tests/ -v
```

## Directory Structure

- `configs/`: Hydra YAML configurations.
- `src/`: Core source code.
  - `chunkers/`: Chunking logic.
  - `corpus/`: Parsing and modeling.
  - `evaluation/`: Metrics and statistical testing.
  - `goldset/`: Dataset generation.
  - `retrieval/`: Indexing and search methods.
  - `utils/`: Caching, LLM clients, and embeddings.
- `tests/`: Pytest suite.
- `data/`: Generated caches, indices, and processed chunks.
- `results/`: CSV benchmarks, routing tables, and visual plots.
