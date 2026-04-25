# Notebooks

Exploratory and analysis notebooks for inspecting corpus structure, chunk quality, and benchmark results.

## Suggested Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_corpus_profile.ipynb` | Load `data/interim/parse_summary.json`; plot node-type distribution (headings, tables, code blocks, paragraphs), token length histogram, heading-depth breakdown |
| `02_chunk_samples.ipynb` | Inspect `data/processed/<strategy>/chunks.jsonl`; compare chunk boundaries side-by-side across all 5 strategies; spot-check table-aware and hierarchical splits |
| `03_gold_set_review.ipynb` | Browse `data/gold/gold_standard.jsonl`; verify anchor spans point to correct corpus locations; review question quality per type (lore / mechanical / tabular / …) |
| `04_results_deepdive.ipynb` | Slice `results/benchmark.csv` by query_type, k-value, and CI bands; plot Recall@k curves and nDCG heatmaps across chunking × retrieval combinations |
| `05_cost_analysis.ipynb` | Parse `results/cost_log.jsonl`; visualise spend per stage, per model, per strategy |

## Usage

```bash
pip install jupyterlab
jupyter lab
```

## Notes

- All notebooks should load data from `data/` and `results/` — **never hard-code corpus text** (see [NOTICE.md](../NOTICE.md))
- Keep outputs cleared before committing (`Cell → Clear All Outputs`)
- Use relative paths from the project root, e.g. `pd.read_json("../data/gold/gold_standard.jsonl", lines=True)`
