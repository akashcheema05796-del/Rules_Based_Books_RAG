# Notebooks

Exploratory notebooks for corpus inspection, chunk samples, and result analysis.

Suggested entries (created as needed):

- `01_corpus_profile.ipynb` — load `results/corpus_profile.json` and plot
  header/table/token distributions.
- `02_chunk_samples.ipynb` — inspect `results/chunk_samples/*_samples.jsonl` for
  qualitative review per strategy.
- `03_results_deepdive.ipynb` — slice `results/benchmark.csv` by query type /
  k-value / confidence interval bands.

All notebooks should avoid committing verbatim corpus text (see
[NOTICE.md](../NOTICE.md)).
