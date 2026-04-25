# Copyright and Usage Notice

This benchmark uses the **AD&D 2nd Edition** rulebook corpus as a challenging,
structure-heavy dataset. That corpus is © Wizards of the Coast / Hasbro and is
NOT redistributed with this repository.

## What is not committed

The following paths are gitignored and must not be added:

- `data/raw/` — original corpus text
- `data/interim/`, `data/processed/` — parsed AST dumps and chunk JSONL
- `data/vector_store/`, `data/bm25_index/` — indices derived from the corpus
- `data/gold/` — gold-set questions reference the corpus directly
- `results/chunk_samples/` — sample chunks contain verbatim corpus text

## Using the benchmark

1. Obtain the corpus independently (see README for expected filename).
2. Place it at `data/raw/DnD_Second_edition__all_26_books.md`.
3. Run the pipeline locally. Do not publish chunks, embeddings, or gold-set
   snippets that reproduce substantial verbatim corpus text.

## Using a different corpus

The pipeline is corpus-agnostic once [configs/books_manifest.yaml](configs/books_manifest.yaml)
is updated. For public demos, use a copyright-clear replacement (Project
Gutenberg, arXiv subsets, Wikipedia dumps).

## Code license

The benchmark code itself is released under the repository's LICENSE (add one
before public release). The corpus is excluded from that license.
