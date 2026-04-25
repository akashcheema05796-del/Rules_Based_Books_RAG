# RAG Chunking × Retrieval Benchmark v3 — Makefile
# Usage: make <target> [CHUNKING=<strategy>] [RETRIEVAL=<method>]

PYTHON ?= python
CHUNKING ?= all
RETRIEVAL ?= all

.PHONY: help setup profile parse goldset chunk index eval-phase1 eval-phase2 eval-phase3 report clean test

help:
	@echo "RAG Benchmark v3 — Available targets:"
	@echo "  setup        Install dependencies"
	@echo "  profile      §4.1  Corpus profile"
	@echo "  parse        §4.2-4.3  AST + book spans"
	@echo "  goldset      §7  Generate gold standard"
	@echo "  chunk        §5  Run chunking strategies (CHUNKING=recursive|markdown_hierarchical|contextual|table_aware|adaptive|all)"
	@echo "  index        Build dense + BM25 indices (CHUNKING=...)"
	@echo "  eval-phase1  §8  Retrieval metrics, dense-only"
	@echo "  eval-phase2  §8  Full retrieval sweep (CHUNKING=..., RETRIEVAL=...)"
	@echo "  eval-phase3  Per-query-type analysis + routing"
	@echo "  report       §9  Stats + plots + writeup"
	@echo "  test         Run test suite"
	@echo "  clean        Remove generated data"

setup:
	pip install -r requirements.txt

profile:
	$(PYTHON) main.py stage=profile

parse:
	$(PYTHON) main.py stage=parse

goldset:
	$(PYTHON) main.py stage=goldset

validate-gold:
	$(PYTHON) main.py stage=validate_gold validate.action=static

review-sheet:
	$(PYTHON) main.py stage=validate_gold validate.action=emit

freeze-gold:
	$(PYTHON) main.py stage=validate_gold validate.action=ingest

chunk:
	$(PYTHON) main.py stage=chunk chunking=$(CHUNKING)

index:
	$(PYTHON) main.py stage=index chunking=$(CHUNKING)

eval-phase1:
	$(PYTHON) main.py stage=eval_phase1 chunking=$(CHUNKING)

eval-phase2:
	$(PYTHON) main.py stage=eval_phase2 chunking=$(CHUNKING) retrieval=$(RETRIEVAL)

eval-phase3:
	$(PYTHON) main.py stage=eval_phase3

report:
	$(PYTHON) main.py stage=report

test:
	pytest tests/ -v --tb=short

clean:
	rm -rf data/interim data/processed data/vector_store data/bm25_index
	rm -rf results/ .cache/
	rm -f process.log
