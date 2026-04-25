"""
Gold standard dataset generation (spec §7).

Generates stratified questions across the 26 detected books, locates each answer
in the corpus to populate `reference_contexts` with char_start/char_end, then
auto-filters by embedding dedup, answer-in-corpus verification, and an
answerable-without-retrieval probe. Target: 200 stratified questions + 10
distractors. Human-validation workflow is in `validator.py`.
"""

import hashlib
import json
import logging
import random
import re
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from src.corpus.models import GoldEntry
from src.utils.cost_tracker import CostTracker
from src.utils.embeddings import EmbeddingClient
from src.utils.llm import LLMClient

logger = logging.getLogger(__name__)

QUERY_TYPE_TARGETS = {
    "lore": 25, "mechanical": 25, "tabular": 20,
    "cross_reference": 15, "monster": 10, "numeric": 5,
}

GENERATION_PROMPT = """You are creating a benchmark dataset for evaluating RAG systems
on AD&D 2nd Edition rulebooks. Generate {count} questions of type "{query_type}"
based ONLY on the provided context. Every answer MUST be verbatim present in the
context (for numeric answers, the exact numeric string must appear).

Context from book "{book_title}":
---
{context}
---

For each question, provide:
1. A clear, specific question answerable from the context above.
2. The exact reference answer as it appears in the context (a short verbatim string,
   typically 1-10 tokens — not a paraphrase).
3. A distinctive anchor phrase (5-15 words) copied verbatim from the context that
   surrounds the answer. This will be used to locate the answer in the source.
4. Whether the question requires multi-hop reasoning across sections.

Return a JSON array of objects with fields:
- "query": the question
- "reference_answer": verbatim short answer
- "anchor": verbatim phrase from context surrounding the answer
- "is_multi_hop": boolean
- "notes": brief source note

{type_guidance}

Return ONLY the JSON array — no markdown fences."""

TYPE_GUIDANCE = {
    "lore": "Focus on narrative, history, social structures, and world-building.",
    "mechanical": "Focus on rules mechanics: THAC0, saving throws, class abilities, turn order.",
    "tabular": "Focus on data found in tables: damage, costs, spell lists, stat blocks. "
               "Pick context that contains an actual markdown table.",
    "cross_reference": "Create questions requiring info from multiple sections. The anchor "
                       "should point to the primary answer location.",
    "monster": "Focus on monster statistics, abilities, habitats, and combat details.",
    "numeric": "Focus on numeric lookups: spell counts, level requirements, ranges.",
}

ANSWERABILITY_PROMPT = """A RAG benchmark question is 'trivially answerable' if a strong
LLM could answer it correctly with no retrieval, purely from general knowledge.
Score 0.0 (needs retrieval from AD&D 2e books) to 1.0 (trivially answerable).

Question: {query}
Claimed answer: {answer}

Examples:
- "What color is the sky?" → 1.0 (general knowledge)
- "What is Lolth's role in Drow society?" → 0.3 (folk D&D knowledge, but edition-specific)
- "What is the THAC0 of a 5th-level Fighter in AD&D 2e?" → 0.0 (edition-specific rule)

Return ONLY a number 0.0-1.0."""


def generate_gold_standard(cfg: DictConfig, project_root: Path) -> None:
    """Generate the gold standard dataset end-to-end."""
    gold_dir = project_root / cfg.paths.gold
    gold_dir.mkdir(parents=True, exist_ok=True)
    gold_path = gold_dir / "gold_standard.jsonl"

    raw_path = project_root / cfg.paths.raw_corpus
    if not raw_path.exists():
        logger.warning(f"Raw corpus not found at {raw_path}. Creating placeholder.")
        _create_placeholder_gold(gold_path)
        return

    corpus_text = raw_path.read_text(encoding="utf-8")

    # Cost tracker enforces spec §10 budget for gold-set generation.
    tracker = CostTracker(
        budget_usd=float(cfg.cost_budget.get("goldset", 10.0)),
        stage="goldset",
    )
    tracker.set_log_path(project_root / cfg.paths.results / "cost_log.jsonl")

    llm = LLMClient(
        model=cfg.llm.model, temperature=0, max_tokens=4096,
        cache_dir=str(project_root / cfg.paths.cache),
        cost_tracker=tracker,
        provider=cfg.llm.get("provider", None),
    )
    embedder = EmbeddingClient(
        model=cfg.embedding.model, dimensions=cfg.embedding.dimensions,
        cache_dir=str(project_root / cfg.paths.cache),
        cost_tracker=tracker,
    )

    all_entries: list[GoldEntry] = []
    all_embeddings: list[np.ndarray] = []
    entry_id = 1

    for query_type, target_count in QUERY_TYPE_TARGETS.items():
        logger.info(f"Generating {target_count} '{query_type}' questions...")
        tracker.check_budget()  # abort early if over

        # Oversample 3× to survive filtering.
        candidates_needed = target_count * 3
        selected_entries: list[GoldEntry] = []
        n_batches = candidates_needed // 5 + 1

        # Sample randomly from anywhere in the corpus.
        for _ in range(n_batches):
            if len(selected_entries) >= target_count:
                break
            section_text, section_offset = _sample_corpus_section(
                corpus_text, query_type=query_type,
            )
            if not section_text:
                continue

            raw = _generate_questions(
                llm, query_type, section_text, "corpus",
                count=min(5, candidates_needed - len(selected_entries)),
            )

            for q in raw:
                if not isinstance(q, dict) or "query" not in q or "anchor" not in q:
                    continue
                span = _locate_anchor(
                    corpus_text, q["anchor"], q.get("reference_answer", ""),
                )
                if span is None:
                    continue  # cannot verify answer in corpus — drop

                entry = GoldEntry(
                    id=f"q_{entry_id:04d}",
                    query_type=query_type,
                    is_multi_hop=bool(q.get("is_multi_hop", False)),
                    query=q["query"],
                    reference_answer=q["reference_answer"],
                    reference_contexts=[{
                        "char_start": span[0],
                        "char_end": span[1],
                    }],
                    notes=q.get("notes", ""),
                    validator="auto",
                )
                entry_id += 1
                selected_entries.append(entry)
                if len(selected_entries) >= target_count * 2:
                    break

        # Embedding dedup (similarity > 0.92) and answerability filter.
        kept = _filter_candidates(selected_entries, all_entries, all_embeddings, embedder, llm)
        kept = kept[:target_count]
        logger.info(f"  kept {len(kept)}/{target_count} for {query_type}")

        for e in kept:
            all_entries.append(e)
            all_embeddings.append(np.array(embedder.embed_single(e.query), dtype=np.float32))

    # Distractors — questions that should NOT be answerable from the corpus.
    for d in _generate_distractors(llm):
        all_entries.append(GoldEntry(
            id=f"q_{entry_id:04d}",
            query_type="distractor",
            query=d["query"],
            reference_answer="NOT_IN_CORPUS",
            reference_contexts=[],
            notes=d.get("notes", "Distractor: answer not in corpus"),
            validator="auto",
        ))
        entry_id += 1

    with open(gold_path, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry.model_dump(), ensure_ascii=False) + "\n")

    content_hash = hashlib.sha256(gold_path.read_bytes()).hexdigest()[:16]
    meta = {
        "count": len(all_entries),
        "content_hash": content_hash,
        "types": {k: sum(1 for e in all_entries if e.query_type == k)
                  for k in set(e.query_type for e in all_entries)},
        "validation_status": "auto_only — run `python main.py stage=validate_gold` "
                             "and a second reviewer before freezing",
    }
    (gold_dir / "gold_meta.json").write_text(json.dumps(meta, indent=2))

    tracker.log_summary()
    logger.info(f"Gold standard: {len(all_entries)} entries → {gold_path}")
    logger.info(f"Content hash: {content_hash}")


# --- Helpers ---------------------------------------------------------------

def _sample_corpus_section(
    corpus_text: str, query_type: str, section_size: int = 3500,
) -> tuple[str, int]:
    """Sample a random section from anywhere in the corpus.

    For `tabular` queries, bias toward sections that contain a pipe table.
    """
    corpus_len = len(corpus_text)
    if corpus_len <= section_size:
        return corpus_text, 0

    for _ in range(8):
        offset = random.randint(0, corpus_len - section_size)
        section = corpus_text[offset:offset + section_size]
        if query_type == "tabular":
            if sum(1 for l in section.split("\n") if l.count("|") >= 2) < 3:
                continue
        return section, offset

    # Fallback: any random position
    offset = random.randint(0, corpus_len - section_size)
    return corpus_text[offset:offset + section_size], offset


def _generate_questions(llm, query_type, context, book_title, count):
    """Call LLM to produce candidate questions; tolerate fenced output."""
    guidance = TYPE_GUIDANCE.get(query_type, "")
    try:
        response = llm.generate(
            system_prompt="You are a benchmark dataset creator. Output valid JSON only.",
            user_prompt=GENERATION_PROMPT.format(
                count=count, query_type=query_type, book_title=book_title,
                context=context, type_guidance=guidance),
            max_tokens=3000,
        )
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
        return json.loads(text)
    except Exception as e:
        logger.warning(f"Generation failed for {query_type}: {e}")
        return []


def _locate_anchor(
    corpus_text: str, anchor: str, answer: str, window: int = 600,
) -> tuple[int, int] | None:
    """Find anchor in corpus; return (char_start, char_end) window around it.

    The window must also contain the reference_answer, otherwise the anchor is
    bogus and the question is dropped. Searches the full corpus.
    """
    if not anchor or len(anchor) < 10:
        return None

    needle = anchor.strip()

    # Exact match first, then a whitespace-tolerant fallback.
    pos = corpus_text.find(needle)
    if pos < 0:
        pattern = re.compile(re.escape(needle[:60]).replace(r"\ ", r"\s+"))
        m = pattern.search(corpus_text)
        pos = m.start() if m else -1
    if pos < 0:
        return None

    win_start = max(0, pos - window // 2)
    win_end = min(len(corpus_text), pos + len(needle) + window // 2)

    if answer and answer.strip() and answer.lower() not in corpus_text[win_start:win_end].lower():
        return None

    return win_start, win_end


def _filter_candidates(
    candidates: list[GoldEntry],
    existing: list[GoldEntry],
    existing_embeddings: list[np.ndarray],
    embedder: EmbeddingClient,
    llm: LLMClient,
    dedup_threshold: float = 0.92,
    answerability_threshold: float = 0.7,
) -> list[GoldEntry]:
    """Embedding dedup + answerable-without-retrieval filter (spec §7.3)."""
    if not candidates:
        return []

    cand_embs = embedder.embed_as_numpy([c.query for c in candidates])
    # Normalize for cosine similarity
    cand_norm = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-9)

    existing_norm = None
    if existing_embeddings:
        ex = np.stack(existing_embeddings)
        existing_norm = ex / (np.linalg.norm(ex, axis=1, keepdims=True) + 1e-9)

    kept: list[GoldEntry] = []
    kept_norm = np.zeros((0, cand_norm.shape[1]), dtype=np.float32)

    for i, entry in enumerate(candidates):
        vec = cand_norm[i]
        if existing_norm is not None:
            if float((existing_norm @ vec).max()) > dedup_threshold:
                continue
        if kept_norm.shape[0] > 0:
            if float((kept_norm @ vec).max()) > dedup_threshold:
                continue

        # Answerable-without-retrieval probe. Skip distractors later by type.
        try:
            resp = llm.generate(
                "You evaluate whether RAG benchmark questions require retrieval.",
                ANSWERABILITY_PROMPT.format(query=entry.query, answer=entry.reference_answer),
                max_tokens=10,
                cache_key=f"answerable:{entry.query}",
            )
            score = float(re.search(r"[0-9.]+", resp).group())
            if score >= answerability_threshold:
                continue
        except Exception:
            pass  # on parse error, keep the candidate

        kept.append(entry)
        kept_norm = np.vstack([kept_norm, vec[None, :]])

    return kept


def _generate_distractors(llm):
    """10 questions whose answers are NOT in the AD&D 2e corpus."""
    try:
        resp = llm.generate(
            "You create questions about fantasy RPGs.",
            """Generate 10 questions that CANNOT be answered from AD&D 2nd Edition books.
They should be about other RPG systems, modern D&D 5e mechanics, or fictional content
not in AD&D 2e. Return a JSON array of {"query": "...", "notes": "..."}.
Return ONLY the JSON array — no markdown fences.""",
            max_tokens=1500,
        )
        text = resp.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
        return json.loads(text)
    except Exception:
        return [{"query": f"Distractor question {i}", "notes": "placeholder"}
                for i in range(10)]


def _create_placeholder_gold(gold_path):
    """Minimal placeholder gold set for smoke tests when corpus is absent."""
    entries = [
        GoldEntry(id="q_0001", query_type="mechanical",
                  query="What is the THAC0 of a 5th-level Fighter?",
                  reference_answer="16", notes="Placeholder"),
        GoldEntry(id="q_0002", query_type="lore",
                  query="What are the primary deities of the Drow?",
                  reference_answer="Lolth", notes="Placeholder"),
    ]
    with open(gold_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.model_dump(), ensure_ascii=False) + "\n")
    logger.info(f"Placeholder gold set: {len(entries)} entries")
