"""
Gold standard dataset generation (spec §7).

Generates stratified questions, auto-filters, and prepares for human validation.
Target: 200 stratified questions + 10 distractors.
"""

import hashlib
import json
import logging
import random
from pathlib import Path

from omegaconf import DictConfig

from src.corpus.models import GoldEntry
from src.utils.llm import LLMClient

logger = logging.getLogger(__name__)

# Stratification targets (spec §7.1)
QUERY_TYPE_TARGETS = {
    "lore": 50, "mechanical": 50, "tabular": 40,
    "cross_reference": 30, "monster": 20, "numeric": 10,
}

GENERATION_PROMPT = """You are creating a benchmark dataset for evaluating RAG systems
on AD&D 2nd Edition rulebooks. Generate {count} questions of type "{query_type}".

Context from the corpus:
{context}

For each question, provide:
1. A clear, specific question
2. The exact reference answer
3. Whether it requires multi-hop reasoning

Return a JSON array of objects with fields:
- "query": the question
- "reference_answer": the exact answer
- "is_multi_hop": boolean
- "notes": brief note about source

{type_guidance}

Return ONLY the JSON array."""

TYPE_GUIDANCE = {
    "lore": "Focus on narrative, history, social structures, and world-building.",
    "mechanical": "Focus on game mechanics, rules, THAC0, saving throws, class abilities.",
    "tabular": "Focus on data found in tables: damage, costs, spell lists, stat blocks.",
    "cross_reference": "Create questions requiring info from multiple sections or books.",
    "monster": "Focus on monster statistics, abilities, habitats, and combat details.",
    "numeric": "Focus on numeric lookups: spell counts, level requirements, ranges.",
}


def generate_gold_standard(cfg: DictConfig, project_root: Path) -> None:
    """Generate the gold standard dataset."""
    gold_dir = project_root / cfg.paths.gold
    gold_dir.mkdir(parents=True, exist_ok=True)
    gold_path = gold_dir / "gold_standard.jsonl"

    raw_path = project_root / cfg.paths.raw_corpus
    if not raw_path.exists():
        logger.warning(f"Raw corpus not found at {raw_path}. "
                       "Creating placeholder gold set.")
        _create_placeholder_gold(gold_path)
        return

    text = raw_path.read_text(encoding="utf-8")
    llm = LLMClient(model=cfg.llm.model, temperature=0, max_tokens=4096,
                     cache_dir=str(project_root / ".cache"))

    all_entries = []
    entry_id = 1

    for query_type, target_count in QUERY_TYPE_TARGETS.items():
        logger.info(f"Generating {target_count} '{query_type}' questions...")

        # Sample sections from corpus
        sections = _sample_sections(text, n=5)

        # Generate candidates (2x target for filtering)
        candidates = _generate_questions(
            llm, query_type, sections, count=target_count * 2)

        # Auto-filter
        filtered = _auto_filter(candidates, all_entries)

        # Take target count
        selected = filtered[:target_count]

        for q in selected:
            entry = GoldEntry(
                id=f"q_{entry_id:04d}",
                query_type=query_type,
                is_multi_hop=q.get("is_multi_hop", False),
                query=q["query"],
                reference_answer=q["reference_answer"],
                notes=q.get("notes", ""),
                validator="auto",
            )
            all_entries.append(entry)
            entry_id += 1

    # Add 10 distractor questions
    distractors = _generate_distractors(llm)
    for d in distractors:
        entry = GoldEntry(
            id=f"q_{entry_id:04d}",
            query_type="distractor",
            query=d["query"],
            reference_answer="NOT_IN_CORPUS",
            notes="Distractor: answer not in corpus",
            validator="auto",
        )
        all_entries.append(entry)
        entry_id += 1

    # Write JSONL
    with open(gold_path, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry.model_dump(), ensure_ascii=False) + "\n")

    # Content hash
    content = gold_path.read_bytes()
    content_hash = hashlib.sha256(content).hexdigest()[:16]
    meta = {"count": len(all_entries), "content_hash": content_hash,
            "types": {k: sum(1 for e in all_entries if e.query_type == k)
                      for k in set(e.query_type for e in all_entries)}}
    (gold_dir / "gold_meta.json").write_text(json.dumps(meta, indent=2))

    logger.info(f"Gold standard: {len(all_entries)} entries → {gold_path}")
    logger.info(f"Content hash: {content_hash}")


def _sample_sections(text, n=5, section_size=3000):
    """Sample random sections from corpus."""
    lines = text.split("\n")
    sections = []
    for _ in range(n):
        start = random.randint(0, max(0, len(lines) - 100))
        section = "\n".join(lines[start:start + 100])
        sections.append(section[:section_size])
    return sections


def _generate_questions(llm, query_type, sections, count):
    """Generate candidate questions using LLM."""
    context = "\n---\n".join(sections[:3])
    guidance = TYPE_GUIDANCE.get(query_type, "")
    try:
        response = llm.generate(
            system_prompt="You are a benchmark dataset creator.",
            user_prompt=GENERATION_PROMPT.format(
                count=count, query_type=query_type,
                context=context, type_guidance=guidance),
            max_tokens=4096)
        return json.loads(response.strip())
    except Exception as e:
        logger.error(f"Generation failed for {query_type}: {e}")
        return []


def _auto_filter(candidates, existing):
    """Filter: remove duplicates and low-quality questions."""
    existing_queries = {e.query.lower() for e in existing}
    filtered = []
    for c in candidates:
        if not isinstance(c, dict) or "query" not in c:
            continue
        if c["query"].lower() in existing_queries:
            continue
        if len(c.get("reference_answer", "")) < 1:
            continue
        existing_queries.add(c["query"].lower())
        filtered.append(c)
    return filtered


def _generate_distractors(llm):
    """Generate 10 distractor questions not answerable from AD&D 2e."""
    try:
        resp = llm.generate(
            "You create questions about fantasy RPGs.",
            """Generate 10 questions that CANNOT be answered from AD&D 2nd Edition books.
They should be about other RPG systems, modern D&D 5e mechanics, or fictional content
not in AD&D 2e. Return a JSON array of {{"query": "...", "notes": "..."}}.
Return ONLY the JSON array.""",
            max_tokens=2000)
        return json.loads(resp.strip())
    except Exception:
        return [{"query": f"Distractor question {i}", "notes": "placeholder"} for i in range(10)]


def _create_placeholder_gold(gold_path):
    """Create a minimal placeholder gold set for testing."""
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
