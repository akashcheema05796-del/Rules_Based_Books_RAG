"""
Cost tracking for API calls.

Tracks USD cost per stage. Reads cost caps from config.
Aborts with projection when budget exceeded (spec §10).
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Approximate costs per 1M tokens (as of 2025-2026)
COST_PER_1M_TOKENS = {
    # OpenAI embeddings
    "text-embedding-3-small": {"input": 0.02},
    "text-embedding-3-large": {"input": 0.13},
    # Anthropic Claude
    "claude-sonnet-4-5-20241022": {"input": 3.00, "output": 15.00},
    # With prompt caching (Anthropic)
    "claude-sonnet-4-5-20241022-cached": {"input": 0.30, "output": 15.00, "cache_write": 3.75},
}


@dataclass
class CostEntry:
    """A single cost event."""
    timestamp: float
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    description: str = ""


@dataclass
class CostTracker:
    """Tracks and enforces API cost budgets.

    Usage:
        tracker = CostTracker(budget_usd=10.0, stage="phase1")
        tracker.record("text-embedding-3-small", input_tokens=1000, output_tokens=0)
        tracker.check_budget()  # raises if over budget
    """

    budget_usd: float = 100.0
    stage: str = "default"
    entries: list[CostEntry] = field(default_factory=list)
    _log_path: Optional[Path] = None

    def set_log_path(self, path: Path) -> None:
        """Set the path for cost log output."""
        self._log_path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def _estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input: bool = False,
    ) -> float:
        """Estimate cost for a model call.

        Args:
            model: Model name.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            cached_input: Whether input uses prompt caching.

        Returns:
            Estimated cost in USD.
        """
        model_key = f"{model}-cached" if cached_input else model

        if model_key not in COST_PER_1M_TOKENS:
            if model not in COST_PER_1M_TOKENS:
                logger.warning(f"Unknown model for cost estimation: {model}")
                return 0.0
            model_key = model

        rates = COST_PER_1M_TOKENS[model_key]
        input_cost = (input_tokens / 1_000_000) * rates.get("input", 0)
        output_cost = (output_tokens / 1_000_000) * rates.get("output", 0)
        return input_cost + output_cost

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int = 0,
        description: str = "",
        cached_input: bool = False,
    ) -> float:
        """Record a cost event.

        Args:
            model: Model name.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            description: Human-readable description.
            cached_input: Whether prompt caching was used.

        Returns:
            Cost of this event in USD.
        """
        cost = self._estimate_cost(model, input_tokens, output_tokens, cached_input)
        entry = CostEntry(
            timestamp=time.time(),
            stage=self.stage,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            description=description,
        )
        self.entries.append(entry)

        if self._log_path:
            with open(self._log_path, "a") as f:
                f.write(json.dumps({
                    "timestamp": entry.timestamp,
                    "stage": entry.stage,
                    "model": entry.model,
                    "input_tokens": entry.input_tokens,
                    "output_tokens": entry.output_tokens,
                    "cost_usd": entry.cost_usd,
                    "description": entry.description,
                }) + "\n")

        return cost

    @property
    def total_cost(self) -> float:
        """Total cost across all entries."""
        return sum(e.cost_usd for e in self.entries)

    @property
    def stage_cost(self) -> float:
        """Cost for the current stage only."""
        return sum(e.cost_usd for e in self.entries if e.stage == self.stage)

    def check_budget(self, projected_additional: float = 0.0) -> None:
        """Check if budget is exceeded. Raises if over budget.

        Args:
            projected_additional: Projected additional cost.

        Raises:
            BudgetExceededError: If total + projected exceeds budget.
        """
        projected_total = self.total_cost + projected_additional
        if projected_total > self.budget_usd:
            msg = (
                f"Budget exceeded! Stage='{self.stage}', "
                f"spent=${self.total_cost:.4f}, projected=${projected_total:.4f}, "
                f"budget=${self.budget_usd:.2f}"
            )
            logger.error(msg)
            raise BudgetExceededError(msg)

    def summary(self) -> dict:
        """Get cost summary."""
        by_model = {}
        for e in self.entries:
            if e.model not in by_model:
                by_model[e.model] = {"count": 0, "cost": 0.0, "input_tokens": 0, "output_tokens": 0}
            by_model[e.model]["count"] += 1
            by_model[e.model]["cost"] += e.cost_usd
            by_model[e.model]["input_tokens"] += e.input_tokens
            by_model[e.model]["output_tokens"] += e.output_tokens

        return {
            "stage": self.stage,
            "total_cost_usd": self.total_cost,
            "budget_usd": self.budget_usd,
            "remaining_usd": self.budget_usd - self.total_cost,
            "num_calls": len(self.entries),
            "by_model": by_model,
        }

    def log_summary(self) -> None:
        """Log cost summary."""
        s = self.summary()
        logger.info(
            f"Cost[{s['stage']}]: ${s['total_cost_usd']:.4f} / ${s['budget_usd']:.2f} "
            f"({s['num_calls']} calls, ${s['remaining_usd']:.4f} remaining)"
        )


class BudgetExceededError(Exception):
    """Raised when cost budget is exceeded."""
    pass
