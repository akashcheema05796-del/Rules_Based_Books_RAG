"""
JSONL structured logging.

Logs per-run data to process.log: strategy, method, chunk count,
tokens, wall time, dollar cost (spec §10).
"""

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RunLog:
    """Structured log entry for a benchmark run."""

    strategy: str = ""
    method: str = ""
    stage: str = ""
    chunk_count: int = 0
    total_tokens: int = 0
    wall_time_seconds: float = 0.0
    cost_usd: float = 0.0
    extra: dict = field(default_factory=dict)


class BenchmarkLogger:
    """JSONL structured logger for benchmark runs."""

    def __init__(self, log_path: str | Path = "process.log"):
        """Initialize the benchmark logger.

        Args:
            log_path: Path to the JSONL log file.
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_run(self, run: RunLog) -> None:
        """Append a run log entry.

        Args:
            run: The RunLog to write.
        """
        entry = {
            "timestamp": time.time(),
            "strategy": run.strategy,
            "method": run.method,
            "stage": run.stage,
            "chunk_count": run.chunk_count,
            "total_tokens": run.total_tokens,
            "wall_time_seconds": round(run.wall_time_seconds, 3),
            "cost_usd": round(run.cost_usd, 6),
            **run.extra,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        logger.info(
            f"Run logged: stage={run.stage}, strategy={run.strategy}, "
            f"method={run.method}, chunks={run.chunk_count}, "
            f"time={run.wall_time_seconds:.1f}s, cost=${run.cost_usd:.4f}"
        )

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log a generic event.

        Args:
            event_type: Type of event.
            data: Event data dictionary.
        """
        entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            **data,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    @contextmanager
    def timed_run(
        self,
        stage: str,
        strategy: str = "",
        method: str = "",
    ):
        """Context manager that times a run and logs it.

        Usage:
            with bench_logger.timed_run("chunk", strategy="recursive") as run:
                # ... do work ...
                run.chunk_count = 1234
                run.total_tokens = 500000

        Args:
            stage: Pipeline stage name.
            strategy: Chunking strategy name.
            method: Retrieval method name.

        Yields:
            RunLog instance to populate during execution.
        """
        run = RunLog(strategy=strategy, method=method, stage=stage)
        start = time.perf_counter()

        try:
            yield run
        finally:
            run.wall_time_seconds = time.perf_counter() - start
            self.log_run(run)
