"""
Tests for CostTracker and BudgetExceededError (src/utils/cost_tracker.py).

Covers: recording costs, budget enforcement, per-stage aggregation,
cached-input pricing, unknown model fallback, log path persistence,
and the summary dict schema.
"""

import json
import pytest
from pathlib import Path

from src.utils.cost_tracker import CostTracker, BudgetExceededError, COST_PER_1M_TOKENS


class TestCostEstimation:
    def test_known_embedding_model(self):
        tracker = CostTracker(budget_usd=100.0, stage="test")
        cost = tracker._estimate_cost("text-embedding-3-small", input_tokens=1_000_000, output_tokens=0)
        # 1M tokens at $0.02/1M = $0.02
        assert abs(cost - 0.02) < 1e-9

    def test_known_llm_model(self):
        tracker = CostTracker(budget_usd=100.0, stage="test")
        cost = tracker._estimate_cost("claude-sonnet-4-5-20241022",
                                      input_tokens=1_000_000, output_tokens=1_000_000)
        # $3.00 input + $15.00 output = $18.00
        assert abs(cost - 18.0) < 1e-9

    def test_cached_input_pricing(self):
        tracker = CostTracker(budget_usd=100.0, stage="test")
        # With caching: $0.30/1M input vs $3.00/1M
        cached = tracker._estimate_cost("claude-sonnet-4-5-20241022",
                                        input_tokens=1_000_000, output_tokens=0,
                                        cached_input=True)
        uncached = tracker._estimate_cost("claude-sonnet-4-5-20241022",
                                          input_tokens=1_000_000, output_tokens=0,
                                          cached_input=False)
        assert cached < uncached
        assert abs(cached - 0.30) < 1e-9

    def test_unknown_model_returns_zero(self):
        tracker = CostTracker(budget_usd=100.0, stage="test")
        cost = tracker._estimate_cost("gpt-unknown-99", input_tokens=10_000, output_tokens=100)
        assert cost == 0.0

    def test_zero_tokens_zero_cost(self):
        tracker = CostTracker(budget_usd=100.0, stage="test")
        cost = tracker._estimate_cost("text-embedding-3-small", input_tokens=0, output_tokens=0)
        assert cost == 0.0


class TestRecord:
    def test_record_adds_entry(self):
        tracker = CostTracker(budget_usd=100.0, stage="test")
        tracker.record("text-embedding-3-small", input_tokens=100_000)
        assert len(tracker.entries) == 1

    def test_record_returns_cost(self):
        tracker = CostTracker(budget_usd=100.0, stage="test")
        cost = tracker.record("text-embedding-3-small", input_tokens=1_000_000)
        assert abs(cost - 0.02) < 1e-9

    def test_multiple_records_accumulate(self):
        tracker = CostTracker(budget_usd=100.0, stage="test")
        tracker.record("text-embedding-3-small", input_tokens=1_000_000)
        tracker.record("text-embedding-3-small", input_tokens=1_000_000)
        assert abs(tracker.total_cost - 0.04) < 1e-9

    def test_stage_cost_only_counts_matching(self):
        tracker = CostTracker(budget_usd=100.0, stage="phase1")
        tracker.record("text-embedding-3-small", input_tokens=1_000_000)
        # Manually insert an entry with different stage
        tracker.stage = "phase2"
        tracker.record("text-embedding-3-small", input_tokens=1_000_000)
        tracker.stage = "phase1"  # restore

        # total should be $0.04, stage_cost for phase1 should be $0.02
        assert abs(tracker.total_cost - 0.04) < 1e-9
        assert abs(tracker.stage_cost - 0.02) < 1e-9

    def test_record_writes_to_log_path(self, tmp_path):
        log_file = tmp_path / "cost_log.jsonl"
        tracker = CostTracker(budget_usd=100.0, stage="test")
        tracker.set_log_path(log_file)
        tracker.record("text-embedding-3-small", input_tokens=100, description="unit test")

        assert log_file.exists()
        line = json.loads(log_file.read_text().strip())
        assert line["model"] == "text-embedding-3-small"
        assert line["input_tokens"] == 100
        assert line["description"] == "unit test"
        assert "cost_usd" in line


class TestBudgetEnforcement:
    def test_within_budget_does_not_raise(self):
        tracker = CostTracker(budget_usd=10.0, stage="test")
        tracker.record("text-embedding-3-small", input_tokens=1_000_000)  # $0.02
        tracker.check_budget()  # Should not raise

    def test_exceeds_budget_raises(self):
        tracker = CostTracker(budget_usd=0.01, stage="test")
        tracker.record("text-embedding-3-small", input_tokens=1_000_000)  # $0.02 > $0.01
        with pytest.raises(BudgetExceededError):
            tracker.check_budget()

    def test_projected_additional_triggers_raise(self):
        tracker = CostTracker(budget_usd=0.05, stage="test")
        tracker.record("text-embedding-3-small", input_tokens=1_000_000)  # $0.02 spent
        with pytest.raises(BudgetExceededError):
            tracker.check_budget(projected_additional=0.05)  # $0.02 + $0.05 > $0.05

    def test_budget_exceeded_error_message(self):
        tracker = CostTracker(budget_usd=0.01, stage="myStage")
        tracker.record("text-embedding-3-small", input_tokens=1_000_000)
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.check_budget()
        assert "myStage" in str(exc_info.value)
        assert "budget" in str(exc_info.value).lower()

    def test_zero_cost_under_budget(self):
        tracker = CostTracker(budget_usd=0.0, stage="test")
        # No records → total=0 ≤ 0 budget → should not raise
        tracker.check_budget()


class TestSummary:
    def test_summary_keys(self):
        tracker = CostTracker(budget_usd=5.0, stage="eval")
        tracker.record("text-embedding-3-small", input_tokens=1_000)
        s = tracker.summary()
        assert "stage" in s
        assert "total_cost_usd" in s
        assert "budget_usd" in s
        assert "remaining_usd" in s
        assert "num_calls" in s
        assert "by_model" in s

    def test_summary_remaining_decreases(self):
        tracker = CostTracker(budget_usd=1.0, stage="eval")
        before = tracker.summary()["remaining_usd"]
        tracker.record("text-embedding-3-small", input_tokens=1_000_000)
        after = tracker.summary()["remaining_usd"]
        assert after < before

    def test_summary_by_model(self):
        tracker = CostTracker(budget_usd=100.0, stage="test")
        tracker.record("text-embedding-3-small", input_tokens=1_000_000)
        tracker.record("text-embedding-3-small", input_tokens=500_000)
        s = tracker.summary()
        assert "text-embedding-3-small" in s["by_model"]
        assert s["by_model"]["text-embedding-3-small"]["count"] == 2

    def test_empty_tracker_summary(self):
        tracker = CostTracker(budget_usd=10.0, stage="test")
        s = tracker.summary()
        assert s["total_cost_usd"] == 0.0
        assert s["num_calls"] == 0
        assert s["remaining_usd"] == 10.0


class TestSetLogPath:
    def test_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "log.jsonl"
        tracker = CostTracker(budget_usd=1.0, stage="test")
        tracker.set_log_path(nested)
        assert nested.parent.exists()
