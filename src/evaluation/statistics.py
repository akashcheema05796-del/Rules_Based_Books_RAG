"""
Statistical analysis (spec §9).

Bootstrap 95% CIs, paired bootstrap significance tests,
Holm-Bonferroni correction, judge variance reporting.
"""

import logging
from itertools import combinations

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute bootstrap confidence interval.

    Args:
        values: Metric values (one per query).
        n_resamples: Number of bootstrap resamples.
        ci: Confidence level.
        seed: Random seed.

    Returns:
        Dict with mean, lower, upper bounds.
    """
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    n = len(arr)

    if n == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "n": 0}

    boot_means = []
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    alpha = (1 - ci) / 2

    return {
        "mean": float(np.mean(arr)),
        "lower": float(np.percentile(boot_means, alpha * 100)),
        "upper": float(np.percentile(boot_means, (1 - alpha) * 100)),
        "std": float(np.std(arr)),
        "n": n,
    }


def paired_bootstrap_test(
    values_a: list[float],
    values_b: list[float],
    n_resamples: int = 10000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap test for significance.

    Tests whether system A is significantly better than system B.

    Args:
        values_a: Metric values for system A.
        values_b: Metric values for system B.
        n_resamples: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        Dict with p-value, mean difference, CI of difference.
    """
    rng = np.random.RandomState(seed)
    a = np.array(values_a)
    b = np.array(values_b)
    assert len(a) == len(b), "Paired test requires equal-length arrays"

    n = len(a)
    observed_diff = np.mean(a) - np.mean(b)

    # Bootstrap the difference
    count_greater = 0
    boot_diffs = []
    for _ in range(n_resamples):
        indices = rng.choice(n, size=n, replace=True)
        boot_diff = np.mean(a[indices]) - np.mean(b[indices])
        boot_diffs.append(boot_diff)
        if boot_diff <= 0:
            count_greater += 1

    p_value = count_greater / n_resamples
    boot_diffs = np.array(boot_diffs)

    return {
        "mean_diff": float(observed_diff),
        "p_value": float(p_value),
        "ci_lower": float(np.percentile(boot_diffs, 2.5)),
        "ci_upper": float(np.percentile(boot_diffs, 97.5)),
        "significant_at_05": p_value < 0.05,
    }


def holm_bonferroni_correction(p_values: dict[str, float], alpha: float = 0.05) -> dict:
    """Apply Holm-Bonferroni correction to multiple comparisons.

    Args:
        p_values: Dict mapping comparison name to p-value.
        alpha: Family-wise error rate.

    Returns:
        Dict with corrected significance decisions.
    """
    m = len(p_values)
    sorted_pairs = sorted(p_values.items(), key=lambda x: x[1])

    results = {}
    for i, (name, p) in enumerate(sorted_pairs):
        adjusted_alpha = alpha / (m - i)
        results[name] = {
            "raw_p": p,
            "adjusted_alpha": adjusted_alpha,
            "significant": p < adjusted_alpha,
            "rank": i + 1,
        }

    return results


def pairwise_significance(
    results_by_system: dict[str, list[float]],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Run pairwise paired bootstrap tests with Holm-Bonferroni correction.

    Args:
        results_by_system: Dict mapping system name to metric values.
        n_resamples: Bootstrap resamples.
        alpha: Significance level.
        seed: Random seed.

    Returns:
        Dict with pairwise comparisons and corrected significance.
    """
    systems = list(results_by_system.keys())
    raw_p_values = {}
    pairwise = {}

    for sys_a, sys_b in combinations(systems, 2):
        comparison = f"{sys_a} vs {sys_b}"
        result = paired_bootstrap_test(
            results_by_system[sys_a],
            results_by_system[sys_b],
            n_resamples=n_resamples,
            seed=seed,
        )
        pairwise[comparison] = result
        raw_p_values[comparison] = result["p_value"]

    # Apply Holm-Bonferroni correction
    corrected = holm_bonferroni_correction(raw_p_values, alpha)

    # Merge corrected results
    for comparison in pairwise:
        pairwise[comparison]["corrected"] = corrected[comparison]

    return pairwise


def judge_variance_report(trial_scores: list[list[float]]) -> dict:
    """Report variance across judge trials.

    Args:
        trial_scores: List of score lists (one per trial).

    Returns:
        Dict with mean, std, per-trial means.
    """
    all_means = [np.mean(trial) for trial in trial_scores]
    return {
        "overall_mean": float(np.mean(all_means)),
        "overall_std": float(np.std(all_means)),
        "per_trial_means": [float(m) for m in all_means],
        "n_trials": len(trial_scores),
    }
