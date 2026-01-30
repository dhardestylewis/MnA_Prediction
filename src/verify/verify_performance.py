# -*- coding: utf-8 -*-
"""
verify_performance.py - Runtime Verification Harness for M&A Pipeline

Verifies performance targets from TODO.md Section 10 (Definition of Done):
1. Inner-loop time: ≤60 seconds per (quarter, horizon) with sampling
2. AutoML overhead: ≤2× single fit time
3. Cache hit rate: 100% on second run
"""

from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class VerificationResult:
    """Result from running performance verification."""
    passed: bool
    inner_loop_passed: bool = True
    automl_passed: bool = True
    cache_hit_passed: bool = True
    details: Dict = None


def verify_inner_loop_time(timing_log: List[Dict], threshold_seconds: float = 60.0) -> tuple:
    """
    Verify that inner-loop train_eval times are within threshold.
    
    Args:
        timing_log: List of timing entries from TimingLogger
        threshold_seconds: Maximum allowed seconds per task
    
    Returns:
        (passed: bool, max_time: float, violating_tasks: List[Dict])
    """
    train_eval_entries = [e for e in timing_log if e.get("phase") == "train_eval"]
    
    if not train_eval_entries:
        return True, 0.0, []
    
    max_time = 0.0
    violating_tasks = []
    
    for entry in train_eval_entries:
        total = entry.get("total_seconds", 0)
        max_time = max(max_time, total)
        if total > threshold_seconds:
            violating_tasks.append({
                "quarter": entry.get("quarter"),
                "horizon": entry.get("horizon_mo"),
                "time_seconds": total
            })
    
    passed = len(violating_tasks) == 0
    return passed, max_time, violating_tasks


def verify_automl_overhead(timing_log: List[Dict], overhead_factor: float = 2.0) -> tuple:
    """
    Verify AutoML completes within 2x single fit time.
    
    Args:
        timing_log: List of timing entries
        overhead_factor: Maximum allowed multiple of single fit time
    
    Returns:
        (passed: bool, actual_factor: float, details: Dict)
    """
    # Get baseline single fit time
    train_eval_entries = [e for e in timing_log if e.get("phase") == "train_eval"]
    automl_entries = [e for e in timing_log if e.get("phase") == "automl_trial"]
    
    if not train_eval_entries or not automl_entries:
        return True, 0.0, {"reason": "no_data"}
    
    # Use median single fit time as baseline
    fit_times = [e.get("fit_seconds", 0) for e in train_eval_entries if e.get("fit_seconds")]
    if not fit_times:
        return True, 0.0, {"reason": "no_fit_times"}
    
    baseline_fit = sorted(fit_times)[len(fit_times) // 2]  # Median
    
    # Check each automl trial
    for entry in automl_entries:
        automl_time = entry.get("total_seconds", 0)
        if baseline_fit > 0:
            actual_factor = automl_time / baseline_fit
            if actual_factor > overhead_factor:
                return False, actual_factor, {
                    "baseline_fit_seconds": baseline_fit,
                    "automl_seconds": automl_time
                }
    
    actual_factor = automl_entries[0].get("total_seconds", 0) / baseline_fit if baseline_fit > 0 else 0
    return True, actual_factor, {"baseline_fit_seconds": baseline_fit}


def verify_cache_hit(compile_artifacts, expected_hit: bool = True) -> tuple:
    """
    Verify cache hit status matches expectation.
    
    Args:
        compile_artifacts: CompileArtifacts instance
        expected_hit: Whether we expect a cache hit (True for 2nd run)
    
    Returns:
        (passed: bool, actual_hit: bool, cache_key: str)
    """
    actual_hit = getattr(compile_artifacts, 'cache_hit', False)
    cache_key = getattr(compile_artifacts, 'cache_key', '')
    passed = actual_hit == expected_hit
    return passed, actual_hit, cache_key


def run_verification_harness(
    timing_log: List[Dict],
    compile_artifacts,
    is_second_run: bool = False,
    inner_loop_threshold: float = 60.0,
    automl_overhead_threshold: float = 2.0
) -> VerificationResult:
    """
    Run full verification harness and return results.
    
    Args:
        timing_log: List of timing entries from TIMING_LOGGER.rows
        compile_artifacts: CompileArtifacts from compile phase
        is_second_run: True if this is a 2nd run (should have cache hit)
        inner_loop_threshold: Max seconds for inner loop (default 60)
        automl_overhead_threshold: Max multiple for AutoML (default 2x)
    
    Returns:
        VerificationResult with pass/fail status and details
    """
    details = {}
    
    # 1. Inner loop time
    inner_passed, max_time, violations = verify_inner_loop_time(timing_log, inner_loop_threshold)
    details["inner_loop"] = {
        "passed": inner_passed,
        "max_time_seconds": max_time,
        "threshold_seconds": inner_loop_threshold,
        "violations": violations
    }
    
    # 2. AutoML overhead
    automl_passed, factor, automl_details = verify_automl_overhead(timing_log, automl_overhead_threshold)
    details["automl"] = {
        "passed": automl_passed,
        "actual_factor": factor,
        "threshold_factor": automl_overhead_threshold,
        **automl_details
    }
    
    # 3. Cache hit (only check on 2nd run)
    if is_second_run:
        cache_passed, actual_hit, cache_key = verify_cache_hit(compile_artifacts, expected_hit=True)
        details["cache"] = {
            "passed": cache_passed,
            "expected_hit": True,
            "actual_hit": actual_hit,
            "cache_key": cache_key
        }
    else:
        cache_passed = True  # Not checking on first run
        details["cache"] = {"passed": True, "reason": "first_run_skip"}
    
    all_passed = inner_passed and automl_passed and cache_passed
    
    return VerificationResult(
        passed=all_passed,
        inner_loop_passed=inner_passed,
        automl_passed=automl_passed,
        cache_hit_passed=cache_passed,
        details=details
    )


def print_verification_summary(result: VerificationResult) -> None:
    """Print human-readable verification summary."""
    print("\n" + "=" * 60)
    print("PERFORMANCE VERIFICATION SUMMARY")
    print("=" * 60)
    
    status = "✅ PASSED" if result.passed else "❌ FAILED"
    print(f"\nOverall: {status}\n")
    
    d = result.details or {}
    
    # Inner loop
    inner = d.get("inner_loop", {})
    inner_status = "✅" if inner.get("passed") else "❌"
    print(f"{inner_status} Inner Loop Time: {inner.get('max_time_seconds', 0):.1f}s (threshold: {inner.get('threshold_seconds', 60)}s)")
    
    # AutoML
    automl = d.get("automl", {})
    automl_status = "✅" if automl.get("passed") else "❌"
    print(f"{automl_status} AutoML Overhead: {automl.get('actual_factor', 0):.2f}x (threshold: {automl.get('threshold_factor', 2)}x)")
    
    # Cache
    cache = d.get("cache", {})
    cache_status = "✅" if cache.get("passed") else "❌"
    if cache.get("reason") == "first_run_skip":
        print(f"{cache_status} Cache Hit: Skipped (first run)")
    else:
        print(f"{cache_status} Cache Hit: {cache.get('actual_hit')} (expected: {cache.get('expected_hit')})")
    
    print("=" * 60)
