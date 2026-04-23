"""
Telemetry wrapper for skill invocations.

Emits structured JSONL events for every verification call,
enforces circuit breakers, and tracks sweep-level metrics.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class InvocationEvent:
    """Per-call telemetry event."""
    event: str = "skill_invocation"
    timestamp: str = ""
    run_id: str = ""

    # Skill metadata
    skill_name: str = "patch-verifier"
    skill_version: str = ""
    version_hash: str = ""

    # Invocation tracking
    invoked: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    parse_success: bool = False
    error: Optional[str] = None

    # Context
    instance_id: str = ""
    patch_source: str = ""
    verifier_model: str = ""
    temperature: float = 0.0
    problem_statement_tokens: int = 0
    diff_tokens: int = 0

    # Output (populated after successful parse)
    scores: dict = field(default_factory=dict)
    overall_score: float = 0.0
    confidence: float = 0.0
    verdict: str = ""
    reasoning: str = ""

    # Gold label
    gold_passed: Optional[bool] = None
    gold_patch_applied: Optional[bool] = None


@dataclass
class CircuitBreaker:
    """Tracks failure patterns and halts sweep when thresholds exceeded."""
    consecutive_failures: int = 0
    total_errors: int = 0
    parse_failures: int = 0
    timeouts: int = 0
    total_calls: int = 0
    total_cost: float = 0.0
    cost_limit: float = 5.0

    def record_success(self):
        self.consecutive_failures = 0
        self.total_calls += 1

    def record_failure(self, error_type: str = "error"):
        self.consecutive_failures += 1
        self.total_errors += 1
        self.total_calls += 1
        if error_type == "parse":
            self.parse_failures += 1
        elif error_type == "timeout":
            self.timeouts += 1

    def record_cost(self, cost: float):
        self.total_cost += cost

    def should_halt(self) -> tuple[bool, str]:
        if self.consecutive_failures >= 3:
            return True, f"3 consecutive failures (total errors: {self.total_errors})"
        if self.total_calls > 5 and self.parse_failures / self.total_calls > 0.2:
            return True, f"Parse failure rate {self.parse_failures}/{self.total_calls} > 20%"
        if self.total_cost > self.cost_limit:
            return True, f"Cost ${self.total_cost:.2f} exceeds limit ${self.cost_limit:.2f}"
        return False, ""


class TelemetryLogger:
    """Append-only JSONL logger for telemetry events."""

    def __init__(self, output_path: str, run_id: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.breaker = CircuitBreaker()
        self.events: list[InvocationEvent] = []

    def version_hash(self, version_path: str) -> str:
        content = Path(version_path).read_text()
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def new_event(self, **kwargs) -> InvocationEvent:
        return InvocationEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            run_id=self.run_id,
            **kwargs,
        )

    def log_event(self, event: InvocationEvent):
        self.events.append(event)

        # Update circuit breaker
        if event.error or not event.invoked:
            error_type = "timeout" if event.error and "timeout" in event.error else (
                "parse" if not event.parse_success and event.invoked else "error"
            )
            self.breaker.record_failure(error_type)
        else:
            self.breaker.record_success()
        self.breaker.record_cost(event.cost_usd)

        # Append to JSONL
        row = asdict(event)
        with open(self.output_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def check_circuit_breaker(self) -> tuple[bool, str]:
        return self.breaker.should_halt()

    def log_sweep_summary(self, config: dict):
        """Emit a sweep-level summary event."""
        invocations = [e for e in self.events if e.invoked and e.parse_success]
        if not invocations:
            return

        gold_labels = [(e.overall_score, e.gold_passed) for e in invocations if e.gold_passed is not None]

        # Compute metrics
        tp = sum(1 for e in invocations if e.verdict == "likely_correct" and e.gold_passed)
        fp = sum(1 for e in invocations if e.verdict == "likely_correct" and e.gold_passed is False)
        fn = sum(1 for e in invocations if e.verdict != "likely_correct" and e.gold_passed)
        tn = sum(1 for e in invocations if e.verdict != "likely_correct" and e.gold_passed is False)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        # F0.5: weights precision 2x over recall (FPs are more costly than FNs)
        beta = 0.5
        f05 = (1 + beta**2) * precision * recall / max(beta**2 * precision + recall, 1e-9)

        # Confident error rate: P(gold_fail | verdict=likely_correct AND confidence > 0.8)
        confident_correct = [e for e in invocations if e.verdict == "likely_correct" and e.confidence > 0.8]
        confident_errors = sum(1 for e in confident_correct if e.gold_passed is False)
        confident_error_rate = confident_errors / max(len(confident_correct), 1)

        # Pass rate metrics
        total_with_gold = sum(1 for e in invocations if e.gold_passed is not None)
        gold_pass_count = sum(1 for e in invocations if e.gold_passed)
        random_pass_rate = gold_pass_count / max(total_with_gold, 1)

        # Top-1 per issue: group by instance_id, pick highest overall_score
        by_issue = {}
        for e in invocations:
            if e.instance_id not in by_issue or e.overall_score > by_issue[e.instance_id].overall_score:
                by_issue[e.instance_id] = e
        top1_passes = sum(1 for e in by_issue.values() if e.gold_passed)
        top1_pass_rate = top1_passes / max(len(by_issue), 1)

        summary = {
            "event": "sweep_complete",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "config": config,
            "metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "f05": round(f05, 4),
                "confident_error_rate": round(confident_error_rate, 4),
                "top1_pass_rate": round(top1_pass_rate, 4),
                "random_baseline_pass_rate": round(random_pass_rate, 4),
                "lift_over_random_pp": round((top1_pass_rate - random_pass_rate) * 100, 2),
                "total_cost_usd": round(self.breaker.total_cost, 4),
                "avg_latency_ms": round(
                    sum(e.latency_ms for e in invocations) / max(len(invocations), 1), 0
                ),
                "patches_evaluated": len(invocations),
            },
            "circuit_breakers": {
                "consecutive_failures": self.breaker.consecutive_failures,
                "parse_failures": self.breaker.parse_failures,
                "timeouts": self.breaker.timeouts,
                "total_errors": self.breaker.total_errors,
            },
        }

        with open(self.output_path, "a") as f:
            f.write(json.dumps(summary) + "\n")

        return summary
