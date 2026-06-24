"""Minimal task factory for the Trinity LiveCodeBench submission."""

from typing import Any

from fugu.tasks.livecodebench import LiveCodeBenchTask


def create_task(task_name: str, *, seed: int = 42, **kwargs: Any) -> LiveCodeBenchTask:
    """Create a LiveCodeBench task instance.

    Only the LiveCodeBench task is required for this submission, so the factory
    validates `task_name` and forwards the remaining keyword arguments to
    `LiveCodeBenchTask`.
    """
    normalized = task_name.lower()
    if normalized not in {"livecodebench", "livecodebench_dataset", "livecodebench_train"}:
        raise ValueError(f"Unsupported task '{task_name}'. Only LiveCodeBench is bundled with this submission.")

    kwargs_copy = dict(kwargs)
    kwargs_copy.setdefault("seed", seed)
    return LiveCodeBenchTask(**kwargs_copy)
