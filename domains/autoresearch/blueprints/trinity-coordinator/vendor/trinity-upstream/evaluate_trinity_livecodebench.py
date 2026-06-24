#!/usr/bin/env python3
"""Minimal Trinity LiveCodeBench evaluation script.

This script loads a trained Trinity router checkpoint, reconstructs the
infrastructure required to run LiveCodeBench, and evaluates the model on the
LiveCodeBench v6 test split. It is designed to be self-contained for the
`trinity_code_submission` package and draws heavily on the original
`experiments/with_training/testing_standalone.py` utilty.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Ensure the repository root (containing `fugu`) is on the import path when this
# script is executed standalone from the submission directory.
SUBMISSION_ROOT = Path(__file__).resolve().parent
if str(SUBMISSION_ROOT) not in sys.path:
    sys.path.append(str(SUBMISSION_ROOT))

from fugu.trainer import RouterInfrastructure
from fugu.algorithms.es import CMAEvolutionTrainer, _calculate_diversity_metrics
from fugu.job_manager import get_job_manager
from fugu.run_tasks import create_task
from fugu.utils import (
    InfrastructureFailure,
    aggregate_token_statistics,
    calculate_agent_stats,
)

# --------------------------------------------------------------------------------------
# Agent registry (subset copied from testing_standalone.py)
# --------------------------------------------------------------------------------------

AVAILABLE_OPEN_AGENTS: Dict[str, Dict[str, object]] = {
    "Qwen/Qwen3-32B (direct)": {
        "model_name": "Qwen/Qwen3-32B",
        "port": 8321,
        "payload": {
            "top_p": 0.8,
            "top_k": 20,
            "presence_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "port": 8322,
        "payload": {},
    },
    "google/gemma-3-27b-it": {
        "model_name": "google/gemma-3-27b-it",
        "port": 8323,
        "payload": {},
    },
    "Qwen/Qwen3-32B (reasoning)": {
        "model_name": "Qwen/Qwen3-32B",
        "port": 8324,
        "payload": {},
    },
    "Qwen/Qwen3-32B (direct-2)": {
        "model_name": "Qwen/Qwen3-32B",
        "port": 8325,
        "payload": {
            "top_p": 0.8,
            "top_k": 20,
            "presence_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B-2": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "port": 8326,
        "payload": {},
    },
    "google/gemma-3-27b-it-2": {
        "model_name": "google/gemma-3-27b-it",
        "port": 8327,
        "payload": {},
    },
    "Qwen/Qwen3-32B (reasoning-2)": {
        "model_name": "Qwen/Qwen3-32B",
        "port": 8328,
        "payload": {},
    },
}

CLOSED_LLM_NAMES = [
    "gpt-4o-mini",
    "claude-3-7-sonnet-20250219",
    "gemini-1.5-pro",
    "deepseek-ai/DeepSeek-V3",
    "gpt-4.1",
    "claude-sonnet-4-20250514",
    "gemini-2.5-pro",
]

TOGETHER_FLAGS = {
    "gpt-4o-mini": False,
    "claude-3-7-sonnet-20250219": False,
    "gemini-1.5-pro": False,
    "deepseek-ai/DeepSeek-V3": True,
    "gpt-4.1": False,
    "claude-sonnet-4-20250514": False,
    "gemini-2.5-pro": False,
}

# --------------------------------------------------------------------------------------
# Default CLI values (aligned with common evaluation usage)
# --------------------------------------------------------------------------------------

DEFAULT_LOG_DIR = Path(
    "logs/ckpt"
)
DEFAULT_ITERATION = 60  # maps to MODEL_CHOICE in example usage
DEFAULT_OPEN_SERVERS = "slurm0us-fugunodeset-3"
DEFAULT_NUM_WORKERS_PER_GPU = 10
DEFAULT_TEST_SIZE = 175

# --------------------------------------------------------------------------------------
# Data classes for structured return values
# --------------------------------------------------------------------------------------

@dataclass
class EvaluationConfig:
    """Container for resolved evaluation settings."""

    task: str
    model_name: str
    llm_names: List[str]
    agent_configs: Dict[str, Dict[str, object]]
    server_map: Dict[str, str]
    port_map: Dict[str, int]
    closed_model_config: Optional[Dict[str, object]]
    seed: int
    temperature: float
    max_tokens: int
    max_turns: int
    diversity_bonus_weight: float
    cost_bonus_weight: float
    turn_bonus_weight: float
    role_bonus_weight: float
    use_structured_router: bool
    use_consultant: bool
    use_verifier: bool
    trinity: bool
    last_token_predict: bool
    valid_ratio: float
    test_ratio: float
    num_repeats: int
    sigma0: float
    test_size: int
    opt_layer_indices: Optional[List[int]]
    model_types: str

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Trinity checkpoint on LiveCodeBench v6 test set.")
    parser.add_argument(
        "log_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_LOG_DIR,
        help="Training log directory that contains es_log.json and models/ (default: %(default)s)",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        help="Optional explicit path to the router checkpoint (.npy or .pt). Overrides --iteration.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=DEFAULT_ITERATION,
        help="Specific iteration to evaluate. Defaults to best validation iteration if not provided (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file. Defaults to <log_dir>/evaluation_livecodebench/performance_trinity_livecodebench.json",
    )
    parser.add_argument(
        "--open-servers",
        type=str,
        default=DEFAULT_OPEN_SERVERS,
        help="Comma separated list of server addresses for open-source agents (default: %(default)s). Example: 10.0.0.1,10.0.0.2",
    )
    parser.add_argument(
        "--num-workers-per-gpu",
        type=int,
        default=DEFAULT_NUM_WORKERS_PER_GPU,
        help="Number of evaluation workers to assign per GPU (default: %(default)s).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma separated list of GPU device IDs to use (default: all visible GPUs).",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=DEFAULT_TEST_SIZE,
        help="Number of test episodes to evaluate (default: %(default)s). If larger than dataset size, will resample sequentially.",
    )
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable progress bars during evaluation for cleaner logs.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging (mirrors training debug mode).",
    )
    return parser.parse_args()


def load_es_log(log_dir: Path) -> Tuple[List[dict], dict]:
    """Load es_log.json and return (log_entries, base_config)."""
    log_file = log_dir / "es_log.json"
    if not log_file.exists():
        raise FileNotFoundError(f"Could not find es_log.json under {log_dir}")

    with log_file.open("r") as f:
        log_data = json.load(f)

    if not log_data or "configs" not in log_data[0]:
        raise ValueError("es_log.json is missing the expected 'configs' entry")

    base_config = log_data[0]["configs"]
    return log_data, base_config


def select_best_iteration(log_entries: Sequence[dict]) -> Tuple[int, float]:
    """Select the iteration with the best validation score."""
    best_iter = None
    best_score = -1.0

    for entry in log_entries:
        if entry.get("type") != "valid":
            continue
        # Prefer explicit best iteration markers if available
        if entry.get("is_new_best"):
            best_iter = entry.get("best_iter", entry.get("iter"))
            best_score = entry.get("best_score", entry.get("test_score", -1.0))

    if best_iter is not None:
        return int(best_iter), float(best_score)

    # Fallback: pick the highest best_score recorded
    for entry in log_entries:
        if entry.get("type") != "valid":
            continue
        score = entry.get("best_score") or entry.get("test_score")
        if score is not None and score > best_score:
            best_score = score
            best_iter = entry.get("best_iter", entry.get("iter"))

    if best_iter is None:
        raise RuntimeError("Could not determine best validation iteration from es_log.json")

    return int(best_iter), float(best_score)


def find_model_file(log_dir: Path, iteration: Optional[int], explicit_path: Optional[Path]) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Specified model file does not exist: {explicit_path}")
        return explicit_path

    models_dir = log_dir / "models"
    if iteration is None:
        best_path = models_dir / "best_model.npy"
        if best_path.exists():
            return best_path
        best_pt = models_dir / "best_model.pt"
        if best_pt.exists():
            return best_pt
        raise FileNotFoundError("No best_model.{npy,pt} found and no iteration specified")

    npy_candidate = models_dir / f"model_iter_{iteration}.npy"
    if npy_candidate.exists():
        return npy_candidate

    pt_candidate = models_dir / f"best_model_iter_{iteration}.pt"
    if pt_candidate.exists():
        return pt_candidate

    raise FileNotFoundError(f"Could not locate model for iteration {iteration} under {models_dir}")


def load_solution(model_path: Path):
    if model_path.suffix == ".npy":
        return np.load(model_path)
    if model_path.suffix == ".pt":
        return torch.load(model_path, map_location="cpu")
    raise ValueError(f"Unsupported model file extension for {model_path}")


def parse_selected_agents(config: dict) -> Tuple[List[str], str]:
    agent_selection = config.get("agent_selection", {})
    model_types = agent_selection.get("model_types", config.get("model_types", "mix"))
    selected_agents = agent_selection.get("selected_agents", config.get("llm_names", []))
    if not selected_agents:
        raise ValueError("No agent list found in training configuration")
    return selected_agents, model_types


def resolve_open_server_map(open_agents: List[str], cli_value: Optional[str]) -> Dict[str, str]:
    if not open_agents:
        return {}

    if not cli_value:
        return {agent: "127.0.0.1" for agent in open_agents}

    servers = [item.strip() for item in cli_value.split(",") if item.strip()]
    if len(servers) == 1:
        return {agent: servers[0] for agent in open_agents}
    if len(servers) == len(open_agents):
        return {agent: servers[idx] for idx, agent in enumerate(open_agents)}

    raise ValueError(
        "Number of provided open-source servers does not match number of open agents; "
        "pass either one server (shared) or one per open agent."
    )


def normalize_opt_layer_indices(indices) -> Optional[List[int]]:
    if indices is None:
        return None
    if isinstance(indices, list):
        return [int(x) for x in indices]
    if isinstance(indices, str):
        return [int(x.strip()) for x in indices.split(",") if x.strip()]
    if isinstance(indices, int):
        return [int(indices)]
    raise ValueError(f"Unsupported opt_layer_indices type: {type(indices)}")


def build_evaluation_config(
    base_config: dict,
    selected_agents: List[str],
    model_types: str,
    open_server_override: Optional[str],
    default_test_size: int,
) -> EvaluationConfig:
    open_agents = [agent for agent in selected_agents if agent in AVAILABLE_OPEN_AGENTS]
    closed_agents = [agent for agent in selected_agents if agent in CLOSED_LLM_NAMES]

    agent_configs: Dict[str, Dict[str, object]] = {}
    port_map: Dict[str, int] = {}

    for agent in open_agents:
        cfg = AVAILABLE_OPEN_AGENTS[agent]
        agent_configs[agent] = cfg
        port_map[agent] = cfg["port"]

    server_map = resolve_open_server_map(open_agents, open_server_override)

    closed_model_config: Optional[Dict[str, object]] = None
    if closed_agents:
        closed_model_config = {
            "model_types": "closed",
            "together_flags": TOGETHER_FLAGS,
        }

    return EvaluationConfig(
        task="livecodebench",
        model_name=base_config.get("model_name"),
        llm_names=selected_agents,
        agent_configs=agent_configs,
        server_map=server_map,
        port_map=port_map,
        closed_model_config=closed_model_config,
        seed=int(base_config.get("seed", 42)),
        temperature=float(base_config.get("temperature", 0.1)),
        max_tokens=int(base_config.get("max_tokens", 4096)),
        max_turns=int(base_config.get("max_turns", 5)),
        diversity_bonus_weight=float(base_config.get("diversity_bonus_weight", 0.0)),
        cost_bonus_weight=float(base_config.get("cost_bonus_weight", 0.0)),
        turn_bonus_weight=float(base_config.get("turn_bonus_weight", 0.0)),
        role_bonus_weight=float(base_config.get("role_bonus_weight", 0.0)),
        use_structured_router=bool(base_config.get("use_structured_router", False)),
        use_consultant=bool(base_config.get("use_consultant", False)),
        use_verifier=bool(base_config.get("use_verifier", False)),
        trinity=bool(base_config.get("trinity", True)),
        last_token_predict=bool(base_config.get("last_token_predict", False)),
        valid_ratio=float(base_config.get("valid_ratio", 0.2)),
        test_ratio=float(base_config.get("test_ratio", 0.2)),
        num_repeats=int(base_config.get("num_repeats", 1)),
        sigma0=float(base_config.get("sigma0", 0.03)),
        test_size=int(base_config.get("test_size", default_test_size)),
        opt_layer_indices=normalize_opt_layer_indices(base_config.get("opt_layer_indices")),
        model_types=model_types,
    )


def allocate_workers(args: argparse.Namespace) -> Tuple[List[int], int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation but no GPUs are available.")

    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    if not gpu_ids:
        raise ValueError("No GPU IDs resolved from --gpus option")

    assignments: List[int] = []
    for gpu_id in gpu_ids:
        assignments.extend([gpu_id] * args.num_workers_per_gpu)

    total_workers = len(assignments)
    if total_workers == 0:
        raise ValueError("Computed zero workers. Increase --num-workers-per-gpu or specify GPUs.")
    return assignments, total_workers


def initialize_infrastructure(
    eval_cfg: EvaluationConfig,
    eval_dir: Path,
    worker_gpu_assignments: List[int],
    total_workers: int,
    debug: bool,
) -> RouterInfrastructure:
    infrastructure = RouterInfrastructure(
        task=eval_cfg.task,
        model_name=eval_cfg.model_name,
        llm_names=eval_cfg.llm_names,
        log_dir=str(eval_dir),
        seed=eval_cfg.seed,
        temperature=eval_cfg.temperature,
        max_tokens=eval_cfg.max_tokens,
        max_turns=eval_cfg.max_turns,
        servers=eval_cfg.server_map,
        ports=eval_cfg.port_map,
        num_workers=total_workers,
        debug=debug,
        test_ratio=eval_cfg.test_ratio,
        valid_ratio=eval_cfg.valid_ratio,
        configure_splits=False,
        worker_gpu_assignments=worker_gpu_assignments,
        trinity=eval_cfg.trinity,
    )
    return infrastructure


def build_cma_trainer(
    infrastructure: RouterInfrastructure,
    eval_cfg: EvaluationConfig,
) -> CMAEvolutionTrainer:
    trainer = CMAEvolutionTrainer(
        infrastructure=infrastructure,
        num_iters=1,
        test_interval=1,
        num_repeats=eval_cfg.num_repeats,
        sigma0=eval_cfg.sigma0,
        seed=eval_cfg.seed,
        num_tests=10,
        test_size=eval_cfg.test_size,
        servers=eval_cfg.server_map,
        opt_layer_indices=eval_cfg.opt_layer_indices,
        diversity_bonus_weight=eval_cfg.diversity_bonus_weight,
        cost_bonus_weight=eval_cfg.cost_bonus_weight,
        turn_bonus_weight=eval_cfg.turn_bonus_weight,
        role_bonus_weight=eval_cfg.role_bonus_weight,
        use_structured_router=eval_cfg.use_structured_router,
        closed_model_config=eval_cfg.closed_model_config,
        agent_configs=eval_cfg.agent_configs,
        use_consultant=eval_cfg.use_consultant,
        use_verifier=eval_cfg.use_verifier,
        trinity=eval_cfg.trinity,
        last_token_predict=eval_cfg.last_token_predict,
    )
    return trainer


def compute_actual_test_size(trainer: CMAEvolutionTrainer) -> int:
    temp_task = create_task(
        trainer.infra.task,
        llm_names=trainer.infra.llm_names,
        seed=trainer.infra.seed,
        max_tokens=trainer.infra.max_tokens,
        temperature=trainer.infra.temperature,
        max_turns=trainer.infra.max_turns,
        servers=trainer.servers,
        ports=trainer.infra.ports,
        valid_ratio=trainer.valid_ratio,
        test_ratio=trainer.test_ratio,
        max_samples=-1,
        trinity=getattr(trainer, "trinity", False),
    )

    temp_task.data_splits = temp_task._load_data(
        seed=trainer.infra.seed,
        split="train",
        validation=True,
        valid_ratio=trainer.valid_ratio,
        test_split=True,
        test_ratio=trainer.test_ratio,
    )

    return len(temp_task.data_splits["test"])


def run_cma_evaluation(
    trainer: CMAEvolutionTrainer,
    solution: np.ndarray,
    requested_test_size: int,
    show_progress: bool,
) -> Dict[str, object]:
    if not hasattr(trainer, "agent_configs") or trainer.agent_configs is None:
        trainer.agent_configs = {}
        for agent_name in trainer.infra.llm_names:
            if agent_name in AVAILABLE_OPEN_AGENTS:
                trainer.agent_configs[agent_name] = AVAILABLE_OPEN_AGENTS[agent_name]

    if not hasattr(trainer, "servers") or not trainer.servers:
        trainer.servers = trainer.infra.servers if hasattr(trainer.infra, "servers") else {}

    actual_test_size = compute_actual_test_size(trainer)
    num_jobs = min(requested_test_size or actual_test_size, actual_test_size)

    worker_config = {
        "router_model_name": trainer.infra.model_name,
        "llm_names": trainer.infra.llm_names,
        "debug": trainer.infra.debug,
        "debug_log_dir": trainer.infra.debug_log_dir,
        "task_name": trainer.infra.task,
        "max_tokens": trainer.infra.max_tokens,
        "temperature": trainer.infra.temperature,
        "max_turns": trainer.infra.max_turns,
        "ports": trainer.infra.ports,
        "servers": trainer.servers,
        "valid_ratio": trainer.valid_ratio,
        "test_ratio": trainer.test_ratio,
        "test_split_enabled": True,
        "seed": trainer.infra.seed,
        "agent_configs": trainer.agent_configs,
        "use_consultant": trainer.use_consultant,
        "use_verifier": getattr(trainer, "use_verifier", False),
        "trinity": getattr(trainer, "trinity", False),
        "worker_gpu_assignments": getattr(trainer.infra, "worker_gpu_assignments", [0]),
        "closed_model_config": getattr(trainer, "closed_model_config", None),
        "using_closed_models": getattr(trainer, "closed_model_config", None) is not None,
        "max_samples": -1,
        "last_token_predict": getattr(trainer, "last_token_predict", False),
    }

    job_manager = get_job_manager()
    job_manager.cleanup()
    job_manager.initialize(trainer.infra.num_workers, worker_config)

    futures = []
    for i in range(num_jobs):
        task_id = i % actual_test_size
        future = job_manager.submit_training_job(
            task_id=int(task_id),
            split="test",
            flat_params=solution.astype(np.float32),
            svd_weights_cpu=trainer.svd_weights_cpu,
            iteration_idx=-1,
            eps_explore=0.0,
            servers_dict=trainer.servers,
            use_structured_router=trainer.use_structured_router,
            closed_model_config=getattr(trainer, "closed_model_config", None),
            agent_configs=trainer.agent_configs,
        )
        futures.append(future)

    results = []
    clean_scores: List[float] = []
    infrastructure_fail_marked = 0
    infrastructure_fail_exceptions = 0
    other_failures = 0

    iterator = futures
    if show_progress:
        iterator = tqdm(futures, desc="Evaluating LiveCodeBench", total=len(futures))

    for future in iterator:
        try:
            result = future.get(timeout=600)
            results.append(result)

            score = result[0] if result else -1.0
            if score == -999.0:
                infrastructure_fail_marked += 1
            elif score == -1.0:
                other_failures += 1
            else:
                clean_scores.append(float(score))
        except InfrastructureFailure:
            infrastructure_fail_exceptions += 1
        except Exception:
            other_failures += 1

    job_manager.cleanup()

    clean_results = [res for res in results if res and res[0] not in (-999.0, -1.0)]

    token_stats = aggregate_token_statistics(clean_results)
    if getattr(trainer, "_calculate_closed_source_costs", None):
        token_stats.update(trainer._calculate_closed_source_costs(clean_results))

    agent_ids = [aid for res in clean_results if len(res) >= 4 for aid in res[3]]
    agent_stats, _ = calculate_agent_stats(agent_ids, trainer.infra.llm_names)

    diversity = _calculate_diversity_metrics(agent_ids, len(trainer.infra.llm_names))
    episodes = [res[3] for res in clean_results if len(res) >= 4]
    episode_diversity = trainer._calculate_episode_diversity_metrics(episodes)

    score = float(np.mean(clean_scores)) if clean_scores else 0.0

    return {
        "test_score": score,
        "num_clean_episodes": len(clean_scores),
        "requested_test_size": int(requested_test_size or actual_test_size),
        "actual_test_dataset_size": actual_test_size,
        "marked_infrastructure_failures": infrastructure_fail_marked,
        "exception_infrastructure_failures": infrastructure_fail_exceptions,
        "other_failures": other_failures,
        "token_stats": token_stats,
        "agent_stats": agent_stats,
        "diversity": diversity,
        "episode_diversity": episode_diversity,
    }


def ensure_output_dir(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------


def main():
    args = parse_args()

    log_entries, base_config = load_es_log(args.log_dir)

    specified_iteration = args.iteration
    validation_score = None
    if specified_iteration is None:
        best_iter, best_score = select_best_iteration(log_entries)
        specified_iteration = best_iter
        validation_score = best_score
        print(f"Auto-selected best validation iteration: {best_iter} (score={best_score:.4f})")
    else:
        print(f"Using user-specified iteration: {specified_iteration}")

    model_path = find_model_file(args.log_dir, specified_iteration, args.model_file)
    print(f"Loading router solution from {model_path}")
    solution = load_solution(model_path)
    if isinstance(solution, torch.Tensor):
        solution = solution.detach().cpu().numpy()

    selected_agents, model_types = parse_selected_agents(base_config)
    print(f"Selected agents ({model_types}): {selected_agents}")

    default_test_size = base_config.get("test_size", 0) or solution.shape[0] if hasattr(solution, "shape") else 0
    eval_cfg = build_evaluation_config(
        base_config,
        selected_agents,
        model_types,
        args.open_servers,
        default_test_size,
    )

    worker_assignments, total_workers = allocate_workers(args)
    print(f"Using {total_workers} workers across GPUs: {worker_assignments}")

    eval_dir = args.output.parent if args.output else args.log_dir / "evaluation_livecodebench"
    eval_dir.mkdir(parents=True, exist_ok=True)

    infrastructure = initialize_infrastructure(
        eval_cfg=eval_cfg,
        eval_dir=eval_dir,
        worker_gpu_assignments=worker_assignments,
        total_workers=total_workers,
        debug=args.debug,
    )

    trainer = build_cma_trainer(infrastructure, eval_cfg)
    trainer.best_solution = solution
    if validation_score is not None:
        trainer.best_score = validation_score

    requested_test_size = args.test_size or eval_cfg.test_size

    start = time.time()
    results = run_cma_evaluation(
        trainer=trainer,
        solution=solution,
        requested_test_size=requested_test_size,
        show_progress=not args.disable_progress,
    )
    duration = time.time() - start
    print(f"Evaluation complete in {duration / 60:.2f} minutes. Score={results['test_score']:.4f}")

    output_path = (
        args.output
        if args.output
        else eval_dir / "performance_trinity_livecodebench.json"
    )
    ensure_output_dir(output_path)

    summary = {
        "iteration": int(specified_iteration),
        "model_path": str(model_path),
        "validation_score": validation_score,
        "evaluation_duration_seconds": duration,
        "config": {
            "task": eval_cfg.task,
            "model_name": eval_cfg.model_name,
            "llm_names": eval_cfg.llm_names,
            "seed": eval_cfg.seed,
            "temperature": eval_cfg.temperature,
            "max_tokens": eval_cfg.max_tokens,
            "max_turns": eval_cfg.max_turns,
            "diversity_bonus_weight": eval_cfg.diversity_bonus_weight,
            "cost_bonus_weight": eval_cfg.cost_bonus_weight,
            "turn_bonus_weight": eval_cfg.turn_bonus_weight,
            "role_bonus_weight": eval_cfg.role_bonus_weight,
            "use_structured_router": eval_cfg.use_structured_router,
            "use_consultant": eval_cfg.use_consultant,
            "use_verifier": eval_cfg.use_verifier,
            "trinity": eval_cfg.trinity,
            "last_token_predict": eval_cfg.last_token_predict,
            "valid_ratio": eval_cfg.valid_ratio,
            "test_ratio": eval_cfg.test_ratio,
            "num_repeats": eval_cfg.num_repeats,
            "sigma0": eval_cfg.sigma0,
            "model_types": eval_cfg.model_types,
            "agent_configs": eval_cfg.agent_configs,
            "server_map": eval_cfg.server_map,
            "port_map": eval_cfg.port_map,
            "closed_model_config": eval_cfg.closed_model_config,
        },
        "results": results,
    }

    with output_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
