"""
CMA-ES Trainer for multi-agent router training with enhanced diagnostics.

This module implements a CMA Evolution Strategy approach for training the router model,
with support for diversity bonuses, structured router approach, agent configuration management,
and closed source cost tracking.
"""

import os
import json
from typing import Dict, Tuple, List, Optional, Any
import multiprocessing as mp
import time
import numpy as np
import cma
from tqdm import tqdm
import torch
import re
import pickle
import signal

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from fugu.trainer import (
    RouterInfrastructure,
    SVDParameterManager,
    EvaluationManager,
    WorkerContext,
    _worker_process_context,
    ParameterApplier
)
from fugu.utils import calculate_agent_stats, aggregate_token_statistics, InfrastructureFailure

# Add debugging import
try:
    from fugu.debug_worker_lifecycle import get_worker_logger
    _DEBUG_LOGGING_AVAILABLE = True
except ImportError:
    _DEBUG_LOGGING_AVAILABLE = False
    print("Warning: Debug logging not available. Create debug_worker_lifecycle.py for detailed logs.")

def _calculate_diversity_metrics(agent_ids: list, num_agents: int) -> dict:
    """
    Calculate diversity metrics for agent selections.

    Args:
        agent_ids: List of agent IDs (zero-indexed) observed in an episode
        num_agents: Total number of agents

    Returns:
        Dictionary with keys 'entropy', 'gini_diversity', and 'unique_ratio'
    """
    # --- 1) Filter out any invalid IDs ---
    valid_ids = [aid for aid in agent_ids if 0 <= aid < num_agents]
    if not valid_ids:
        return {
            "entropy": 0.0,
            "gini_diversity": 0.0,
            "unique_ratio": 0.0
        }

    # --- 2) Count occurrences of each agent ---
    counts = np.zeros(num_agents, dtype=float)
    for aid in valid_ids:
        counts[aid] += 1.0

    # --- 3) Convert to probabilities ---
    total = counts.sum()
    probs = counts / total if total > 0 else counts

    # --- 4) Entropy ---
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * np.log(p)

    # --- 5) Gini diversity coefficient ---
    gini = 1.0 - np.sum(probs ** 2)

    # --- 6) Unique agent ratio ---
    unique_ratio = np.count_nonzero(counts) / num_agents

    return {
        "entropy": float(entropy),
        "gini_diversity": float(gini),
        "unique_ratio": float(unique_ratio)
    }


class CMAEvolutionTrainer:
    """
    CMA-ES trainer for router optimization that:
    - Flattens all SVD scale factors plus the action layer weight
    - Runs multiple evaluation rollouts per candidate solution
    - Performs periodic validation on test data
    - Supports agent configuration management
    - Supports wandb logging
    - Tracks closed source agent costs
    - Supports multi-GPU worker distribution
    """

    def __init__(
            self,
            infrastructure: RouterInfrastructure,
            num_iters: int = 1000,
            test_interval: int = 10,
            save_interval: int = 1,
            num_repeats: int = 16,
            popsize_override: int = 0,
            sigma0: float = 0.03,
            seed: int = 42,
            num_tests: int = 100,
            test_size: int = 100,
            servers: Dict[str, str] = None,
            opt_layer_indices: Optional[List[int]] = None,
            diversity_bonus_weight: float = 0.0,
            cost_bonus_weight: float = 0.0,
            turn_bonus_weight: float = 0.0,
            role_bonus_weight: float = 0.0,
            use_structured_router: bool = False,
            closed_model_config: Optional[Dict] = None,
            agent_configs: Optional[Dict] = None,
            use_consultant: bool = True,
            resume_from_training: bool = True,
            last_token_predict: bool = False,
            wandb_run=None,
            use_verifier: bool = False,
            worker_batch_size: int = 1,
            head_type: str = "linear",
            trinity: bool = False,  # NEW: Trinity mode flag
    ):
        """
        Initialize the CMA-ES trainer.

        Args:
            infrastructure: RouterInfrastructure object for environment setup
            num_iters: Total CMA-ES iterations
            test_interval: Evaluate on validation set every N iterations
            save_interval: Save the cma solver ckpt every N iterations
            num_repeats: Number of training episodes per candidate
            popsize_override: If >0, force CMA popsize to this value
            sigma0: CMA initial sigma
            seed: Random seed
            num_tests: Number of tasks to sample for validation
            test_size: Number of tasks to sample for final test evaluation
            servers: Server mapping for agent models
            opt_layer_indices: Layer indices to optimize (e.g., [0, 4, 8, 12])
            diversity_bonus_weight: Weight for diversity bonus (0 to disable)
            cost_bonus_weight: Weight for cost penalty (higher values penalize cost more)
            turn_bonus_weight: Weight for turn bonus/penalty
            role_bonus_weight: Weight for role bonus/penalty (trinity mode only)
            use_structured_router: Whether to use the hybrid structured router approach
            closed_model_config: Configuration for closed API models
            agent_configs: Dictionary mapping agent names to their configurations
            use_consultant: Whether to enable the consultant feature
            use_verifier: Whether to enable the verifier feature
            wandb_run: Weights & Biases run instance for logging
            worker_batch_size: task batch size for each training jobs
            head_type: Type of head architecture
            trinity: Whether to enable trinity mode (solver, thinker, verifier roles)
        """
        self.infra = infrastructure
        self.num_iters = num_iters
        self.test_interval = test_interval
        self.save_interval = save_interval
        self.num_repeats = num_repeats
        self.sigma0 = sigma0
        self.seed = seed
        self.num_tests = num_tests
        self.test_size = test_size
        self.popsize_override = popsize_override
        self.servers = servers or {}
        self.opt_layer_indices = opt_layer_indices
        self.resume_from_training = resume_from_training
        self.last_token_predict = last_token_predict
        self.worker_batch_size = worker_batch_size
        self.use_verifier = use_verifier
        self.head_type = head_type
        self.trinity = trinity  # NEW: Store trinity flag

        # Reward coeff.
        self.diversity_bonus_weight = diversity_bonus_weight
        self.cost_bonus_weight = cost_bonus_weight
        self.turn_bonus_weight = turn_bonus_weight
        self.role_bonus_weight = role_bonus_weight

        self.use_structured_router = use_structured_router
        self.closed_model_config = closed_model_config
        self.agent_configs = agent_configs or {}
        self.use_consultant = use_consultant
        self.wandb_run = wandb_run

        # Add cost tracking for closed source models
        self.cumulative_closed_source_cost = 0.0

        # Explicitly store valid_ratio and test_ratio from infrastructure
        self.valid_ratio = getattr(self.infra, 'valid_ratio', 0.5)
        self.test_ratio = getattr(self.infra, 'test_ratio', 0.2)

        # Log special settings
        if self.diversity_bonus_weight > 0:
            print(f"[CMA-ES] Using diversity bonus with weight: {self.diversity_bonus_weight}")
        if self.cost_bonus_weight > 0:
            print(f"[CMA-ES] Using cost penalty with weight: {self.cost_bonus_weight}")
        if self.turn_bonus_weight > 0:
            print(f"[CMA-ES] Using turn bonus with weight: {self.turn_bonus_weight}")
        if self.role_bonus_weight > 0:
            print(f"[CMA-ES] Using role bonus with weight: {self.role_bonus_weight}")
        if self.use_structured_router:
            print(f"[CMA-ES] Using hybrid structured router approach with action layer and task descriptions")
        if self.closed_model_config:
            print(f"[CMA-ES] Using closed model configuration with API calls")
        if self.agent_configs:
            print(f"[CMA-ES] Using custom agent configurations for {len(self.agent_configs)} agents")
        if self.trinity:
            print(f"[CMA-ES] Using trinity mode with solver/thinker/verifier roles")
            # Trinity mode overrides consultant and verifier settings
            if self.use_consultant:
                print(f"[CMA-ES] Warning: Trinity mode overrides consultant setting")
            if self.use_verifier:
                print(f"[CMA-ES] Warning: Trinity mode overrides verifier setting")
        print(f"[CMA-ES] Using {self.head_type} head architecture")

        # Log layer selection if specified
        if self.opt_layer_indices:
            print(f"[CMA-ES] Selectively training layers: {self.opt_layer_indices}")

        # Log GPU configuration
        if hasattr(self.infra, 'worker_gpu_assignments') and self.infra.worker_gpu_assignments:
            gpu_counts = {}
            for gpu_id in self.infra.worker_gpu_assignments:
                gpu_counts[gpu_id] = gpu_counts.get(gpu_id, 0) + 1
            print(f"[CMA-ES] Worker distribution across GPUs:")
            for gpu_id, count in sorted(gpu_counts.items()):
                print(f"  cuda:{gpu_id}: {count} workers")

        # Set up SVD weights and parameter counting
        self.model_config, self.num_learnable_params, self.svd_weights_cpu = self._setup_svd_info()

        self.diag_dir = os.path.join(self.infra.log_dir, "es_diagnostics")
        os.makedirs(self.diag_dir, exist_ok=True)

        # Ckpt point folder for cma solver
        self.ckpt_dir = os.path.join(self.infra.log_dir, "es_ckpts")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.log_file = os.path.join(self.infra.log_dir, "es_log.json")
        self.action_weights_file = os.path.join(self.infra.log_dir, "action_weights_evolution.json")

        # Initialize action weights data
        self.action_weights_data = {}

        # Initialize model tracking variables
        model_save_dir = os.path.join(self.infra.log_dir, "models")
        os.makedirs(model_save_dir, exist_ok=True)
        self.best_model_path = os.path.join(model_save_dir, "best_model.npy")
        self.best_score = -np.inf
        self.best_solution = None
        self.best_iter = -1

        # Create initial log entry with enhanced config
        self.log_data = [
            {
                "configs": {
                    "task": self.infra.task,
                    "model_name": self.infra.model_name,
                    "llm_names": self.infra.llm_names,
                    "log_dir": self.infra.log_dir,
                    "num_iters": self.num_iters,
                    "test_interval": self.test_interval,
                    "num_repeats": self.num_repeats,
                    "sigma0": self.sigma0,
                    "seed": self.seed,
                    "num_tests": self.num_tests,
                    "test_size": self.test_size,
                    "opt_layer_indices": self.opt_layer_indices,
                    "diversity_bonus_weight": self.diversity_bonus_weight,
                    "cost_bonus_weight": self.cost_bonus_weight,
                    "turn_bonus_weight": self.turn_bonus_weight,
                    "role_bonus_weight": self.role_bonus_weight,
                    "use_structured_router": self.use_structured_router,
                    "hybrid_approach": self.use_structured_router,
                    "closed_model_config": self.closed_model_config is not None,
                    "valid_ratio": self.valid_ratio,
                    "test_ratio": self.test_ratio,
                    "temperature": self.infra.temperature,
                    "max_tokens": self.infra.max_tokens,
                    "max_turns": self.infra.max_turns,
                    "use_consultant": self.use_consultant,
                    "use_verifier": self.use_verifier,
                    "trinity": self.trinity,  # NEW: Add trinity flag
                    "agent_configs": self.agent_configs,
                    "num_agents": len(self.infra.llm_names),
                    "last_token_predict": self.last_token_predict,
                    "gpu_config": {
                        "router_gpu": "cuda:0",
                        "worker_gpu_assignments": getattr(self.infra, 'worker_gpu_assignments', [1]),
                        "total_workers": self.infra.num_workers,
                    },
                    "head_type": self.head_type,
                    "head_parameter_count": self._get_head_parameter_count(),
                }
            }
        ]

        # Load existing log data and restore state if resuming
        if resume_from_training and os.path.exists(self.log_file):
            with open(self.log_file, "r") as f:
                self.log_data = json.load(f)
            print("[CMA-ES] load the existing log file")

            # Restore best model state from log data
            try:
                best_validation_entry = None
                for entry in self.log_data:
                    if (entry.get("type") == "valid" and
                            entry.get("is_new_best", False) and
                            entry.get("best_score") is not None):
                        # Find the most recent best validation entry
                        if (best_validation_entry is None or
                                entry.get("iter", -1) > best_validation_entry.get("iter", -1)):
                            best_validation_entry = entry

                if best_validation_entry:
                    self.best_score = best_validation_entry["best_score"]
                    self.best_iter = best_validation_entry["best_iter"]

                    # Try to load the corresponding model file if it exists
                    if os.path.exists(self.best_model_path):
                        try:
                            self.best_solution = np.load(self.best_model_path)
                            print(f"[CMA-ES] Restored best model: score={self.best_score:.4f}, iter={self.best_iter}")
                        except Exception as e:
                            print(f"[CMA-ES] Warning: Could not load best model file: {e}")
                            # Reset to defaults if loading fails
                            self.best_score = -np.inf
                            self.best_solution = None
                            self.best_iter = -1
                    else:
                        print(
                            f"[CMA-ES] Best model state restored from logs: score={self.best_score:.4f}, iter={self.best_iter}")
                        print(f"[CMA-ES] Model file not found at: {self.best_model_path}")
                else:
                    print("[CMA-ES] No previous best validation results found in log")

            except Exception as e:
                print(f"[CMA-ES] Warning: Error during resume state restoration: {e}")
                print("[CMA-ES] Continuing with fresh state")

        # Initialize log files if not resuming
        if not resume_from_training:
            with open(self.log_file, "w") as f:
                json.dump(self.log_data, f, indent=2)

            with open(self.action_weights_file, "w") as f:
                json.dump({}, f)
            print("[CMA-ES] build a fresh log file")

    def _get_head_parameter_count(self) -> int:
        """Get the parameter count for the current head configuration."""
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(self.infra.model_name)
        hidden_size = model_config.hidden_size

        # NEW: Determine output features based on trinity mode
        if self.trinity:
            # Trinity mode: agents + 3 roles (solver, thinker, verifier)
            num_outputs = len(self.infra.llm_names) + 3
        else:
            # Normal mode: agents + consultant (if enabled)
            consultant_outputs = 1 if self.use_consultant else 0
            num_outputs = len(self.infra.llm_names) + consultant_outputs

        from fugu.head_modules import create_router_head
        temp_head = create_router_head(hidden_size, num_outputs, self.head_type, device="cpu")
        return temp_head.get_parameter_count()

    def _setup_svd_info(self) -> Tuple[Dict, int, Dict[str, torch.Tensor]]:
        """
        Load and prepare SVD weights for training, counting learnable parameters.
        Works even when main process router model is not initialized.
        """
        # Load SVD weights directly from disk (don't depend on router model)
        svd_weights_disk = SVDParameterManager.load_svd_weights(
            self.infra.model_name, device="cpu"
        )

        if self.opt_layer_indices is not None:
            svd_weights_cpu = SVDParameterManager.filter_svd_weights_by_layers(
                svd_weights_disk, self.opt_layer_indices
            )
            filtered_count = len(svd_weights_disk) - len(svd_weights_cpu)
            print(f"[CMA-ES] Filtered out {filtered_count} SVD weight components based on layer selection")
            print(f"[CMA-ES] Keeping {len(svd_weights_cpu)} SVD weight components for training")
        else:
            svd_weights_cpu = {k: v.cpu() for k, v in svd_weights_disk.items()}

        # Count how many singular values there are
        num_singular = 0
        for name, tensor in svd_weights_cpu.items():
            if name.endswith(".S"):
                num_singular += tensor.numel()

        # Calculate action-layer parameter count from config (not from actual model)
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(self.infra.model_name)
        hidden_size = model_config.hidden_size

        # Determine output features based on trinity mode
        if self.trinity:
            # Trinity mode: agents + 3 roles (solver, thinker, verifier)
            num_outputs = len(self.infra.llm_names) + 3
        else:
            # Normal mode: agents + consultant (if enabled)
            consultant_outputs = 1 if self.use_consultant else 0
            num_outputs = len(self.infra.llm_names) + consultant_outputs

        # Create a temporary head to get parameter count
        from fugu.head_modules import create_router_head
        temp_head = create_router_head(hidden_size, num_outputs, self.head_type, device="cpu")
        action_param_count = temp_head.get_parameter_count()

        num_learnable_params = num_singular + action_param_count

        print(f"[CMA-ES] Trinity mode: {self.trinity}")
        if self.trinity:
            print(f"[CMA-ES] Output dimensions: {len(self.infra.llm_names)} agents + 3 roles = {num_outputs}")
        else:
            print(
                f"[CMA-ES] Output dimensions: {len(self.infra.llm_names)} agents + {1 if self.use_consultant else 0} consultant = {num_outputs}")
        print(f"[CMA-ES] Head type: {self.head_type}")
        print(f"[CMA-ES] Head parameter count: {action_param_count}")
        print(
            f"[CMA-ES] Total learnable parameters: {num_learnable_params} (SVD: {num_singular}, Head: {action_param_count})")

        return model_config, num_learnable_params, svd_weights_cpu

    def _calculate_episode_diversity_metrics(self, episodes: list) -> dict:
        """Episode-level agent-diversity metrics.

        NOTE: this method is referenced by the vendored es.py (run_test) and
        evaluate_trinity_livecodebench.py but was absent from the OpenReview code
        submission. Reconstructed faithfully from the module-level
        `_calculate_diversity_metrics` contract: `episodes` is a list where each
        item is one episode's list of agent IDs. We report the mean of per-episode
        diversity (entropy / gini / unique-ratio) plus mean episode length, all
        diagnostic-only (not used in the reward or score path).
        """
        n_agents = len(self.infra.llm_names)
        valid = [ep for ep in episodes if ep]
        if not valid:
            return {
                "episode_mean_entropy": 0.0,
                "episode_mean_gini_diversity": 0.0,
                "episode_mean_unique_ratio": 0.0,
                "episode_mean_length": 0.0,
                "num_episodes": 0,
            }
        ents, ginis, uniqs, lengths = [], [], [], []
        for ep in valid:
            m = _calculate_diversity_metrics(list(ep), n_agents)
            ents.append(m["entropy"])
            ginis.append(m["gini_diversity"])
            uniqs.append(m["unique_ratio"])
            lengths.append(len(ep))
        return {
            "episode_mean_entropy": float(np.mean(ents)),
            "episode_mean_gini_diversity": float(np.mean(ginis)),
            "episode_mean_unique_ratio": float(np.mean(uniqs)),
            "episode_mean_length": float(np.mean(lengths)),
            "num_episodes": len(valid),
        }

    def run_test(self, solution=None):
        """
        Run evaluation on the test set using the existing unified job manager.

        Args:
            solution: Optional specific solution to test. If None, uses best saved model.

        Returns:
            Dict with test results
        """
        print("[CMA-ES] Running test evaluation...")

        # Try to load the best model if no solution provided
        if solution is None:
            if hasattr(self, 'best_solution') and self.best_solution is not None:
                solution = self.best_solution
            elif os.path.exists(self.best_model_path):
                try:
                    solution = np.load(self.best_model_path)
                    print(f"[CMA-ES] Loaded best model from {self.best_model_path}")
                except Exception as e:
                    print(f"[CMA-ES] Error loading best model: {e}")
                    if hasattr(self, 'solver') and hasattr(self.solver, 'result'):
                        solution = self.solver.result.xfavorite
                        print("[CMA-ES] Using final model instead")
                    else:
                        print("[CMA-ES] No model available for testing")
                        return None
            elif hasattr(self, 'solver') and hasattr(self.solver, 'result'):
                solution = self.solver.result.xfavorite
                print("[CMA-ES] No saved best model found, using final model")
            else:
                print("[CMA-ES] No model available for testing")
                return None

        # Use the existing unified job manager instead of creating a new pool
        from fugu.job_manager import get_job_manager
        job_manager = get_job_manager()

        if job_manager.pool is None:
            raise RuntimeError("Unified job manager not initialized for testing")

        print(f"[CMA-ES] Using existing unified job manager for test evaluation")

        # Sample test indices and submit jobs to the existing unified job manager
        np_random = np.random.RandomState(seed=self.seed)
        test_futures = []
        num_test_samples = self.test_size

        for _ in range(num_test_samples):
            tid = np_random.randint(0, self.infra.test_dataset_size)

            future = job_manager.submit_training_job(
                task_id=int(tid),
                split="test",
                flat_params=solution.astype(np.float32),
                svd_weights_cpu=self.svd_weights_cpu,
                iteration_idx=-1,  # No iteration (final test)
                eps_explore=0.0,  # No exploration
                servers_dict=self.servers,
                use_structured_router=self.use_structured_router,
                closed_model_config=getattr(self, 'closed_model_config', None),
                agent_configs=self.agent_configs
            )
            test_futures.append(future)

        # Collect results from the unified job manager
        from tqdm import tqdm
        test_results = []
        infrastructure_failures = 0

        for future in tqdm(test_futures, total=len(test_futures), desc=f"[CMA-ES] Test Evaluation"):
            try:
                result = future.get(timeout=600)  # 10 min timeout per job
                test_results.append(result)
            except Exception as e:
                print(f"Test job failed: {e}")
                infrastructure_failures += 1
                # Add failed result for other errors
                test_results.append((-1.0, 0, [], [], "Error occurred", {
                    "router_tokens": 0, "agent_input_tokens": 0,
                    "agent_output_tokens": 0, "total_tokens": 0, "num_turns": 0
                }, [], []))

        if infrastructure_failures > 0:
            print(f"[CMA-ES] Test evaluation: {infrastructure_failures} jobs failed")

        # Process results (unchanged from original)
        test_token_stats = aggregate_token_statistics(test_results)

        # Calculate test costs
        test_closed_source_costs = self._calculate_closed_source_costs(test_results)
        test_token_stats.update(test_closed_source_costs)

        test_agent_ids = [aid for r in test_results if len(r) >= 4 and r[0] != -1.0 for aid in r[3]]
        test_agent_stats, _ = calculate_agent_stats(test_agent_ids, self.infra.llm_names)
        test_diversity = _calculate_diversity_metrics(test_agent_ids, len(self.infra.llm_names))
        test_episodes = [r[3] for r in test_results if len(r) >= 4 and r[0] != -1.0]
        test_episode_diversity = self._calculate_episode_diversity_metrics(test_episodes)

        test_scores = [r[0] for r in test_results if r[0] != -1.0]
        test_score = float(np.mean(test_scores)) if test_scores else 0.0

        # Create test results entry
        test_entry = {
            "type": "test",
            "test_score": test_score,
            "num_samples": len(test_scores),
            "validation_best_score": self.best_score if hasattr(self, 'best_score') else None,
            "infrastructure_failures": infrastructure_failures,
            **test_agent_stats,
            **test_diversity,
            **test_episode_diversity,
            "token_stats": test_token_stats,
        }

        self.log_data.append(test_entry)

        # Log test results to wandb
        if self.wandb_run is not None and _WANDB_AVAILABLE:
            wandb_test_log = {
                "test/score": test_score,
                "test/num_samples": len(test_scores),
                "test/agent_diversity_entropy": test_diversity.get("entropy", 0.0),
                "test/total_tokens": test_token_stats.get("total_tokens", 0),
                "test/avg_tokens_per_episode": test_token_stats.get("avg_tokens_per_episode", 0.0),
                "test/infrastructure_failures": infrastructure_failures,
            }

            self.wandb_run.log(wandb_test_log)

        # Log and save results
        print(f"[CMA-ES] Test evaluation complete. Score: {test_score:.4f}")
        with open(self.log_file, "w") as f:
            import json
            json.dump(self.log_data, f, indent=2)

        return test_entry
