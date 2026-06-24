"""
Unified job manager to eliminate nested multiprocessing.
Handles both training episodes and code execution in a single worker pool.
"""

import os
import time
import signal
import threading
import multiprocessing as mp
from enum import Enum
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
import traceback
import numpy as np
import torch
import atexit

class JobType(Enum):
    TRAINING_EPISODE = "training_episode"
    CODE_EXECUTION = "code_execution"
    ROUTER_INFERENCE = "router_inference"

@dataclass
class JobResult:
    """Wrapper for job results with metadata."""
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    worker_pid: Optional[int] = None

class UnifiedJobManager:
    """
    Single multiprocessing pool that handles all job types.
    Eliminates nested multiprocessing issues entirely.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.pool = None
        self.num_workers = None
        self.worker_config = None
        self._cleanup_registered = False

    def initialize(self, num_workers: int, worker_config: Dict):
        """Initialize the unified worker pool."""
        if self.pool is not None:
            print("[JobManager] Pool already initialized, skipping")
            return

        self.num_workers = num_workers
        self.worker_config = worker_config.copy()

        try:
            # Ensure spawn method is set
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                # Already set, continue
                pass

            # Create the unified pool
            self.pool = mp.Pool(
                num_workers,
                initializer=_init_unified_worker,
                initargs=(worker_config,),
            )

            # Register cleanup
            if not self._cleanup_registered:
                atexit.register(self.cleanup)
                self._cleanup_registered = True

            print(f"[JobManager] Initialized unified pool with {num_workers} workers")

        except Exception as e:
            print(f"[JobManager] Failed to initialize pool: {e}")
            raise

    def submit_training_job(self,
                            task_id: int,
                            split: str,
                            flat_params: np.ndarray,
                            iteration_idx: int,
                            eps_explore: float,
                            servers_dict: Dict,
                            use_structured_router: bool,
                            closed_model_config: Optional[Dict] = None,
                            agent_configs: Optional[Dict] = None,
                            svd_weights_cpu: Optional[Dict] = None,  # OLD interface (ES)
                            svd_weights_path: Optional[str] = None) -> mp.pool.AsyncResult:  # NEW interface (RL)
        """Submit a training episode job to the unified pool. Supports both old (ES) and new (RL) interfaces."""
        if self.pool is None:
            raise RuntimeError("JobManager not initialized. Call initialize() first.")

        # Determine which interface is being used
        if svd_weights_cpu is not None and svd_weights_path is not None:
            raise ValueError("Cannot specify both svd_weights_cpu and svd_weights_path")
        elif svd_weights_cpu is None and svd_weights_path is None:
            raise ValueError("Must specify either svd_weights_cpu or svd_weights_path")

        # Set up job data based on interface
        job_data = {
            'task_id': task_id,
            'split': split,
            'flat_params': flat_params,
            'iteration_idx': iteration_idx,
            'eps_explore': eps_explore,
            'servers_dict': servers_dict,
            'use_structured_router': use_structured_router,
            'closed_model_config': closed_model_config,
            'agent_configs': agent_configs,
        }

        if svd_weights_cpu is not None:
            # ES interface - direct SVD weights
            job_data['svd_weights_cpu'] = svd_weights_cpu
            job_data['use_lazy_loading'] = False
        else:
            # RL interface - lazy loading with path
            job_data['svd_weights_path'] = svd_weights_path
            job_data['use_lazy_loading'] = True

        return self.pool.apply_async(
            _execute_unified_job,
            (JobType.TRAINING_EPISODE, job_data)
        )

    def _is_running_in_worker(self) -> bool:
        """
        Detect if we're currently running inside a worker process.

        Returns:
            True if running in a worker process, False otherwise
        """
        try:
            import multiprocessing as mp
            current_process = mp.current_process()

            # Check if this is a worker process by looking at the process name
            if hasattr(current_process, 'name'):
                process_name = current_process.name
                if ('SpawnPoolWorker' in process_name or
                        'ForkPoolWorker' in process_name or
                        'PoolWorker' in process_name):
                    return True

            # Also check if we have worker globals initialized
            global _worker_initialized
            if _worker_initialized:
                return True

            return False

        except Exception:
            # If we can't determine, assume we're not in a worker
            return False

    def submit_code_execution_job(self,
                                  sample: Dict,
                                  generation: str,
                                  timeout: int,
                                  debug: bool = False,
                                  using_closed_models: bool = False) -> mp.pool.AsyncResult:
        """Submit a code execution job to the unified pool."""

        # Create a mock AsyncResult class for direct execution
        class MockAsyncResult:
            def __init__(self, result):
                self._result = result

            def get(self, timeout=None):
                return self._result

            def ready(self):
                return True

            def successful(self):
                return True

        # NEW: Check if we're already running inside a worker process
        # If so, execute directly instead of submitting another job
        if self._is_running_in_worker():
            if debug:
                print("[JobManager] Detected nested job submission, using direct execution")

            # Execute directly using the worker's code executor
            try:
                # Try to access the worker's global code executor
                import sys
                current_module = sys.modules[__name__]

                if hasattr(current_module, '_code_executor') and current_module._code_executor is not None:
                    code_executor = current_module._code_executor
                    result, metadata = code_executor.check_correctness(
                        sample, generation, timeout, debug, using_closed_models
                    )
                    return MockAsyncResult((result, metadata))
                else:
                    # Fallback: create a direct executor
                    from fugu.tasks.livecodebench_direct_executor import DirectCodeExecutor
                    direct_executor = DirectCodeExecutor()
                    result, metadata = direct_executor.check_correctness(
                        sample, generation, timeout, debug, using_closed_models
                    )
                    return MockAsyncResult((result, metadata))

            except Exception as e:
                if debug:
                    print(f"[JobManager] Direct execution failed: {e}")
                # Fall through to normal job submission

        if self.pool is None:
            raise RuntimeError("JobManager not initialized. Call initialize() first.")

        job_data = {
            'sample': sample,
            'generation': generation,
            'timeout': timeout,
            'debug': debug,
            'using_closed_models': using_closed_models,
        }

        return self.pool.apply_async(
            _execute_unified_job,
            (JobType.CODE_EXECUTION, job_data)
        )

    def submit_batch_jobs(self, job_type: JobType, job_data_list: List[Dict]) -> List[mp.pool.AsyncResult]:
        """Submit multiple jobs of the same type."""
        if self.pool is None:
            raise RuntimeError("JobManager not initialized. Call initialize() first.")

        futures = []
        for job_data in job_data_list:
            future = self.pool.apply_async(
                _execute_unified_job,
                (job_type, job_data)
            )
            futures.append(future)

        return futures

    def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status for monitoring."""
        if self.pool is None:
            return {"status": "not_initialized"}

        try:
            # Try to get some basic info about the pool
            return {
                "status": "active",
                "num_workers": self.num_workers,
                "pool_object": str(self.pool),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def cleanup(self):
        """Clean up the worker pool."""
        if self.pool is not None:
            try:
                print("[JobManager] Terminating unified pool...")
                self.pool.terminate()
                self.pool.join()
                self.pool = None
                print("[JobManager] Unified pool cleaned up successfully")
            except Exception as e:
                print(f"[JobManager] Error during cleanup: {e}")
                self.pool = None

    def force_cleanup(self):
        """Force cleanup even if workers are stuck."""
        if self.pool is not None:
            try:
                print("[JobManager] Force terminating unified pool...")
                self.pool.terminate()
                # Don't wait for join in force mode
                self.pool = None
                print("[JobManager] Unified pool force cleaned up")
            except Exception as e:
                print(f"[JobManager] Error during force cleanup: {e}")
                self.pool = None

# Global instance
_job_manager = UnifiedJobManager()

def get_job_manager() -> UnifiedJobManager:
    """Get the global job manager instance."""
    return _job_manager

# Worker process globals - these will be initialized in each worker
_worker_context = None
_code_executor = None
_worker_initialized = False


def _init_unified_worker(config: Dict):
    """Initialize a unified worker with SVD weight caching capability."""
    global _worker_context, _code_executor, _worker_initialized

    if _worker_initialized:
        return  # Already initialized

    worker_pid = os.getpid()
    logger = None

    try:
        # Initialize logger if available
        try:
            from fugu.debug_worker_lifecycle import get_worker_logger
            logger = get_worker_logger()
            logger.log_event("worker_init_start", {
                "config_keys": list(config.keys()),
                "worker_pid": worker_pid
            })
        except Exception:
            # Continue without logging
            pass

        # Import here to avoid circular imports
        from fugu.trainer import WorkerContext

        # Initialize worker context for training
        _worker_context = WorkerContext()
        _worker_context.initialize_from_config(config)

        # LAZY LOADING: Initialize SVD weights caching (only for RL)
        _worker_context._cached_svd_weights = None
        _worker_context._cached_svd_weights_path = None

        # Initialize code executor for LiveCodeBench
        try:
            from fugu.tasks.livecodebench_direct_executor import DirectCodeExecutor
            _code_executor = DirectCodeExecutor()
        except ImportError:
            print(f"[Worker {worker_pid}] Warning: DirectCodeExecutor not available, code execution will be limited")
            _code_executor = None

        # Set up agent configurations if provided
        if config.get("agent_configs"):
            try:
                from fugu.utils import set_worker_agent_configs
                set_worker_agent_configs(config["agent_configs"])
            except ImportError:
                print(f"[Worker {worker_pid}] Warning: Could not set agent configs")

        # FIXED: Use multiprocessing process identity for GPU assignment
        worker_gpu_assignments = config.get('worker_gpu_assignments', [1])

        # Get worker index from multiprocessing process identity
        import multiprocessing as mp
        current_process = mp.current_process()

        worker_index = 0
        if hasattr(current_process, '_identity') and current_process._identity:
            # Identity is a tuple like (1,) for first worker, (2,) for second, etc.
            worker_index = current_process._identity[0] - 1
        elif hasattr(current_process, 'name') and 'SpawnPoolWorker' in current_process.name:
            # Try to extract from name like "SpawnPoolWorker-1", "SpawnPoolWorker-2", etc.
            try:
                worker_index = int(current_process.name.split('-')[-1]) - 1
            except (ValueError, IndexError):
                # Fallback: use PID-based assignment (less reliable but better than all-same-GPU)
                worker_index = worker_pid % len(worker_gpu_assignments)
        elif hasattr(current_process, 'name') and 'ForkPoolWorker' in current_process.name:
            # Handle ForkPoolWorker naming
            try:
                worker_index = int(current_process.name.split('-')[-1]) - 1
            except (ValueError, IndexError):
                worker_index = worker_pid % len(worker_gpu_assignments)
        else:
            # Fallback: use PID-based assignment
            worker_index = worker_pid % len(worker_gpu_assignments)

        # Assign GPU based on worker index with bounds checking
        if worker_index < len(worker_gpu_assignments):
            assigned_gpu = worker_gpu_assignments[worker_index]
        else:
            # Round-robin assignment if we have more workers than explicit assignments
            assigned_gpu = worker_gpu_assignments[worker_index % len(worker_gpu_assignments)]

        # Store GPU assignment in worker context
        _worker_context.assigned_gpu = assigned_gpu

        # Set up agent configurations if provided (duplicate but kept as-is)
        if "agent_configs" in config and config["agent_configs"]:
            from fugu.utils import set_worker_agent_configs
            set_worker_agent_configs(config["agent_configs"])

            if logger:
                logger.log_event("agent_configs_loaded", {
                    "agent_count": len(config["agent_configs"]),
                    "agent_names": list(config["agent_configs"].keys())
                })

            # Print agent configuration info if debug is enabled
            if config.get("debug", False):
                print(f"[Worker {worker_pid}] Agent configurations loaded:")
                for agent_name, agent_config in config["agent_configs"].items():
                    print(f"  {agent_name} -> {agent_config['model_name']}")
                    if agent_config.get("payload"):
                        print(f"    Payload: {agent_config['payload']}")

        # Print initialization info with detailed worker assignment info
        trinity_mode = config.get('trinity', False)
        head_type = config.get('head_type', 'linear')
        trinity_str = f"with trinity mode ({head_type} head)" if trinity_mode else f"with {head_type} head"

        print(
            f"[Worker {worker_pid}] Worker initialized with SVD caching on GPU cuda:{assigned_gpu} "
            f"(index {worker_index}/{len(worker_gpu_assignments)}, process: {current_process.name}) {trinity_str}"
        )

        # Print debug info if enabled
        if config.get("debug", False):
            print(f"[Worker {worker_pid}] Process identity: {getattr(current_process, '_identity', 'None')}")
            print(f"[Worker {worker_pid}] Process name: {getattr(current_process, 'name', 'None')}")
            print(f"[Worker {worker_pid}] Worker index: {worker_index}/{len(worker_gpu_assignments)}")
            print(f"[Worker {worker_pid}] GPU assignment: cuda:{assigned_gpu}")
            print(f"[Worker {worker_pid}] Available GPUs: {worker_gpu_assignments}")
            print(f"[Worker {worker_pid}] SVD caching enabled")
            print(f"[Worker {worker_pid}] Trinity mode: {trinity_mode}")
            print(f"[Worker {worker_pid}] Head type: {head_type}")

        _worker_initialized = True

    except Exception as e:
        if logger:
            logger.log_event("worker_init_error", {
                "error": str(e),
                "error_type": type(e).__name__,
                "worker_pid": worker_pid
            })
        print(f"Error in worker initialization: {e}")
        raise

def _execute_unified_job(job_type: JobType, job_data: Dict) -> Any:
    """Execute a job based on its type."""
    global _worker_context, _code_executor

    worker_pid = os.getpid()
    start_time = time.time()

    try:
        if job_type == JobType.TRAINING_EPISODE:
            result = _execute_training_episode(job_data)
        elif job_type == JobType.CODE_EXECUTION:
            result = _execute_code_execution(job_data)
        else:
            raise ValueError(f"Unknown job type: {job_type}")

        execution_time = time.time() - start_time
        return result

    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Error executing {job_type.value} job in worker {worker_pid}: {e}"
        print(error_msg)
        traceback.print_exc()

        # Return appropriate error result based on job type
        if job_type == JobType.TRAINING_EPISODE:
            return (-1.0, 0, [], [], "Error occurred", {
                "router_tokens": 0, "agent_input_tokens": 0,
                "agent_output_tokens": 0, "total_tokens": 0, "num_turns": 0
            }, [], [])
        elif job_type == JobType.CODE_EXECUTION:
            return [[-1]], {"error": error_msg, "error_code": -5, "execution_time": execution_time}
        else:
            raise


def _execute_training_episode(job_data: Dict) -> Tuple:
    """Execute a training episode with support for both direct SVD weights (ES) and lazy loading (RL)."""
    global _worker_context

    # Extract job data
    task_id = job_data['task_id']
    split = job_data['split']
    flat_params = job_data['flat_params']
    iteration_idx = job_data['iteration_idx']
    eps_explore = job_data['eps_explore']
    servers_dict = job_data['servers_dict']
    use_structured_router = job_data['use_structured_router']
    closed_model_config = job_data.get('closed_model_config')
    agent_configs = job_data.get('agent_configs')
    use_lazy_loading = job_data.get('use_lazy_loading', False)

    worker_pid = os.getpid()

    # Set up agent configurations if provided
    if agent_configs:
        try:
            from fugu.utils import set_worker_agent_configs
            set_worker_agent_configs(agent_configs)
        except ImportError:
            pass  # Continue without agent configs

    # Store closed model configuration in worker context
    if closed_model_config:
        _worker_context.using_closed_models = True
        os.environ["USING_CLOSED_MODELS"] = "1"
    else:
        _worker_context.using_closed_models = False
        os.environ.pop("USING_CLOSED_MODELS", None)

    # Lazy initialization of models
    if _worker_context.router_model is None:
        _initialize_worker_models()

    # Handle SVD weights based on interface
    svd_weights_cpu = None

    if use_lazy_loading:
        # RL interface - lazy loading with caching
        svd_weights_path = job_data['svd_weights_path']

        # Check if we have cached SVD weights for this path
        if (hasattr(_worker_context, '_cached_svd_weights_path') and
            _worker_context._cached_svd_weights_path == svd_weights_path):
            # Use cached weights
            svd_weights_cpu = _worker_context._cached_svd_weights
        else:
            # Load new weights and cache them
            try:
                start_time = time.time()
                svd_weights_cpu = torch.load(svd_weights_path, map_location='cpu')
                load_time = time.time() - start_time

                # Cache the loaded weights
                _worker_context._cached_svd_weights = svd_weights_cpu
                _worker_context._cached_svd_weights_path = svd_weights_path

            except Exception as e:
                print(f"[Worker {worker_pid}] Failed to load SVD weights from {svd_weights_path}: {e}")
                # Return error result
                return (-1.0, 0, [], [], "SVD weights loading failed", {
                    "router_tokens": 0, "agent_input_tokens": 0,
                    "agent_output_tokens": 0, "total_tokens": 0, "num_turns": 0
                })
    else:
        # ES interface - direct SVD weights (no caching needed)
        svd_weights_cpu = job_data['svd_weights_cpu']

    # Apply parameters to models
    try:
        from fugu.trainer import ParameterApplier
        ParameterApplier.apply_params_to_model(
            _worker_context.router_model,
            _worker_context.linear_layer,
            svd_weights_cpu,
            flat_params,
            use_structured_router
        )
    except Exception as e:
        print(f"[Worker {worker_pid}] Error applying parameters: {e}")
        raise

    # Run the training episode evaluation
    try:
        from fugu.trainer import EvaluationManager

        # Set up debug logging if enabled
        debug_log_file = None
        if _worker_context.debug and _worker_context.debug_log_dir is not None:
            iter_dir = EvaluationManager._mk_iter_dir(_worker_context.debug_log_dir, iteration_idx)
            log_filename = f"debug_unified_{task_id}_{split}.txt"
            debug_log_file = os.path.join(iter_dir, log_filename)
            with open(debug_log_file, "w") as f:
                f.write(f"Unified job manager debug log for task_id: {task_id}, split: {split}\n")
                f.write(f"Using {'lazy loading' if use_lazy_loading else 'direct SVD weights'}\n")
                # NEW: Log trinity mode status
                trinity_mode = getattr(_worker_context, 'trinity', False)
                f.write(f"Trinity mode: {trinity_mode}\n")

        # NEW: Choose evaluation method based on trinity flag
        trinity_mode = getattr(_worker_context, 'trinity', False)

        if trinity_mode:
            # Use trinity evaluation
            return EvaluationManager.evaluate_episode_trinity(
                context=_worker_context,
                task_id=task_id,
                split=split,
                iter_idx=iteration_idx,
                eps_explore=eps_explore,
                debug_log_file=debug_log_file
            )
        else:
            # Use standard evaluation
            return EvaluationManager.evaluate_episode(
                context=_worker_context,
                task_id=task_id,
                split=split,
                iter_idx=iteration_idx,
                eps_explore=eps_explore,
                debug_log_file=debug_log_file
            )

    except Exception as e:
        print(f"[Worker {worker_pid}] Error in training episode evaluation: {e}")
        import traceback
        traceback.print_exc()
        return (-1.0, 0, [], [], "Error occurred", {
            "router_tokens": 0, "agent_input_tokens": 0,
            "agent_output_tokens": 0, "total_tokens": 0, "num_turns": 0
        })

def _execute_code_execution(job_data: Dict) -> Tuple:
    """Execute code execution directly (replaces nested multiprocessing)."""
    global _code_executor

    if _code_executor is None:
        return [[-1]], {
            "error": "Code executor not available",
            "error_code": -5,
            "error_message": "DirectCodeExecutor not initialized"
        }

    sample = job_data['sample']
    generation = job_data['generation']
    timeout = job_data['timeout']
    debug = job_data['debug']
    using_closed_models = job_data['using_closed_models']

    try:
        # Use direct code execution instead of subprocess
        return _code_executor.check_correctness(
            sample, generation, timeout, debug, using_closed_models
        )
    except Exception as e:
        print(f"Error in code execution: {e}")
        traceback.print_exc()
        return [[-1]], {
            "error": str(e),
            "error_code": -5,
            "error_message": f"Code execution failed: {e}"
        }

def _initialize_worker_models():
    """Initialize models in the worker process."""
    global _worker_context

    worker_pid = os.getpid()
    assigned_gpu = getattr(_worker_context, 'assigned_gpu', 1)
    device_str = f"cuda:{assigned_gpu}"

    try:
        # Initialize router model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _worker_context.router_model = AutoModelForCausalLM.from_pretrained(
            _worker_context.router_model_name,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            device_map=device_str,
        )

        # Apply Qwen2 forward patch if needed
        if "qwen" in _worker_context.router_model_name.lower():
            from fugu.model_mods.modeling_qwen2 import forward as qwen2_forward
            _worker_context.router_model.forward = qwen2_forward.__get__(
                _worker_context.router_model,
                type(_worker_context.router_model)
            )

        # Initialize tokenizer
        _worker_context.tokenizer = AutoTokenizer.from_pretrained(
            _worker_context.router_model_name
        )

        # Initialize action layer with trinity mode support
        use_consultant = getattr(_worker_context, 'use_consultant', True)
        trinity_mode = getattr(_worker_context, 'trinity', False)
        head_type = getattr(_worker_context, 'head_type', 'linear')

        # Determine output features based on trinity mode
        if trinity_mode:
            # Trinity mode: agents + 3 roles (solver, thinker, verifier)
            output_features = len(_worker_context.llm_names) + 3
        else:
            # Normal mode: agents + consultant (if enabled)
            consultant_outputs = 1 if use_consultant else 0
            output_features = len(_worker_context.llm_names) + consultant_outputs

        # Use the new RouterHead instead of nn.Linear
        from fugu.head_modules import create_router_head
        _worker_context.linear_layer = create_router_head(
            hidden_size=_worker_context.router_model.config.hidden_size,
            num_agents=output_features,
            head_type=head_type,
            max_turns=_worker_context.max_turns,
            device=device_str,
            dtype=torch.bfloat16
        )

        # Initialize task instance
        from fugu.run_tasks import create_task

        valid_ratio = getattr(_worker_context, 'valid_ratio', 0.5)
        test_ratio = getattr(_worker_context, 'test_ratio', 0.2)

        create_task_args = {
            "task_name": _worker_context.task_name,
            "llm_names": _worker_context.llm_names,
            "max_tokens": _worker_context.max_tokens,
            "temperature": _worker_context.temperature,
            "max_turns": _worker_context.max_turns,
            "servers": _worker_context.servers,
            "ports": _worker_context.ports,
            "valid_ratio": valid_ratio,
            "test_ratio": test_ratio,
            "seed": _worker_context.seed,
            "trinity": trinity_mode,  # Pass trinity flag
        }

        worker_log_dir = getattr(_worker_context, 'log_dir', None)
        if worker_log_dir:
            create_task_args["log_dir"] = worker_log_dir

        _worker_context.task_instance = create_task(**create_task_args)

        trinity_str = f"with {head_type} head (trinity: {trinity_mode})" if trinity_mode else f"with {head_type} head"
        print(f"[Worker {worker_pid}] Models initialized successfully on {device_str} {trinity_str}")

    except Exception as e:
        print(f"[Worker {worker_pid}] Error initializing models: {e}")
        import traceback
        traceback.print_exc()
        raise

def cleanup_job_manager():
    """Clean up the global job manager (can be called from external code)."""
    global _job_manager
    if _job_manager is not None:
        _job_manager.cleanup()

def force_cleanup_job_manager():
    """Force cleanup the global job manager (for emergency situations)."""
    global _job_manager
    if _job_manager is not None:
        _job_manager.force_cleanup()

# Register cleanup on module exit
atexit.register(cleanup_job_manager)