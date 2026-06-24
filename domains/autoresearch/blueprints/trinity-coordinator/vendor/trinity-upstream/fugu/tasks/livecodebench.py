import json
import zlib
import pickle
import base64
import os
import signal
import time
import tempfile
from enum import Enum
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import numpy as np
from datasets import load_dataset

from fugu.core import Task
from fugu.utils import extract_answer

# Import the unified job manager
from fugu.job_manager import get_job_manager


def extract_code(model_output: str):
    """
    Extract and validate code from model output with security checks.
    Returns error message instead of executing dangerous code.
    """
    if not model_output:
        return ""

    # Remove any stray backticks that might still appear
    content = model_output.replace('```python', '').replace('```', '').strip()

    if not content:
        return ""

    # Security validation - check for dangerous imports and function calls
    content_lower = content.lower()

    # Check for dangerous OS operations
    dangerous_os_functions = [
        'os.kill', 'os.system', 'os.putenv', 'os.remove', 'os.removedirs',
        'os.rmdir', 'os.fchdir', 'os.setuid', 'os.fork', 'os.forkpty',
        'os.killpg', 'os.rename', 'os.renames', 'os.truncate', 'os.replace',
        'os.unlink', 'os.fchmod', 'os.fchown', 'os.chmod', 'os.chown',
        'os.chroot', 'os.lchflags', 'os.lchmod', 'os.lchown', 'os.getcwd',
        'os.chdir'
    ]

    for func in dangerous_os_functions:
        if func in content_lower:
            return f"CONTENT_REMOVED_DUE_TO_DANGEROUS_OS_FUNCTION: {func}"

    # Check for dangerous shutil operations
    dangerous_shutil_functions = ['shutil.rmtree', 'shutil.move', 'shutil.chown']
    for func in dangerous_shutil_functions:
        if func in content_lower:
            return f"CONTENT_REMOVED_DUE_TO_DANGEROUS_SHUTIL_FUNCTION: {func}"

    # Check for subprocess operations
    if 'subprocess.popen' in content_lower:
        return "CONTENT_REMOVED_DUE_TO_SUBPROCESS_OPERATIONS: subprocess.popen"
    if 'subprocess.call' in content_lower:
        return "CONTENT_REMOVED_DUE_TO_SUBPROCESS_OPERATIONS: subprocess.call"
    if 'subprocess.run' in content_lower:
        return "CONTENT_REMOVED_DUE_TO_SUBPROCESS_OPERATIONS: subprocess.run"

    # Check for dangerous built-in functions
    dangerous_builtins = ['quit()', 'exit()', 'eval(', 'exec(', 'compile(']
    for builtin in dangerous_builtins:
        if builtin in content_lower:
            return f"CONTENT_REMOVED_DUE_TO_DANGEROUS_BUILTIN: {builtin}"

    # Check for dangerous module imports
    dangerous_modules = [
        'import os', 'from os import', 'import subprocess', 'from subprocess import',
        'import shutil', 'from shutil import', 'import resource', 'from resource import',
        'import psutil', 'from psutil import', 'import tkinter', 'from tkinter import',
        'import joblib', 'from joblib import', 'import ipdb', 'from ipdb import'
    ]

    for module in dangerous_modules:
        if module in content_lower:
            return f"CONTENT_REMOVED_DUE_TO_DANGEROUS_MODULE_IMPORT: {module}"

    # Check for file operations that could be dangerous
    dangerous_file_ops = ['open(']
    if any(op in content_lower for op in dangerous_file_ops):
        # Allow stdin operations but be careful about file opens
        if 'open(' in content_lower:
            # Check if it's not just sys.stdin or input()
            lines = content.split('\n')
            for line in lines:
                if 'open(' in line.lower() and 'sys.stdin' not in line.lower():
                    # Allow only read-only opens for common safe patterns
                    if not any(safe_pattern in line.lower() for safe_pattern in ['"r"', "'r'", 'mode="r"', "mode='r'"]):
                        if not line.strip().startswith('#'):  # Ignore comments
                            return f"CONTENT_REMOVED_DUE_TO_DANGEROUS_FILE_OPERATION: {line.strip()}"

    # Check for network operations
    dangerous_network = ['socket.', 'urllib.', 'requests.', 'http.', 'ftplib.', 'smtplib.']
    for net_op in dangerous_network:
        if net_op in content_lower:
            return f"CONTENT_REMOVED_DUE_TO_NETWORK_OPERATION: {net_op}"

    # Check for system-level operations
    if '__import__' in content_lower:
        return "CONTENT_REMOVED_DUE_TO_DYNAMIC_IMPORTS: __import__"
    if 'globals(' in content_lower:
        return "CONTENT_REMOVED_DUE_TO_GLOBALS_ACCESS: globals()"
    if 'locals(' in content_lower:
        return "CONTENT_REMOVED_DUE_TO_LOCALS_ACCESS: locals()"
    if 'vars(' in content_lower:
        return "CONTENT_REMOVED_DUE_TO_VARS_ACCESS: vars()"
    if 'dir(' in content_lower:
        return "CONTENT_REMOVED_DUE_TO_INTROSPECTION: dir()"

    # Check for potential code injection
    if 'exec(' in content_lower:
        return "CONTENT_REMOVED_DUE_TO_CODE_EXECUTION: exec()"
    if 'eval(' in content_lower:
        return "CONTENT_REMOVED_DUE_TO_CODE_EVALUATION: eval()"

    return content


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)


@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)

        self.public_test_cases = json.loads(self.public_test_cases)  # type: ignore
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        try:
            self.private_test_cases = json.loads(self.private_test_cases)  # type: ignore
        except:
            self.private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(self.private_test_cases.encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = json.loads(self.metadata)  # type: ignore

    def insert_output(self, output_list: list[str], code_list: list[str]) -> dict:
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "output_list": output_list,
            "code_list": code_list,
        }

    def insert_output_evaluation(
            self,
            output_list: list[str],
            code_list: list[str],
            graded_list: list[bool],
            **kwargs,
    ) -> dict:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        for k, v in kwargs.items():
            output[k] = v
        return output

    def get_evaluation_sample(self):
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "outputs": [
                        t.output
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }


class LiveCodeBenchTask(Task):
    """Implementation of the LiveCodeBench task using the unified Task framework with job manager."""

    # Prompt for stdin/stdout problems with configurable thinking tag
    STDIN_PROMPT_TEMPLATE = (
        "Question:\n{question_content}\n\n"
        "Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). "
        "Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n\n"
        "Show your work in <THINKING_TAG> </THINKING_TAG> tags. "
        "And return ONLY the final Python code in <answer> </answer> tags, with NO markdown formatting or backticks.\n\n"
        "Also, do not include anything other than Python code between the tags.\n\n"
        "{starter_code}"
        "FORBIDDEN (do NOT do this):\n"
        "<answer>\n"
        "```python\n"
        "n = int(input())\n"
        "a = list(map(int, input().split()))\n"
        "print(sum(a))\n"
        "```\n"
        "</answer>\n\n"
        "FORBIDDEN (do NOT do this):\n"
        "<answer>\n"
        "Here is the best solution ... blablabla (content that is not Python code)\n"
        "n = int(input())\n"
        "a = list(map(int, input().split()))\n"
        "print(sum(a))\n"
        "</answer>\n\n"
        "CORRECT (DO THIS):\n"
        "<answer>\n"
        "n = int(input())\n"
        "a = list(map(int, input().split()))\n"
        "print(sum(a))\n"
        "</answer>\n\n"
        "Think step by step inside <THINKING_TAG> tags, but keep it short."
    )

    FUNCTION_PROMPT_TEMPLATE = (
        "Question:""\n{question_content}\n\n"
        "Format: Implement the function `{function_name}` that solves this problem. "
        "Your solution should be a complete Python function definition wrapped in a Solution class.\n\n"
        "CRITICAL: You MUST use the EXACT function name '{function_name}' - do not change, modify, or rename it in any way.\n\n"
        "Show your work in <THINKING_TAG> </THINKING_TAG> tags. "
        "And return ONLY the final Python code in <answer> </answer> tags, with NO markdown formatting or backticks.\n\n"
        "Your code should follow this structure:\n"
        "- Import any necessary modules at the top\n"
        "- Define a Solution class\n"
        "- Implement the {function_name} method inside the Solution class with the EXACT name '{function_name}'\n"
        "- Do NOT include test cases or example usage\n"
        "- Do NOT change the function name to snake_case, camelCase, or any other format\n\n"
        "{starter_code}"
        "FORBIDDEN (do NOT do this):\n"
        "<answer>\n"
        "```python\n"
        "class Solution:\n"
        "    def {function_name}(...):\n"
        "        ..."
        "```\n"
        "</answer>\n\n"
        "FORBIDDEN (do NOT do this):\n"
        "<answer>\n"
        "Here is the best solution ... blablabla (content that is not Python code)\n"
        "class Solution:\n"
        "    def {function_name}(...):\n"
        "        ..."
        "</answer>\n\n"
        "CORRECT (DO THIS):\n"
        "<answer>\n"
        "class Solution:\n"
        "    def {function_name}(...):\n"
        "        ..."
        "</answer>\n\n"
        "Think step by step inside <THINKING_TAG> tags, but keep it short."
    )

    def __init__(
            self,
            llm_names: List[str],
            seed: int = 42,
            max_tokens: int = 1024,
            temperature: float = 0.8,
            max_turns: int = 3,
            servers: Dict[str, str] = None,
            ports: Dict[str, int] = None,
            track_costs: bool = True,
            debug: bool = False,
            together: bool = True,
            valid_ratio: float = 0.2,
            max_samples: int = -1,
            test_ratio: float = 0.2,
            use_consultant: bool = True,
            use_verifier: bool = False,
            log_dir: Optional[str] = None,
            trinity: bool = False,  # NEW: Add trinity parameter
    ):
        super().__init__(
            llm_names,
            seed=seed,
            max_tokens=max_tokens,
            temperature=temperature,
            max_turns=max_turns,
            servers=servers,
            ports=ports,
            track_costs=track_costs,
            debug=debug,
            together=together,
            valid_ratio=valid_ratio,
            max_samples=max_samples,
            test_ratio=test_ratio,
            use_consultant=use_consultant,
            use_verifier=use_verifier,
            log_dir=log_dir,
            trinity=trinity,  # NEW: Pass trinity parameter to parent class
        )

        # Flag to indicate if we're using closed models (for timeout handling)
        self._using_closed_models = False

    def _get_user_prompt_template(self) -> str:
        """Return the user prompt template with thinking tag substituted."""
        return self.STDIN_PROMPT_TEMPLATE.replace("THINKING_TAG", self.thinking_tag)

    def _format_base_prompt(self, task_data: Dict) -> str:
        """
        Format the base user prompt for a specific task instance.
        Chooses between stdin/stdout and function-based format based on problem type.
        """

        if task_data.get("starter_code"):
            starter_code = "Here is the starter code for this question:\n\n" + task_data["starter_code"] + "\n\n"
        else:
            starter_code = ""

        # Check if this is a call-based problem by looking at metadata
        try:
            if 'metadata' in task_data:
                metadata = json.loads(task_data['metadata'])
                fn_name = metadata.get('func_name')

                if fn_name:
                    # This is a call-based problem - use function template with thinking tag substituted
                    template = self.FUNCTION_PROMPT_TEMPLATE.replace("THINKING_TAG", self.thinking_tag)
                    return template.format(
                        question_content=task_data["question_content"],
                        function_name=fn_name,
                        starter_code=starter_code
                    )
        except (json.JSONDecodeError, KeyError) as e:
            # If metadata parsing fails, fall back to stdin format
            if self.debug:
                print(f"Warning: Could not parse metadata, using stdin format: {e}")

        # Default to stdin/stdout format for standard input problems with thinking tag substituted
        template = self.STDIN_PROMPT_TEMPLATE.replace("THINKING_TAG", self.thinking_tag)
        return template.format(
            question_content=task_data["question_content"],
            starter_code=starter_code
        )

    def _load_data(
            self,
            seed: int,
            split: str = "train",
            validation: bool = False,
            valid_ratio: float = None,
            max_samples: int = None,
            test_split: bool = False,
            test_ratio: float = None
    ) -> Dict[str, Any]:
        """
        Load the dataset for this task with improved handling of data splits.
        """
        # Use provided values or fall back to instance defaults
        if valid_ratio is None:
            valid_ratio = self.valid_ratio
        if max_samples is None:
            max_samples = self.max_samples
        if test_ratio is None:
            test_ratio = self.test_ratio

        # Load the livecodebench v1 dataset 400 samples for training 
        # And v6 dataset 175 samples for testing
        # https://github.com/LiveCodeBench/LiveCodeBench/tree/main?tab=readme-ov-file#dataset-versions
        ds = load_dataset("livecodebench/code_generation_lite", split="test", version_tag="release_v1")
        ds_test = load_dataset("livecodebench/code_generation_lite", split="test", version_tag="v6")

        # Obtain deterministic train/valid/test indices
        from fugu.utils import get_or_create_indices

        split_indices = get_or_create_indices(
            task_name="livecodebenchv6",
            dataset_len=len(ds),
            seed=self.split_seed,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio
        )

        def _take(idxs: List[int]):
            return ds.select(idxs)

        data_splits = {
            "train": _take(split_indices["train"]),
            "valid": _take(split_indices["valid"]),
            # "test": _take(split_indices["test"]) # Replace v6 samples for testing.
            "test": ds_test
        }

        # Optional down-sampling per split
        if max_samples != -1:
            data_splits["train"] = data_splits["train"] \
                .shuffle(seed=seed) \
                .select(range(min(max_samples, len(data_splits["train"]))))
            valid_cap = int(max_samples * valid_ratio / (1.0 - valid_ratio)) if valid_ratio < 1.0 else len(
                data_splits["valid"])
            data_splits["valid"] = data_splits["valid"] \
                .shuffle(seed=seed) \
                .select(range(min(valid_cap, len(data_splits["valid"]))))
            test_cap = int(max_samples * test_ratio / (1.0 - test_ratio)) if test_ratio < 1.0 else len(
                data_splits["test"])
            data_splits["test"] = data_splits["test"] \
                .shuffle(seed=seed) \
                .select(range(min(test_cap, len(data_splits["test"]))))

        # Debug/log the final sizes
        for k, v in data_splits.items():
            print(f"Split {k} contains {len(v)} examples")

        return data_splits

    def _calculate_reward(self, completions: List[str], task_data: List[Dict], debug: bool = False) -> List[float]:
        """
        Calculate rewards using the unified job manager approach.
        """
        rewards = []

        for completion, data in zip(completions, task_data):
            try:
                reward = self._calculate_livecodebench_reward_unified(completion, data, debug=debug)
                rewards.append(reward)
            except Exception as e:
                if debug:
                    print(f"ERROR processing completion: {e}")
                    import traceback
                    traceback.print_exc()
                rewards.append(0.0)

        return rewards

    """
    Fix for the LiveCodeBench nested job manager issue.

    The problem: LiveCodeBench task tries to submit code execution jobs to the job manager
    while already running inside a worker process, causing "JobManager not initialized" errors.

    The solution: Check if we're running inside a worker process and use direct code execution
    instead of submitting another job.
    """

    def _calculate_livecodebench_reward_unified(self, completion: str, task_data: Dict[str, Any],
                                                debug: bool = False) -> float:
        """
        Calculate reward for LiveCodeBench task using the unified job manager.
        This replaces the old context-aware dual approach with a single unified path.
        """

        debug = debug or getattr(self, 'debug',
                                 False) or True  # This forces debug=True always # TODO: make it adjustable

        # Set up debug logging exactly like the previous working version
        if debug:
            import os
            import time
            import tempfile

            base_log_dir = self.log_dir
            if base_log_dir is None:
                base_log_dir = tempfile.mkdtemp(prefix="livecodebench_debug_")

            # Create debug directory with the same structure as before
            debug_dir = os.path.join(base_log_dir, "debug_livecodebench")
            os.makedirs(debug_dir, exist_ok=True)

            pid = os.getpid()
            debug_file = os.path.join(debug_dir, f"livecodebench_debug_{int(time.time())}_{pid}.txt")

            def log_debug(message):
                try:
                    with open(debug_file, "a") as f:
                        f.write(f"{time.strftime('%H:%M:%S')} - {message}\n")
                        f.flush()
                    # Removed console output - no print statements
                except Exception:
                    # Removed console output fallback - silent operation
                    pass

            log_debug("=== LiveCodeBench evaluation started (CMA-ES TRAINING VERSION - UNIFIED) ===")
            log_debug(f"Debug file location: {debug_file}")
            log_debug(f"Base log directory: {base_log_dir}")
            log_debug(f"Completion length: {len(completion)}")
            log_debug(f"Task data keys: {list(task_data.keys())}")
            log_debug(f"Problem title: {task_data.get('question_title', 'N/A')}")
            log_debug(f"Platform: {task_data.get('platform', 'N/A')}")
            log_debug(f"Difficulty: {task_data.get('difficulty', 'N/A')}")
        else:
            def log_debug(message):
                pass

        try:
            # Extract the generated code
            log_debug("=== FULL COMPLETION ===")
            log_debug(completion)
            log_debug("=== END COMPLETION ===")

            # First extract from answer tags
            answer_content = extract_answer(completion)
            if answer_content is None:
                log_debug("WARNING: No code found in <answer> tags!")
                return 0.0

            log_debug(f"Extracted answer content length: {len(answer_content)}")
            log_debug("=== EXTRACTED ANSWER CONTENT ===")
            log_debug(answer_content)
            log_debug("=== END ANSWER CONTENT ===")

            # Then clean the extracted code
            extracted_code = extract_code(answer_content)
            if not extracted_code or extracted_code.startswith("CONTENT_REMOVED_DUE_TO"):
                log_debug(f"Code extraction failed or dangerous code detected: {extracted_code}")
                return 0.0

            log_debug(f"Cleaned code length: {len(extracted_code)}")
            log_debug("=== CLEANED CODE ===")
            log_debug(extracted_code)
            log_debug("=== END CLEANED CODE ===")

            # Create task data sample
            try:
                problem = CodeGenerationProblem(**task_data)
                evaluation_sample = problem.get_evaluation_sample()
                sample = {"input_output": evaluation_sample["input_output"]}
                log_debug("Successfully transformed task data to evaluation format")
            except Exception as e:
                log_debug(f"Error transforming task data: {e}")
                import traceback
                log_debug(f"Transform traceback: {traceback.format_exc()}")
                return 0.0

            # Log the test cases
            try:
                io_data = json.loads(sample['input_output'])
                log_debug(f"Number of test cases: {len(io_data.get('inputs', []))}")
                log_debug(f"Function name (if call-based): {io_data.get('fn_name', 'N/A')}")
                log_debug(f"Test type: {'call-based' if io_data.get('fn_name') else 'stdin-based'}")

                # Log first test case for debugging
                if io_data.get('inputs') and len(io_data['inputs']) > 0:
                    log_debug(f"First test input: {io_data['inputs'][0][:200]}...")
                if io_data.get('outputs') and len(io_data['outputs']) > 0:
                    log_debug(f"First expected output: {io_data['outputs'][0][:200]}...")
            except Exception as e:
                log_debug(f"Error parsing input_output: {e}")
                return 0.0

            # NEW: Check if we're running inside a worker process and use direct execution
            try:
                # First try to use direct code execution if we're in a worker process
                log_debug("Attempting direct code execution (worker process detection)")

                # Try to access the worker's global code executor
                try:
                    # Import the job manager module to access worker globals
                    from fugu import job_manager

                    # Check if we can access the worker's global _code_executor
                    if hasattr(job_manager, '_code_executor') and job_manager._code_executor is not None:
                        log_debug("Using direct code executor (running in worker process)")

                        # Use direct code execution
                        code_executor = job_manager._code_executor
                        result, metadata = code_executor.check_correctness(
                            sample=sample,
                            generation=extracted_code,
                            timeout=6,
                            debug=debug,
                            using_closed_models=getattr(self, '_using_closed_models', False)
                        )

                        log_debug(f"Direct execution returned: result={result}, metadata={metadata}")

                    else:
                        log_debug("No direct code executor available in worker, trying job manager")
                        # Fall through to job manager approach
                        raise AttributeError("Direct executor not available")

                except (ImportError, AttributeError) as e:
                    log_debug(f"Direct execution not available ({e}), trying job manager approach")

                    # Fallback to unified job manager for code execution
                    log_debug("Using unified job manager for code execution")

                    try:
                        from fugu.job_manager import get_job_manager
                        job_manager = get_job_manager()

                        # Check if job manager is initialized
                        if job_manager.pool is None:
                            log_debug("Job manager not initialized, creating direct executor as fallback")
                            # Create a direct executor as last resort
                            from fugu.tasks.livecodebench_direct_executor import DirectCodeExecutor
                            direct_executor = DirectCodeExecutor()
                            result, metadata = direct_executor.check_correctness(
                                sample=sample,
                                generation=extracted_code,
                                timeout=6,
                                debug=debug,
                                using_closed_models=getattr(self, '_using_closed_models', False)
                            )
                            log_debug(f"Fallback direct execution returned: result={result}, metadata={metadata}")
                        else:
                            # Submit code execution job to the unified manager
                            future = job_manager.submit_code_execution_job(
                                sample=sample,
                                generation=extracted_code,
                                timeout=6,
                                debug=debug,
                                using_closed_models=getattr(self, '_using_closed_models', False)
                            )

                            # Get the result with timeout
                            result, metadata = future.get(timeout=300)  # 5 minute timeout
                            log_debug(f"Job manager returned: result={result}, metadata={metadata}")

                    except Exception as e:
                        log_debug(f"Error during unified job manager execution: {e}")
                        import traceback
                        log_debug(f"Job manager traceback: {traceback.format_exc()}")

                        # Last resort: create and use direct executor
                        log_debug("Creating direct executor as final fallback")
                        try:
                            from fugu.tasks.livecodebench_direct_executor import DirectCodeExecutor
                            direct_executor = DirectCodeExecutor()
                            result, metadata = direct_executor.check_correctness(
                                sample=sample,
                                generation=extracted_code,
                                timeout=6,
                                debug=debug,
                                using_closed_models=getattr(self, '_using_closed_models', False)
                            )
                            log_debug(f"Final fallback execution returned: result={result}, metadata={metadata}")
                        except Exception as final_e:
                            log_debug(f"Final fallback also failed: {final_e}")
                            return 0.0

            except Exception as outer_e:
                log_debug(f"Outer exception in code execution: {outer_e}")
                import traceback
                log_debug(f"Outer traceback: {traceback.format_exc()}")
                return 0.0

            # Log individual test results with the exact format from previous version
            if isinstance(result, list):
                for i, test_result in enumerate(result):
                    if test_result == True:
                        log_debug(f"Test {i + 1}: PASSED")
                    elif test_result == -2:
                        log_debug(f"Test {i + 1}: WRONG ANSWER")
                    elif test_result == -3:
                        log_debug(f"Test {i + 1}: TIME LIMIT EXCEEDED")
                    elif test_result == -4:
                        log_debug(f"Test {i + 1}: RUNTIME ERROR")
                    elif test_result == -5:
                        log_debug(f"Test {i + 1}: TEST RUNNER ERROR")
                    else:
                        log_debug(f"Test {i + 1}: FAILED (code: {test_result})")

            # Log metadata if available
            if metadata:
                log_debug(f"Metadata: {metadata}")
                if 'error_message' in metadata:
                    log_debug(f"Error message: {metadata['error_message']}")
                if 'error' in metadata:
                    log_debug(f"Error details: {metadata['error']}")

            # Calculate pass rate
            if isinstance(result, list) and result:
                # Count only True values (successful tests)
                passed_tests = sum(1 for r in result if r == True)
                total_tests = len(result)

                # Only give reward if ALL test cases pass
                reward = 1.0 if passed_tests == total_tests and total_tests > 0 else 0.0

                log_debug(f"=== FINAL RESULTS ===")
                log_debug(f"Individual test results: {result}")
                log_debug(f"Tests passed: {passed_tests}/{total_tests}")
                log_debug(f"Final reward: {reward}")
                log_debug(f"=== END RESULTS ===")

                return reward
            else:
                log_debug(f"Invalid result format: {result}")
                return 0.0

        except Exception as e:
            log_debug(f"Error in unified LiveCodeBench evaluation: {e}")
            import traceback
            log_debug(f"Traceback: {traceback.format_exc()}")
            return 0.0
        finally:
            if debug:
                log_debug("=== LiveCodeBench evaluation completed ===")

    def set_using_closed_models(self, using_closed_models: bool):
        """Set flag indicating whether we're using closed models (for timeout handling)."""
        self._using_closed_models = using_closed_models


# Legacy functions for backward compatibility - now use unified approach
def codegen_metrics(
        samples_list,
        generations_list,
        k_list=[1, 5, 10, 20, 40, 50, 75, 100, 125, 150, 200, 500, 1000],
        num_process_evaluate=16,
        timeout=6,
        debug=False,
        using_closed_models=False,
):
    """
    Legacy function that now uses the unified job manager approach.
    Maintains backward compatibility with existing code.
    """
    try:
        from collections import defaultdict
        import json
        from fugu.tasks.livecodebench_utils import compute_metrics_from_results

        # Get the unified job manager
        job_manager = get_job_manager()

        # Prepare jobs for batch submission
        futures = []
        remap_index = []

        for idx, (sample, generation_list) in enumerate(zip(samples_list, generations_list)):
            assert isinstance(generation_list, list), generations_list[0]
            for generation in generation_list:
                assert isinstance(generation, str), generations_list[0]

                # Submit to unified job manager
                future = job_manager.submit_code_execution_job(
                    sample=sample,
                    generation=generation,
                    timeout=timeout,
                    debug=debug,
                    using_closed_models=using_closed_models
                )
                futures.append(future)
                remap_index.append(idx)

        # Collect results
        results = defaultdict(list)
        metadatas = defaultdict(list)

        print(f"Evaluating {len(futures)} code generations using unified job manager...")

        for future_idx, future in enumerate(futures):
            try:
                result, metadata = future.get(timeout=300)  # 5 min timeout

                # Process result to match expected format
                if isinstance(result, list):
                    processed_result = []
                    for r in result:
                        if isinstance(r, bool):
                            processed_result.append(r)
                        elif r == True:
                            processed_result.append(True)
                        else:
                            processed_result.append(False)  # Any error code becomes False
                    result = processed_result
                else:
                    result = [False]  # Failed execution

                original_idx = remap_index[future_idx]
                results[original_idx].append(result)
                metadatas[original_idx].append(metadata)

            except Exception as e:
                if debug:
                    print(f"Error evaluating future {future_idx}: {e}")

                # Add failed result
                original_idx = remap_index[future_idx]
                results[original_idx].append([False])
                metadatas[original_idx].append({"error": str(e), "error_code": -5})

        # Compute metrics
        metrics = compute_metrics_from_results(results, k_list=k_list)

        # Format metadata for backward compatibility
        final_metadata = []
        for key in sorted(list(metadatas.keys())):
            final_metadata.append(metadatas[key])

        for i in range(len(final_metadata)):
            if type(final_metadata[i]) is not list:
                final_metadata[i] = [json.dumps(final_metadata[i])]
            else:
                final_metadata[i] = [json.dumps(x) for x in final_metadata[i]]

        return [metrics, results, final_metadata]

    except Exception as e:
        if debug:
            print(f"Error in unified codegen_metrics: {e}")
            import traceback
            traceback.print_exc()

        # Return safe fallback
        fallback_metrics = {"pass@1": 0.0}
        fallback_results = defaultdict(list)
        fallback_metadata = []

        return [fallback_metrics, fallback_results, fallback_metadata]


if __name__ == "__main__":
    # Example usage for debugging
    llms = ["claude-3-7-sonnet-20250219", "gpt-4o-mini"]
    task = LiveCodeBenchTask(llm_names=llms, max_turns=3, max_tokens=4096, debug=True)
    obs = task.reset(split="test", task_id=2)
    done = False

    while not done:
        # action = np.random.rand(len(llms))
        action = np.array([0, 1])
        obs, reward, done, _, _ = task.step(action)

    print("Final response:", task.response)
    print("Reward:", reward)