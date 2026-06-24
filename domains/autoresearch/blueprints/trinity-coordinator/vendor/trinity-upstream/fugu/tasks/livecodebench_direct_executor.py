"""
Direct code execution for LiveCodeBench without subprocesses.
Used by the unified job manager to eliminate nested multiprocessing.
"""

import json
import signal
import sys
import time
import ast
import threading
from io import StringIO
from types import ModuleType
from typing import Tuple, List, Dict, Any
from unittest.mock import patch, mock_open

# Import existing utilities
from fugu.tasks.livecodebench_utils import (
    import_string, timeout_handler, TimeoutException, Capturing,
    MockStdinWithBuffer, clean_if_name, make_function, call_method,
    get_function, validate_code_safety, truncatefn, get_stripped_lines,
    convert_line_to_decimals, CODE_TYPE
)


class DirectCodeExecutor:
    """
    Direct code executor that runs in the same process.
    No subprocess creation - uses signal alarms for timeouts.
    """

    def __init__(self):
        self.active_timeouts = set()

    def check_correctness(self, sample: Dict, generation: str, timeout: int,
                          debug: bool = False, using_closed_models: bool = False) -> Tuple[List, Dict]:
        """
        Check correctness of generated code using direct execution.

        Args:
            sample: Problem sample with input/output data
            generation: Generated code to test
            timeout: Timeout per test case in seconds
            debug: Enable debug output
            using_closed_models: Whether we're using API models (affects timeout handling)

        Returns:
            Tuple of (results_list, metadata_dict)
        """
        # Save current signal handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)

        try:
            # Parse test cases
            try:
                in_outs = json.loads(sample["input_output"])
            except ValueError as e:
                return [[-1]], {"error": f"Invalid input_output: {e}", "error_code": -5}

            # Determine test type
            if in_outs.get("fn_name") is None:
                test_type = CODE_TYPE.standard_input
                method_name = None
            else:
                test_type = CODE_TYPE.call_based
                method_name = in_outs["fn_name"]

            # Set global timeout only if not using closed models
            if not using_closed_models:
                total_timeout = (timeout + 1) * len(in_outs["inputs"]) + 5
                signal.alarm(total_timeout)

            try:
                if test_type == CODE_TYPE.call_based:
                    results, metadata = self._grade_call_based_direct(
                        code=generation,
                        all_inputs=in_outs["inputs"],
                        all_outputs=in_outs["outputs"],
                        fn_name=method_name,
                        timeout=timeout,
                        using_closed_models=using_closed_models
                    )
                else:
                    results, metadata = self._grade_stdio_direct(
                        code=generation,
                        all_inputs=in_outs["inputs"],
                        all_outputs=in_outs["outputs"],
                        timeout=timeout,
                        using_closed_models=using_closed_models
                    )

                return results, metadata

            except TimeoutException:
                return [[-1 for _ in range(len(in_outs["inputs"]))]], {
                    "error": "Global timeout",
                    "error_code": -3,
                    "error_message": "Time Limit Exceeded (Global)"
                }
            except Exception as e:
                error_str = str(e)
                if "not enough values to unpack" in error_str or "too many values to unpack" in error_str:
                    return [[-1 for _ in range(len(in_outs["inputs"]))]], {
                        "error": repr(e),
                        "error_code": -6,
                        "error_message": "TestRunnerError (Unpacking)"
                    }
                else:
                    return [[-1 for _ in range(len(in_outs["inputs"]))]], {
                        "error": repr(e),
                        "error_code": -5,
                        "error_message": "TestRunnerError"
                    }
            finally:
                signal.alarm(0)

        finally:
            # Restore original handler
            signal.signal(signal.SIGALRM, old_handler)

    def _compile_code_direct(self, code: str, timeout: int, using_closed_models: bool = False) -> ModuleType:
        """Compile code directly in this process."""
        try:
            # Validate code safety
            is_safe, safety_error = validate_code_safety(code)
            if not is_safe:
                return None

            # Only set timeout for local models
            if not using_closed_models:
                signal.alarm(timeout)

            tmp_sol = ModuleType("tmp_sol", "")

            # Track thread exceptions
            thread_exceptions = []
            original_excepthook = threading.excepthook

            def capture_thread_exception(args):
                thread_exceptions.append({
                    'thread': args.thread,
                    'exception': args.exc_value,
                    'traceback': args.exc_traceback
                })

            threading.excepthook = capture_thread_exception

            try:
                exec(code, tmp_sol.__dict__)
                time.sleep(0.1)  # Brief pause for thread detection

                if thread_exceptions:
                    return None

                if "class Solution" in code:
                    compiled_sol = tmp_sol.Solution()
                else:
                    compiled_sol = tmp_sol

                return compiled_sol

            finally:
                threading.excepthook = original_excepthook

        except Exception as e:
            error_msg = str(e)
            if "not enough values to unpack" in error_msg or "too many values to unpack" in error_msg:
                print(f"Compilation unpacking error: {error_msg}")
            return None
        finally:
            signal.alarm(0)

    def _grade_call_based_direct(self, code: str, all_inputs: List, all_outputs: List,
                                 fn_name: str, timeout: int, using_closed_models: bool = False) -> Tuple[List, Dict]:
        """Grade call-based problems directly."""
        code = import_string + "\n\n" + code
        compiled_sol = self._compile_code_direct(code, timeout, using_closed_models)

        if compiled_sol is None:
            return [-4], {"error_code": -4, "error_message": "Compilation failed"}

        method = get_function(compiled_sol, fn_name)
        if method is None:
            return [-4], {"error_code": -4, "error_message": f"Function {fn_name} not found"}

        # Parse inputs and outputs
        all_inputs = [[json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs]
        all_outputs = [json.loads(output) for output in all_outputs]

        all_results = []
        total_execution = 0

        for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
            if not using_closed_models:
                signal.alarm(timeout)

            try:
                start = time.time()
                prediction = method(*gt_inp)
                total_execution += time.time() - start
                signal.alarm(0)

                # Handle tuple vs list
                if isinstance(prediction, tuple):
                    prediction = list(prediction)

                tmp_result = prediction == gt_out
                all_results.append(tmp_result)

                if not tmp_result:
                    return all_results, {
                        "output": truncatefn(prediction),
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                        "error_code": -2,
                        "error_message": "Wrong Answer",
                    }

            except Exception as e:
                signal.alarm(0)
                error_str = repr(e).lower()
                if "timeoutexception" in error_str:
                    all_results.append(-3)
                    return all_results, {
                        "error": repr(e),
                        "error_code": -3,
                        "error_message": "Time Limit Exceeded",
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                    }
                elif "not enough values to unpack" in error_str or "too many values to unpack" in error_str:
                    all_results.append(-4)
                    return all_results, {
                        "error": repr(e),
                        "error_code": -4,
                        "error_message": "Runtime Error (Unpacking)",
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                    }
                else:
                    all_results.append(-4)
                    return all_results, {
                        "error": repr(e),
                        "error_code": -4,
                        "error_message": "Runtime Error",
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                    }
            finally:
                signal.alarm(0)

        return all_results, {"execution time": total_execution}

    def _grade_stdio_direct(self, code: str, all_inputs: List, all_outputs: List,
                            timeout: int, using_closed_models: bool = False) -> Tuple[List, Dict]:
        """Grade stdio problems directly."""
        code = clean_if_name(code)
        code = make_function(code)

        compiled_sol = self._compile_code_direct(code, timeout, using_closed_models)
        if compiled_sol is None:
            return [-4], {"error_code": -4, "error_message": "Compilation failed"}

        method = get_function(compiled_sol, "wrapped_function")
        if method is None:
            return [-4], {"error_code": -4, "error_message": "Function wrapped_function not found"}

        all_results = []
        total_execution_time = 0

        for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
            if not using_closed_models:
                signal.alarm(timeout)

            with Capturing() as captured_output:
                try:
                    start = time.time()
                    call_method(method, gt_inp)
                    total_execution_time += time.time() - start
                    signal.alarm(0)
                except Exception as e:
                    signal.alarm(0)
                    error_str = repr(e).lower()
                    if "timeoutexception" in error_str:
                        all_results.append(-3)
                        return all_results, {
                            "error": repr(e),
                            "error_code": -3,
                            "error_message": "Time Limit Exceeded",
                            "inputs": truncatefn(gt_inp),
                            "expected": truncatefn(gt_out),
                        }
                    elif "not enough values to unpack" in error_str or "too many values to unpack" in error_str:
                        all_results.append(-4)
                        return all_results, {
                            "error": repr(e),
                            "error_code": -4,
                            "error_message": "Runtime Error (Unpacking)",
                            "inputs": truncatefn(gt_inp),
                            "expected": truncatefn(gt_out),
                        }
                    else:
                        all_results.append(-4)
                        return all_results, {
                            "error": repr(e),
                            "error_code": -4,
                            "error_message": "Runtime Error",
                            "inputs": truncatefn(gt_inp),
                            "expected": truncatefn(gt_out),
                        }
                finally:
                    signal.alarm(0)

            # Process output
            prediction = captured_output[0]
            stripped_prediction_lines = get_stripped_lines(prediction)
            stripped_gt_out_lines = get_stripped_lines(gt_out)

            # Compare outputs (same logic as original)
            WA_send_args = {
                "output": truncatefn(prediction),
                "inputs": truncatefn(gt_inp),
                "expected": truncatefn(gt_out),
                "error_code": -2,
            }

            if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
                all_results.append(-2)
                WA_send_args["error_message"] = "Wrong answer: mismatched output length"
                return all_results, WA_send_args

            for output_line_idx, (stripped_prediction_line, stripped_gt_out_line) in enumerate(
                    zip(stripped_prediction_lines, stripped_gt_out_lines)
            ):
                WA_send_args["error_message"] = (
                    f"Wrong answer at {output_line_idx=}: "
                    f"{truncatefn(stripped_prediction_line)} != {truncatefn(stripped_gt_out_line)}"
                )

                if stripped_prediction_line == stripped_gt_out_line:
                    continue

                # Try decimal comparison
                success, decimal_prediction_line = convert_line_to_decimals(stripped_prediction_line)
                if not success:
                    all_results.append(-2)
                    return all_results, WA_send_args

                success, decimal_gtout_line = convert_line_to_decimals(stripped_gt_out_line)
                if not success:
                    all_results.append(-2)
                    return all_results, WA_send_args

                if decimal_prediction_line == decimal_gtout_line:
                    continue

                all_results.append(-2)
                return all_results, WA_send_args

            all_results.append(True)

        return all_results, {"execution time": total_execution_time}