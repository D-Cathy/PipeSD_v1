"""Explicit wrapper for the official HumanEval evaluator.

The evaluator executes generated Python code. It is deliberately gated behind a
command-line acknowledgement and should be run only in an isolated environment.
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run official HumanEval functional correctness evaluation")
    parser.add_argument("samples", help="JSONL containing task_id and completion")
    parser.add_argument(
        "--allow-code-execution",
        action="store_true",
        help="Acknowledge that generated and dataset code will be executed",
    )
    parser.add_argument("--problem-file", default=None)
    args = parser.parse_args()
    if not args.allow_code_execution:
        parser.error("Refusing to execute code without --allow-code-execution. Use an isolated environment.")
    command = [sys.executable, "-m", "human_eval.evaluate_functional_correctness", args.samples]
    if args.problem_file:
        command.extend(["--problem_file", args.problem_file])
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
