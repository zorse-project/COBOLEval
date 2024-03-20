# Adapted from: https://github.com/openai/human-eval/blob/master/human_eval/evaluate_functional_correctness.py

import sys

import fire
from evaluation import evaluate_functional_correctness

from data import HUMAN_EVAL


def entrypoint():
    all_results = []
    run_folders = ["gpt-4"]  # edit
    for folder in run_folders:
        all_results.append(eval(f"preds/{folder}", "1"))

    for res, folder in zip(all_results, run_folders):
        print(f"{folder}: {res}")


def eval(
    pred_path: str,
    k: str = "1,10,100",
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k))
    results = evaluate_functional_correctness(pred_path, k, problem_file)
    print(results)
    return results


def main():
    fire.Fire(entrypoint)


sys.exit(main())
