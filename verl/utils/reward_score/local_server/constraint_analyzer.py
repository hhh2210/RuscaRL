import concurrent.futures
import time

from .code_executer import execute_code
from .llm_call import llm_score


def mean_func(numbers):
    mean_num = sum(numbers) / len(numbers)
    return mean_num

# def all_func(numbers):
#     if 0 in numbers:
#         return 0
#     return 1


def _evaluate_reward(instruction, response, functions, reduction="mean"):
    def process_function(function):
        result, _ = execute_code(instruction, response, function)
        return 1 if result else 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_function, functions))
    if reduction == "mean":
        return mean_func(results)
    else:
        return results



def evaluate_if_reward_multi(
    instruction,
    answers,
    checker_names=None,
    functions=None,
    skip_rules=False,
    ignore_rules=False,
):
    t_start = time.perf_counter()

    llm_checkers, llm_functions, rule_checkers, rule_functions = [], [], [], []
    for checker, function in zip(checker_names, functions):
        if "[llm]" in checker:
            llm_checkers.append(checker)
            llm_functions.append(function)
        elif "[rule]" in checker:
            rule_checkers.append(checker)
            rule_functions.append(function)
        else:
            pass

    if ignore_rules:
        # Match the "soft-constraint-only" behavior: drop rule-based constraints entirely.
        rule_functions = []
        rule_checkers = []
    elif skip_rules:
        # Re-route rule constraints to the LLM judge instead of executing code functions.
        # This preserves the constraint text while disabling hard checks.
        llm_checkers.extend([c.replace("[rule]", "").strip() for c in rule_checkers])
        llm_functions.extend(rule_functions)
        rule_functions = []
        rule_checkers = []

    t_after_parse = time.perf_counter()

    # single processing
    scores = []
    if len(rule_functions) != 0:
        for answer in answers:
            scores.append(_evaluate_reward(instruction, answer, rule_functions, reduction="mean"))
    else:
        scores = [1] * len(answers)

    t_after_rule = time.perf_counter()

    # quality scores
    q_scores = []
    for answer in answers:
        q_scores.append(llm_score(instruction, answer, llm_checkers))

    t_after_llm = time.perf_counter()

    # Timing summary (only print if llm was actually called)
    if llm_checkers:
        rule_time = t_after_rule - t_after_parse
        llm_time = t_after_llm - t_after_rule
        total_time = t_after_llm - t_start
        print(
            f"  [constraint_analyzer] rule={rule_time:.3f}s, llm={llm_time:.3f}s, "
            f"total={total_time:.3f}s (llm_checkers={len(llm_checkers)}, rule_funcs={len(rule_functions)})"
        )

    # return scores
    return {
        "overall": [_scores * q_scores[i] for i, _scores in enumerate(scores)]
    }
