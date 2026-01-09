from .constraint_analyzer import evaluate_if_reward_multi
import json


def local_serve(request):
    item = json.loads(request["labels"])
    checkers, functions = item["checkers"], item["functions"]
    skip_rules = bool(request.get("skip_rules", False))
    result = evaluate_if_reward_multi(
        request["instruction"],
        request["answers"],
        checkers,
        functions,
        skip_rules=skip_rules,
    )
    return {"result": result["overall"]}
