from .constraint_analyzer import evaluate_if_reward_multi
import ast
import json


def local_serve(request):
    labels = request.get("labels")
    item = None
    if isinstance(labels, dict):
        item = labels
    elif labels is not None:
        try:
            item = json.loads(labels)
        except Exception:
            try:
                item = ast.literal_eval(labels)
            except Exception:
                item = None

    if not isinstance(item, dict):
        return {"result": [0]}

    checkers = item.get("checkers", [])
    functions = item.get("functions", [])
    if not isinstance(checkers, list) or not isinstance(functions, list):
        return {"result": [0]}
    skip_rules = bool(request.get("skip_rules", False))
    result = evaluate_if_reward_multi(
        request["instruction"],
        request["answers"],
        checkers,
        functions,
        skip_rules=skip_rules,
    )
    return {"result": result["overall"]}
