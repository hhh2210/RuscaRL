import ast
import os
import sys
import traceback

# Ensure local_server modules (e.g., llm_call) are importable even if the dataset
# function strings don't append the path.
_LOCAL_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
if _LOCAL_SERVER_DIR not in sys.path:
    sys.path.append(_LOCAL_SERVER_DIR)

def extract_imports(code):
    tree = ast.parse(code)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            for alias in node.names:
                imports.append(f"from {module} import {alias.name}")
    return imports

def execute_code(instruction, response, function=None):
    global_context = {}
    local_vars = {"response": response}
    
    try:
        import_statements = extract_imports(function)
        for statement in import_statements:
            exec(statement, global_context)

        # 执行传入的 function 代码
        exec(function, global_context, local_vars)

        # 检查 check_following 是否存在且可调用
        if 'check_following' in local_vars and callable(local_vars['check_following']):
            result = local_vars['check_following']("instruction", response)
            return result, None
        else:
            return None, "Function 'check_following' is missing or not callable"

    except Exception as e:
        error_message = f"Execution error: {e}\n{traceback.format_exc()}"
        print(function + "\n" + error_message)
        return None, error_message
