"""Check essential serving invariants without importing Modal or starting GPUs.

This is an AST-based guardrail, not a substitute for runtime integration tests.
"""

import ast
import sys
from pathlib import Path


def call_name(node):
    if isinstance(node, ast.Call):
        return call_name(node.func)
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def check(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Inference"]
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "web_app"]
    problems = []
    if len(classes) != 1:
        problems.append("expected exactly one Inference class")
    if len(functions) != 1:
        problems.append("expected exactly one web_app function")
    if classes:
        modal_cls = [d for d in classes[0].decorator_list if isinstance(d, ast.Call) and call_name(d.func) == "cls"]
        if not modal_cls:
            problems.append("Inference is missing a Modal @app.cls decorator")
        else:
            kwargs = {k.arg: k.value for k in modal_cls[0].keywords}
            min_count = kwargs.get("min_containers")
            if not isinstance(min_count, ast.Constant) or min_count.value != 0:
                problems.append("GPU min_containers must be explicitly set to 0")
            gpu = kwargs.get("gpu")
            if not isinstance(gpu, ast.Constant) or gpu.value != "T4":
                problems.append("expected GPU type T4; review changes intentionally")
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1
    print("PASS: GPU scale-to-zero, T4, and expected entry points confirmed")
    return 0


if __name__ == "__main__":
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "modal_app.py")
    sys.exit(check(source))
