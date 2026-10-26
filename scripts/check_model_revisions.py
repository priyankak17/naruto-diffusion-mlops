"""Static checks for reproducible Hugging Face model loading.

Runs without importing Modal, downloading weights, or starting a GPU.
"""

import ast
import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "modal_app.py"

EXPECTED_REPOSITORIES = {
    "MODEL_ID": "pynk17/sd-naruto-fp32-baseline",
    "MODEL_REVISION": "f32002b0aad5378174e3db49d47143cec9f79d95",
    "TOKENIZER_REVISION": "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
}


def check_model_revisions(source=SOURCE):
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))

    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value

    for name, expected in EXPECTED_REPOSITORIES.items():
        if constants.get(name) != expected:
            raise AssertionError(f"{name} is missing or has an unexpected value")

    for name in ("MODEL_REVISION", "TOKENIZER_REVISION"):
        revision = constants[name]
        assert re.fullmatch(r"[0-9a-f]{40}", revision), f"Invalid SHA: {name}"

    expected_loads = {
        "StableDiffusionPipeline": ("MODEL_ID", "MODEL_REVISION"),
        "CLIPTokenizer": (None, "TOKENIZER_REVISION"),
    }

    for loader, (model_name, revision_name) in expected_loads.items():
        matches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == loader
        ]

        assert len(matches) == 1, f"Expected one {loader}.from_pretrained call"

        call = matches[0]
        revision = next(
            (kw.value for kw in call.keywords if kw.arg == "revision"),
            None,
        )

        assert (
            isinstance(revision, ast.Name)
            and revision.id == revision_name
        ), f"{loader} must use revision={revision_name}"

        if model_name:
            assert (
                call.args
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == model_name
            ), f"{loader} must load {model_name}"

    return True


if __name__ == "__main__":
    check_model_revisions()
    print("PASS: Model and tokenizer revisions are pinned")
