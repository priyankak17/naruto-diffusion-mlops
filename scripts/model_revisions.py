"""Inspect and resolve Hugging Face model revisions without importing Modal or using a GPU.

Usage:
  python scripts/model_revisions.py inspect
  python scripts/model_revisions.py resolve
"""
import argparse
import ast
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "modal_app.py"
EXPECTED = {
    "fine_tuned": "pynk17/sd-naruto-fp32-baseline",
    "tokenizer": "stable-diffusion-v1-5/stable-diffusion-v1-5",
}


def inspect_source(source=SOURCE):
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "from_pretrained" or not node.args:
            continue
        model = node.args[0]
        revision = next((k.value for k in node.keywords if k.arg == "revision"), None)
        calls.append({
            "repository": model.value if isinstance(model, ast.Constant) and isinstance(model.value, str) else None,
            "revision": revision.value if isinstance(revision, ast.Constant) and isinstance(revision.value, str) else None,
            "line": node.lineno,
        })
    return calls


def resolve_revision(repo):
    url = "https://huggingface.co/api/models/" + urllib.parse.quote(repo, safe="/")
    request = urllib.request.Request(url, headers={"User-Agent": "pynk17-model-revision-audit/1.0"})
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.load(response)
    sha = payload.get("sha")
    if not isinstance(sha, str) or len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha.lower()):
        raise ValueError("Hugging Face did not return a valid 40-character revision SHA")
    return sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["inspect", "resolve"])
    args = parser.parse_args()
    calls = inspect_source()
    if args.action == "inspect":
        print(json.dumps({"loads": calls, "fully_pinned": all(c["revision"] for c in calls)}, indent=2))
        return 0
    try:
        resolved = {name: {"repo": repo, "revision": resolve_revision(repo)} for name, repo in EXPECTED.items()}
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        print(f"Could not resolve Hugging Face revisions: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(resolved, indent=2))
    print("Note: resolve is informational; no deployment files were changed.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
