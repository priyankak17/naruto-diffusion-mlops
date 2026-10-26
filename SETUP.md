# Test + CI installation
Copy `tests/`, `.github/`, and `requirements-test.txt` into the same directory as your *current working* `modal_app.py`. Do NOT replace your `modal_app.py` with the older project archive copy.

In Git Bash:
```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```
Expected: 12 passed (depending on any subsequent changes to your API contract).

Then commit and push these additions to your GitHub repository:
```bash
git add tests requirements-test.txt .github/workflows/ci.yml
git commit -m "test: add GPU-free API contract tests and CI"
git push
```

GitHub → Actions → `API unit tests` confirms CI status. Neither pytest nor this workflow performs Modal deployment or real GPU inference. No Modal API token needed in GitHub Secrets.

Note: Tests isolate the actual `web_app()` AST so they execute the real HTTP handlers but not Modal decorators or the GPU backend. This is unit/contract coverage, not a real-deployment smoke test.
