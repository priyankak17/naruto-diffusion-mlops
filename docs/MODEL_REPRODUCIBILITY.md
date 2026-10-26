# Model reproducibility — revision audit

The GPU-serving dependencies in `modal_app.py` are already pinned to exact versions. However, loading a Hugging Face repository without `revision=` means the model can change independently of the GitHub commit.

## 1. Inspect (offline, GPU-free)

```bash
python scripts/model_revisions.py inspect
```

This lists each `from_pretrained` call and whether it specifies a literal revision. A `null` revision means the load is not pinned. This is an audit, not a blocking CI check yet.

## 2. Resolve current revisions (network, GPU-free)

```bash
python scripts/model_revisions.py resolve
```

Copy the reported 40-character commit SHAs for the fine-tuned model and SD 1.5 tokenizer. The script reads Hugging Face's model metadata; it does **not** download weights or alter the deployed service.

## 3. Apply revisions in a separate reviewed change

Add `revision="<fine-tuned SHA>"` to the fine-tuned model `StableDiffusionPipeline.from_pretrained(...)` call and `revision="<tokenizer SHA>"` to the base-model `CLIPTokenizer.from_pretrained(...)` call.

Test with `python -m pytest -q`, the existing static guardrails, and a controlled deployment/smoke test when your GPU budget allows. Only enforce revision pinning in CI **after** these values have been applied. Keep `min_containers=0`.

## Limits

This script audits literal `from_pretrained` calls, not arbitrary dynamic loaders, and records the default branch tip returned at audit time. It cannot guarantee the particular commit was used for an earlier image-generation request.
