# Diffusion Inference | Production-Oriented ML Serving

**A cost-conscious MLOps project serving a fine-tuned Stable Diffusion 1.5 model through an authenticated API on a serverless NVIDIA T4 GPU.**

[Model on Hugging Face](https://huggingface.co/pynk17/sd-naruto-fp32-baseline) · [Source code](https://github.com/priyankak17/naruto-diffusion-mlops) · [CI runs](https://github.com/priyankak17/naruto-diffusion-mlops/actions)

## Engineering highlights

- **Inference serving:** FastAPI `/generate` returns PNG images from a fine-tuned Stable Diffusion 1.5 pipeline hosted on Modal T4; the `/health` endpoint reports API gateway health.
- **Cost controls:** GPU workers use `min_containers=0` (scale to zero) and `max_containers=1`. A persistent Modal Volume caches Hugging Face artifacts across container lifecycles.
- **API contracts and security:** `X-API-Key` authentication, Pydantic request validation, bounded generation parameters, and controlled worker-failure responses.
- **Reproducibility:** Pinned serving-library versions, immutable Hugging Face model/tokenizer revisions, and a static CI guard checking that revision pins remain in place.
- **Quality gates:** 13 GPU-free pytest checks verified locally; GitHub Actions **API unit tests** and **Repository quality** both passed on commit `3c99d63`.

## Measured results

Measurements from individual authenticated requests at **256 × 256**, **12 diffusion steps**, on Modal T4; these are *observations*, not throughput or p95 claims.

| Metric | Observed |
| --- | ---: |
| GPU inference | **1.14–1.49 s** |
| End-to-end HTTP response | **18.22–32.64 s** |
| Peak allocated GPU memory | **2.73–2.75 GiB** |
| Model load (one measured startup) | **9.09 s** |

**Key lesson:** GPU generation accounts for only part of user-visible latency. Serverless startup, model loading, and other request overhead require separate measurement. The API intentionally scales to zero to avoid continuously running a GPU, trading lower idle cost for cold-start latency.

## Architecture

```text
Client
  │ POST /generate  ·  X-API-Key
  ▼
FastAPI gateway (Modal CPU)
  │ validate request + authenticate
  ▼
Modal Inference worker (NVIDIA T4; scale to zero)
  │ fine-tuned SD 1.5 pipeline, FP16 serving, DPMSolver scheduler
  ├── Hugging Face model + base SD 1.5 tokenizer
  └── Modal Volume for cached model artifacts
  ▼
PNG + X-Inference-Ms + X-Peak-VRAM-GiB
```

## Reproduce API checks without a GPU

Use **Python 3.11**. From the repository root:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
python scripts/check_serving_config.py modal_app.py
python scripts/check_model_revisions.py
```

Testing & Validation: GitHub Actions runs automated API tests using a mocked GPU worker, keeping CI fast and cost-efficient. End-to-end GPU inference was independently validated on Modal T4 with successful PNG generation.

## API example

The deployed endpoint requires a private API key stored in Modal Secrets. To generate an image from a shell where `API_URL` and `API_KEY` are already configured:

```bash
curl -sS -L --max-time 300 \
  "$API_URL/generate" \
  -H "X-API-Key: $API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a ninja character with blue armor","steps":12,"seed":42,"width":256,"height":256}' \
  -o output.png
```

**Health & Deployment:** `GET /health` provides a lightweight API gateway health check. GPU model readiness is validated through inference requests. The deployment uses scale-to-zero to optimize compute costs, with cold-start latency when the GPU worker is inactive.

## Production engineering: implemented vs. next

| Implemented and tested | Planned / not yet claimed |
| --- | --- |
| Modal T4 inference and authenticated PNG response | Structured JSON logging and end-to-end request IDs |
| CPU gateway liveness endpoint and input validation | GPU-aware operational readiness and failure instrumentation |
| Scale-to-zero GPU and persistent model cache | Deployment approval, rollback and release automation |
| Unit tests, static checks and GitHub Actions CI | Expanded reliability testing, cost model and monitoring |
| Immutable model and tokenizer revision configuration | Documented Docker/AWS portability design |

## Design trade-offs

- **Serverless over always-on GPUs:** lower idle spend at the cost of cold starts.
- **FP16 at inference:** serves FP32-trained weights in half precision to reduce inference memory usage; no accuracy-equivalence claim is made.
- **CPU-only CI:** fast feedback on API and configuration changes, while actual GPU behavior requires separate integration verification.

**Performance & Roadmap:** Initial Modal T4 benchmarks measured 1.14–1.49s GPU inference latency and 2.73–2.75 GiB peak VRAM. Next priorities include p95 latency and throughput benchmarking, cost-per-image analysis, and defining production service-level objectives (SLOs).

**Stack:** Python 3.11 · PyTorch · Diffusers · Transformers · FastAPI · Pydantic · Modal · Hugging Face · pytest · Ruff · GitHub Actions.
