# pynk17 Naruto Diffusion Inference Service

A cost-aware Modal serving adaptation for an existing Hugging Face Stable Diffusion v1.5 model: [`pynk17/sd-naruto-fp32-baseline`](https://huggingface.co/pynk17/sd-naruto-fp32-baseline).

## Verified vs. unverified

The Hugging Face model card identifies the model as an SD v1.5 full pipeline with `model_index.json`, `unet`, `vae`, `text_encoder`, `tokenizer`, `scheduler`, `safety_checker` and a training `checkpoint-50/unet`. The card reports successful evaluation with `StableDiffusionPipeline`. Its FP32 baseline loader is:

```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(
    'pynk17/sd-naruto-fp32-baseline', torch_dtype=torch.float32
).to('cuda')
```

This repository deliberately casts the trained FP32 weights to FP16 *for serving*. GPU viability, actual peak memory, model quality and Modal deployment have **not** been experimentally verified by the authoring assistant. Perform the initial smoke test below.

## Why this differs from Modal's upstream example

The [original Modal text-to-image example](https://modal.com/docs/examples/text_to_image) serves SD3.5 Turbo using `StableDiffusion3Pipeline`, H100, 4 diffusion steps and zero guidance. This project uses your *Stable Diffusion v1.5* weights with `StableDiffusionPipeline`, T4, 20 steps, guidance 7.5 and a default 256×256 output (matching the evaluation resolution documented in your model card). It separates the HTTP gateway (CPU) from the GPU worker, validates request inputs, enforces an API key and exposes response-time metadata.

## Before creating cloud resources

1. Review the Modal [pricing page](https://modal.com/pricing) and [billing settings](https://modal.com/settings). Pricing changes; currently T4 is ~$0.59 per active GPU hour, plus CPU and RAM. Starter advertises $30 monthly included compute credits, subject to account and usage conditions.
2. Configure a **Workspace Budget** with the smallest spending cap you're willing to accept, e.g. $5 or $10, before your first GPU run. This is a cap, not a promise of zero billed usage. Never deploy if entering a payment card is financially unsuitable.
3. The `modal_app.py` configuration uses `gpu='T4'`, `max_containers=1`, `min_containers=0` and `scaledown_window=60`. T4 has 16 GB VRAM; verify actual usage rather than assuming the model fits.
4. Do **not** run the original SD3.5 H100 example unmodified.

## Set up

Requires Python 3.11+, Git and an authenticated Modal account.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade 'modal>=1.3,<2'
modal setup
```

Create a long random API key **locally** and store it in Modal Secrets. Do not commit it into your repo.

```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
# Copy the output, then replace the placeholder below with that random value:
modal secret create pynk17-naruto-api-key API_KEY=YOUR_RANDOM_KEY
```

## Inspect Hugging Face repository inventories (metadata only, free)

Before any GPU execution, list the exact files for all three Naruto experiments without downloading weights:

```bash
python inspect_hf.py
```

This inspects Hugging Face metadata for `model_index.json`, pipeline components and checkpoint weight filenames. It gives an independent file-level check of the FP16 repos which the authoring assistant could not fetch directly.

## First proof: a single GPU-backed image

Only after setting a spending cap:

```bash
modal run modal_app.py --steps 20 --seed 42
```

Expect `sample.png` in the current directory, along with measured GPU inference time and peak memory. You will pay for all billed cloud compute involved in the cold start and image-generation workload, not only the measured inference milliseconds. If the run fails, inspect the logs before attempting again.

To compare to the model card, re-run with 30 steps:

```bash
modal run modal_app.py --steps 30 --seed 42
```

Be aware that serving with FP16 is a new experiment relative to the model card's FP32 reference. Qualitatively compare the resulting image to the expected content. **Do not claim numerical or performance improvements without measured results.**

## Deploy only after the smoke test succeeds

```bash
modal deploy modal_app.py
```

Modal prints a public gateway URL for `web_app`. Open `YOUR_GATEWAY_URL/docs` for the OpenAPI schema and `YOUR_GATEWAY_URL/health` for CPU-gateway liveness. The health route does **not** load the GPU model.

A request (replace URL and key):

```bash
export API_URL='https://YOUR_MODAL_GATEWAY_URL'
export API_KEY='YOUR_RANDOM_KEY'
curl -f -X POST "$API_URL/generate" \
  -H "X-API-Key: $API_KEY" \
  -H 'Content-Type: application/json' \
  --data '{"prompt":"a ninja character with blue armor and spiky hair","steps":20,"seed":42,"width":256,"height":256}' \
  -D headers.txt -o output.png
```

Check `output.png`, `X-Inference-Ms` and `X-Peak-VRAM-GiB` in `headers.txt`. `401` means API key mismatch, `422` means invalid input, and `502` means the inference worker failed. Check logs for details.

## Controlled benchmarking

Run only after the API succeeds and while you have remaining budget. Every request incurs compute:

```bash
export API_KEY="YOUR_RANDOM_KEY"
python benchmark_client.py --url "$API_URL/generate" --count 3 --steps 20
```

This writes `benchmark_results.csv` and reports client latency, GPU inference time, peak GPU memory and response status. Increase to `--count 10` only if budget allows.

## Validation checklist

- [ ] One image generated by the first `modal run` test
- [ ] `/health` returns 200 and does not start the GPU worker
- [ ] `/generate` rejects a missing or wrong `X-API-Key` with 401
- [ ] `/generate` rejects an invalid step count or resolution with 422
- [ ] Authenticated `/generate` returns a valid PNG
- [ ] GPU peak VRAM recorded; no OOM on T4
- [ ] Warm vs cold request times measured separately
- [ ] Workspace usage inspected before/after each session
- [ ] Final README describes costs, limitations and how to stop deployment

## Benchmark plan

Measure ten requests with a fixed prompt, resolution, steps and seed. Report initial request end-to-end latency separately from nine warm responses. Vary steps (10, 20, 30), resolution (256 and 512) and precision (FP32 vs FP16 only if hardware allows), one variable at a time. Record: `client_duration_ms`, `X-Inference-Ms`, `X-Peak-VRAM-GiB`, output status and cost observed on your Modal usage dashboard. Don't present training gradient checkpointing as an inference optimization: it mainly reduces training activation memory; the architecture may be identical at inference.

## Troubleshooting

**HF download/model missing:** inspect Modal logs and verify HF repository permissions and exact components. The model card describes a full pipeline but the authoring environment couldn't fetch its raw files. For a partial repo, use the original base model and override `unet` with `UNet2DConditionModel.from_pretrained(..., subfolder='...')` after verifying checkpoint contents.

**GPU out of memory:** first try 256×256, steps 20, batch size 1 and FP16. If T4 still fails, stop; inspect the actual peak and model allocation. An L4 (24 GB) is a pricier fallback, not the default.

**Very slow first request:** cold starts can include downloading model files and container startup. `model_cache.commit()` persists downloaded HF files for future instances; the container still loads the weights into GPU memory on every cold start.

**API endpoint appears publicly accessible:** the URL and docs are public, but `/generate` requires the secret `X-API-Key`. Do not log or share that secret. Public users can still access `/health` and `/docs`; disable public deployment when not demonstrating it.

## Primary sources

* [Your model and model card](https://huggingface.co/pynk17/sd-naruto-fp32-baseline)
* [Modal's upstream human-written deployment guide](https://modal.com/docs/examples/text_to_image)
* [Modal GPU pricing](https://modal.com/pricing)
* [Modal Volume persistence semantics](https://modal.com/docs/sdk/py/latest/Volume)
* [Modal secrets CLI](https://modal.com/docs/cli/latest/secret)
