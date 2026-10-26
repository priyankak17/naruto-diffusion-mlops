"""Cost-conscious Modal deployment for pynk17/sd-naruto-fp32-baseline.

Based on Modal's Stable Diffusion serving example, adapted to Stable Diffusion v1.5.
This code is statically checked but not GPU-executed in the authoring environment.
"""

import modal
import time 

APP_NAME = "pynk17-naruto-inference"
MODEL_ID = "pynk17/sd-naruto-fp32-baseline"
MODEL_REVISION = "f32002b0aad5378174e3db49d47143cec9f79d95"
TOKENIZER_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
CACHE_DIR = "/model-cache"

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name("pynk17-naruto-hf-cache", create_if_missing=True)

# Dependency pins follow the published Modal example's tested-generation era.
# After the first successful smoke test, update them together and re-run tests.
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.5.1",
        "diffusers==0.31.0",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "huggingface-hub==0.36.0",
        "safetensors==0.4.5",
        "Pillow==11.0.0",
        "numpy<2",
    )
    .env({"HF_HUB_CACHE": CACHE_DIR})
)
api_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "fastapi[standard]==0.115.4", "pydantic==2.9.2"
)


@app.cls(
    image=gpu_image,
    gpu="T4",                       # 16 GB VRAM; validate actual peak memory
    volumes={CACHE_DIR: model_cache},
    timeout=600,
    min_containers=0,             # no permanently warm GPU
    max_containers=1,             # cap concurrent GPU workers
    scaledown_window=60,         # idle GPU containers scale down
)
class Inference:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import StableDiffusionPipeline
        from transformers import CLIPTokenizer
        from diffusers import DPMSolverMultistepScheduler

        # Load the original SD 1.5 tokenizer.
        # Keep your fine-tuned Naruto model weights.
        start = time.perf_counter()

        tokenizer = CLIPTokenizer.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            subfolder="tokenizer",
            revision=TOKENIZER_REVISION
        )

        self.pipe = StableDiffusionPipeline.from_pretrained(
            "pynk17/sd-naruto-fp32-baseline",
            revision=MODEL_REVISION,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            cache_dir="/model-cache",
        ).to("cuda")

        print("Naruto diffusion pipeline loaded successfully")
        

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        print(
                    f"MODEL_LOAD_SECONDS={time.perf_counter() - start:.2f}",
                    flush=True,
                )
        
        self.pipe.set_progress_bar_config(disable=True)
        model_cache.commit()  # persist downloaded HF files across cold starts
        print("Model loaded on GPU", torch.cuda.get_device_name(0))



    @modal.method()
    def run(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted",
        steps: int = 12,
        seed: int = 42,
        width: int = 256,
        height: int = 256,
    ) -> tuple[bytes, float, float]:
        import io
        import time
        import torch

        torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        started = time.perf_counter()
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=7.5,
                width=width,
                height=height,
                generator=generator,
            )
        torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - started) * 1000
        peak_vram_gib = torch.cuda.max_memory_allocated() / 1024**3

        # Retain the pipeline's safety checker. Never publish flagged content.
        if result.nsfw_content_detected and any(result.nsfw_content_detected):
            raise ValueError("Safety checker rejected generated image")

        output = io.BytesIO()
        result.images[0].save(output, format="PNG")
        return output.getvalue(), inference_ms, peak_vram_gib


@app.function(
    image=api_image,
    secrets=[modal.Secret.from_name("pynk17-naruto-api-key")],
    timeout=600,                   # allow an initial cold GPU model download
    min_containers=0,
    max_containers=1,
    scaledown_window=300,
)
@modal.asgi_app()
def web_app():
    import logging
    import os
    import secrets
    from typing import Literal

    from fastapi import FastAPI, Header, HTTPException
    from fastapi.responses import Response
    from pydantic import BaseModel, Field

    api = FastAPI(title="pynk17 Naruto Diffusion API", version="0.1.0")

    class GenerationRequest(BaseModel):
        prompt: str = Field(min_length=1, max_length=256)
        negative_prompt: str = Field(default="blurry, low quality, distorted", max_length=256)
        steps: int = Field(default=12, ge=10, le=40)
        seed: int = Field(default=42, ge=0, le=2**32 - 1)
        width: Literal[256, 512] = 256
        height: Literal[256, 512] = 256

    @api.get("/health")
    def health():
        # This only confirms API process health, not GPU readiness.
        return {"status": "ok", "model": MODEL_ID, "gpu_ready": "not_checked"}

    @api.post("/generate", responses={200: {"content": {"image/png": {}}}})
    def generate(
        request: GenerationRequest,
        x_api_key: str | None = Header(default=None),
    ):
        expected_key = os.environ["API_KEY"]
        if not x_api_key or not secrets.compare_digest(x_api_key, expected_key):
            raise HTTPException(status_code=401, detail="Invalid API key")

        try:
            png, inference_ms, peak_vram_gib = Inference().run.remote(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                steps=request.steps,
                seed=request.seed,
                width=request.width,
                height=request.height,
            )
        except Exception as exc:
            logging.exception("GPU inference failed")
            raise HTTPException(
                status_code=502,
                detail="Inference worker failed; inspect Modal logs",
            ) from exc

        return Response(
            content=png,
            media_type="image/png",
            headers={
                "X-Inference-Ms": f"{inference_ms:.1f}",
                "X-Peak-VRAM-GiB": f"{peak_vram_gib:.2f}",
                "X-Model-Id": MODEL_ID,
                "Cache-Control": "no-store",
            },
        )

    return api


@app.local_entrypoint()
def main(
    prompt: str = "a ninja character with blue armor and spiky hair",
    steps: int = 12,
    seed: int = 42,
):
    # An inexpensive *single* functional smoke test before public deployment.
    from pathlib import Path

    png, inference_ms, peak_vram_gib = Inference().run.remote(
        prompt=prompt, steps=steps, seed=seed
    )
    path = Path("sample.png")
    path.write_bytes(png)
    print(f"Wrote {path} | GPU time: {inference_ms:.1f} ms | VRAM: {peak_vram_gib:.2f} GiB")
