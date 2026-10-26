"""Manual, bounded HTTP benchmark. Each request may incur GPU charges."""
import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--url", required=True, help="Modal API gateway URL ending in /generate")
p.add_argument("--count", type=int, default=3)
p.add_argument("--steps", type=int, default=20)
args = p.parse_args()
if not (1 <= args.count <= 10):
    p.error("--count must be from 1 to 10 to limit accidental GPU usage")
api_key = os.environ.get("API_KEY")
if not api_key:
    p.error("Export API_KEY first")

payload = json.dumps({
    "prompt": "a ninja character with blue armor and spiky hair",
    "steps": args.steps,
    "seed": 42,
    "width": 256,
    "height": 256,
}).encode("utf-8")

rows = []
for i in range(args.count):
    req = urllib.request.Request(
        args.url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            image = resp.read()
            status = resp.status
            inference_ms = resp.headers.get("X-Inference-Ms", "")
            vram_gib = resp.headers.get("X-Peak-VRAM-GiB", "")
    except urllib.error.HTTPError as exc:
        status, image, inference_ms, vram_gib = exc.code, b"", "", ""
    elapsed_ms = (time.perf_counter() - started) * 1000
    rows.append({"request": i + 1, "status": status,
                 "client_ms": round(elapsed_ms, 1),
                 "gpu_inference_ms": inference_ms,
                 "peak_vram_gib": vram_gib,
                 "bytes": len(image)})
    print(rows[-1])

with open("benchmark_results.csv", "w", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print("Saved benchmark_results.csv. First request may include GPU cold-start time.")
