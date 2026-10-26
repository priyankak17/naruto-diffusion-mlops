"""Metadata-only Hugging Face checkpoint inventory. Does not download model weights."""
import json
import urllib.parse
import urllib.request

REPOS = (
    "pynk17/sd-naruto-fp32-baseline",
    "pynk17/sd-naruto-fp16",
    "pynk17/sd-naruto-fp16-grad-checkpointing",
)

for repo in REPOS:
    print(f"\n{repo}\n" + "=" * len(repo))
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            tree = json.load(response)
        files = [x for x in tree if x.get("type") == "file"]
        for f in files:
            path = f.get("path", "")
            if path.endswith((".json", ".safetensors", ".bin", ".pt", ".ckpt")):
                print(f"  {path} ({f.get('size', '?')} bytes)")
        model_index_url = f"https://huggingface.co/{repo}/resolve/main/model_index.json"
        try:
            with urllib.request.urlopen(model_index_url, timeout=30) as response:
                index = json.load(response)
            print("  pipeline:", index.get("_class_name"))
        except Exception:
            print("  No accessible model_index.json; inspect the component layout")
    except Exception as exc:
        print(f"  Could not retrieve repository inventory: {exc}")
