"""Offline tests for the model revision audit; never load model weights."""
from pathlib import Path
from scripts.model_revisions import inspect_source


def test_detects_unpinned_and_pinned_loads(tmp_path):
    path = tmp_path / "app.py"
    path.write_text(
        'a = Pipe.from_pretrained("owner/model", revision="012345")\n'
        'b = Tokenizer.from_pretrained("owner/base")\n',
        encoding="utf-8",
    )
    assert inspect_source(path) == [
        {"repository": "owner/model", "revision": "012345", "line": 1},
        {"repository": "owner/base", "revision": None, "line": 2},
    ]


def test_project_has_fine_tuned_and_tokenizer_loads():
    calls = inspect_source(Path(__file__).resolve().parents[1] / "modal_app.py")
    assert any(c["repository"] == "pynk17/sd-naruto-fp32-baseline" for c in calls)
    assert any(c["repository"] == "stable-diffusion-v1-5/stable-diffusion-v1-5" for c in calls)
