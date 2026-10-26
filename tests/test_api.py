"""FastAPI contract tests; never import or initialize the Modal GPU runtime.

Extract only web_app() from the deployment module. This permits CI to validate
actual request validation, authentication and response logic without creating
Modal images, loading model weights, or calling GPU functions.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

MODEL_ID = 'pynk17/sd-naruto-fp32-baseline'
TEST_KEY = 'unit-test-key-only'
PNG_BYTES = b'\x89PNG\r\n\x1a\nmock-png-payload'


@pytest.fixture()
def api(monkeypatch):
    monkeypatch.setenv('API_KEY', TEST_KEY)
    source_path = Path(__file__).resolve().parents[1] / 'modal_app.py'
    module_ast = ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
    web_function = next(
        (node for node in module_ast.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'web_app'),
        None,
    )
    assert web_function is not None, 'modal_app.py must define web_app()'
    web_function.decorator_list = []  # Do not execute @modal.asgi_app()

    calls = []

    def fake_remote(**kwargs):
        calls.append(kwargs)
        return PNG_BYTES, 1234.5, 2.75

    fake_inference = lambda: SimpleNamespace(run=SimpleNamespace(remote=fake_remote))
    namespace = {'MODEL_ID': MODEL_ID, 'Inference': fake_inference}
    isolated = ast.Module(body=[web_function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(isolated), str(source_path), 'exec'), namespace)
    return TestClient(namespace['web_app']()), calls


def test_health(api):
    client, calls = api
    res = client.get('/health')
    assert res.status_code == 200
    assert res.json() == {'status': 'ok', 'model': MODEL_ID, 'gpu_ready': 'not_checked'}
    assert not calls


@pytest.mark.parametrize('headers', [{}, {'X-API-Key': 'incorrect-key'}])
def test_missing_or_invalid_key_rejected(api, headers):
    client, calls = api
    res = client.post('/generate', json={'prompt': 'naruto ninja'}, headers=headers)
    assert res.status_code == 401
    assert not calls


@pytest.mark.parametrize('payload', [
    {'prompt': ''},
    {'prompt': 'a' * 257},
    {'prompt': 'ninja', 'steps': 9},
    {'prompt': 'ninja', 'steps': 41},
    {'prompt': 'ninja', 'width': 384},
    {'prompt': 'ninja', 'height': 1024},
    {'prompt': 'ninja', 'seed': -1},
])
def test_invalid_payload_rejected_without_gpu(api, payload):
    client, calls = api
    res = client.post('/generate', json=payload, headers={'X-API-Key': TEST_KEY})
    assert res.status_code == 422
    assert not calls


def test_valid_request_returns_mocked_png_and_headers(api):
    client, calls = api
    res = client.post('/generate', json={
        'prompt': 'ninja with blue armor', 'steps': 12, 'seed': 42,
        'width': 256, 'height': 256,
    }, headers={'X-API-Key': TEST_KEY})
    assert res.status_code == 200
    assert res.headers['content-type'] == 'image/png'
    assert res.content == PNG_BYTES
    assert float(res.headers['x-inference-ms']) == 1234.5
    assert float(res.headers['x-peak-vram-gib']) == 2.75
    assert res.headers['cache-control'] == 'no-store'
    assert len(calls) == 1
    assert calls[0]['prompt'] == 'ninja with blue armor'
    assert calls[0]['steps'] == 12


def test_worker_error_returns_safe_502(api):
    client, calls = api
    # Modify the generated fake to raise via its captured callable namespace.
    # Calls are recorded only by the original mocked worker, so here we replace
    # the reference inside the route's isolated web_app global namespace.
    route = next(r for r in client.app.routes if getattr(r, 'path', '') == '/generate')
    class BrokenWorker:
        def remote(self, **kwargs):
            raise RuntimeError('fake GPU unavailable')
    route.endpoint.__globals__['Inference'] = lambda: SimpleNamespace(run=BrokenWorker())
    res = client.post('/generate', json={'prompt': 'ninja'}, headers={'X-API-Key': TEST_KEY})
    assert res.status_code == 502
    assert res.json()['detail'] == 'Inference worker failed; inspect Modal logs'
