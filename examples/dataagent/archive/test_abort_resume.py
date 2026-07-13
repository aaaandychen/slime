"""End-to-end abort/resume test: SGLang + DataAgent + Slime API.

Full chain:
  Slime custom_generate → SSE → DataAgent → Slime API → SGLang
                                                         │
                                        pause_generation ←┘ (mid-generation)
                                        continue_generation
                                        notify_resume

Usage:
  export DEEPSEEK_API_KEY=sk-xxx   # for DataAgent model config (Embedding)
  SGLANG_PORT=30000 HF_CHECKPOINT=/path/to/model \\
    python examples/dataagent/test_abort_resume.py
"""

import asyncio
import os
import sys
import time
from unittest.mock import MagicMock, patch

import httpx

sys.path.insert(0, ".")

# ── config ────────────────────────────────────────────────────────────
SGLANG_HOST = os.environ.get("SGLANG_HOST", "127.0.0.1")
SGLANG_PORT = int(os.environ.get("SGLANG_PORT", "30000"))
HF_CHECKPOINT = os.environ.get("HF_CHECKPOINT", "")
API_PORT = int(os.environ.get("API_PORT", "19876"))

DATAAGENT_DIR = os.environ.get("DATAAGENT_DIR", "/mnt/cephfs/chenzhenyang/DataAgent")
DATAAGENT_PORT = int(os.environ.get("DATAAGENT_PORT", "8065"))
DATAAGENT_AGENT_ID = int(os.environ.get("DATAAGENT_AGENT_ID", "20"))
EMBEDDING_PORT = int(os.environ.get("EMBEDDING_PORT", "8765"))
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

SGLANG_URL = f"http://{SGLANG_HOST}:{SGLANG_PORT}"
DATAAGENT_URL = f"http://127.0.0.1:{DATAAGENT_PORT}"

if not HF_CHECKPOINT:
    print("ERROR: HF_CHECKPOINT required.")
    sys.exit(1)


# ── DataAgent helpers ──────────────────────────────────────────────────
def pause_engine():
    resp = httpx.post(f"{SGLANG_URL}/pause_generation", json={})
    print(f"  pause_generation → {resp.status_code}")


def continue_engine():
    resp =httpx.post(f"{SGLANG_URL}/continue_generation", json={})
    print(f"  continue_generation → {resp.status_code}")


def ensure_dataagent_ready():
    """Ensure DataAgent backend + Embedding are up."""
    # Embedding
    import subprocess
    if not _http_ok(f"http://127.0.0.1:{EMBEDDING_PORT}/health"):
        print("  Starting Embedding service...")
        subprocess.Popen(
            [f"{DATAAGENT_DIR}/.venv/bin/python3", f"{DATAAGENT_DIR}/embedding_server.py", str(EMBEDDING_PORT)],
            stdout=open("/tmp/embedding_test.log", "w"),
            stderr=subprocess.STDOUT,
        )
        _wait_url(f"http://127.0.0.1:{EMBEDDING_PORT}/health", "Embedding")

    # Backend
    if not _http_ok(f"{DATAAGENT_URL}/echo/ok"):
        print("  Starting DataAgent backend...")
        subprocess.Popen(
            ["java", "-jar",
             f"{DATAAGENT_DIR}/data-agent-management/target/spring-ai-alibaba-data-agent-management-1.0.0-SNAPSHOT.jar",
             "--spring.profiles.active=h2", f"--server.port={DATAAGENT_PORT}"],
            env={**os.environ, "PATH": f"{DATAAGENT_DIR}/.venv/bin:{os.environ['PATH']}"},
            stdout=open("/tmp/dataagent_test.log", "w"),
            stderr=subprocess.STDOUT,
        )
        _wait_url(f"{DATAAGENT_URL}/echo/ok", "DataAgent backend")


def configure_dataagent():
    """Register Slime API as DataAgent's LLM backend, set up datasource + agent."""
    api = DATAAGENT_URL

    # Model: Slime API as CHAT backend
    print("  Registering Slime API as CHAT model...")
    _curl(f"{api}/api/model-config/add", {
        "provider": "custom",
        "baseUrl": f"http://127.0.0.1:{API_PORT}",
        "apiKey": "",
        "modelName": "slime-rl",
        "modelType": "CHAT",
        "maxTokens": 8192,
        "completionsPath": "/v1/chat/completions",
                "proxyEnabled": False,
    })
    list_resp = _curl_get(f"{api}/api/model-config/list")
    chat_id = _find_model_id(list_resp, "CHAT")
    _curl_post(f"{api}/api/model-config/activate/{chat_id}")
    print(f"  CHAT model id={chat_id} activated")

    # Model: local Embedding
    print("  Registering Embedding model...")
    _curl(f"{api}/api/model-config/add", {
        "provider": "custom",
        "baseUrl": f"http://127.0.0.1:{EMBEDDING_PORT}",
        "apiKey": "",
        "modelName": "bge-small-zh-v1.5",
        "modelType": "EMBEDDING",
        "embeddingsPath": "/v1/embeddings",
                "proxyEnabled": False,
    })
    emb_id = _find_model_id(_curl_get(f"{api}/api/model-config/list"), "EMBEDDING")
    _curl_post(f"{api}/api/model-config/activate/{emb_id}")
    print(f"  EMBEDDING model id={emb_id} activated")

    # Datasource
    print("  Setting up datasource...")
    ds_id = _ensure_datasource(api)

    # Agent
    print("  Setting up agent...")
    agent_id = _ensure_agent(api, ds_id)
    print(f"  Agent id={agent_id} ready")
    return agent_id


def _http_ok(url):
    try:
        httpx.get(url, timeout=httpx.Timeout(5))
        return True
    except Exception:
        return False


def _wait_url(url, desc, max_retries=60):
    for _ in range(max_retries):
        if _http_ok(url):
            return
        time.sleep(2)
    raise RuntimeError(f"{desc} not ready: {url}")


def _curl(url, body):
    resp = httpx.post(url, json=body, timeout=httpx.Timeout(10))
    resp.raise_for_status()


def _curl_get(url):
    resp = httpx.get(url, timeout=httpx.Timeout(10))
    resp.raise_for_status()
    return resp.json()


def _curl_post(url):
    resp = httpx.post(url, timeout=httpx.Timeout(10))
    resp.raise_for_status()
    return resp.json()


def _find_model_id(list_json, model_type):
    import json as _json
    data = list_json.get("data", [])
    for m in data:
        if m.get("modelType") == model_type:
            return m["id"]
    raise RuntimeError(f"No {model_type} model in list")


def _ensure_datasource(api):
    resp = _curl_get(f"{api}/api/datasource")
    items = resp if isinstance(resp, list) else resp.get("data", [])
    for ds in items:
        if ds.get("name") == "demo_sales" and ds.get("status") == "active":
            return ds["id"]
    resp2 = httpx.post(f"{api}/api/datasource", json={
        "name": "demo_sales", "type": "mysql",
        "host": "127.0.0.1", "port": 3306,
        "databaseName": "demo_sales", "username": "root", "password": "",
    }, timeout=httpx.Timeout(10))
    return resp2.json()["id"]


def _ensure_agent(api, ds_id):
    # Always delete and recreate to guarantee clean state.
    resp = httpx.get(f"{api}/api/agent/list", timeout=httpx.Timeout(10))
    items = resp.json() if isinstance(resp.json(), list) else resp.json().get("data", [])
    for a in items:
        if a.get("name") == "Demo":
            httpx.delete(f"{api}/api/agent/{a['id']}", timeout=httpx.Timeout(10))

    resp2 = httpx.post(f"{api}/api/agent", json={
        "name": "Demo", "description": "Test", "category": "demo",
    }, timeout=httpx.Timeout(10))
    aid = resp2.json()["id"]
    _curl_post(f"{api}/api/agent/{aid}/datasources/{ds_id}")
    httpx.post(f"{api}/api/agent/{aid}/datasources/tables", json={
        "datasourceId": ds_id, "tables": ["products", "orders"],
    }, timeout=httpx.Timeout(10))
    _curl_post(f"{api}/api/agent/{aid}/datasources/init")
    _curl_post(f"{api}/api/agent/{aid}/publish")
    return aid


# ── main ──────────────────────────────────────────────────────────────
async def main():
    # 1. Build args.
    args = MagicMock()
    args.slime_api_host = "127.0.0.1"
    args.slime_api_port = API_PORT
    args.sglang_router_ip = SGLANG_HOST
    args.sglang_router_port = SGLANG_PORT
    args.hf_checkpoint = HF_CHECKPOINT
    args.sglang_server_concurrency = 128
    args.rollout_max_response_len = 2048
    args.rollout_temperature = 1.0
    args.rollout_top_p = 1.0
    args.rollout_top_k = 50
    args.rollout_stop = None
    args.rollout_stop_token_ids = None
    args.rollout_skip_special_tokens = False
    args.rollout_seed = 42
    args.n_samples_per_prompt = 1
    args.sglang_dp_size = 1
    args.sglang_enable_deterministic_inference = False

    with patch("slime.utils.http_utils.get_rollout_num_engines", return_value=1):
        from examples.dataagent import slime_api
        from examples.dataagent import custom_generate
        from slime.utils import http_utils
        from slime.utils.types import Sample

        if http_utils._http_client is None:
            http_utils._http_client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=128),
                timeout=httpx.Timeout(None),
                trust_env=False,
            )

        # 2. Ensure DataAgent is configured.
        print("=== [0] Ensuring DataAgent is ready ===")
        ensure_dataagent_ready()
        agent_id = configure_dataagent()

        # 3. Start Slime API server.
        print("=== [1] Starting Slime API server ===")
        slime_api.ensure_server(args)
        for _ in range(30):
            try:
                httpx.get(f"http://127.0.0.1:{API_PORT}/health", timeout=httpx.Timeout(2))
                print("  Slime API ready")
                break
            except Exception:
                await asyncio.sleep(2)
        else:
            raise RuntimeError(f"Slime API did not start on :{API_PORT}")

        # 4. Create sample and call custom_generate (SSE → DataAgent).
        print("=== [2] Calling custom_generate.generate() ===")
        sample = Sample()
        sample.prompt = "各区域销售额排名"
        sample.index = 0
        sample.tokens = []
        sample.loss_mask = []
        sample.rollout_log_probs = []
        sample.metadata = {}
        sample.status = Sample.Status.PENDING

        # Override env vars that custom_generate reads at call time.
        os.environ["DATAAGENT_AGENT_ID"] = str(agent_id)
        os.environ["DATAAGENT_BASE_URL"] = DATAAGENT_URL

        # Start generation in background.
        gen_task = asyncio.create_task(
            custom_generate.generate(args, sample, {})
        )

        # Wait for the first LLM call to begin, then trigger pause.
        await asyncio.sleep(5.0)

        print("=== [3] pause_generation ===")
        pause_engine()

        print("=== [4] Waiting 10s (simulate weight sync) ===")
        await asyncio.sleep(10.0)

        print("=== [5] continue_generation ===")
        continue_engine()

        print("=== [6] notify_resume() ===")
        slime_api.notify_resume()

        print("=== [7] Waiting for generation to complete ===")
        result_sample = await asyncio.wait_for(gen_task, timeout=120.0)

        # 5. Verify.
        print()
        print("=== Results ===")
        print(f"  sample.status          = {result_sample.status}")
        print(f"  reward                 = {result_sample.reward}")
        print(f"  metadata               = {result_sample.metadata}")
        print(f"  len(sample.tokens)     = {len(result_sample.tokens or [])}")
        print(f"  response_length        = {result_sample.response_length}")
        if result_sample.status.value == "failed":
            print(f"  dataagent_error        = {result_sample.metadata.get('dataagent_error', 'N/A')}")

        if result_sample.loss_mask:
            zeros = sum(1 for m in result_sample.loss_mask if m == 0)
            ones = sum(1 for m in result_sample.loss_mask if m == 1)
            print(f"  loss_mask zeros (prompt) = {zeros}")
            print(f"  loss_mask ones  (output) = {ones}")

        assert len(result_sample.tokens or []) > 0, "Sample tokens empty"
        assert result_sample.response_length > 0, "No response tokens"
        print()
        print("All checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
