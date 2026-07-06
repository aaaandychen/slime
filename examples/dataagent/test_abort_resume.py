"""Integration test: abort → resume with a real SGLang engine.

Starts slime_api server, sends a long-generation request, triggers
pause_generation on the SGLang engine, waits 10s, continues, and
verifies the handler retries correctly and the sample is complete.

Requires a running SGLang engine.  Usage:

  SGLANG_ENGINE_URL=http://127.0.0.1:30000 \\
  HF_CHECKPOINT=/path/to/model \\
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
SGLANG_URL = os.environ.get("SGLANG_ENGINE_URL", "http://127.0.0.1:30000")
HF_CHECKPOINT = os.environ.get("HF_CHECKPOINT", "")
API_PORT = int(os.environ.get("API_PORT", "19876"))

if not HF_CHECKPOINT:
    print("ERROR: HF_CHECKPOINT must point to a valid model directory.")
    print("  HF_CHECKPOINT=/path/to/Qwen3-4B python examples/dataagent/test_abort_resume.py")
    sys.exit(1)


# ── helpers ────────────────────────────────────────────────────────────
def pause_engine():
    """POST /pause_generation to the SGLang engine."""
    resp = httpx.post(f"{SGLANG_URL}/pause_generation", json={})
    print(f"  pause_generation → {resp.status_code}")
    resp.raise_for_status()


def continue_engine():
    """POST /continue_generation to the SGLang engine."""
    resp = httpx.post(f"{SGLANG_URL}/continue_generation", json={})
    print(f"  continue_generation → {resp.status_code}")
    resp.raise_for_status()


async def send_chat_request(thread_id: str, port: int) -> dict:
    """Simulate DataAgent calling the slime_api endpoint."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(120)) as client:
        resp = await client.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            headers={
                "X-DataAgent-Thread-Id": thread_id,
                "Content-Type": "application/json",
            },
            json={
                "model": "slime-rl",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Write a 500-word essay on the importance of testing in software engineering."},
                ],
                "max_tokens": 512,
            },
        )
        return resp.json()


# ── main ──────────────────────────────────────────────────────────────
async def main():
    # 1. Build fake args pointing at the real SGLang engine.
    sglang_host, sglang_port_str = SGLANG_URL.replace("http://", "").split(":")
    sglang_port = int(sglang_port_str)

    args = MagicMock()
    args.slime_api_host = "127.0.0.1"
    args.slime_api_port = API_PORT
    args.sglang_router_ip = sglang_host
    args.sglang_router_port = sglang_port
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
        from slime.utils.types import Sample

        print(f"=== Using SGLang at {SGLANG_URL} ===")
        print(f"=== Model: {HF_CHECKPOINT} ===")
        print()

        # 2. Start server + register sample.
        slime_api.ensure_server(args)
        await asyncio.sleep(1.0)  # let tokenizer load + server start

        sample = Sample()
        sample.prompt = "Write a 500-word essay"
        sample.index = 0
        sample.tokens = []
        sample.loss_mask = []
        sample.rollout_log_probs = []

        thread_id = "test-abort-001"
        slime_api._thread_samples[thread_id] = sample

        # 3. Start generation.
        print("=== [1] Sending chat request ===")
        task = asyncio.create_task(send_chat_request(thread_id, API_PORT))

        # Give SGLang time to start generating.
        await asyncio.sleep(3.0)

        # 4. Trigger pause → 10s wait → continue.
        print("=== [2] pause_generation ===")
        pause_engine()

        print("=== [3] Waiting 10s (simulate weight sync) ===")
        await asyncio.sleep(10.0)

        print("=== [4] continue_generation ===")
        continue_engine()

        print("=== [5] notify_resume() ===")
        slime_api.notify_resume()

        # 5. Wait for handler to finish.
        print("  Waiting for handler to complete...")
        result = await asyncio.wait_for(task, timeout=60.0)
        finish = result["choices"][0]["finish_reason"]
        content = result["choices"][0]["message"]["content"]
        print(f"  finish_reason = {finish}")
        print(f"  content (first 200 chars) = {content[:200]}")

        slime_api._thread_samples.pop(thread_id, None)

        # 6. Verify sample.
        print()
        print("=== Results ===")
        print(f"  sample.status          = {sample.status}")
        print(f"  len(sample.tokens)     = {len(sample.tokens or [])}")
        print(f"  len(sample.loss_mask)  = {len(sample.loss_mask or [])}")
        print(f"  response_length        = {sample.response_length}")

        if sample.loss_mask:
            prompt_zeros = sum(1 for m in sample.loss_mask if m == 0)
            response_ones = sum(1 for m in sample.loss_mask if m == 1)
            print(f"  loss_mask zeros (prompt) = {prompt_zeros}")
            print(f"  loss_mask ones  (output) = {response_ones}")

        assert len(sample.tokens or []) > 0, "Sample tokens empty"
        assert sample.response_length > 0, "No response tokens"
        assert response_ones == sample.response_length, \
            f"response_length ({sample.response_length}) != loss_mask ones ({response_ones})"
        assert len(sample.rollout_log_probs or []) == len(sample.loss_mask or []), \
            "rollout_log_probs / loss_mask length mismatch"

        total_trainable = response_ones
        print(f"  total trainable tokens   = {total_trainable}")
        print()
        print("All checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
