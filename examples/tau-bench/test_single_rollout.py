#!/usr/bin/env python3
"""
Single rollout test for tau-bench agent training.
Starts a local sglang server, runs one tau-bench task, and prints the full conversation.
Usage:
    source .venv/bin/activate
    python examples/tau-bench/test_single_rollout.py
"""
import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.dirname(__file__))

from tau_bench.envs import get_env
from tau_bench.types import RunConfig
from trainable_agents import InteractionResult, Status, agent_factory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_rollout")

# --- Config ---
HF_CHECKPOINT = os.environ.get("HF_CHECKPOINT", "/mnt/cephfs/chenzhenyang/models/Qwen3-4B-Instruct")
SGLANG_PORT = int(os.environ.get("SGLANG_PORT", "30000"))
GPU_ID = int(os.environ.get("GPU_ID", "0"))
TASK_INDEX = int(os.environ.get("TASK_INDEX", "0"))

TAU_CONFIG = RunConfig(
    env="retail",
    agent_strategy="tool-calling",
    user_model="deepseek-chat",
    user_model_provider="deepseek",
    user_strategy="llm",
    task_split="train",
    model="qwen3-4b",
    model_provider="auto_router",
    temperature=1.0,
)


def start_sglang_server(hf_checkpoint: str, port: int, gpu_id: int) -> subprocess.Popen:
    """Launch sglang server on the specified GPU."""
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", hf_checkpoint,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--tp-size", "1",
        "--mem-fraction-static", "0.7",
        "--cuda-device-index", str(gpu_id),
        "--trust-remote-code",
    ]
    logger.info(f"Starting sglang server: {' '.join(cmd)}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
    return proc


def wait_for_sglang(port: int, timeout: int = 120) -> bool:
    """Wait for sglang server to be ready."""
    import urllib.request
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                logger.info("sglang server is ready")
                return True
        except Exception:
            pass
        time.sleep(2)
    logger.error("sglang server did not become ready in time")
    return False


async def run_single_rollout(port: int, task_index: int) -> InteractionResult:
    """Run a single tau-bench rollout."""
    logger.info(f"Creating tau-bench env: retail, task {task_index}")
    env = get_env(
        env_name=TAU_CONFIG.env,
        user_strategy=TAU_CONFIG.user_strategy,
        user_model=TAU_CONFIG.user_model,
        user_provider=TAU_CONFIG.user_model_provider,
        task_split=TAU_CONFIG.task_split,
        task_index=task_index,
    )

    logger.info(f"Task goal: {env.tasks[task_index].goal}")
    logger.info(f"Tools available: {[t['name'] for t in env.tools_info]}")

    rollout_args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=port,
        hf_checkpoint=HF_CHECKPOINT,
        sglang_server_concurrency=32,
    )
    sampling_params = {
        "temperature": 1.0,
        "max_new_tokens": 1024,
        "top_p": 0.9,
        "top_k": 50,
    }

    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=TAU_CONFIG,
        rollout_args=rollout_args,
        sampling_params=sampling_params,
    )

    result = await agent.asolve(env, rollout_args, sampling_params, task_index)

    return result


def print_result(result: InteractionResult, task_index: int):
    """Pretty-print the rollout result."""
    print("\n" + "=" * 80)
    print(f"  Rollout Result - Task #{task_index}")
    print("=" * 80)
    print(f"  Status : {result.status.value}")
    print(f"  Reward : {result.reward}")
    print(f"  Response length : {result.response_length} tokens")
    print(f"  Loss mask length: {len(result.loss_mask)}")
    if result.info:
        print(f"  Info   : {json.dumps(result.info, indent=2, default=str)[:500]}")
    print("-" * 80)
    print("  Conversation:")
    print("-" * 80)
    for i, msg in enumerate(result.messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        name = msg.get("name", "")
        prefix = f"  [{i}] {role}"
        if name:
            prefix += f" ({name})"
        # Truncate long content for readability
        if len(content) > 500:
            content = content[:500] + "...[truncated]"
        print(f"{prefix}: {content}")
        print()
    print("-" * 80)
    if result.response:
        print(f"  Full agent response text ({len(result.response)} chars):")
        print(result.response[:1000])
    print("=" * 80)


async def main():
    print("\n" + "=" * 80)
    print("  Tau-Bench Single Rollout Test")
    print(f"  Model : {HF_CHECKPOINT}")
    print(f"  GPU   : {GPU_ID}")
    print(f"  Task  : retail train #{TASK_INDEX}")
    print(f"  Port  : {SGLANG_PORT}")
    print("=" * 80 + "\n")

    # 1. Start sglang server
    proc = start_sglang_server(HF_CHECKPOINT, SGLANG_PORT, GPU_ID)
    try:
        if not wait_for_sglang(SGLANG_PORT, timeout=180):
            logger.error("Failed to start sglang server")
            return

        # 2. Run single rollout
        result = await run_single_rollout(SGLANG_PORT, TASK_INDEX)

        # 3. Print result
        print_result(result, TASK_INDEX)

    finally:
        logger.info("Stopping sglang server...")
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
