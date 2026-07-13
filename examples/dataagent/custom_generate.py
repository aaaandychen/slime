"""Custom generate function bridging Slime to DataAgent via SSE.

Phase 3: fully leverages slime's OpenAIAdapter for LLM interception.
DataAgent calls the adapter's /v1/chat/completions instead of a hand-rolled
proxy; the adapter's TrajectoryManager records every turn with correct
loss_mask and linearises the session into list[Sample] on finish_session.

Architecture (aligned with coding_agent_rl):

    generate(args, sample, sampling_params)
      ├── AdapterService(args)          ← singleton: tokenizer + adapter + aiohttp
      ├── adapter.open_session(tid)     ← register session before DataAgent runs
      ├── _query_dataagent(query, tid)  ← SSE → DataAgent → adapter (auto)
      ├── score(nodes, label)           ← compute reward from node trace
      └── adapter.finish_session(tid, reward=…) → list[Sample]
            (DataAgent linear conversation → 1 Sample; branches → N)

Usage:
  --custom-generate-function-path examples.dataagent.custom_generate.generate
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import httpx

from examples.dataagent.adapter import AdapterService
from examples.dataagent.reward_func import score

logger = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────

DATAAGENT_BASE_URL = os.environ.get("DATAAGENT_BASE_URL", "http://localhost:8065")
DATAAGENT_TIMEOUT = int(os.environ.get("DATAAGENT_TIMEOUT", "600"))
# Wall-clock guard for the whole rollout (boot + SSE + reward).
ROLLOUT_GUARD_SEC = int(os.environ.get("DATAAGENT_ROLLOUT_GUARD_SEC", "0") or 0)


# ── SSE consumer ──────────────────────────────────────────────────────

async def _query_dataagent(query: str, thread_id: str) -> dict[str, Any]:
    """Send query to DataAgent, collect SSE events, return structured result.

    DataAgent internally makes LLM calls to the adapter's
    ``/v1/chat/completions``.  Each call carries ``X-DataAgent-Thread-Id``
    → adapter routes to the correct session and the TrajectoryManager
    records the turn.
    """
    url = f"{DATAAGENT_BASE_URL}/api/stream/search"
    params = {
        "agentId": int(os.environ.get("DATAAGENT_AGENT_ID", "20")),
        "query": query,
        "threadId": thread_id,
    }
    headers = {"Accept": "text/event-stream"}

    nodes: dict[str, dict[str, Any]] = {}
    node_order: list[str] = []
    error_msg: str | None = None

    async with httpx.AsyncClient(timeout=httpx.Timeout(DATAAGENT_TIMEOUT)) as client:
        async with client.stream("GET", url, params=params, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload_str = line[5:].strip()
                if not payload_str or payload_str == "[DONE]":
                    continue
                try:
                    d = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue

                if d.get("error"):
                    error_msg = d.get("text", "unknown")
                    break
                if d.get("complete"):
                    break

                node_name = d.get("nodeName", "unknown")
                text_type = d.get("textType", "TEXT")
                text_chunk = d.get("text", "")
                key = f"{node_name}|{text_type}"

                if key not in nodes:
                    nodes[key] = {"node": node_name, "type": text_type, "text": ""}
                    node_order.append(key)
                nodes[key]["text"] += text_chunk

    if error_msg:
        raise RuntimeError(f"DataAgent query failed: {error_msg}")

    return {"nodes": [nodes[k] for k in node_order]}


# ── slime entrypoint ───────────────────────────────────────────────────

async def generate(args: Any, sample: Any, sampling_params: dict) -> list:
    """Slime custom generate entrypoint.

    Uses the standard adapter session lifecycle:

        open_session → [DataAgent calls adapter] → finish_session

    Returns ``list[Sample]`` for slime's ``generate_and_rm`` contract
    (which accepts both single ``Sample`` and ``list[Sample]``).
    DataAgent's linear conversation produces exactly 1 Sample — all turns
    concatenated with correct loss_mask by TrajectoryManager.  If the
    conversation branches (compaction forks, sub-agents), the manager
    automatically fans out into multiple Samples.
    """
    from slime.utils.types import Sample

    # Start adapter (idempotent singleton — first call boots the HTTP server).
    svc = AdapterService(args)
    adapter = svc.adapter

    # Session ID: deterministic, maps to X-DataAgent-Thread-Id header.
    thread_id = f"slime-{sample.index}"
    adapter.open_session(
        thread_id,
        sampling_defaults=sampling_params,
        max_context_tokens=svc.max_context_len,
    )

    query = str(sample.prompt).strip()
    logger.info("DataAgent generate: threadId=%s query=%r", thread_id, query[:120])
    t0 = time.time()

    nodes: list[dict] = []
    try:
        if ROLLOUT_GUARD_SEC > 0:
            async with asyncio.timeout(ROLLOUT_GUARD_SEC):
                result = await _query_dataagent(query, thread_id)
        else:
            result = await _query_dataagent(query, thread_id)
        nodes = result["nodes"]
    except Exception:
        logger.exception("DataAgent generate failed: threadId=%s", thread_id)
        await adapter.drop_session(thread_id)
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
        return [sample]

    # Compute reward from the DataAgent node trace + ground-truth label.
    reward = score(nodes, getattr(sample, "label", "") or "")

    # finish_session drains the session's trajectory tree into list[Sample]
    # with correct tokens/loss_mask/log_probs.  The reward is set on each
    # emitted sample (linear → 1 sample gets full reward; branches → split).
    samples = await adapter.finish_session(
        thread_id,
        base_sample=sample,
        reward=float(reward),
        extra_metadata={
            "dataagent_nodes": nodes,
            "dataagent_thread_id": thread_id,
        },
    )

    if not samples:
        # Edge case: DataAgent answered without any LLM calls (e.g. cached).
        # No turns were recorded, so finish_session returns [].  Emit the
        # base sample with the reward so the group still trains.
        sample.reward = reward
        sample.status = Sample.Status.COMPLETED
        sample.metadata = {
            **(sample.metadata or {}),
            "dataagent_nodes": nodes,
            "dataagent_thread_id": thread_id,
        }
        return [sample]

    logger.info(
        "DataAgent generate: done (threadId=%s, nodes=%d, reward=%.2f, segments=%d, elapsed=%.1fs)",
        thread_id,
        len(nodes),
        reward,
        len(samples),
        time.time() - t0,
    )
    return samples
