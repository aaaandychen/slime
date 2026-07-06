"""Custom generate function bridging Slime to DataAgent via SSE.

Phase 2: fully-async with Slime LLM API interception.
DataAgent calls Slime's /v1/chat/completions instead of DeepSeek;
Slime proxies to SGLang and captures tokens/logprobs into Sample.

Usage:
  --custom-generate-function-path examples.dataagent.custom_generate.generate
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

import httpx

from examples.dataagent import slime_api

logger = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────

DATAAGENT_BASE_URL = os.environ.get("DATAAGENT_BASE_URL", "http://localhost:8065")
DATAAGENT_AGENT_ID = int(os.environ.get("DATAAGENT_AGENT_ID", "20"))
DATAAGENT_TIMEOUT = int(os.environ.get("DATAAGENT_TIMEOUT", "600"))


# ── SSE consumer ──────────────────────────────────────────────────────

async def _query_dataagent(query: str, thread_id: str) -> dict[str, Any]:
    """Send query to DataAgent, collect SSE events, return structured result.

    *thread_id* is set by slime so the LLM API can route requests back
    to the correct sample via ``X-DataAgent-Thread-Id``.
    """
    url = f"{DATAAGENT_BASE_URL}/api/stream/search"
    params = {"agentId": DATAAGENT_AGENT_ID, "query": query, "threadId": thread_id}
    headers = {"Accept": "text/event-stream"}

    nodes: dict[str, dict[str, Any]] = {}  # "nodeName|textType" → {node, type, text}
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

async def generate(args: Any, sample: Any, sampling_params: dict) -> Any:
    """Slime custom generate entrypoint.

    Called by :func:`generate_and_rm` for each sample.  Starts the Slime
    LLM API server (idempotent), registers the sample under a unique
    threadId, streams the query to DataAgent, waits for completion,
    scores the result, and cleans up.
    """
    from slime.utils.types import Sample

    # Ensure the Slime LLM API is running (idempotent, first call starts it).
    slime_api.ensure_server(args)

    # Register sample under a unique threadId so handle_chat can find it.
    thread_id = f"slime-{sample.index}-{uuid.uuid4().hex[:8]}"
    slime_api._thread_samples[thread_id] = sample

    # Ensure mutable lists are initialised before handle_chat starts
    # appending via append_response_tokens.
    if sample.tokens is None:
        sample.tokens = []
    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    if sample.loss_mask is None:
        sample.loss_mask = []

    query = str(sample.prompt).strip()
    logger.info("DataAgent generate: threadId=%s query=%r", thread_id, query[:120])

    try:
        result = await _query_dataagent(query, thread_id)
    except Exception:
        logger.exception("DataAgent generate failed: threadId=%s", thread_id)
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
        slime_api._thread_samples.pop(thread_id, None)
        return sample

    # Score the final output.
    from examples.dataagent.reward_func import score

    sample.reward = score(result["nodes"], getattr(sample, "label", ""))
    sample.metadata = getattr(sample, "metadata", None) or {}
    sample.metadata["dataagent_nodes"] = result["nodes"]
    sample.metadata["dataagent_thread_id"] = thread_id
    sample.status = Sample.Status.COMPLETED

    slime_api._thread_samples.pop(thread_id, None)
    logger.info(
        "DataAgent generate: done (threadId=%s, nodes=%d, reward=%.2f)",
        thread_id,
        len(result["nodes"]),
        sample.reward,
    )
    return sample
