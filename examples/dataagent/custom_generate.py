"""Custom generate function that bridges Slime to DataAgent via SSE.

Phase 1: Slime → DataAgent → DeepSeek (remote API).
DataAgent calls the real DeepSeek API for LLM inference; Slime only
consumes the SSE event stream and records the workflow trace.

Usage:
  --custom-generate-function-path examples.dataagent.custom_generate.generate
"""

import asyncio
import json
import logging
import os
from typing import Any

import httpx

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# ── config (override via env or args) ─────────────────────────────────

DATAAGENT_BASE_URL = os.environ.get("DATAAGENT_BASE_URL", "http://localhost:8065")
DATAAGENT_AGENT_ID = int(os.environ.get("DATAAGENT_AGENT_ID", "20"))
DATAAGENT_TIMEOUT = int(os.environ.get("DATAAGENT_TIMEOUT", "300"))  # seconds


# ── SSE consumer ──────────────────────────────────────────────────────

async def _query_dataagent(query: str) -> dict[str, Any]:
    """Send a query to DataAgent and collect the full SSE event stream.

    Returns a dict with ``threadId`` and ``nodes`` (a list of per-node
    accumulated outputs).
    """
    url = f"{DATAAGENT_BASE_URL}/api/stream/search"
    params = {"agentId": DATAAGENT_AGENT_ID, "query": query}
    headers = {"Accept": "text/event-stream"}

    nodes: dict[str, dict[str, Any]] = {}  # keyed by "nodeName|textType"
    node_order: list[str] = []
    thread_id: str | None = None
    error_msg: str | None = None

    async with httpx.AsyncClient(timeout=httpx.Timeout(DATAAGENT_TIMEOUT)) as client:
        async with client.stream("GET", url, params=params, headers=headers) as resp:
            resp.raise_for_status()

            async for line in resp.aiter_lines():
                if not line:
                    continue

                # SSE events can have an optional "event:" prefix line followed by "data:"
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                    continue

                if not line.startswith("data:"):
                    continue

                payload_str = line[5:].strip()
                if not payload_str or payload_str == "[DONE]":
                    continue

                try:
                    d = json.loads(payload_str)
                except json.JSONDecodeError:
                    logger.warning("DataAgent SSE: skipping non-JSON: %r", payload_str[:120])
                    continue

                if d.get("error"):
                    error_msg = d.get("text", "Unknown error")
                    logger.error("DataAgent SSE error: %s", error_msg)
                    break

                if d.get("complete"):
                    logger.info("DataAgent SSE: complete (threadId=%s)", d.get("threadId"))
                    break

                thread_id = d.get("threadId")
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

    return {
        "threadId": thread_id,
        "nodes": [nodes[k] for k in node_order],
    }


# ── slime entrypoint ───────────────────────────────────────────────────

async def generate(args: Any, sample: Sample, sampling_params: dict) -> Sample:
    """Slime custom generate entrypoint.

    Called by :func:`generate_and_rm` for each sample.  Sends
    ``sample.prompt`` as a natural-language query to DataAgent, collects
    the SSE workflow trace, and stores it on the sample.
    """
    query = sample.prompt
    if isinstance(query, (list, tuple)):
        # If prompt was tokenized (list of ints), decode it.
        # But typically for data-agent we pass raw text via --input-key.
        query = str(query)
    query = str(query).strip()

    logger.info("DataAgent generate: query=%r", query[:120])

    sample.metadata = getattr(sample, "metadata", None) or {}

    try:
        result = await _query_dataagent(query)
    except Exception as exc:
        logger.exception("DataAgent generate failed: %s", exc)
        sample.status = Sample.Status.FAILED
        sample.metadata["dataagent_error"] = str(exc)
        sample.response = ""
        sample.reward = 0.0
        return sample

    # Store the full workflow trace for inspection.
    sample.metadata["dataagent_thread_id"] = result["threadId"]
    sample.metadata["dataagent_nodes"] = result["nodes"]

    # Build a text response from the final node (report) or the full trace.
    report_text = ""
    for n in result["nodes"]:
        if n["type"] in ("MARK_DOWN", "HTML", "TEXT"):
            report_text += n["text"] + "\n"

    sample.response = report_text or json.dumps(result["nodes"], ensure_ascii=False, indent=2)
    sample.status = Sample.Status.COMPLETED
    # Reward is filled by a separate reward function (--custom-rm-path).
    sample.reward = None

    logger.info(
        "DataAgent generate: done (threadId=%s, nodes=%d, response_len=%d)",
        result["threadId"],
        len(result["nodes"]),
        len(sample.response),
    )
    return sample
