"""Custom generate function bridging Slime to Data-Analysis-Agent (DAA) via HTTP SSE.

DAA runs as a separate Flask/waitress process. Each Slime sample:
  1. Creates a DAA session
  2. Connects the demo_sales SQLite database
  3. Sends the query via POST /api/session/<sid>/chat
  4. Consumes SSE events to extract final text + tool audit events
  5. Scores via reward_func

The thread_id flows: custom_generate → DAA HTTP header → ContextVar →
OpenAI client default_headers → /v1/chat/completions → slime_api →
Sample token capture.

Usage:
  --custom-generate-function-path examples.data_analysis_agent.custom_generate.generate
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx

from examples.dataagent import slime_api
from slime.rollout.fully_async_rollout import _global_worker

logger = logging.getLogger(__name__)

# ── config ──────────────────────────────────────────────────────────
DAA_BASE_URL = os.environ.get("DAA_BASE_URL", "http://localhost:5001")
DAA_TIMEOUT = int(os.environ.get("DAA_TIMEOUT", "600"))
DAA_PROVIDER = os.environ.get("DAA_PROVIDER", "slime-rl")
DEMO_SALES_DB = os.environ.get(
    "DEMO_SALES_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_sales.db"),
)

# Tables available for analysis
_DEMO_TABLES = [
    "products", "orders", "customers", "campaigns",
    "returns", "daily_traffic", "suppliers",
]


# ── DAA HTTP helpers ────────────────────────────────────────────────

async def _create_session(client: httpx.AsyncClient) -> str:
    """Create a new DAA session, return its sid."""
    resp = await client.post(f"{DAA_BASE_URL}/api/session/new")
    resp.raise_for_status()
    return resp.json()["session_id"]


async def _configure_session(client: httpx.AsyncClient, sid: str) -> None:
    """Set LLM provider and connect the demo_sales SQLite database."""
    # Set model provider to slime-rl custom model pointing at slime_api
    await client.post(
        f"{DAA_BASE_URL}/api/session/{sid}/model",
        json={"provider": DAA_PROVIDER},
    )

    # Connect SQLite as read-only to prevent RL agent from corrupting data
    db_uri = f"sqlite:///{DEMO_SALES_DB}?mode=ro"
    resp = await client.post(
        f"{DAA_BASE_URL}/api/session/{sid}/connect",
        json={"connection_string": db_uri, "name": "demo_sales"},
    )
    resp.raise_for_status()
    data = resp.json()
    source_id = data.get("source_id") or data.get("id")

    # Select all tables for analysis
    if source_id:
        await client.post(
            f"{DAA_BASE_URL}/api/session/{sid}/sources/{source_id}/analysis-tables",
            json={"tables": _DEMO_TABLES},
        )


async def _chat_stream(
    client: httpx.AsyncClient, sid: str, query: str, thread_id: str,
) -> dict[str, Any]:
    """Send query to DAA, consume SSE stream, return parsed result.

    Returns:
        dict with keys:
          - text: concatenated final answer
          - tool_events: list of tool_audit events
          - error: error message if failed (only present on failure)
    """
    url = f"{DAA_BASE_URL}/api/session/{sid}/chat"
    headers = {
        "Content-Type": "application/json",
        "X-DataAgent-Thread-Id": thread_id,
        "Accept": "text/event-stream",
    }

    text_parts: list[str] = []
    tool_events: list[dict] = []
    error_msg: str | None = None

    async with client.stream(
        "POST", url, json={"message": query}, headers=headers,
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload_str = line[6:].strip()
            if not payload_str:
                continue
            try:
                event = json.loads(payload_str)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")

            if etype == "error":
                error_msg = event.get("message", "DAA chat error")
                break
            if etype == "done":
                break
            if etype == "text_delta":
                text_parts.append(str(event.get("content", "")))
            elif etype == "tool_audit":
                tool_events.append({
                    "tool": event.get("tool", ""),
                    "ok": event.get("ok", True),
                    "error": event.get("error", ""),
                    "summary": str(event.get("summary", ""))[:500],
                    "content_len": len(str(event.get("content", ""))),
                })

    if error_msg:
        return {"error": error_msg}

    return {"text": "".join(text_parts), "tool_events": tool_events}


# ── slime entrypoint ─────────────────────────────────────────────────

async def generate(args: Any, sample: Any, sampling_params: dict) -> Any:
    """Slime custom generate entrypoint — Data-Analysis-Agent integration.

    Called by :func:`generate_and_rm` for each sample.  Starts the Slime
    LLM API server (idempotent), registers the sample under a unique
    threadId, sends the query to DAA, waits for completion, scores the
    result, and cleans up.
    """
    from slime.utils.types import Sample

    # Ensure the Slime LLM API is running (idempotent, first call starts it).
    slime_api.ensure_server(args)

    # Bridge to the worker's shared dict.
    if _global_worker is not None:
        slime_api._sample_map = _global_worker.sample_map

    # Register sample under slime-{sample.index}
    thread_id = f"slime-{sample.index}"
    slime_api._sample_map[thread_id] = sample

    if sample.tokens is None:
        sample.tokens = []

    query = str(sample.prompt).strip()
    logger.info("DAA generate: threadId=%s query=%r", thread_id, query[:120])

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(DAA_TIMEOUT)) as client:
            sid = await _create_session(client)
            logger.debug("DAA session created: sid=%s threadId=%s", sid, thread_id)

            try:
                await _configure_session(client, sid)
                result = await _chat_stream(client, sid, query, thread_id)
            finally:
                try:
                    await client.post(f"{DAA_BASE_URL}/api/session/{sid}/stop")
                except Exception:
                    pass

        if "error" in result:
            raise RuntimeError(result["error"])

    except Exception:
        logger.exception("DAA generate failed: threadId=%s", thread_id)
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
        slime_api._sample_map.pop(thread_id, None)
        return sample

    # ── Score ─────────────────────────────────────────────────────────
    from examples.data_analysis_agent.reward_func import score

    sample.reward = score(
        text=result["text"],
        tool_events=result["tool_events"],
        label=getattr(sample, "label", ""),
    )
    sample.metadata = getattr(sample, "metadata", None) or {}
    sample.metadata["daa_text"] = result["text"]
    sample.metadata["daa_tool_events"] = result["tool_events"]
    sample.metadata["daa_thread_id"] = thread_id
    sample.status = Sample.Status.COMPLETED

    slime_api._sample_map.pop(thread_id, None)
    logger.info(
        "DAA generate: done (threadId=%s, text_len=%d, tools=%d, reward=%.2f)",
        thread_id,
        len(result["text"]),
        len(result["tool_events"]),
        sample.reward,
    )
    return sample
