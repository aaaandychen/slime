"""Slime LLM API — OpenAI-compatible /v1/chat/completions endpoint.

DataAgent calls this as its LLM backend.  Each request carries an
``X-DataAgent-Thread-Id`` header that maps to a :class:`Sample` in the
shared ``_thread_samples`` dict.  Token IDs and logprobs from SGLang
are appended directly to that sample.

Public API::

    slime_api.ensure_server(args)     # start aiohttp (idempotent)
    slime_api._thread_samples[tid] = sample   # custom_generate writes directly
    slime_api.notify_resume()         # update_weights calls after continue_generation
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from typing import Any

from aiohttp import web

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post

logger = logging.getLogger(__name__)

# ── shared state ──────────────────────────────────────────────────────
_thread_samples: dict[str, Any] = {}  # threadId → Sample  (custom_generate writes directly)
_args: Any = None
_server_started = False
_server_lock = threading.Lock()
_resume_generation: int = 0           # monotonic counter, notify_resume increments


# ── public API ────────────────────────────────────────────────────────

def ensure_server(args: Any) -> None:
    """Store *args* and start the aiohttp server (idempotent)."""
    global _args, _server_started
    _args = args
    if _server_started:
        return
    with _server_lock:
        if _server_started:
            return
        host = getattr(args, "slime_api_host", "0.0.0.0")
        port = getattr(args, "slime_api_port", 18080)

        t = threading.Thread(
            target=_run_server, args=(host, port), name="slime-api", daemon=True
        )
        t.start()
        _server_started = True
        logger.info("slime_api: server starting on %s:%d", host, port)


def notify_resume() -> None:
    """Wake up all handlers blocked on weight-sync abort.

    Called from ``update_weights()`` right after ``continue_generation.remote()``.
    """
    global _resume_generation
    _resume_generation += 1
    logger.debug("slime_api: notify_resume → generation=%d", _resume_generation)


# ── request handler ───────────────────────────────────────────────────

async def handle_chat_completions(request: web.Request) -> web.Response:
    """POST /v1/chat/completions — OpenAI-compatible."""
    thread_id = request.headers.get("X-DataAgent-Thread-Id", "")
    sample = _thread_samples.get(thread_id)
    if sample is None:
        return _error(400, f"Unknown threadId: {thread_id!r}")

    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools")
    max_tokens = body.get("max_tokens", 2048)

    state = GenerateState(_args)
    sampling_params = state.sampling_params.copy()
    sampling_params["max_new_tokens"] = min(max_tokens, sampling_params["max_new_tokens"])

    # Build prompt: chat template → tokenize → SGLang input_ids.
    prompt_text = state.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, tools=tools
    )
    input_ids = state.tokenizer.encode(prompt_text, add_special_tokens=False)

    # Write prompt tokens into the sample as non-trainable context
    # (loss_mask=0).  The trainer needs them for the forward pass;
    # without them rollout_logprobs and training-time logprobs misalign.
    # append_response_tokens unconditionally bumps response_length (even
    # for trainable=False), so we save & restore to keep it response-only.
    resp_len_before = sample.response_length or 0
    sample.append_response_tokens(
        _args, tokens=input_ids, trainable=False
    )
    sample.response_length = resp_len_before

    url = f"http://{_args.sglang_router_ip}:{_args.sglang_router_port}/generate"
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    # Generate (with abort-retry for partial rollout).
    # Concurrency is controlled by generate_and_rm's outer semaphore
    # (async with state.semaphore) — DataAgent calls are serial within
    # each Sample, so no inner semaphore is needed here.
    # We deliberately do NOT engage GenerateState's semaphore or
    # dp_rank_context here because handle_chat runs in a different
    # asyncio event loop (daemon thread) than the Worker thread where
    # those primitives were created.
    while True:
        output = await post(url, payload)

        meta = output.get("meta_info", {})
        finish = meta.get("finish_reason", {}).get("type", "")

        if finish == "abort":
            # Weight sync in progress.  Save partial tokens (old
            # weights) to the sample AND extend input_ids so the
            # retry continues from where SGLang left off.
            partial_tokens = _extract_response_tokens(output)
            if partial_tokens:
                partial_logprobs = _extract_response_logprobs(output)
                sample.append_response_tokens(
                    _args,
                    tokens=partial_tokens,
                    log_probs=partial_logprobs,
                    trainable=True,
                    meta_info=meta,
                )
                input_ids = input_ids + partial_tokens
                payload["input_ids"] = input_ids

            target = _resume_generation + 1
            await _wait_resume(target)
            continue

        break

    # Append generated tokens (or continuation, if abort-then-retry).
    new_tokens = _extract_response_tokens(output)
    if new_tokens:
        new_logprobs = _extract_response_logprobs(output)
        sample.append_response_tokens(
            _args,
            tokens=new_tokens,
            log_probs=new_logprobs,
            trainable=True,
            meta_info=meta,
        )

    # Assemble OpenAI-format response
    text = output.get("text", "")
    finish_reason = _map_finish_reason(finish)

    return web.json_response({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "slime-rl",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": len(input_ids) - len(new_tokens),
            "completion_tokens": len(new_tokens),
            "total_tokens": len(input_ids),
        },
    })


# ── internals ─────────────────────────────────────────────────────────

async def _wait_resume(target: int) -> None:
    while _resume_generation < target:
        await asyncio.sleep(0.1)


def _extract_response_tokens(output: dict) -> list[int]:
    token_logprobs = output.get("meta_info", {}).get("output_token_logprobs")
    if not token_logprobs:
        return []
    return [item[1] for item in token_logprobs]


def _extract_response_logprobs(output: dict) -> list[float]:
    token_logprobs = output.get("meta_info", {}).get("output_token_logprobs")
    if not token_logprobs:
        return []
    return [item[0] for item in token_logprobs]


def _map_finish_reason(meta_type: str) -> str:
    if meta_type == "stop":
        return "stop"
    if meta_type == "length":
        return "length"
    if meta_type == "abort":
        return "length"
    return "stop"


def _error(status: int, message: str) -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": "invalid_request_error"}},
        status=status,
    )


def _run_server(host: str, port: int) -> None:
    """Blocking entry-point for the background daemon thread."""
    GenerateState(_args)  # warm singleton in this thread's context

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/health", lambda r: web.json_response({"status": "ok"}))

    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, host, port)
    loop.run_until_complete(site.start())
    logger.info("slime_api: listening on %s:%d", host, port)

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(runner.cleanup())
        loop.close()
