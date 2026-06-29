"""
Fully async multi-turn search agent loop.

State machine::

    PENDING -> GENERATING -> PROCESSING_TOOLS -> ... -> TERMINATED

Each turn:
1. Send ``sample.tokens`` (full history) as ``input_ids`` to SGLang.
2. Parse model output for ``<search>`` / ``<answer>`` action.
3. Execute tool or finalize.

Vehicle token boundaries are marked with ``loss_mask``: model tokens = 1,
tool / environment tokens = 0.  Those masks flow into training so only
model tokens contribute to the PPO loss, while tool tokens provide
in-context conditioning.

Wire via::

    --custom-generate-function-path examples.fully_async.search_agent.fully_async_agent_loop.generate

Configuration
-------------
Edit ``SEARCH_CONFIG`` at the top of this file to set up your search
backend.  See ``examples/search-r1/`` for reference search
implementations (local retrieval, Google Search).
"""

from __future__ import annotations

import asyncio
import logging
import queue
import re
import time
from argparse import Namespace

from slime.rollout.fully_async_rollout import _get_global_worker
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.async_utils import run
from slime.utils.http_utils import post
from slime.utils.types import Sample

__all__ = ["generate_rollout_fully_async", "generate"]

logger = logging.getLogger(__name__)

# ============================================================
# Configuration — edit for your search backend / turn budget.
# ============================================================

SEARCH_CONFIG: dict = {
    "max_turns": 3,
    "topk": 3,
    "search_concurrency": 256,
    # Backend selection: "local" or "google" (add more as needed).
    "search_backend": "local",
    # Local retrieval server config (used when search_backend == "local").
    "local": {
        "search_url": "http://127.0.0.1:8000/retrieve",
        "proxy": None,
    },
    # Google Search config (used when search_backend == "google").
    "google": {
        "api_key": "your_api_key_here",
        "snippet_only": True,
        "proxy": None,
    },
}

_SEMAPHORE = asyncio.Semaphore(SEARCH_CONFIG["search_concurrency"])

# ============================================================
# Agent state (persisted in sample.metadata["__agent_state__"])
# ============================================================

_AGENT_STATE_KEY = "__agent_state__"


def _init_agent_state() -> dict:
    return {
        "turn": 0,
        "phase": "generating",  # "generating" | "done"
        "searched_queries": [],  # for dedup
        "search_result_sigs": [],  # for near-duplicate detection
    }


def _get_agent_state(sample: Sample) -> dict:
    state = sample.metadata.get(_AGENT_STATE_KEY)
    if state is None:
        state = _init_agent_state()
        sample.metadata[_AGENT_STATE_KEY] = state
    return state


# ============================================================
# Action parsing
# ============================================================

_ACTION_RE = re.compile(r"<(search|answer)>\s*(.*?)\s*</\1>", re.DOTALL)


def _parse_last_action(text: str) -> tuple[str | None, str]:
    """Return ``(action, content)`` for the last ``<search>`` or ``<answer>`` in *text*.

    Returns ``(None, "")`` when no valid action tag is found.
    """
    matches = list(_ACTION_RE.finditer(text))
    if not matches:
        return None, ""
    m = matches[-1]
    return m.group(1), m.group(2).strip()


# ============================================================
# Search execution
# ============================================================


def _format_search_results(results) -> str:
    """Format raw search results into an ``<information>`` block."""
    if not results:
        return ""
    parts: list[str] = []
    for idx, doc in enumerate(results):
        if isinstance(doc, dict):
            content = doc.get("document", {}).get("contents", str(doc))
        else:
            content = str(doc)
        title = content.split("\n")[0] if content else ""
        parts.append(f"Doc {idx + 1}(Title: {title}) {content}")
    body = "\n".join(parts)
    return f"\n\n<information>{body}</information>\n\n" if body else ""


async def _execute_search(query: str) -> str:
    """Execute a search query and return formatted ``<information>`` text.

    Override this function to plug in your own search backend.
    The default implementation calls a local retrieval server
    (``SEARCH_CONFIG["local"]["search_url"]``).
    """
    import aiohttp

    backend = SEARCH_CONFIG.get("search_backend", "local")

    if backend == "local":
        cfg = SEARCH_CONFIG.get("local", {})
        async with aiohttp.ClientSession() as session:
            async with session.post(
                cfg.get("search_url", "http://127.0.0.1:8000/retrieve"),
                json={"queries": [query], "topk": SEARCH_CONFIG.get("topk", 3)},
                proxy=cfg.get("proxy"),
            ) as resp:
                if resp.status != 200:
                    logger.warning("Search HTTP %d for query: %s", resp.status, query)
                    return ""
                results = await resp.json()
    elif backend == "google":
        # Placeholder — users should implement their own Google Search client.
        logger.warning("Google search backend not implemented; returning empty results.")
        results = []
    else:
        raise ValueError(f"Unknown search backend: {backend!r}")

    return _format_search_results(results)


# ============================================================
# Custom rollout function (--rollout-function-path)
# ============================================================
# Drop-in replacement for fully_async_rollout._generate_rollout_async that
# takes exactly ``rollout_batch_size`` groups without draining the whole
# output queue — leftovers stay queued for the next rollout call.
#
# Wire via::
#
#     --rollout-function-path examples.fully_async.search_agent.fully_async_agent_loop.generate_rollout_fully_async


async def _collect_rollout(args, rollout_id: int, data_buffer) -> list[list[Sample]]:
    assert args.rollout_global_dataset
    worker = _get_global_worker(args, data_buffer)

    target: int = args.rollout_batch_size
    logger.info(
        "fully-async rollout %d: target=%d queue_warm=%d",
        rollout_id,
        target,
        worker.queue_size(),
    )

    collected: dict[int, list[Sample]] = {}
    started = time.time()
    last_log = started
    LOG_EVERY = 30.0

    while len(collected) < target:
        # Pull one group at a time — never more than we need.
        # Leftovers stay in the queue for the next rollout call.
        try:
            gid, group = worker.output_queue.get_nowait()
            collected[gid] = group
        except queue.Empty:
            await asyncio.sleep(0.05)

        now = time.time()
        if now - last_log > LOG_EVERY:
            logger.info(
                "fully-async rollout %d: collected %d/%d, queue=%d, elapsed=%.1fs",
                rollout_id,
                len(collected),
                target,
                worker.queue_size(),
                now - started,
            )
            last_log = now

    # Order by sample.index for determinism (slime convention).
    # Fan-out: group[0] may be list[Sample] rather than Sample.
    def _key(group) -> int:
        first = group[0]
        if isinstance(first, list):
            return int(first[0].index) if first else 0
        return int(first.index) if first else 0

    out = sorted(collected.values(), key=_key)
    logger.info(
        "fully-async rollout %d: done in %.1fs, queue_left=%d",
        rollout_id,
        time.time() - started,
        worker.queue_size(),
    )
    return out


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """Slime ``--rollout-function-path`` entrypoint for fully async + search agent.

    Compared to the stock ``fully_async_rollout.generate_rollout_fully_async``,
    this variant never discards completed groups — it drains exactly
    ``args.rollout_batch_size`` from the output queue and leaves the rest
    buffered for the next call.
    """
    if evaluation:
        raise ValueError("fully-async rollout doesn't support evaluation mode")
    return run(_collect_rollout(args, rollout_id, data_buffer))


# ============================================================
# Main agent loop entry point (--custom-generate-function-path)
# ============================================================


async def generate(args: Namespace, sample: Sample, sampling_params: dict, evaluation: bool = False) -> Sample:
    """Multi-turn search agent loop.

    Called by ``generate_and_rm`` → ``generate_and_rm_group`` for each
    sample group.  Must adhere to the ``--custom-generate-function-path``
    contract::

        async def generate(args, sample, sampling_params, evaluation=False) -> Sample
    """
    agent = _get_agent_state(sample)
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # ---- prepare stop tags --------------------------------------------------
    _stop_tags = ["</search>", "</answer>"]
    _existing = sampling_params.get("stop") or []
    if isinstance(_existing, str):
        _existing = [_existing]
    turn_sp: dict = {
        **sampling_params,
        "stop": list(dict.fromkeys([*_existing, *_stop_tags])),
    }

    max_turns: int = SEARCH_CONFIG.get("max_turns", 3)
    per_turn_budget: int = max(sampling_params.get("max_new_tokens", 512) // max_turns, 64)

    # ---- encode prompt on first entry ---------------------------------------
    if agent["turn"] == 0 and not sample.tokens:
        prompt_ids = state.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
        sample.tokens = list(prompt_ids)

    last_meta_info: dict = {}
    non_generation_start: float = time.time()

    # ---- main multi-turn loop -----------------------------------------------
    while agent["turn"] < max_turns:
        payload: dict = {
            "input_ids": sample.tokens,
            "sampling_params": {**turn_sp, "max_new_tokens": per_turn_budget},
            "return_logprob": True,
        }

        output = await post(url, payload)
        last_meta_info = output.get("meta_info", {})

        # Abort → keep partial state; worker will requeue to data_buffer.
        if last_meta_info.get("finish_reason", {}).get("type") == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        # Extract new tokens and log-probs from this turn.
        if "output_token_logprobs" in last_meta_info:
            cur_tokens = [item[1] for item in last_meta_info["output_token_logprobs"]]
            cur_logprobs = [item[0] for item in last_meta_info["output_token_logprobs"]]
        else:
            cur_tokens, cur_logprobs = [], []

        if not cur_tokens:
            break

        cur_text: str = output.get("text", "")
        sample.append_response_tokens(
            args,
            tokens=cur_tokens,
            log_probs=cur_logprobs,
            trainable=True,
            meta_info=last_meta_info,
            text=cur_text,
        )

        # Length limit reached → stop.
        if last_meta_info.get("finish_reason", {}).get("type") == "length":
            break

        # ---- action parsing -------------------------------------------------
        action, content = _parse_last_action(cur_text)

        if action == "answer":
            agent["phase"] = "done"
            break

        if action == "search":
            async with _SEMAPHORE:
                obs_text = await _execute_search(content)

            if obs_text:
                obs_ids = state.tokenizer(obs_text, add_special_tokens=False)["input_ids"]
                sample.append_response_tokens(args, tokens=obs_ids, trainable=False, text=obs_text)

            agent["turn"] += 1
            agent["searched_queries"].append(content)
        else:
            # Parse error — inject a brief retry hint so the model can recover.
            hint = "\nInvalid format. Use <search>query</search> or <answer>result</answer>.\n"
            hint_ids = state.tokenizer(hint, add_special_tokens=False)["input_ids"]
            sample.append_response_tokens(args, tokens=hint_ids, trainable=False, text=hint)
            agent["turn"] += 1

    # ---- finalize status ----------------------------------------------------
    sample.non_generation_time += time.time() - non_generation_start

    if sample.status != Sample.Status.ABORTED:
        finish_type = last_meta_info.get("finish_reason", {}).get("type")
        if finish_type == "length":
            sample.status = Sample.Status.TRUNCATED
        elif agent["phase"] == "done":
            sample.status = Sample.Status.COMPLETED
        else:
            sample.status = Sample.Status.COMPLETED  # max turns exhausted

    return sample
