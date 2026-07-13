"""DataAgent adapter — reuses slime's OpenAIAdapter for black-box RL.

DataAgent speaks the OpenAI Chat Completions protocol, so it can drive the
shared adapter directly.  The adapter intercepts each LLM call, proxies to
SGLang, and records tokens into a per-session accumulator.

Two things DataAgent needs that the base adapter doesn't provide:

1. **Session routing via ``X-DataAgent-Thread-Id`` header.**  DataAgent's
   Java backend stamps every LLM request with this header so slime can
   attribute tokens to the right sample.

2. **Simplified trajectory manager (no fork/realign).**  DataAgent's
   multi-turn conversation has completely independent prompts per turn
   (each node constructs its own prompt from scratch — no shared prefix
   with previous turns).  The base ``TrajectoryManager`` uses prefix
   matching to merge turns into one Sample; with no shared prefix, every
   turn FORKs → N segments → reward split N ways → no gradient.

   We replace it with ``DataAgentTrajectoryManager``: a simple
   per-session accumulator that appends each turn's full prompt
   (loss_mask=0) + output (loss_mask=1) into one Sample, no prefix
   matching, no fork.  This mirrors the original ``slime_api.py`` approach.

3. **Transparent abort→resume during weight sync.**  In fully-async
   training, SGLang ``pause_generation`` fires mid-turn and the in-flight
   ``/generate`` returns ``finish_reason="abort"`` with partial tokens.
   We retry: record partial tokens as loss_mask=0 context, wait for
   ``GenerateState.aborted`` to clear, re-invoke with extended prompt.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
from typing import Any

from aiohttp import web

from slime.agent.adapters.common import (
    Session,
    TurnRecord,
    _render_token_ids,
    call_sglang_generate,
)
from slime.agent.adapters.openai import OpenAIAdapter
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.agent.parsing import parse_model_output
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


# ── simplified trajectory manager ───────────────────────────────────────


class _SessionAccumulator:
    """Per-session token accumulator — one DataAgent conversation = one Sample.

    Appends each turn's full prompt (loss_mask=0) + output (loss_mask=1).
    No prefix matching, no fork.  The first turn's prompt is recorded as
    ``leading_prompt_len`` and stripped from the training region on export.
    """

    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.loss_mask: list[int] = []
        self.logprobs: list[float] = []
        self.leading_prompt_len: int = 0

    def record_turn(
        self,
        prompt_ids: list[int],
        output_ids: list[int],
        output_logprobs: list[float] | None = None,
    ) -> None:
        """Append one turn: full prompt (loss_mask=0) + output (loss_mask=1)."""
        if not self.leading_prompt_len:
            self.leading_prompt_len = len(prompt_ids)
        # Prompt — context, not trained.
        self.tokens.extend(prompt_ids)
        self.loss_mask.extend([0] * len(prompt_ids))
        self.logprobs.extend([0.0] * len(prompt_ids))
        # Output — model-generated, trained.
        self.tokens.extend(output_ids)
        self.loss_mask.extend([1] * len(output_ids))
        self.logprobs.extend(
            output_logprobs if output_logprobs else [0.0] * len(output_ids)
        )

    def record_partial(self, partial_ids: list[int]) -> None:
        """Append aborted partial output as loss_mask=0 context."""
        self.tokens.extend(partial_ids)
        self.loss_mask.extend([0] * len(partial_ids))
        self.logprobs.extend([0.0] * len(partial_ids))

    def to_sample(
        self,
        base_sample: Sample,
        reward: float,
        extra_metadata: dict[str, Any] | None,
    ) -> Sample:
        """Emit one Sample, stripping the first-turn prompt from training."""
        start = self.leading_prompt_len
        return Sample(
            index=base_sample.index,
            group_index=base_sample.group_index,
            rollout_id=base_sample.rollout_id if base_sample.rollout_id is not None else base_sample.index,
            prompt=base_sample.prompt,
            label=base_sample.label,
            tokens=list(self.tokens),
            response_length=len(self.loss_mask) - start,
            loss_mask=self.loss_mask[start:],
            rollout_log_probs=self.logprobs[start:],
            reward=reward,
            status=Sample.Status.COMPLETED,
            metadata=dict(extra_metadata or {}),
        )


class DataAgentTrajectoryManager:
    """Simplified manager: one session → one Sample, no fork.

    API-compatible with ``TrajectoryManager`` (``record_turn`` /
    ``get_trajectory`` / ``drop_session``) so ``BaseAdapter.finish_session``
    works unchanged.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionAccumulator] = {}

    def has_session(self, sid: str) -> bool:
        return sid in self._sessions

    def turn_count(self, sid: str) -> int:
        return 0  # not tracked; base adapter only uses this for logging

    def record_turn(
        self,
        sid: str,
        *,
        turn: TurnRecord,
        prompt_messages: list[dict[str, Any]] | None = None,
        response_message: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record one turn's prompt + output into the session accumulator.

        ``prompt_messages`` / ``response_message`` are ignored — we only
        need the token ids from ``turn``.  If ``output_ids`` is non-empty
        and ``prompt_ids`` is empty, this is an abort partial (loss_mask=0).
        """
        acc = self._sessions.setdefault(sid, _SessionAccumulator())
        is_partial = metadata.get("abort") if metadata else False
        if is_partial and not turn.prompt_ids:
            # Abort partial: partial_ids are in output_ids, record as context.
            acc.record_partial(list(turn.output_ids))
        else:
            acc.record_turn(
                list(turn.prompt_ids),
                list(turn.output_ids),
                list(turn.output_log_probs) if turn.output_log_probs else None,
            )

    def get_trajectory(
        self,
        sid: str,
        *,
        base_sample: Sample,
        reward: float = 0.0,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Sample]:
        """Return exactly 1 Sample with the full reward (never split)."""
        acc = self._sessions.pop(sid, None)
        if acc is None or not acc.tokens:
            return []
        return [acc.to_sample(base_sample, reward, extra_metadata)]

    def drop_session(self, sid: str) -> None:
        self._sessions.pop(sid, None)


# ── adapter ─────────────────────────────────────────────────────────────


class DataAgentAdapter(OpenAIAdapter):
    """``OpenAIAdapter`` subclass for DataAgent's wire conventions.

    Overrides:
    - ``__init__``: replace ``TrajectoryManager`` with ``DataAgentTrajectoryManager``
    - ``_session_id``: read ``X-DataAgent-Thread-Id`` header
    - ``_run_turn``: wrap SGLang call in abort→resume retry + use simplified manager
    """

    log_prefix = "dataagent_adapter"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Replace the base TrajectoryManager — DataAgent's independent prompts
        # per turn break prefix matching, so we use a simple accumulator.
        self.manager = DataAgentTrajectoryManager()

    def _session_id(self, request: web.Request, body: dict) -> str:
        """Resolve sid: ``X-DataAgent-Thread-Id`` header first, then base."""
        tid = request.headers.get("X-DataAgent-Thread-Id", "")
        if tid:
            return tid
        return super()._session_id(request, body)

    async def _run_turn(self, request: web.Request) -> web.StreamResponse:
        """One agent turn with transparent abort→resume retry."""
        body = await request.json()
        self._preprocess_body(body)
        sid = self._session_id(request, body)
        if sid in self.closed:
            return web.Response(status=503, text="session closed")
        capped = self._check_turn_cap(sid)
        if capped is not None:
            return capped

        tok = self.tokenizer
        s = self.store.setdefault(sid, Session())
        task = asyncio.current_task()
        self.inflight.setdefault(sid, set()).add(task)
        try:
            translated, tools_schema = self._translate(body)
            prompt_ids = _render_token_ids(
                translated, tok, tools=tools_schema, add_generation_prompt=True
            )

            turn, full_text = await self._call_sglang_with_retry(
                prompt_ids, s, body, sid
            )

            raw_output = full_text or (
                tok.decode(turn.output_ids, skip_special_tokens=False)
                if turn.output_ids
                else ""
            )
            parsed = parse_model_output(
                raw_output,
                tools_schema=tools_schema,
                tool_parser_name=self.tool_parser,
                reasoning_parser_name=self.reasoning_parser,
            )
            reply = self._build_reply(parsed, turn.finish_reason, translated, tools_schema)
            turn = dataclasses.replace(turn, finish_reason=reply.finish_reason)

            self._run_debug_callback(sid, translated, tools_schema, reply.manager_message, turn)

            # Record into the simplified manager (prompt=loss_mask=0, output=loss_mask=1).
            self.manager.record_turn(
                sid,
                turn=turn,
                prompt_messages=translated,
                response_message=reply.manager_message,
                metadata={"sid": sid},
            )
            in_tok, out_tok = len(prompt_ids), len(turn.output_ids)

            stream = body.get("stream") is True or "text/event-stream" in request.headers.get("Accept", "")
            return await self._respond(request, body, reply, in_tok, out_tok, stream)
        finally:
            self.inflight.get(sid, set()).discard(task)

    async def _call_sglang_with_retry(
        self,
        prompt_ids: list[int],
        session: Any,
        body: dict,
        sid: str,
    ) -> tuple[TurnRecord, str]:
        """Call SGLang; on abort, record partial as loss_mask=0, wait, retry."""
        from slime.rollout.sglang_rollout import GenerateState

        tok = self.tokenizer
        accumulated_text = ""
        while True:
            turn = await call_sglang_generate(
                prompt_ids, session, body, adapter=self, session_id=sid,
            )
            if turn.finish_reason != "abort":
                new_text = (
                    tok.decode(turn.output_ids, skip_special_tokens=False)
                    if turn.output_ids
                    else ""
                )
                return turn, accumulated_text + new_text

            partial = list(turn.output_ids)
            if partial:
                accumulated_text += tok.decode(partial, skip_special_tokens=False)
                self.logger.debug(
                    "[%s] sid=%s abort with %d partial tokens; recording as context",
                    self.log_prefix, sid, len(partial),
                )
                # Record partial as loss_mask=0 context (no prompt_ids →
                # manager treats it as abort partial).
                self.manager.record_turn(
                    sid,
                    turn=TurnRecord(
                        prompt_ids=[],
                        output_ids=partial,
                        finish_reason="abort",
                        output_log_probs=turn.output_log_probs,
                    ),
                    prompt_messages=[],
                    response_message=None,
                    metadata={"sid": sid, "abort": True},
                )
                prompt_ids = list(prompt_ids) + partial

            await self._wait_resume()

    async def _wait_resume(self, poll_interval: float = 0.1) -> None:
        """Block until SGLang resumes after weight sync (``GenerateState.aborted`` clears)."""
        from slime.rollout.sglang_rollout import GenerateState
        state = GenerateState(None)
        while state.aborted:
            await asyncio.sleep(poll_interval)


# ── singleton service ───────────────────────────────────────────────────


class AdapterService(metaclass=SingletonMeta):
    """Per-process singleton: tokenizer + adapter + aiohttp server thread."""

    def __init__(self, args) -> None:
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"

        tool_parser = getattr(args, "sglang_tool_call_parser", None) or None
        reasoning_parser = getattr(args, "sglang_reasoning_parser", None) or None
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)

        self.adapter = DataAgentAdapter(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=tool_parser,
            reasoning_parser=reasoning_parser,
        )

        host = os.environ.get("SLIME_API_HOST", "0.0.0.0")
        port = int(os.environ.get("SLIME_API_PORT", "18080") or 18080)

        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=host,
            port=port,
            thread_name="dataagent-adapter",
            runner_kwargs={
                "handler_cancellation": True,
                "access_log_class": FilteredAccessLogger,
            },
        )
        self.adapter_url = f"http://127.0.0.1:{self.app_handle.port}"
        logger.info(
            "DataAgentAdapter: listening on %s (sglang=%s tool_parser=%s reasoning_parser=%s)",
            self.adapter_url,
            sglang_url,
            tool_parser,
            reasoning_parser,
        )
