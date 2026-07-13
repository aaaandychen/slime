"""DataAgent adapter — reuses slime's OpenAIAdapter for black-box RL.

Replaces the hand-rolled ``slime_api.py`` (~260 lines of manual HTTP/token
management) with a thin ``OpenAIAdapter`` subclass (~90 lines).  DataAgent
speaks the OpenAI Chat Completions protocol, so it can drive the shared
adapter directly; the adapter's ``TrajectoryManager`` records every turn
with correct ``loss_mask`` (model outputs=1, tool/user context=0) and
linearises the session into ``list[Sample]`` on ``finish_session``.

Two things DataAgent needs that the base adapter doesn't provide:

1. **Session routing via ``X-DataAgent-Thread-Id`` header.**  DataAgent's
   Java backend stamps every LLM request with this header so slime can
   attribute tokens to the right sample.  The base adapter resolves sid
   from ``Authorization: Bearer`` / ``metadata.session_id`` / ``user``;
   we add the custom header as the first lookup.

2. **Transparent abort→resume during weight sync.**  In fully-async
   training, SGLang ``pause_generation`` (weight update) fires mid-turn
   and the in-flight ``/generate`` returns ``finish_reason="abort"`` with
   partial tokens.  The base adapter would hand that truncated response
   back to DataAgent, corrupting its reasoning context.  We wrap the SGLang
   call in a retry loop: record partial tokens as loss_mask=0 context,
   wait for ``GenerateState.aborted`` to clear, then re-invoke SGLang with
   the extended prompt so DataAgent sees one coherent response.

The ``AdapterService`` singleton mirrors ``coding_agent_rl._AdapterService``:
one tokenizer + one adapter + one aiohttp thread per process, shared across
all concurrent ``generate()`` calls.
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

logger = logging.getLogger(__name__)


class DataAgentAdapter(OpenAIAdapter):
    """``OpenAIAdapter`` subclass for DataAgent's wire conventions.

    Overrides only ``_session_id`` (header lookup) and ``_run_turn``
    (abort-resume loop); everything else — chat-template rendering,
    SGLang proxying, tool/reasoning parsing, trajectory recording,
    session lifecycle, SSE streaming — is inherited.
    """

    log_prefix = "dataagent_adapter"

    def _session_id(self, request: web.Request, body: dict) -> str:
        """Resolve sid: ``X-DataAgent-Thread-Id`` header first, then base."""
        tid = request.headers.get("X-DataAgent-Thread-Id", "")
        if tid:
            return tid
        return super()._session_id(request, body)

    async def _run_turn(self, request: web.Request) -> web.StreamResponse:
        """One agent turn with transparent abort→resume retry.

        Identical to ``BaseAdapter._run_turn`` except the SGLang call is
        wrapped in a retry loop (see ``_call_sglang_with_retry``).  The
        retry returns ``(turn, full_text)`` where ``turn.output_ids`` is
        new-only (for trajectory recording with loss_mask=1) and
        ``full_text`` is the decoded partial+new (for the response to
        DataAgent, so it sees one coherent response).
        """
        body = await request.json()
        self._preprocess_body(body)
        sid = self._session_id(request, body)
        if sid in self.closed:
            self.logger.debug("[%s] sid=%s request after session closed", self.log_prefix, sid)
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
            prompt_ids = _render_token_ids(translated, tok, tools=tools_schema, add_generation_prompt=True)

            turn, full_text = await self._call_sglang_with_retry(prompt_ids, s, body, sid)

            # Use full_text (partial + new) for parsing so DataAgent sees
            # the complete response.  turn.output_ids is new-only so
            # record_turn trains only on the post-resume tokens.
            raw_output = full_text or (tok.decode(turn.output_ids, skip_special_tokens=False) if turn.output_ids else "")
            parsed = parse_model_output(
                raw_output,
                tools_schema=tools_schema,
                tool_parser_name=self.tool_parser,
                reasoning_parser_name=self.reasoning_parser,
            )
            reply = self._build_reply(parsed, turn.finish_reason, translated, tools_schema)
            turn = dataclasses.replace(turn, finish_reason=reply.finish_reason)

            self._run_debug_callback(sid, translated, tools_schema, reply.manager_message, turn)

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
        """Call SGLang; on ``finish_reason="abort"``, wait for resume and retry.

        SGLang's ``pause_generation`` (weight sync) aborts in-flight
        ``/generate`` calls with partial tokens.  We:

        1. Record the partial output as a non-trainable context turn
           (``response_message=None`` → ``loss_mask=0``) so the trajectory
           keeps token continuity without training on the aborted fragment.
        2. Extend ``prompt_ids`` with the partial tokens so the retry
           continues from where SGLang left off.
        3. Accumulate the decoded partial text so DataAgent receives the
           full response (partial + new), not just the post-resume tail.
        4. Wait for ``GenerateState.aborted`` to clear (weight sync done).
        5. Re-invoke; the completed turn's ``output_ids`` contains only
           the *new* tokens from the retry — the partial tokens are already
           in the trajectory from step 1.  The returned text is the
           concatenation of all partial texts + the final text.

        Returns ``(turn, full_text)`` where ``turn.output_ids`` is new-only
        (for ``record_turn`` with loss_mask=1) and ``full_text`` is the
        decoded partial+new (for the response to DataAgent).
        """
        from slime.rollout.sglang_rollout import GenerateState

        tok = self.tokenizer
        accumulated_text = ""
        while True:
            turn = await call_sglang_generate(
                prompt_ids, session, body, adapter=self, session_id=sid,
            )
            if turn.finish_reason != "abort":
                # Final completion: decode the new tokens and append to
                # accumulated partial text so DataAgent sees the full response.
                new_text = tok.decode(turn.output_ids, skip_special_tokens=False) if turn.output_ids else ""
                full_text = accumulated_text + new_text
                return turn, full_text

            partial = list(turn.output_ids)
            if partial:
                partial_text = tok.decode(partial, skip_special_tokens=False)
                accumulated_text += partial_text
                self.logger.debug(
                    "[%s] sid=%s abort with %d partial tokens; recording as context",
                    self.log_prefix, sid, len(partial),
                )
                # Record partial as routing-only context (loss_mask=0).
                self.manager.record_turn(
                    sid,
                    turn=TurnRecord(
                        prompt_ids=list(prompt_ids),
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
        """Block until SGLang resumes after weight sync.

        ``GenerateState`` is a process-wide singleton (``SingletonMeta``);
        by the time the adapter handles requests, the rollout worker has
        already initialised it with real args, so ``GenerateState(None)``
        returns the cached instance without re-running ``__init__``.
        """
        from slime.rollout.sglang_rollout import GenerateState
        state = GenerateState(None)
        while state.aborted:
            await asyncio.sleep(poll_interval)


class AdapterService(metaclass=SingletonMeta):
    """Per-process singleton: tokenizer + adapter + aiohttp server thread.

    Mirrors ``coding_agent_rl.generate._AdapterService``.  The first
    ``AdapterService(args)`` call starts the HTTP server; subsequent calls
    return the same instance, so concurrent ``generate()`` invocations
    share one adapter.
    """

    def __init__(self, args) -> None:
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"

        tool_parser = getattr(args, "sglang_tool_call_parser", None) or None
        reasoning_parser = getattr(args, "sglang_reasoning_parser", None) or None
        fork_threshold = (
            int(v) if (v := os.environ.get("SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS")) else None
        )
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)

        self.adapter = DataAgentAdapter(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=tool_parser,
            reasoning_parser=reasoning_parser,
            fork_threshold_tokens=fork_threshold,
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
