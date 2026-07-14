"""LongDS per-sample rollout for slime black-box RL training.

    --custom-generate-function-path examples.longds_rl.generate.generate

Phase 4: Full pipeline — AnthropicAdapter → SGLang → token capture.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import secrets
import time
import traceback
from dataclasses import dataclass
from typing import Any

from slime.agent.adapters import AnthropicAdapter
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.agent.sandbox import E2BSandbox
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from examples.longds_rl import longds_task
from examples.longds_rl.longds_harness import LongDSHarness

logger = logging.getLogger(__name__)
logging.getLogger("e2b").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LongDSConfig:
    adapter_public_host: str | None
    adapter_bind_host: str
    adapter_port: int
    fork_merge_threshold: int | None
    agent_time_budget_sec: int
    rollout_guard_sec: int
    boot_concurrency: int
    boot_retries: int

    @classmethod
    def from_env(cls) -> LongDSConfig:
        agent_total = int(os.environ.get("LONGDS_AGENT_TIME_BUDGET_SEC", "3600"))
        guard = int(os.environ.get("LONGDS_ROLLOUT_GUARD_SEC", "0") or 0) or agent_total + 300
        fork = int(v) if (v := os.environ.get("SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS")) else None
        return cls(
            adapter_public_host=os.environ.get("ADAPTER_PUBLIC_HOST"),
            adapter_bind_host=os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"),
            adapter_port=int(os.environ.get("ADAPTER_PORT", "18001")),
            fork_merge_threshold=fork,
            agent_time_budget_sec=agent_total,
            rollout_guard_sec=guard,
            boot_concurrency=int(os.environ.get("LONGDS_BOOT_CONCURRENCY", "16")),
            boot_retries=int(os.environ.get("LONGDS_BOOT_RETRIES", "2")),
        )


CONFIG = LongDSConfig.from_env()
_BOOT_SEM = asyncio.Semaphore(CONFIG.boot_concurrency)


# ---------------------------------------------------------------------------
# Adapter service (singleton per Ray worker)
# ---------------------------------------------------------------------------


class _AdapterService(metaclass=SingletonMeta):
    def __init__(self, args) -> None:
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        self.tool_parser = getattr(args, "sglang_tool_call_parser", None) or None
        self.reasoning_parser = getattr(args, "sglang_reasoning_parser", None) or None

        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        if not CONFIG.adapter_public_host:
            raise RuntimeError(
                "ADAPTER_PUBLIC_HOST is not set. Export it to the host IP that "
                "sandboxes can reach for reverse-connection to the adapter."
            )

        self.adapter = AnthropicAdapter(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=self.tool_parser,
            reasoning_parser=self.reasoning_parser,
            fork_threshold_tokens=CONFIG.fork_merge_threshold,
        )
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=CONFIG.adapter_bind_host,
            port=CONFIG.adapter_port,
            thread_name="anthropic-adapter",
            runner_kwargs={
                "handler_cancellation": True,
                "access_log_class": FilteredAccessLogger,
            },
        )
        self.adapter_url = f"http://{CONFIG.adapter_public_host}:{self.app_handle.port}"
        logger.info(
            "longds_generate: tokenizer=%s adapter=%s max_ctx=%s",
            args.hf_checkpoint, self.adapter_url, self.max_context_len,
        )


# ---------------------------------------------------------------------------
# Sandbox boot
# ---------------------------------------------------------------------------


async def _boot_sandbox(image: str, instance_id: str, harness: LongDSHarness) -> E2BSandbox:
    sb = None
    last_err: Exception | None = None
    for attempt in range(CONFIG.boot_retries):
        cand = E2BSandbox(image)
        try:
            async with _BOOT_SEM:
                await cand.__aenter__()
                try:
                    await harness.install_cli(cand)
                except BaseException:
                    await cand.__aexit__(None, None, None)
                    raise
            sb = cand
            break
        except Exception as e:
            last_err = e
            logger.warning(
                "longds_generate: %s boot attempt %d/%d: %s: %s",
                instance_id, attempt + 1, CONFIG.boot_retries,
                type(e).__name__, str(e)[:200],
            )
            await asyncio.sleep(1 + attempt + random.random())
    if sb is None:
        raise last_err  # type: ignore[misc]
    return sb


# ---------------------------------------------------------------------------
# generate() — the entry point
# ---------------------------------------------------------------------------


async def generate(
    args: Any,
    base_sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> list[Sample]:
    state = _AdapterService(args)
    md = _parse_metadata(base_sample)
    instance_id = md.get("task_id", base_sample.label or "unknown")

    reason = longds_task.validate_metadata(md)
    if reason:
        return _abort(base_sample, f"invalid_metadata:{reason}", instance_id)

    session_id = base_sample.session_id or _new_session_id(base_sample, instance_id)
    state.adapter.open_session(
        session_id,
        sampling_defaults=sampling_params,
        max_context_tokens=state.max_context_len,
    )

    harness = LongDSHarness()
    image = md.get("image") or "python:3.11-slim"
    workdir = md.get("workdir") or "/workspace"

    t0 = time.time()
    try:
        async with asyncio.timeout(CONFIG.rollout_guard_sec):
            sb = await _boot_sandbox(image, instance_id, harness)
            try:
                await _upload_data_files(sb, workdir, md)

                history: list[dict] = []
                rewards: list[float] = []

                for idx, turn in enumerate(md["turns"]):
                    first_turn = idx == 0
                    prompt = longds_task.build_prompt(
                        turn, first_turn=first_turn, history=history,
                    )

                    exit_code = await harness.run(
                        sb,
                        workdir=workdir,
                        session_id=session_id,
                        adapter_url=state.adapter_url,
                        time_budget_sec=CONFIG.agent_time_budget_sec,
                        prompt=prompt,
                        resume=not first_turn,
                    )

                    result = await _read_answer_json(sb, workdir)
                    reward = longds_task.evaluate_answer(
                        result.get("answer", ""),
                        turn["ground_truth"],
                        turn.get("answer_type", "auto"),
                    )
                    rewards.append(reward)
                    history.append({
                        "turn_id": turn["turn_id"],
                        "question": turn["question"],
                        "answer": result.get("answer", ""),
                    })

                    logger.info(
                        "longds_generate: %s turn %d/%d reward=%.1f exit=%d",
                        instance_id, idx + 1, len(md["turns"]), reward, exit_code,
                    )

                aggregate_reward = sum(rewards) / len(rewards) if rewards else 0.0

            finally:
                await sb.__aexit__(None, None, None)

    except asyncio.TimeoutError:
        logger.warning("longds_generate: %s timeout after %.1fs", instance_id, time.time() - t0)
        return _abort(base_sample, "wall_clock_timeout", instance_id)
    except Exception:
        logger.warning("longds_generate: %s failed:\n%s", instance_id, traceback.format_exc())
        return _abort(base_sample, f"exception:{traceback.format_exc()[:200]}", instance_id)

    # ── eval path: no token trajectory ──────────────────────────────
    if evaluation:
        logger.info(
            "longds_generate: %s eval reward=%.2f elapsed=%.1fs",
            instance_id, aggregate_reward, time.time() - t0,
        )
        base_sample.reward = float(aggregate_reward)
        base_sample.status = Sample.Status.COMPLETED
        base_sample.metadata = {
            **(base_sample.metadata or {}),
            "instance_id": instance_id,
            "per_turn_rewards": rewards,
            "num_turns": len(md["turns"]),
            "grading_solved": aggregate_reward == 1.0,
        }
        return [base_sample]

    # ── train path: drain trajectory into Samples ───────────────────
    samples = await state.adapter.finish_session(
        session_id,
        base_sample=base_sample,
        reward=float(aggregate_reward),
        extra_metadata={
            "instance_id": instance_id,
            "num_turns": len(md["turns"]),
            "per_turn_rewards": rewards,
            "grading_solved": aggregate_reward == 1.0,
        },
    )

    if not samples:
        return _abort(base_sample, "adapter_session_empty", instance_id)

    logger.info(
        "longds_generate: %s reward=%.2f turns=%d segments=%d elapsed=%.1fs",
        instance_id, aggregate_reward, len(md["turns"]), len(samples), time.time() - t0,
    )
    return samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_metadata(sample: Sample) -> dict:
    md = dict(sample.metadata or {})
    if "turns" not in md:
        raise KeyError("sample.metadata missing 'turns'")
    return md


async def _upload_data_files(sb, workdir: str, md: dict) -> None:
    """Upload LongDS data files into sandbox data/."""
    data_root = md.get("data_root", "")
    files = md.get("data_files", [])
    if not files:
        return
    data_dir = f"{workdir}/data"
    await sb.exec(f"mkdir -p {data_dir}", user="agent")
    for fname in files:
        local = f"{data_root}/{fname}" if data_root else fname
        await sb.write_file(f"{data_dir}/{fname}", local, user="agent")


async def _read_answer_json(sb, workdir: str) -> dict:
    try:
        raw = await sb.read_file(f"{workdir}/answer.json", user="agent")
    except Exception:
        return {"answer": "", "reasoning": ""}
    if not raw or not raw.strip():
        return {"answer": "", "reasoning": ""}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {
                "answer": str(data.get("answer", "")),
                "reasoning": str(data.get("reasoning", data.get("reasoning_summary", ""))),
            }
        return {"answer": str(data), "reasoning": ""}
    except json.JSONDecodeError:
        return {"answer": raw.strip(), "reasoning": ""}


def _new_session_id(sample: Sample, instance_id: str) -> str:
    if sample.index is not None and sample.group_index is not None:
        return f"longds-{instance_id}-{sample.index}-{sample.group_index}"
    return f"longds-{instance_id}-{secrets.token_hex(8)}"


def _abort(base_sample: Sample, reason: str, instance_id: str) -> list[Sample]:
    base_sample.tokens = [0, 0]
    base_sample.response = ""
    base_sample.response_length = 1
    base_sample.loss_mask = [0]
    base_sample.rollout_log_probs = [0.0]
    base_sample.reward = 0.0
    base_sample.remove_sample = True
    base_sample.status = Sample.Status.ABORTED
    base_sample.metadata = {
        **(base_sample.metadata or {}),
        "abort_reason": reason,
        "instance_id": instance_id,
    }
    logger.warning("longds_generate: %s aborted: %s", instance_id, reason)
    return [base_sample]
