"""LongDS per-sample rollout for slime black-box RL training.

    --custom-generate-function-path examples.longds_rl.generate.generate

Each LongDS task runs as N consecutive ``claude -p`` invocations inside a
temporary workspace directory. Turn 1 starts a new Claude Code session;
turns 2+ use --resume to continue within the same workspace. Claude Code
connects to the AnthropicAdapter (via ANTHROPIC_BASE_URL env) so every
API call is intercepted and token logprobs are captured by SGLang.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import shutil
import subprocess
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slime.agent.adapters import AnthropicAdapter
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from examples.longds_rl import longds_task

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LongDSConfig:
    adapter_public_host: str | None
    adapter_bind_host: str
    adapter_port: int
    fork_merge_threshold: int | None
    turn_timeout_sec: int
    rollout_timeout_sec: int

    @classmethod
    def from_env(cls) -> LongDSConfig:
        turn_to = int(os.environ.get("LONGDS_TURN_TIMEOUT_SEC", "600"))
        rollout_to = int(os.environ.get("LONGDS_ROLLOUT_TIMEOUT_SEC", "0") or 0) or (turn_to * 20)
        fork = int(v) if (v := os.environ.get("SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS")) else None
        return cls(
            adapter_public_host=os.environ.get("ADAPTER_PUBLIC_HOST"),
            adapter_bind_host=os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"),
            adapter_port=int(os.environ.get("ADAPTER_PORT", "18001")),
            fork_merge_threshold=fork,
            turn_timeout_sec=turn_to,
            rollout_timeout_sec=rollout_to,
        )


CONFIG = LongDSConfig.from_env()

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
                "Claude Code can reach for connecting to the adapter."
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
# generate() — the entry point slime calls
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

    workdir = tempfile.mkdtemp(prefix=f"longds_{instance_id}_")
    _prepare_workspace(workdir, md)

    t0 = time.time()
    try:
        history: list[dict] = []
        rewards: list[float] = []

        for idx, turn in enumerate(md["turns"]):
            first_turn = idx == 0
            prompt = longds_task.build_prompt(
                turn, first_turn=first_turn, history=history,
            )

            exit_code = _run_claude_code(
                prompt=prompt,
                session_id=session_id,
                adapter_url=state.adapter_url,
                workdir=workdir,
                resume=not first_turn,
                timeout=CONFIG.turn_timeout_sec,
            )

            result = _read_answer_json(workdir)
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

            if time.time() - t0 > CONFIG.rollout_timeout_sec:
                logger.warning("longds_generate: %s rollout timeout after turn %d", instance_id, idx + 1)
                break

        aggregate_reward = sum(rewards) / len(rewards) if rewards else 0.0

    except Exception:
        logger.warning(
            "longds_generate: %s rollout failed:\n%s",
            instance_id, traceback.format_exc(),
        )
        return _abort(base_sample, f"exception:{traceback.format_exc()[:200]}", instance_id)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

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
# Workspace & subprocess helpers
# ---------------------------------------------------------------------------


def _prepare_workspace(workdir: str, md: dict) -> None:
    """Copy LongDS data files into workdir/data/."""
    data_root = md.get("data_root", "")
    files = md.get("data_files", [])
    if not files:
        return
    data_dir = os.path.join(workdir, "data")
    os.makedirs(data_dir, exist_ok=True)
    for fname in files:
        src = os.path.join(data_root, fname) if data_root else fname
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(data_dir, fname))


def _run_claude_code(
    *,
    prompt: str,
    session_id: str,
    adapter_url: str,
    workdir: str,
    resume: bool,
    timeout: int,
) -> int:
    """Run one turn of ``claude -p`` and return its exit code."""
    cmd = [
        "claude", "-p", prompt,
        "--permission-mode", "bypassPermissions",
        "--output-format", "stream-json",
        "--verbose",
    ]
    if resume:
        cmd += ["--resume", session_id]
    else:
        cmd += ["--session-id", session_id]

    env = {
        **os.environ,
        "ANTHROPIC_BASE_URL": adapter_url,
        "ANTHROPIC_AUTH_TOKEN": session_id,
        "ANTHROPIC_MODEL": "slime-actor",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "NO_COLOR": "1",
    }

    try:
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        return proc.returncode
    except subprocess.TimeoutExpired:
        logger.warning("longds_generate: claude -p timeout after %ds", timeout)
        return -1


def _read_answer_json(workdir: str) -> dict:
    """Read answer.json produced by Claude Code in the workspace."""
    path = os.path.join(workdir, "answer.json")
    try:
        raw = Path(path).read_text()
    except FileNotFoundError:
        return {"answer": "", "reasoning": ""}
    if not raw.strip():
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_metadata(sample: Sample) -> dict:
    md = dict(sample.metadata or {})
    if "turns" not in md:
        raise KeyError("sample.metadata missing 'turns'")
    return md


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
