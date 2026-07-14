"""LongDS per-sample rollout for slime black-box RL training.

Registered as: --custom-generate-function-path examples.longds_rl.generate.generate

Phase 3: Full orchestration with sandbox + harness, no Adapter yet.
Phase 4: Add AnthropicAdapter for token trajectory capture.

Reuses:
  - slime.agent.harness.ClaudeCodeHarness (install_cli, write_config)
  - slime.agent.harness.common.run_agent (detached launch + poll)
  - slime.agent.sandbox.Sandbox protocol + E2BSandbox
  - slime.utils.types.Sample

Only the task layer (longds_task) and resume-aware harness
(longds_harness.LongDSHarness) are LongDS-specific.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
import traceback
from typing import Any

from slime.agent.sandbox import E2BSandbox
from slime.utils.types import Sample

from examples.longds_rl import longds_task
from examples.longds_rl.longds_harness import LongDSHarness

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (overridable via env)
# ---------------------------------------------------------------------------

_DEFAULT_IMAGE = "python:3.11-slim"  # placeholder — set per-cluster
_DEFAULT_WORKDIR = "/workspace"
_DEFAULT_TURN_BUDGET_SEC = 600   # 10 min per turn
_DEFAULT_BOOT_TIMEOUT_SEC = 300  # 5 min for sandbox boot + CLI install


# ---------------------------------------------------------------------------
# generate() — the entry point slime calls
# ---------------------------------------------------------------------------


async def generate(
    args: Any,
    base_sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> list[Sample]:
    """Run one LongDS task (N turns) in a sandbox via Claude Code.

    Turn 1 starts a new Claude Code session; turns 2+ use --resume to
    continue inside the same sandbox (preserving files and conversation).
    """
    md = _parse_metadata(base_sample)
    instance_id = md.get("task_id", base_sample.label or "unknown")

    reason = longds_task.validate_metadata(md)
    if reason:
        return _abort(base_sample, f"invalid_metadata:{reason}", instance_id)

    session_id = base_sample.session_id or _new_session_id(base_sample, instance_id)
    harness = LongDSHarness()
    image = md.get("image") or _DEFAULT_IMAGE
    workdir = md.get("workdir") or _DEFAULT_WORKDIR

    t0 = time.time()
    try:
        sb = await _boot_sandbox(image, instance_id, harness)
        try:
            # ── per-turn loop ──────────────────────────────────────
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
                    adapter_url="",  # Phase 4: set real adapter URL
                    time_budget_sec=_DEFAULT_TURN_BUDGET_SEC,
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

    except Exception:
        logger.warning(
            "longds_generate: %s rollout failed:\n%s",
            instance_id, traceback.format_exc(),
        )
        return _abort(base_sample, f"exception:{traceback.format_exc()[:200]}", instance_id)

    # ── build result ──────────────────────────────────────────────
    base_sample.reward = float(aggregate_reward)
    base_sample.status = Sample.Status.COMPLETED
    base_sample.metadata = {
        **(base_sample.metadata or {}),
        "instance_id": instance_id,
        "per_turn_rewards": rewards,
        "num_turns": len(md["turns"]),
        "grading_solved": aggregate_reward == 1.0,
        "elapsed_sec": time.time() - t0,
    }

    logger.info(
        "longds_generate: %s reward=%.2f turns=%d elapsed=%.1fs",
        instance_id, aggregate_reward, len(md["turns"]), time.time() - t0,
    )
    return [base_sample]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_metadata(sample: Sample) -> dict:
    md = dict(sample.metadata or {})
    if "turns" not in md:
        raise KeyError("sample.metadata missing 'turns'")
    return md


async def _read_answer_json(sb, workdir: str) -> dict:
    import json
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


async def _boot_sandbox(image: str, instance_id: str, harness: LongDSHarness):
    """Boot sandbox and install harness CLI. Caller must __aexit__."""
    sb = E2BSandbox(image)
    await sb.__aenter__()
    try:
        await harness.install_cli(sb)
    except BaseException:
        await sb.__aexit__(None, None, None)
        raise
    return sb


def _new_session_id(sample: Sample, instance_id: str) -> str:
    if sample.index is not None and sample.group_index is not None:
        return f"longds-{instance_id}-{sample.index}-{sample.group_index}"
    return f"longds-{instance_id}-{secrets.token_hex(8)}"


def _abort(base_sample: Sample, reason: str, instance_id: str) -> list[Sample]:
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
