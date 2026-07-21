"""DataMind RL: per-sample generate() for slime.

    --custom-generate-function-path examples.datamind_rl.generate.generate

Flow: Claude Code (local, no sandbox) → write answer.json → evaluate against ground_truth.
Reuses the LongDS runner pattern without fixed-turn prompts.
"""

from __future__ import annotations

import json
import logging
import os
import re
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

logger = logging.getLogger(__name__)

# ── System prompt that Claude Code will wrap into a session ────────────
SYSTEM_PROMPT = """\
You are an expert data analyst. Given a data analysis task and data files in the `data/` directory, inspect the data, run analysis, and produce the final answer.

**Workflow**:
1. Start by inspecting the data: for SQL tasks call `get_db_info()`; for CSV tasks read a sample with `head`.
2. Write and execute your analysis code.
3. After each execution, judge: "Do I already have the information needed to answer?" If yes, immediately output the answer. If the code failed or the result is unexpected, you may fix and retry, but avoid re-running queries that already succeeded.
4. When ready, write the final answer:
   echo '{"answer":"<your final answer>","reasoning":"<brief method>"}' > answer.json

**Rules**:
- Do NOT modify files under `data/`.
- Do NOT repeat the same query or file inspection once you already have the result.
- If you find yourself exploring without making progress, stop and provide your best answer based on what you already know.
- For SQL queries, use `sqlite3` or `python3` with the database.
- The final answer.json is required to complete the task."""


@dataclass(frozen=True)
class DataMindConfig:
    adapter_public_host: str | None
    adapter_bind_host: str
    adapter_port: int
    fork_merge_threshold: int | None
    timeout_sec: int

    @classmethod
    def from_env(cls) -> DataMindConfig:
        fork = int(v) if (v := os.environ.get("SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS")) else None
        return cls(
            adapter_public_host=os.environ.get("ADAPTER_PUBLIC_HOST"),
            adapter_bind_host=os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"),
            adapter_port=int(os.environ.get("ADAPTER_PORT", "18001")),
            fork_merge_threshold=fork,
            timeout_sec=int(os.environ.get("DATAMIND_TIMEOUT_SEC", "1200")),
        )


CONFIG = DataMindConfig.from_env()


class _AdapterService(metaclass=SingletonMeta):
    def __init__(self, args) -> None:
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        self.tool_parser = getattr(args, "sglang_tool_call_parser", None) or None
        self.reasoning_parser = getattr(args, "sglang_reasoning_parser", None) or None

        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        if not CONFIG.adapter_public_host:
            raise RuntimeError("ADAPTER_PUBLIC_HOST is not set.")

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
        logger.info("datamind_rl: adapter=%s max_ctx=%s", self.adapter_url, self.max_context_len)


async def generate(
    args: Any,
    base_sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> list[Sample]:
    state = _AdapterService(args)
    md = dict(base_sample.metadata or {})
    instance_id = md.get("task_id", base_sample.label or "unknown")

    session_id = base_sample.session_id or _new_session_id(base_sample, instance_id)
    state.adapter.open_session(
        session_id,
        sampling_defaults=sampling_params,
        max_context_tokens=state.max_context_len,
    )

    workdir = tempfile.mkdtemp(prefix=f"datamind_{instance_id.replace('/', '_')}_")
    _prepare_workspace(workdir, md)

    t0 = time.time()
    try:
        prompt = md.get("prompt") or base_sample.prompt or ""
        exit_code = _run_claude_code(
            prompt=prompt,
            session_id=session_id,
            adapter_url=state.adapter_url,
            workdir=workdir,
            timeout=CONFIG.timeout_sec,
        )

        result = _read_answer_json(workdir)
        answer = result.get("answer", "")
        reasoning = result.get("reasoning", "")

        ground_truth = md.get("ground_truth", "")

        # Delegate detailed scoring to the reward function
        logger.info(
            "datamind_rl: %s exit=%d answer_len=%d elapsed=%.1fs",
            instance_id,
            exit_code,
            len(answer),
            time.time() - t0,
        )

    except Exception:
        logger.warning("datamind_rl: %s failed:\n%s", instance_id, traceback.format_exc())
        return _abort(base_sample, f"exception:{traceback.format_exc()[:200]}", instance_id)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    # ── eval path ──
    if evaluation:
        base_sample.reward = 0.0  # Will be filled by eval separately
        base_sample.status = Sample.Status.COMPLETED
        base_sample.metadata = {
            **(base_sample.metadata or {}),
            "instance_id": instance_id,
            "answer": answer,
            "reasoning": reasoning,
            "ground_truth": ground_truth,
            "agent_exit_code": exit_code,
        }
        return [base_sample]

    # ── train path ──
    samples = await state.adapter.finish_session(
        session_id,
        base_sample=base_sample,
        reward=None,
        extra_metadata={
            "instance_id": instance_id,
            "answer": answer,
            "reasoning": reasoning,
            "ground_truth": ground_truth,
            "agent_exit_code": exit_code,
            "data_source": md.get("data_source", ""),
        },
    )

    if not samples:
        return _abort(base_sample, "adapter_session_empty", instance_id)

    logger.info(
        "datamind_rl: %s exit=%d segments=%d elapsed=%.1fs",
        instance_id, exit_code, len(samples), time.time() - t0,
    )
    return samples


# ── Workspace & execution ────────────────────────────────────────────


def _ensure_claude_perms() -> None:
    """Ensure the claude user can access the claude binary and its native deps.

    Called before every rollout to protect against Claude Code auto-updates
    that create new files without world-readable permissions.
    """
    try:
        subprocess.run(["chmod", "-R", "o+rX", "/root/.nvm"], capture_output=True, timeout=10)
        subprocess.run(["chmod", "-R", "o+rX", "/root/.cache"], capture_output=True, timeout=10)
    except Exception:
        pass


def _ensure_workdir_perms(workdir: str) -> None:
    """Give the claude user read/write access to the workspace."""
    try:
        subprocess.run(["chown", "-R", "claude:claude", workdir], capture_output=True, timeout=10)
    except Exception:
        pass


def _prepare_workspace(workdir: str, md: dict) -> None:
    data_root = md.get("data_root", "")
    if not data_root:
        return
    # Extract the data file path from the prompt
    prompt = md.get("prompt", "")
    m = re.search(r"data/files/(\S+)", prompt)
    if not m:
        return
    rel_path = m.group(1).rstrip(".")
    src = os.path.join(data_root, rel_path)
    if not os.path.exists(src):
        logger.warning("datamind_rl: data file not found: %s", src)
        return
    # Create the directory structure in workspace
    dst_dir = os.path.join(workdir, "data", "files")
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dst_dir, rel_path))


def _run_claude_code(
    *,
    prompt: str,
    session_id: str,
    adapter_url: str,
    workdir: str,
    timeout: int,
) -> int:
    # Claude Code refuses --dangerously-skip-permissions as root.
    # Run via sudo to the 'claude' user instead.
    # Ensure claude user can access the binary and workspace.
    _ensure_claude_perms()
    _ensure_workdir_perms(workdir)

    cmd = [
        "sudo", "-E", "-u", "claude",
        "env", "HOME=/home/claude",
        "/root/.nvm/versions/node/v22.23.1/bin/claude", "-p", prompt,
        "--permission-mode", "bypassPermissions",
        "--output-format", "stream-json",
        "--verbose",
        "--session-id", session_id,
    ]

    env = {
        **os.environ,
        "ANTHROPIC_BASE_URL": adapter_url,
        "ANTHROPIC_AUTH_TOKEN": session_id,
        "ANTHROPIC_MODEL": "slime-actor",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "NO_COLOR": "1",
    }

    try:
        proc = subprocess.run(cmd, cwd=workdir, env=env, timeout=timeout, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.warning("datamind_rl: claude stderr: %s", proc.stderr[-500:])
        return proc.returncode
    except subprocess.TimeoutExpired:
        logger.warning("datamind_rl: claude -p timeout after %ds", timeout)
        return -1


def _read_answer_json(workdir: str) -> dict:
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


# ── Helpers ──────────────────────────────────────────────────────────


def _new_session_id(sample: Sample, instance_id: str) -> str:
    if sample.index is not None and sample.group_index is not None:
        return f"datamind-{instance_id}-{sample.index}-{sample.group_index}"
    return f"datamind-{instance_id}-{secrets.token_hex(8)}"


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
    logger.warning("datamind_rl: %s aborted: %s", instance_id, reason)
    return [base_sample]
