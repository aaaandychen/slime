"""LongDS-flavored ClaudeCodeHarness — adds --resume for multi-turn state.

Reuses ClaudeCodeHarness.install_cli() and write_config() unchanged.
Only launch_and_wait and run are overridden to thread the resume flag
across consecutive turns inside the same sandbox session.
"""

from __future__ import annotations

import json
import os
import shlex

from slime.agent.harness.claude_code import ClaudeCodeHarness
from slime.agent.harness.common import HarnessContext, run_agent
from slime.agent.sandbox import Sandbox, ensure_agent_user


class LongDSHarness(ClaudeCodeHarness):
    """ClaudeCodeHarness variant that passes --resume on turn 2+.

    install_cli and write_config are inherited unchanged.
    """

    name = "claude_code_longds"

    async def launch_and_wait(
        self,
        sb: Sandbox,
        ctx: HarnessContext,
        prompt: str,
        time_budget_sec: int,
        *,
        resume: bool = False,
    ) -> int:
        """Run one turn of Claude Code inside the sandbox.

        Args:
            resume: If True, pass --resume <session_id> so Claude Code
                    picks up the previous turn's state. If False (first
                    turn), pass --session-id <session_id> to start new.
        """
        cmd = f"/usr/local/bin/claude -p {shlex.quote(prompt)} {self.launch_flags}"

        extra = os.environ.get(self.extra_args_env, "").strip()
        if extra:
            cmd = f"{cmd} {extra}"

        if resume:
            cmd = f"{cmd} --resume {shlex.quote(ctx.session_id)}"
        else:
            cmd = f"{cmd} --session-id {shlex.quote(ctx.session_id)}"

        env = {
            "ANTHROPIC_BASE_URL": ctx.adapter_url,
            "ANTHROPIC_AUTH_TOKEN": ctx.session_id,
            "ANTHROPIC_MODEL": ctx.model_label,
            **self.static_env,
        }
        extra_envs = os.environ.get(self.extra_envs_env, "").strip()
        if extra_envs:
            env.update(json.loads(extra_envs))

        return await run_agent(
            sb,
            workdir=ctx.workdir,
            start_cmd=cmd,
            env=env,
            time_budget_sec=time_budget_sec,
        )

    async def run(
        self,
        sb: Sandbox,
        *,
        workdir: str,
        session_id: str,
        adapter_url: str,
        time_budget_sec: int,
        prompt: str,
        resume: bool = False,
    ) -> int:
        """Run one turn: ensure user → write config → launch and wait."""
        await ensure_agent_user(sb, workdir)
        ctx = HarnessContext(
            workdir=workdir,
            session_id=session_id,
            adapter_url=adapter_url,
        )
        await self.write_config(sb, ctx)
        return await self.launch_and_wait(
            sb, ctx, prompt, time_budget_sec, resume=resume,
        )
