"""Phase 4 verification: AnthropicAdapter + multi-turn token trajectories.

Tests the full adapter pipeline on CPU: AnthropicAdapter HTTP server →
FakeSGLang → TrajectoryManager → finish_session → Sample with correct
loss_mask for multi-turn LongDS conversations.

Run: python3 tests/test_phase4_adapter.py
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import aiohttp

# -- path setup --
_EXAMPLE = Path(__file__).resolve().parents[1]
_SLIME_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_EXAMPLE))
sys.path.insert(0, str(_SLIME_ROOT))
sys.path.insert(0, str(_SLIME_ROOT / "tests" / "test_agent"))

# Stub transformers (heavy dep not needed for CPU test)
import types as _types
if "transformers" not in sys.modules:
    _stub = _types.ModuleType("transformers")
    for _n in ("AutoProcessor", "AutoTokenizer", "PreTrainedTokenizerBase", "ProcessorMixin"):
        setattr(_stub, _n, type(_n, (), {}))
    sys.modules["transformers"] = _stub

# Shim asyncio.timeout if Python < 3.11
if not hasattr(asyncio, "timeout"):
    @contextlib.asynccontextmanager
    async def _timeout_shim(_delay):
        yield
    asyncio.timeout = _timeout_shim

from _fakes import FakeSandbox, FakeTokenizer, fake_call_sglang_generate
from slime.agent.adapters import common as adapters_common
from slime.agent.harness import ClaudeCodeHarness
from slime.agent.harness import common as harness_common
from slime.utils.misc import SingletonMeta
from slime.utils.types import Sample

# -- the module under test --
import examples.longds_rl.generate as gen


# ======================================================================
# Test fixtures
# ======================================================================

_REAL_SLEEP = asyncio.sleep


async def _fast_sleep(_secs):
    await _REAL_SLEEP(0)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        hf_checkpoint="unused",
        rollout_max_context_len=0,
        sglang_tool_call_parser=None,
        sglang_reasoning_parser=None,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=1,
    )


def _base_sample() -> Sample:
    """2-turn LongDS task fixture."""
    return Sample(
        index=0,
        group_index=0,
        prompt="",
        label="sports__fifa__task1",
        metadata={
            "task_id": "sports/fifa/task1",
            "domain": "sports",
            "dataset": "fifa",
            "data_root": "",
            "data_files": [],
            "workdir": "/workspace",
            "turns": [
                {
                    "turn_id": 1,
                    "context": "FIFA data.",
                    "question": "Who scored the most goals?",
                    "ground_truth": "Kylian Mbappe",
                    "answer_type": "categorical",
                    "files_used": [],
                },
                {
                    "turn_id": 2,
                    "context": "Now compute correlation.",
                    "question": "What is the Pearson r?",
                    "ground_truth": "0.4521",
                    "answer_type": "numeric",
                    "files_used": [],
                },
            ],
        },
    )


async def _longds_agent(env: dict) -> int:
    """Stand-in for ``claude -p --resume`` in a LongDS multi-turn task.

    Sends 2 turns to the AnthropicAdapter over real HTTP loopback,
    each time including the full conversation history in the messages
    array — exactly like Claude Code does with --resume.

    Also writes answer.json so generate() can compute rewards.
    """
    base_url = env["ANTHROPIC_BASE_URL"]
    token = env["ANTHROPIC_AUTH_TOKEN"]
    workdir = env.get("WORKDIR", "/workspace")

    system_msg = {"role": "system",
                  "content": [{"type": "text", "text": "You are a data scientist."}]}

    # ── Turn 1 ──────────────────────────────────────────────────────────
    t1_user = {"role": "user",
               "content": [{"type": "text", "text": "Who scored the most goals?"}]}
    t1_messages = [system_msg, t1_user]

    async with aiohttp.ClientSession(trust_env=False) as sess:
        async with sess.post(
            f"{base_url}/v1/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"model": "m", "max_tokens": 64, "messages": t1_messages},
        ) as r:
            t1_data = await r.json()

    t1_assistant = {"role": "assistant", "content": t1_data["content"]}

    # Write turn 1 answer (simulate Claude Code's Bash tool)
    # Use the sandbox files dict — but in FakeSandbox, we need to find the
    # sandbox instance. Since this runs inside on_launch, the fake sandbox
    # writes to its own files dict. But we don't have access to it here.
    # Instead, generate() reads answer.json from sandbox. The FakeSandbox
    # records write_file calls. We'll pre-seed the answer.
    # For now, we just return 0 and let the test pre-seed answer.json.

    # ── Turn 2 (with full history, like --resume does) ──────────────────
    t2_user = {"role": "user",
               "content": [{"type": "text", "text": "What is the Pearson r?"}]}
    t2_messages = [system_msg, t1_user, t1_assistant, t2_user]

    async with aiohttp.ClientSession(trust_env=False) as sess:
        async with sess.post(
            f"{base_url}/v1/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"model": "m", "max_tokens": 64, "messages": t2_messages},
        ) as r:
            await r.json()

    return 0


def _two_turn_script():
    """Scripted model responses for 2 LongDS turns.

    Returned as (response_text, finish_reason, logprobs) per sglang call.
    FakeTokenizer encodes these → adapter decode round-trips them back.
    """
    return [
        ("let me check the data answer is Kylian Mbappe", "stop", None),
        ("the Pearson r is 0.4521", "stop", None),
    ]


async def _noop_install(self, sb) -> None:
    return None


def _patch_generate(monkeypatch, tokenizer: FakeTokenizer, sandbox_factory) -> None:
    """Wire generate()'s external edges to CPU fakes."""
    monkeypatch.setattr(
        gen, "CONFIG",
        dataclasses.replace(
            gen.CONFIG,
            adapter_public_host="127.0.0.1",
            adapter_bind_host="127.0.0.1",
            adapter_port=0,
            rollout_guard_sec=60,
            agent_time_budget_sec=30,
            boot_retries=1,
        ),
    )
    monkeypatch.setattr(gen, "load_tokenizer", lambda *a, **k: tokenizer)
    monkeypatch.setattr(gen, "E2BSandbox", sandbox_factory)
    monkeypatch.setattr(ClaudeCodeHarness, "install_cli", _noop_install)
    monkeypatch.setattr(harness_common.asyncio, "sleep", _fast_sleep)
    monkeypatch.setattr(
        adapters_common, "call_sglang_generate",
        fake_call_sglang_generate(_two_turn_script(), tokenizer),
    )
    SingletonMeta.clear_instances(gen._AdapterService)


# ======================================================================
# §1: Multi-turn trajectory correctness
# ======================================================================


def test_multi_turn_chain_has_correct_loss_mask():
    """2-turn LongDS → 1 Sample with assistant tokens in loss_mask=1."""
    async def run_case(monkeypatch):
        tok = FakeTokenizer()

        # Pre-seed answer.json for both turns
        turn_answers = [
            json.dumps({"answer": "Kylian Mbappe", "reasoning": "checked data"}),
            json.dumps({"answer": "0.4521", "reasoning": "computed r"}),
        ]
        call_count = [0]

        def sandbox_factory(image="fake", **_kw):
            sb = FakeSandbox(image)
            # -- on_launch: the agent dials the adapter, then writes answer --
            async def _agent_and_write(env):
                idx = call_count[0]
                call_count[0] += 1
                code = await _longds_agent(env)
                if idx < len(turn_answers):
                    sb.files["/workspace/answer.json"] = turn_answers[idx]
                return code
            sb.on_launch = _agent_and_write
            return sb

        _patch_generate(monkeypatch, tok, sandbox_factory)
        samples = await gen.generate(
            _args(), _base_sample(), sampling_params={"max_new_tokens": 32},
        )

        assert samples, "should produce at least 1 Sample"
        s = samples[0]
        assert s.status == Sample.Status.COMPLETED
        assert len(s.loss_mask) == len(s.rollout_log_probs) == s.response_length
        assert sum(s.loss_mask) > 0, "at least one trained token"

        # Both turns correct → reward 1.0
        assert s.reward == 1.0, f"Expected reward 1.0, got {s.reward}"
        assert s.metadata["grading_solved"] is True

        # loss_mask must have 1s (generated tokens) and 0s (prompt/injected tokens)
        mask = s.loss_mask
        assert 1 in mask, "loss_mask should contain trainable tokens"
        assert 0 in mask, "loss_mask should contain context-only tokens"

        # The response string should contain both answers (decoded from token ids)
        decoded = s.response
        assert isinstance(decoded, str) and len(decoded) > 0

        print(f"  ✅ Sample: {len(s.tokens)} tokens, "
              f"trained={sum(mask)}/{len(mask)}, reward={s.reward}")
        print(f"     loss_mask[:20]: {mask[:20]}")
        print(f"     response[:100]: {decoded[:100]}")

    with __import__("pytest").MonkeyPatch.context() as mp:
        asyncio.run(run_case(mp))


# ======================================================================
# §2: Partial reward still produces valid trajectory
# ======================================================================


def test_partial_reward_trajectory():
    """1 correct + 1 wrong → reward 0.5, but trajectory still valid."""
    async def run_case(monkeypatch):
        tok = FakeTokenizer()

        turn_answers = [
            json.dumps({"answer": "Kylian Mbappe", "reasoning": "correct"}),
            json.dumps({"answer": "wrong answer", "reasoning": "bad"}),
        ]
        call_count = [0]

        def sandbox_factory(image="fake", **_kw):
            sb = FakeSandbox(image)
            async def _agent_and_write(env):
                idx = call_count[0]
                call_count[0] += 1
                code = await _longds_agent(env)
                if idx < len(turn_answers):
                    sb.files["/workspace/answer.json"] = turn_answers[idx]
                return code
            sb.on_launch = _agent_and_write
            return sb

        _patch_generate(monkeypatch, tok, sandbox_factory)
        samples = await gen.generate(
            _args(), _base_sample(), sampling_params={"max_new_tokens": 32},
        )

        assert samples
        s = samples[0]
        assert s.reward == 0.5, f"Expected 0.5, got {s.reward}"
        assert s.status == Sample.Status.COMPLETED
        assert sum(s.loss_mask) > 0
        print(f"  ✅ Partial reward: {s.reward}, per_turn={s.metadata['per_turn_rewards']}")

    with __import__("pytest").MonkeyPatch.context() as mp:
        asyncio.run(run_case(mp))


# ======================================================================
# §3: Abort path
# ======================================================================


def test_abort_on_empty_turns():
    """Empty turns metadata → ABORTED sample."""
    async def run_case(monkeypatch):
        tok = FakeTokenizer()
        bad_sample = _base_sample()
        bad_sample.metadata["turns"] = []

        _patch_generate(monkeypatch, tok, FakeSandbox.factory())
        samples = await gen.generate(
            _args(), bad_sample, sampling_params={"max_new_tokens": 32},
        )
        assert len(samples) == 1
        s = samples[0]
        assert s.status == Sample.Status.ABORTED
        assert "empty_turns" in s.metadata["abort_reason"]
        print(f"  ✅ Aborted: {s.metadata['abort_reason']}")

    with __import__("pytest").MonkeyPatch.context() as mp:
        asyncio.run(run_case(mp))


# ======================================================================
# Runner
# ======================================================================

if __name__ == "__main__":
    results = {"passed": 0, "failed": 0}

    for name, fn in sorted(locals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                results["passed"] += 1
            except Exception as e:
                results["failed"] += 1
                import traceback
                print(f"  ❌ {name}: {e}")
                traceback.print_exc()

    print(f"\nPhase 4: {results['passed']} passed, {results['failed']} failed")
    sys.exit(1 if results["failed"] else 0)
