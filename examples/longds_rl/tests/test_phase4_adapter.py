"""Phase 4 verification: AnthropicAdapter + multi-turn token trajectories.

Tests the full pipeline on CPU: AnthropicAdapter HTTP server →
FakeSGLang → TrajectoryManager → finish_session → Sample with correct
loss_mask for multi-turn LongDS conversations.

The ``claude -p`` subprocess is mocked — instead of launching the real
CLI, the mock writes answer.json and sends Anthropic Messages API
requests to the adapter (simulating what Claude Code would do).

Run: python3 tests/test_phase4_adapter.py
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

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

from _fakes import FakeTokenizer, fake_call_sglang_generate
from slime.agent.adapters import common as adapters_common
from slime.utils.misc import SingletonMeta
from slime.utils.types import Sample

# -- the module under test --
import examples.longds_rl.generate as gen


# ======================================================================
# Fixtures
# ======================================================================

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
    return Sample(
        index=0,
        group_index=0,
        prompt="",
        label="sports__fifa__task1",
        metadata={
            "task_id": "sports/fifa/task1",
            "domain": "sports",
            "data_root": "",
            "data_files": [],
            "turns": [
                {"turn_id": 1, "context": "FIFA data.",
                 "question": "Who scored the most goals?",
                 "ground_truth": "Kylian Mbappe",
                 "answer_type": "categorical", "files_used": []},
                {"turn_id": 2, "context": "Now correlation.",
                 "question": "What is the Pearson r?",
                 "ground_truth": "0.4521",
                 "answer_type": "numeric", "files_used": []},
            ],
        },
    )


# ======================================================================
# Mock claude -p: writes answer.json AND dials the adapter
# ======================================================================

def _make_mock_run_claude(turn_answers: list[str], adapter_url: str,
                           session_id: str):
    """Return a mock for ``subprocess.run`` that simulates one ``claude -p``

    Each call:
    1. Sends one Anthropic Messages request to the adapter (simulating
       what Claude Code would do)
    2. Writes answer.json into the workdir (simulating CC's Bash tool)
    """
    call_count = [0]
    history: list[dict] = []

    def mock_run(cmd, *, cwd, env, timeout, capture_output, text, **kwargs):
        idx = call_count[0]
        call_count[0] += 1

        # Step 1: dial the adapter synchronously (subprocess.run is sync)
        import urllib.request
        base = env.get("ANTHROPIC_BASE_URL", adapter_url)
        token = env.get("ANTHROPIC_AUTH_TOKEN", session_id)

        system = {"role": "system",
                  "content": [{"type": "text", "text": "You are a data scientist."}]}
        questions = [
            [{"type": "text", "text": "Who scored the most goals?"}],
            [{"type": "text", "text": "What is the Pearson r?"}],
        ]

        messages = [system] + list(history)
        if idx < len(questions):
            messages.append({"role": "user", "content": questions[idx]})

        body = json.dumps({"model": "m", "max_tokens": 64, "messages": messages})
        req = urllib.request.Request(
            f"{base}/v1/messages",
            data=body.encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())

        # Record this exchange in accumulated history
        if idx < len(questions):
            history.append({"role": "user", "content": questions[idx]})
        history.append({"role": "assistant", "content": data["content"]})

        # Step 2: write answer.json (like CC's Bash tool)
        if idx < len(turn_answers):
            answer_path = os.path.join(cwd, "answer.json")
            with open(answer_path, "w") as f:
                f.write(turn_answers[idx])

        return SimpleNamespace(returncode=0)

    return mock_run


def _patch_generate(monkeypatch, tokenizer: FakeTokenizer, mock_run) -> str:
    """Wire generate()'s external edges to CPU fakes. Returns workdir path."""
    monkeypatch.setattr(
        gen, "CONFIG",
        dataclasses.replace(
            gen.CONFIG,
            adapter_public_host="127.0.0.1",
            adapter_bind_host="127.0.0.1",
            adapter_port=0,
            rollout_timeout_sec=60,
            turn_timeout_sec=30,
        ),
    )
    monkeypatch.setattr(gen, "load_tokenizer", lambda *a, **k: tokenizer)
    monkeypatch.setattr(gen.subprocess, "run", mock_run)
    monkeypatch.setattr(
        adapters_common, "call_sglang_generate",
        fake_call_sglang_generate(_two_turn_script(), tokenizer),
    )

    # Fixed temp dir so the mock can write answer.json to it
    workdir = tempfile.mkdtemp(prefix="longds_test_")
    monkeypatch.setattr(gen.tempfile, "mkdtemp", lambda prefix="": workdir)
    # copytree is a noop (data_files is empty in test fixture)
    monkeypatch.setattr(gen.shutil, "copytree", lambda *a, **kw: None)
    monkeypatch.setattr(gen, "_prepare_workspace", lambda w, m: None)
    # rmtree is a noop (cleanup after test)
    monkeypatch.setattr(gen.shutil, "rmtree", lambda *a, **kw: None)

    SingletonMeta.clear_instances(gen._AdapterService)
    return workdir


def _two_turn_script():
    return [
        ("answer is Kylian Mbappe", "stop", None),
        ("the Pearson r is 0.4521", "stop", None),
    ]


# ======================================================================
# §1: Multi-turn trajectory correctness
# ======================================================================


def test_multi_turn_chain_has_correct_loss_mask():
    """2-turn → 1 Sample with correct loss_mask and reward."""
    async def run_case(monkeypatch):
        tok = FakeTokenizer()
        turn_answers = [
            json.dumps({"answer": "Kylian Mbappe", "reasoning": "checked data"}),
            json.dumps({"answer": "0.4521", "reasoning": "computed r"}),
        ]

        # Need adapter_url before patching — _AdapterService reads CONFIG
        mock_run = _make_mock_run_claude(
            turn_answers, adapter_url="http://127.0.0.1:1", session_id="test-sid",
        )
        _patch_generate(monkeypatch, tok, mock_run)

        samples = await gen.generate(
            _args(), _base_sample(), sampling_params={"max_new_tokens": 32},
        )

        assert samples, "should produce at least 1 Sample"
        s = samples[0]
        assert s.status == Sample.Status.COMPLETED
        assert len(s.loss_mask) == len(s.rollout_log_probs) == s.response_length
        assert sum(s.loss_mask) > 0, "at least one trained token"
        assert s.reward == 1.0, f"Expected 1.0, got {s.reward}"
        assert s.metadata["grading_solved"] is True
        assert 1 in s.loss_mask and 0 in s.loss_mask

        print(f"  ✅ Sample: {len(s.tokens)} tokens, "
              f"trained={sum(s.loss_mask)}/{len(s.loss_mask)}, reward={s.reward}")

    with __import__("pytest").MonkeyPatch.context() as mp:
        asyncio.run(run_case(mp))


# ======================================================================
# §2: Partial reward
# ======================================================================


def test_partial_reward_trajectory():
    """1 correct + 1 wrong → reward 0.5."""
    async def run_case(monkeypatch):
        tok = FakeTokenizer()
        turn_answers = [
            json.dumps({"answer": "Kylian Mbappe", "reasoning": "correct"}),
            json.dumps({"answer": "wrong answer", "reasoning": "bad"}),
        ]

        mock_run = _make_mock_run_claude(
            turn_answers, adapter_url="http://127.0.0.1:1", session_id="test-sid",
        )
        _patch_generate(monkeypatch, tok, mock_run)

        samples = await gen.generate(
            _args(), _base_sample(), sampling_params={"max_new_tokens": 32},
        )
        assert samples
        s = samples[0]
        assert s.reward == 0.5, f"Expected 0.5, got {s.reward}"
        assert s.metadata["per_turn_rewards"] == [1.0, 0.0]
        print(f"  ✅ Partial: reward={s.reward}, per_turn={s.metadata['per_turn_rewards']}")

    with __import__("pytest").MonkeyPatch.context() as mp:
        asyncio.run(run_case(mp))


# ======================================================================
# §3: Abort path
# ======================================================================


def test_abort_on_empty_turns():
    """Empty turns → ABORTED."""
    async def run_case(monkeypatch):
        tok = FakeTokenizer()
        mock_run = _make_mock_run_claude(
            [], adapter_url="http://127.0.0.1:1", session_id="test-sid",
        )
        _patch_generate(monkeypatch, tok, mock_run)

        bad = _base_sample()
        bad.metadata["turns"] = []
        samples = await gen.generate(
            _args(), bad, sampling_params={"max_new_tokens": 32},
        )
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
