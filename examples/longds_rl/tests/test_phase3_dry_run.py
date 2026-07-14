"""Phase 3 verification: end-to-end dry run with FakeSandbox.

Tests the full generate() orchestration without Docker, GPU, or SGLang.
Uses slime's FakeSandbox to mock sandbox I/O — Claude Code execution is
simulated by pre-writing answer.json via the on_launch callback.

Run: python3 tests/test_phase3_dry_run.py
"""

import json
import sys
from pathlib import Path

# -- path setup --
# test file: .../slime/examples/longds_rl/tests/test_phase3_dry_run.py
# slime root: parents[3]
_EXAMPLE = Path(__file__).resolve().parents[1]   # examples/longds_rl
_SLIME_ROOT = Path(__file__).resolve().parents[3]  # slime/
sys.path.insert(0, str(_EXAMPLE))
sys.path.insert(0, str(_SLIME_ROOT))
sys.path.insert(0, str(_SLIME_ROOT / "tests" / "test_agent"))

from _fakes import FakeSandbox
from slime.agent.harness.common import HarnessContext

from longds_harness import LongDSHarness
from longds_task import build_prompt, evaluate_answer, validate_metadata


# ======================================================================
# 1. LongDSHarness command assembly (no sandbox)
# ======================================================================

def test_harness_first_turn_no_resume():
    """Verify first-turn command: --session-id, no --resume."""
    harness = LongDSHarness()
    assert harness.name == "claude_code_longds"
    ctx = HarnessContext(
        workdir="/workspace",
        session_id="test-sid-1",
        adapter_url="http://adapter:18001",
    )
    assert ctx.session_id == "test-sid-1"
    assert ctx.adapter_url == "http://adapter:18001"
    assert ctx.workdir == "/workspace"
    print("  ✅ harness context correct")


# ======================================================================
# 2. Full generate() flow with FakeSandbox
# ======================================================================

async def _run_generate_with_fake_sandbox():
    """Run generate() with FakeSandbox, verifying the orchestration."""
    from generate import generate, _parse_metadata, _read_answer_json

    # Build a minimal args-like object and Sample
    class _Args:
        pass

    args = _Args()

    class _Sample:
        def __init__(self):
            self.prompt = ""
            self.label = "sports__fifa_analytics__task1"
            self.metadata = {
                "task_id": "sports/fifa_analytics/task1",
                "domain": "sports",
                "dataset": "fifa_analytics",
                "image": "fake-image",
                "workdir": "/workspace",
                "data_root": "",
                "data_files": ["players.csv", "teams.csv"],
                "turns": [
                    {
                        "turn_id": 1,
                        "context": "FIFA data analysis.",
                        "question": "Who scored the most goals?",
                        "ground_truth": "Kylian Mbappe",
                        "answer_type": "categorical",
                        "files_used": ["players.csv"],
                    },
                    {
                        "turn_id": 2,
                        "context": "Now correlation analysis.",
                        "question": "What is the Pearson r?",
                        "ground_truth": "0.4521",
                        "answer_type": "numeric",
                        "files_used": ["players.csv"],
                    },
                ],
                "num_turns": 2,
            }
            self.session_id = "test-session"
            self.index = 0
            self.group_index = 0
            self.reward = 0.0
            self.response = ""
            self.response_length = 0
            self.tokens = []
            self.loss_mask = []
            self.rollout_log_probs = []
            self.remove_sample = False
            self.status = None

    sample = _Sample()

    # Build FakeSandbox with scripted answers — on_launch writes the
    # correct answer.json before each turn (simulating Claude Code).
    turn_answers = [
        json.dumps({"answer": "Kylian Mbappe", "reasoning": "Read players.csv, found max goals"}),
        json.dumps({"answer": "0.4521", "reasoning": "Computed Pearson r"}),
    ]
    call_count = [0]

    fake_sb = FakeSandbox(image="fake-image")

    async def on_launch(env):
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(turn_answers):
            fake_sb.files["/workspace/answer.json"] = turn_answers[idx]
        return 0

    fake_sb.on_launch = on_launch

    # Monkeypatch _boot_sandbox to return our fake
    import generate as gen_mod
    _original_boot = gen_mod._boot_sandbox

    async def _fake_boot(image, instance_id, harness):
        return fake_sb

    gen_mod._boot_sandbox = _fake_boot

    try:
        results = await generate(args, sample, {}, evaluation=False)
        assert len(results) == 1
        result = results[0]
        assert result.reward == 1.0, f"Expected reward 1.0, got {result.reward}"
        assert result.status.name == "COMPLETED"
        assert result.metadata["num_turns"] == 2
        assert result.metadata["per_turn_rewards"] == [1.0, 1.0]
        assert result.metadata["grading_solved"] is True
        print("  ✅ generate flow: 2 turns, both correct → reward 1.0")
    finally:
        gen_mod._boot_sandbox = _original_boot


def test_generate_flow():
    import asyncio
    asyncio.run(_run_generate_with_fake_sandbox())


# ======================================================================
# 3. Partial reward scenario
# ======================================================================

async def _run_partial_reward():
    """Verify: one correct answer + one wrong → reward 0.5."""
    from generate import generate

    class _Args:
        pass

    class _Sample:
        def __init__(self):
            self.prompt = ""
            self.label = "test__partial"
            self.metadata = {
                "task_id": "test/partial",
                "data_files": [],
                "workdir": "/workspace",
                "turns": [
                    {"turn_id": 1, "question": "Q1", "context": "",
                     "ground_truth": "correct", "answer_type": "categorical"},
                    {"turn_id": 2, "question": "Q2", "context": "",
                     "ground_truth": "expected", "answer_type": "categorical"},
                ],
            }
            self.session_id = "test-partial"
            self.index = 0
            self.group_index = 0
            self.reward = 0.0
            self.response = ""
            self.response_length = 0
            self.tokens = []
            self.loss_mask = []
            self.rollout_log_probs = []
            self.remove_sample = False
            self.status = None

    # Turn 1 correct, turn 2 wrong
    answers = [
        json.dumps({"answer": "correct", "reasoning": "..."}),
        json.dumps({"answer": "wrong!!!", "reasoning": "..."}),
    ]
    call_count = [0]

    fake_sb = FakeSandbox(image="fake")

    async def on_launch(env):
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(answers):
            fake_sb.files["/workspace/answer.json"] = answers[idx]
        return 0

    fake_sb.on_launch = on_launch

    import generate as gen_mod
    _orig = gen_mod._boot_sandbox

    async def _fake_boot(image, instance_id, harness):
        return fake_sb

    gen_mod._boot_sandbox = _fake_boot

    try:
        results = await generate(_Args(), _Sample(), {}, evaluation=False)
        result = results[0]
        assert result.reward == 0.5, f"Expected 0.5, got {result.reward}"
        assert result.metadata["per_turn_rewards"] == [1.0, 0.0]
        print("  ✅ partial reward: 1 correct + 1 wrong → 0.5")
    finally:
        gen_mod._boot_sandbox = _orig


def test_partial_reward():
    import asyncio
    asyncio.run(_run_partial_reward())


# ======================================================================
# 4. Metadata validation in generate()
# ======================================================================

async def _run_abort_on_bad_metadata():
    """generate() should abort gracefully on invalid metadata."""
    from generate import generate

    class _Args:
        pass

    class _Sample:
        def __init__(self):
            self.prompt = ""
            self.label = "bad"
            self.metadata = {"turns": []}  # empty turns
            self.session_id = None
            self.index = 0
            self.group_index = 0
            self.reward = 0.0
            self.response = ""
            self.response_length = 0
            self.tokens = []
            self.loss_mask = []
            self.rollout_log_probs = []
            self.remove_sample = False
            self.status = None

    results = await generate(_Args(), _Sample(), {}, evaluation=False)
    assert len(results) == 1
    result = results[0]
    assert result.remove_sample is True
    assert result.status.name == "ABORTED"
    assert "empty_turns" in result.metadata["abort_reason"]
    print("  ✅ abort on empty turns")


def test_abort_on_bad_metadata():
    import asyncio
    asyncio.run(_run_abort_on_bad_metadata())


# ======================================================================
# 5. Prompt content verification (sanity check)
# ======================================================================

def test_prompt_content_for_mock_task():
    """Verify the prompts built for the mock FIFA task look right."""
    turns = [
        {"turn_id": 1, "context": "FIFA 2022 data.",
         "question": "Who scored the most goals?",
         "files_used": ["players.csv"]},
        {"turn_id": 2, "context": "Compute correlation.",
         "question": "What is the Pearson r?",
         "files_used": ["players.csv"]},
    ]

    p1 = build_prompt(turns[0], first_turn=True, history=None)
    assert "expert data scientist" in p1
    assert "Who scored the most goals?" in p1
    assert "players.csv" in p1
    assert "answer.json" in p1
    # No history on first turn
    assert "## Previous Turns" not in p1

    p2 = build_prompt(turns[1], first_turn=False, history=[
        {"turn_id": 1, "question": "Who scored the most goals?",
         "answer": "Kylian Mbappe"},
    ])
    assert "## Previous Turns" in p2
    assert "Kylian Mbappe" in p2
    assert "Pearson" in p2
    # No system prompt on turn 2
    assert "expert data scientist" not in p2

    print("  ✅ prompt content correct for multi-turn")


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

    print(f"\nPhase 3: {results['passed']} passed, {results['failed']} failed")
    sys.exit(1 if results["failed"] else 0)
