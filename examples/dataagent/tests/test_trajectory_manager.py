"""Unit tests for DataAgentTrajectoryManager and _SessionAccumulator.

Verifies:
1. Multi-turn conversation → exactly 1 Sample (no fork).
2. loss_mask: first-turn prompt stripped, subsequent prompts=0, outputs=1.
3. Reward not split (full reward on the single Sample).
4. Abort partial tokens recorded as loss_mask=0.
5. Independent prompts per turn (no shared prefix) still produce 1 Sample.
6. drop_session cleans up.
7. Multiple concurrent sessions don't interfere.

Run:
  cd /path/to/slime
  python -m pytest examples/dataagent/tests/test_trajectory_manager.py -v
  # or without pytest:
  python examples/dataagent/tests/test_trajectory_manager.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from examples.dataagent.adapter import (
    DataAgentTrajectoryManager,
    _SessionAccumulator,
)
from slime.agent.adapters.common import TurnRecord
from slime.utils.types import Sample


def _base_sample(idx: int = 0) -> Sample:
    """Minimal base sample for get_trajectory."""
    s = Sample()
    s.index = idx
    s.group_index = 0
    s.prompt = "test query"
    s.label = ""
    return s


def _turn(prompt_ids: list[int], output_ids: list[int], logprobs: list[float] | None = None) -> TurnRecord:
    return TurnRecord(
        prompt_ids=list(prompt_ids),
        output_ids=list(output_ids),
        finish_reason="stop",
        output_log_probs=logprobs or [],
    )


# ── _SessionAccumulator tests ───────────────────────────────────────────


def test_accumulator_single_turn():
    """One turn: prompt=loss_mask 0, output=loss_mask 1, first prompt stripped."""
    acc = _SessionAccumulator()
    # prompt = [1,2,3], output = [4,5]
    acc.record_turn([1, 2, 3], [4, 5], [0.1, 0.2])
    s = acc.to_sample(_base_sample(), reward=0.5, extra_metadata={"k": "v"})

    assert s.tokens == [1, 2, 3, 4, 5]
    # First prompt (3 tokens) stripped from training region.
    assert s.loss_mask == [1, 1]  # only output
    assert s.response_length == 2
    assert s.rollout_log_probs == [0.1, 0.2]
    assert s.reward == 0.5
    assert s.metadata == {"k": "v"}


def test_accumulator_multi_turn_no_shared_prefix():
    """Two turns with completely different prompts → still 1 Sample.

    This is the DataAgent case: each turn's prompt is independent, no
    shared prefix with previous turns.
    """
    acc = _SessionAccumulator()
    # Turn 1: prompt=[1,2,3] output=[4,5]
    acc.record_turn([1, 2, 3], [4, 5], [0.1, 0.2])
    # Turn 2: prompt=[10,20,30,40] (completely different) output=[50,60]
    acc.record_turn([10, 20, 30, 40], [50, 60], [0.3, 0.4])
    s = acc.to_sample(_base_sample(), reward=0.8, extra_metadata=None)

    # All tokens concatenated.
    assert s.tokens == [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60]
    # loss_mask: [0,0,0, 1,1, 0,0,0,0, 1,1] — first 3 (first prompt) stripped.
    assert s.loss_mask == [1, 1, 0, 0, 0, 0, 1, 1]
    assert s.response_length == 8  # 11 total - 3 leading prompt
    assert s.reward == 0.8  # full reward, not split


def test_accumulator_abort_partial():
    """Abort partial tokens recorded as loss_mask=0 context."""
    acc = _SessionAccumulator()
    # Turn 1: prompt=[1,2] output=[3,4]
    acc.record_turn([1, 2], [3, 4], [0.1, 0.2])
    # Abort partial: partial_ids=[5,6] — recorded as loss_mask=0
    acc.record_partial([5, 6])
    # Turn 2 (after resume): prompt=[10,20] output=[30]
    acc.record_turn([10, 20], [30], [0.3])
    s = acc.to_sample(_base_sample(), reward=0.5, extra_metadata=None)

    assert s.tokens == [1, 2, 3, 4, 5, 6, 10, 20, 30]
    # loss_mask: [0,0, 1,1, 0,0, 0,0, 1] — first 2 stripped.
    assert s.loss_mask == [1, 1, 0, 0, 0, 0, 1]
    assert s.reward == 0.5


def test_accumulator_empty():
    """Empty accumulator → to_sample still works (no trainable tokens)."""
    acc = _SessionAccumulator()
    s = acc.to_sample(_base_sample(), reward=0.0, extra_metadata=None)
    assert s.tokens == []
    assert s.loss_mask == []
    assert s.response_length == 0


# ── DataAgentTrajectoryManager tests ────────────────────────────────────


def test_manager_single_session_multi_turn():
    """One session, 3 turns → 1 Sample with full reward."""
    mgr = DataAgentTrajectoryManager()
    sid = "test-sid-1"

    # Turn 1
    mgr.record_turn(sid, turn=_turn([1, 2, 3], [4, 5], [0.1, 0.2]))
    # Turn 2 (different prompt — no shared prefix)
    mgr.record_turn(sid, turn=_turn([10, 20], [30, 40], [0.3, 0.4]))
    # Turn 3 (yet another different prompt)
    mgr.record_turn(sid, turn=_turn([100], [200], [0.5]))

    samples = mgr.get_trajectory(sid, base_sample=_base_sample(), reward=0.9)

    assert len(samples) == 1  # exactly 1, not 3
    s = samples[0]
    assert s.reward == 0.9  # full reward, not split
    assert s.tokens == [1, 2, 3, 4, 5, 10, 20, 30, 40, 100, 200]
    # loss_mask: [0,0,0, 1,1, 0,0, 1,1, 0, 1] — first 3 stripped
    assert s.loss_mask == [1, 1, 0, 0, 1, 1, 0, 1]
    assert s.response_length == 8


def test_manager_abort_partial_via_record_turn():
    """record_turn with metadata abort + empty prompt_ids → partial loss_mask=0."""
    mgr = DataAgentTrajectoryManager()
    sid = "test-sid-abort"

    # Normal turn 1
    mgr.record_turn(sid, turn=_turn([1, 2], [3, 4], [0.1, 0.2]))
    # Abort partial: prompt_ids=[] (empty), output_ids=[5,6] (partial), metadata abort
    mgr.record_turn(
        sid,
        turn=_turn([], [5, 6]),
        metadata={"sid": sid, "abort": True},
    )
    # Normal turn 2 (after resume)
    mgr.record_turn(sid, turn=_turn([10, 20], [30], [0.3]))

    samples = mgr.get_trajectory(sid, base_sample=_base_sample(), reward=0.5)

    assert len(samples) == 1
    s = samples[0]
    assert s.tokens == [1, 2, 3, 4, 5, 6, 10, 20, 30]
    # [0,0, 1,1, 0,0, 0,0, 1] — first 2 stripped
    assert s.loss_mask == [1, 1, 0, 0, 0, 0, 1]
    assert s.reward == 0.5


def test_manager_reward_not_split():
    """Even with 10 turns, reward stays on 1 Sample — no reward/K split."""
    mgr = DataAgentTrajectoryManager()
    sid = "test-sid-reward"

    for i in range(10):
        mgr.record_turn(sid, turn=_turn([i], [i + 100], [0.1 * i]))

    samples = mgr.get_trajectory(sid, base_sample=_base_sample(), reward=0.42)

    assert len(samples) == 1
    assert samples[0].reward == 0.42  # NOT 0.42/10


def test_manager_multiple_sessions_independent():
    """Two sessions don't interfere; each produces 1 Sample."""
    mgr = DataAgentTrajectoryManager()

    mgr.record_turn("sid-A", turn=_turn([1], [2], [0.1]))
    mgr.record_turn("sid-B", turn=_turn([3], [4], [0.2]))
    mgr.record_turn("sid-A", turn=_turn([5], [6], [0.3]))
    mgr.record_turn("sid-B", turn=_turn([7], [8], [0.4]))

    samples_a = mgr.get_trajectory("sid-A", base_sample=_base_sample(), reward=0.5)
    samples_b = mgr.get_trajectory("sid-B", base_sample=_base_sample(), reward=0.6)

    assert len(samples_a) == 1
    assert len(samples_b) == 1
    assert samples_a[0].tokens == [1, 2, 5, 6]
    assert samples_b[0].tokens == [3, 4, 7, 8]
    assert samples_a[0].reward == 0.5
    assert samples_b[0].reward == 0.6


def test_manager_drop_session():
    """drop_session removes the accumulator."""
    mgr = DataAgentTrajectoryManager()
    sid = "test-drop"

    mgr.record_turn(sid, turn=_turn([1], [2]))
    assert mgr.has_session(sid)

    mgr.drop_session(sid)
    assert not mgr.has_session(sid)

    # get_trajectory on dropped session returns []
    assert mgr.get_trajectory(sid, base_sample=_base_sample()) == []


def test_manager_get_trajectory_consumes_session():
    """get_trajectory pops the session — second call returns []."""
    mgr = DataAgentTrajectoryManager()
    sid = "test-consume"

    mgr.record_turn(sid, turn=_turn([1, 2], [3, 4], [0.1, 0.2]))
    samples1 = mgr.get_trajectory(sid, base_sample=_base_sample(), reward=0.5)
    samples2 = mgr.get_trajectory(sid, base_sample=_base_sample(), reward=0.5)

    assert len(samples1) == 1
    assert samples2 == []  # session consumed


def test_manager_empty_session_returns_empty():
    """get_trajectory on a session with no turns returns []."""
    mgr = DataAgentTrajectoryManager()
    sid = "test-empty"

    # Never recorded any turn
    assert mgr.get_trajectory(sid, base_sample=_base_sample()) == []


def test_manager_has_session_and_turn_count():
    """has_session / turn_count basic checks."""
    mgr = DataAgentTrajectoryManager()

    assert not mgr.has_session("nope")
    mgr.record_turn("sid", turn=_turn([1], [2]))
    assert mgr.has_session("sid")


# ── runner ─────────────────────────────────────────────────────────────

_ALL_TESTS = [
    test_accumulator_single_turn,
    test_accumulator_multi_turn_no_shared_prefix,
    test_accumulator_abort_partial,
    test_accumulator_empty,
    test_manager_single_session_multi_turn,
    test_manager_abort_partial_via_record_turn,
    test_manager_reward_not_split,
    test_manager_multiple_sessions_independent,
    test_manager_drop_session,
    test_manager_get_trajectory_consumes_session,
    test_manager_empty_session_returns_empty,
    test_manager_has_session_and_turn_count,
]


def main():
    passed = 0
    failed = 0
    for test in _ALL_TESTS:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {test.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
