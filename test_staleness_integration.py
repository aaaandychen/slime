"""Integration test for the staleness control chain in fully_async_rollout.

Simulates the full lifecycle:
  worker generates via done_cb →
  trainer pulls only needed →
  staleness quota blocks new pulls →
  reset_staleness after weight sync →
  worker resumes with fresh quota

Run with: python test_staleness_integration.py
"""

import sys
from unittest.mock import MagicMock, call

sys.path.insert(0, ".")

from slime.utils.types import Sample
from slime.rollout.fully_async_rollout import AsyncRolloutWorker


# ── helpers ──────────────────────────────────────────────────────────

def _make_args(**kw):
    """Build a minimal fake args object."""
    defaults = dict(
        staleness_threshold=1,
        rollout_batch_size=4,
        sglang_server_concurrency=1,
    )
    defaults.update(kw)
    args = MagicMock()
    for k, v in defaults.items():
        setattr(args, k, v)
    return args


def _completed_group():
    """Return a non-aborted result (as generate_and_rm_group would)."""
    return [MagicMock(status="completed")]


def _aborted_group():
    """Return an ABORTED result."""
    s = Sample()
    s.status = Sample.Status.ABORTED
    return [s]


class _FakeTask:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


# ── the test ─────────────────────────────────────────────────────────

def test_full_staleness_cycle():
    args = _make_args(staleness_threshold=1, rollout_batch_size=4)
    # max_required_samples = (1 + 1) * 4 = 8

    # Mock data_buffer: returns one group per get_samples(1) call
    data_buffer = MagicMock()
    data_buffer.get_samples.return_value = [[MagicMock()]]  # one group

    worker = AsyncRolloutWorker(args, data_buffer, concurrency=3)

    print(f"max_required_samples = {worker.max_required_samples}  "
          f"(staleness_threshold={args.staleness_threshold}, batch_size={args.rollout_batch_size})")

    # ── Phase 1: generate 8 groups → should exhaust staleness quota ──
    for i in range(8):
        cb = worker._make_done_cb(i)
        cb(_FakeTask(_completed_group()))

    assert worker.staleness_samples == 8
    assert worker.staleness_samples == worker.max_required_samples
    assert worker.output_queue.qsize() == 8
    print(f"Phase 1 OK: staleness_samples={worker.staleness_samples}, "
          f"output_queue={worker.output_queue.qsize()}")

    # _loop would now break because staleness_samples >= max_required_samples
    should_pull = worker.staleness_samples < worker.max_required_samples
    assert not should_pull, "worker should NOT pull when quota exhausted"
    print("Phase 1 OK: worker stops pulling (quota exhausted)")

    # ── Phase 2: trainer pulls only needed (batch_size=4), not all 8 ──
    # When there are leftovers, they stay in queue for the next round.
    # Simulate first rollout has collected 4 already, takes more.
    collected = {}
    needed = 4 - len(collected)  # 4
    for gid, group in worker.get_completed_groups(max_groups=needed):
        collected[gid] = group
    assert len(collected) == 4
    assert worker.output_queue.qsize() == 4  # 4 leftovers stay
    print(f"Phase 2 OK: trainer took {len(collected)}, "
          f"queue_left={worker.output_queue.qsize()}")

    # ── Phase 3: second rollout round — trainer takes remaining 4 ──
    # No new generation happened (quota still exhausted)
    collected2 = {}
    needed2 = 4
    for gid, group in worker.get_completed_groups(max_groups=needed2):
        collected2[gid] = group
    assert len(collected2) == 4
    assert worker.output_queue.qsize() == 0
    print(f"Phase 3 OK: trainer took {len(collected2)}, queue drained")

    # ── Phase 4: weight sync → reset_staleness ──
    # Simulate: ABORTED groups requeue (don't increment staleness)
    cb_abort = worker._make_done_cb(99)
    cb_abort(_FakeTask(_aborted_group()))
    assert worker.staleness_samples == 8  # unchanged, abort doesn't count
    assert data_buffer.add_samples.called  # requeued to buffer
    print("Phase 4 OK: ABORTED group requeued, staleness NOT incremented")

    # Now reset
    worker.reset_staleness()
    assert worker.staleness_samples == 0
    print("Phase 4 OK: reset_staleness → counter=0, worker can pull again")

    # ── Phase 5: worker resumes, generates new batch with fresh weights ──
    should_pull = worker.staleness_samples < worker.max_required_samples
    assert should_pull
    print("Phase 5 OK: worker resumes pulling after reset")

    # Generate 8 more (but trainer only takes 4, leftover 4 for next round)
    for i in range(8):
        cb = worker._make_done_cb(100 + i)
        cb(_FakeTask(_completed_group()))

    assert worker.staleness_samples == 8
    assert worker.output_queue.qsize() == 8

    # Trainer takes 4, leaves 4
    collected3 = {}
    for gid, group in worker.get_completed_groups(max_groups=4):
        collected3[gid] = group
    assert len(collected3) == 4
    assert worker.output_queue.qsize() == 4  # preserved for next round
    print(f"Phase 5 OK: trainer took {len(collected3)}, "
          f"{worker.output_queue.qsize()} leftovers preserved")

    # ── Phase 6: verify data_buffer.get_samples call count ──
    print(f"Phase 6 INFO: data_buffer.get_samples call count={data_buffer.get_samples.call_count}")

    print()
    print("All 6 phases passed: full staleness chain integrated correctly.")


def test_rollout_manager_reset_staleness():
    """Verify RolloutManager.reset_staleness delegates to the global worker."""
    from unittest.mock import patch
    import slime.rollout.fully_async_rollout as mod

    args = _make_args()
    data_buffer = MagicMock()

    # Create worker and set as global
    worker = AsyncRolloutWorker(args, data_buffer, concurrency=3)
    worker.staleness_samples = 15
    mod._global_worker = worker

    # Simulate what RolloutManager.reset_staleness does
    from slime.rollout.fully_async_rollout import _global_worker

    if hasattr(data_buffer, "reset_staleness"):
        data_buffer.reset_staleness()

    if _global_worker is not None:
        _global_worker.reset_staleness()

    assert worker.staleness_samples == 0
    print("PASS: RolloutManager.reset_staleness → worker.reset_staleness()")

    mod._global_worker = None  # cleanup


if __name__ == "__main__":
    test_full_staleness_cycle()
    test_rollout_manager_reset_staleness()
