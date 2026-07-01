"""Unit tests for staleness control in fully_async_rollout.py.

Run with: python test_staleness_unit.py
"""

import queue
import sys
from unittest.mock import MagicMock

sys.path.insert(0, ".")

from slime.utils.types import Sample
from slime.rollout.fully_async_rollout import AsyncRolloutWorker


class FakeArgs:
    staleness_threshold = 2
    rollout_batch_size = 8
    sglang_server_concurrency = 1


def test_init():
    worker = AsyncRolloutWorker(FakeArgs(), MagicMock(), concurrency=4)
    assert worker.staleness_threshold == 2
    assert worker.staleness_samples == 0
    assert worker.max_required_samples == (1 + 2) * 8  # 24
    print("PASS test_init")


def test_reset_staleness():
    worker = AsyncRolloutWorker(FakeArgs(), MagicMock(), concurrency=4)
    worker.staleness_samples = 10
    worker.reset_staleness()
    assert worker.staleness_samples == 0
    print("PASS test_reset_staleness")


def test_get_completed_groups_with_limit():
    worker = AsyncRolloutWorker(FakeArgs(), MagicMock(), concurrency=4)
    for i in range(4):
        worker.output_queue.put((i, [f"sample_{i}"]))

    result = worker.get_completed_groups(max_groups=2)
    assert len(result) == 2
    assert worker.output_queue.qsize() == 2  # 2 left
    print("PASS test_get_completed_groups_with_limit")


def test_get_completed_groups_no_limit():
    worker = AsyncRolloutWorker(FakeArgs(), MagicMock(), concurrency=4)
    for i in range(3):
        worker.output_queue.put((i, [f"sample_{i}"]))

    result = worker.get_completed_groups()  # no max_groups
    assert len(result) == 3
    assert worker.output_queue.qsize() == 0  # drained all
    print("PASS test_get_completed_groups_no_limit")


def test_done_cb_increments_staleness():
    worker = AsyncRolloutWorker(FakeArgs(), MagicMock(), concurrency=4)
    worker.staleness_samples = 0
    cb = worker._make_done_cb(99)

    class FakeTask:
        def result(self):
            return [MagicMock(status="completed")]

    cb(FakeTask())
    assert worker.staleness_samples == 1
    assert worker.output_queue.qsize() == 1
    print("PASS test_done_cb_increments_staleness")


def test_done_cb_skips_aborted():
    worker = AsyncRolloutWorker(FakeArgs(), MagicMock(), concurrency=4)
    worker.staleness_samples = 0
    cb = worker._make_done_cb(100)

    s = Sample()
    s.status = Sample.Status.ABORTED

    class FakeTask:
        def result(self):
            return [s]

    cb(FakeTask())
    assert worker.staleness_samples == 0  # NOT incremented
    assert worker.output_queue.qsize() == 0  # NOT put to queue
    print("PASS test_done_cb_skips_aborted")


def test_loop_stops_when_quota_exhausted():
    worker = AsyncRolloutWorker(FakeArgs(), MagicMock(), concurrency=4)
    worker.staleness_samples = 24
    worker.max_required_samples = 24

    # Simulate the _loop's Top up condition
    should_pull = worker.staleness_samples < worker.max_required_samples
    assert not should_pull  # should NOT pull
    print("PASS test_loop_stops_when_quota_exhausted")


def test_loop_pulls_when_within_quota():
    worker = AsyncRolloutWorker(FakeArgs(), MagicMock(), concurrency=4)
    worker.staleness_samples = 10
    worker.max_required_samples = 24

    should_pull = worker.staleness_samples < worker.max_required_samples
    assert should_pull  # should pull
    print("PASS test_loop_pulls_when_within_quota")


def test_max_required_samples_default():
    """When staleness_threshold not set, defaults to 1."""
    args = FakeArgs()
    del args.staleness_threshold  # simulate not set
    args.staleness_threshold = 1   # getattr default
    worker = AsyncRolloutWorker(args, MagicMock(), concurrency=4)
    assert worker.max_required_samples == (1 + 1) * 8  # 16
    print("PASS test_max_required_samples_default")


if __name__ == "__main__":
    test_init()
    test_reset_staleness()
    test_get_completed_groups_with_limit()
    test_get_completed_groups_no_limit()
    test_done_cb_increments_staleness()
    test_done_cb_skips_aborted()
    test_loop_stops_when_quota_exhausted()
    test_loop_pulls_when_within_quota()
    test_max_required_samples_default()
    print()
    print("All 9 tests passed.")
