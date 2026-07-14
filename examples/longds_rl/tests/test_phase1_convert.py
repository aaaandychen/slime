"""Phase 1 verification: LongDS → slime JSONL conversion.

Tests convert_longds_to_slime.py with mock fixtures.
No Docker/GPU/SGLang required.

Run: python3 tests/test_phase1_convert.py
"""

import json
import sys
import tempfile
from pathlib import Path

# -- path setup --
_EXAMPLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXAMPLE))
_FIXTURES = _EXAMPLE / "tests" / "fixtures" / "mock_tasks"

import convert_longds_to_slime as cvt


# ── answer type detection ──────────────────────────────────────────────

def test_detect_numeric():
    assert cvt.detect_answer_type("42") == "numeric"
    assert cvt.detect_answer_type("3.1415") == "numeric"
    assert cvt.detect_answer_type("$1,234.56") == "numeric"
    assert cvt.detect_answer_type("98.6%") == "numeric"
    assert cvt.detect_answer_type("-15") == "numeric"
    assert cvt.detect_answer_type("0.4521") == "numeric"

def test_detect_categorical():
    assert cvt.detect_answer_type("Kylian Mbappe") == "categorical"
    assert cvt.detect_answer_type("France") == "categorical"

def test_detect_json():
    assert cvt.detect_answer_type('{"key":"value"}') == "json"
    assert cvt.detect_answer_type('[1,2,3]') == "json"

def test_detect_list():
    assert cvt.detect_answer_type("1. First\n2. Second\n3. Third") == "list"
    assert cvt.detect_answer_type("apple, banana, cherry") == "list"

def test_detect_empty():
    assert cvt.detect_answer_type("") == "exact"
    assert cvt.detect_answer_type("  ") == "exact"


# ── conversion ──────────────────────────────────────────────────────────

def test_convert_one_task():
    row = cvt.convert_one_task(
        {"task_domain": "sports", "dataset_name": "fifa_analytics", "task_id": "task1"},
        task_root=_FIXTURES, data_root=_FIXTURES,
    )
    assert row is not None
    assert row["prompt"] == ""
    assert row["label"] == "sports__fifa_analytics__task1"

    md = row["metadata"]
    assert md["task_id"] == "sports/fifa_analytics/task1"
    assert md["domain"] == "sports"
    assert md["num_turns"] == 3
    assert "players.csv" in md["data_files"]
    assert "teams.csv" in md["data_files"]

    t1 = md["turns"][0]
    assert t1["turn_id"] == 1
    assert t1["answer_type"] == "categorical"
    assert "Kylian Mbappe" in t1["ground_truth"]
    assert "players.csv" in t1["files_used"]

    t2 = md["turns"][1]
    assert t2["answer_type"] == "numeric"
    assert "0.4521" in t2["ground_truth"]

    t3 = md["turns"][2]
    assert t3["answer_type"] == "categorical"

def test_convert_missing_task():
    row = cvt.convert_one_task(
        {"task_domain": "sports", "dataset_name": "nonexistent", "task_id": "task99"},
        task_root=_FIXTURES, data_root=_FIXTURES,
    )
    assert row is None

def test_full_pipeline():
    """Write JSONL, read back, validate slime-compatible keys."""
    task_list = cvt.load_json(_FIXTURES / "task_list.json")
    rows = []
    for info in task_list:
        r = cvt.convert_one_task(info, _FIXTURES, _FIXTURES)
        if r:
            rows.append(r)
    assert len(rows) == 1

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp = Path(f.name)
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    try:
        with open(tmp) as f:
            loaded = json.loads(f.readline())
        assert loaded["label"] == "sports__fifa_analytics__task1"
        for k in ("prompt", "label", "metadata"):
            assert k in loaded
        for k in ("turns", "data_files", "data_root", "domain"):
            assert k in loaded["metadata"]
    finally:
        tmp.unlink(missing_ok=True)


# ── runner ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {"passed": 0, "failed": 0}
    for name, fn in sorted(locals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                results["passed"] += 1
                print(f"  ✅ {name}")
            except Exception as e:
                results["failed"] += 1
                print(f"  ❌ {name}: {e}")
    print(f"\nPhase 1: {results['passed']} passed, {results['failed']} failed")
    sys.exit(1 if results["failed"] else 0)
