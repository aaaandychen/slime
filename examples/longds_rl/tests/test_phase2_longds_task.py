"""Phase 2 verification: test longds_task.py pure functions.

Tests answer evaluation (numeric/categorical/json/list), prompt building,
and metadata validation. All CPU-only, no Docker/GPU/SGLang needed.

Run: python3 tests/test_phase2_longds_task.py
"""

import sys
from pathlib import Path

_EXAMPLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXAMPLE))

import longds_task as lt


# ── answer evaluation: numeric ──────────────────────────────────────────

def test_numeric_exact():
    assert lt.evaluate_answer("42", "42", "numeric") == 1.0
    assert lt.evaluate_answer("3.1415", "3.1415", "numeric") == 1.0

def test_numeric_trailing_zeros():
    """JUDGE_PROMPT rule: insignificant trailing zeros."""
    assert lt.evaluate_answer("22245.00", "22245", "numeric") == 1.0
    assert lt.evaluate_answer("22245", "22245.00", "numeric") == 1.0
    assert lt.evaluate_answer("25.7600", "25.76", "numeric") == 1.0

def test_numeric_no_rounding():
    """JUDGE_PROMPT rule: no rounding. 0.125 ≠ 0.12."""
    assert lt.evaluate_answer("0.125", "0.12", "numeric") == 0.0
    assert lt.evaluate_answer("100", "101", "numeric") == 0.0

def test_numeric_symbols():
    assert lt.evaluate_answer("$1,234.56", "1234.56", "numeric") == 1.0
    assert lt.evaluate_answer("98.6%", "98.6", "numeric") == 1.0
    assert lt.evaluate_answer("€50", "50", "numeric") == 1.0


# ── answer evaluation: categorical ──────────────────────────────────────

def test_categorical_exact():
    assert lt.evaluate_answer("Kylian Mbappe", "Kylian Mbappe", "categorical") == 1.0

def test_categorical_case():
    assert lt.evaluate_answer("kylian mbappe", "Kylian Mbappe", "categorical") == 1.0
    assert lt.evaluate_answer("FRANCE", "France", "categorical") == 1.0

def test_categorical_punctuation():
    assert lt.evaluate_answer("Kylian Mbappe.", "Kylian Mbappe", "categorical") == 1.0
    assert lt.evaluate_answer("  extra spaces  ", "extra spaces", "categorical") == 1.0

def test_categorical_wrong():
    assert lt.evaluate_answer("Messi", "Mbappe", "categorical") == 0.0
    assert lt.evaluate_answer("", "something", "categorical") == 0.0


# ── answer evaluation: JSON ─────────────────────────────────────────────

def test_json_match():
    assert lt.evaluate_answer(
        '{"name":"Alice","age":30}', '{"name":"Alice","age":30}', "json") == 1.0

def test_json_trailing_zeros():
    assert lt.evaluate_answer('{"value":3.00}', '{"value":3}', "json") == 1.0

def test_json_extra_fields_ignored():
    """Only ground-truth keys matter (JUDGE_PROMPT rule #1)."""
    assert lt.evaluate_answer(
        '{"name":"Alice","age":30,"extra":"x"}',
        '{"name":"Alice","age":30}', "json") == 1.0

def test_json_wrong():
    assert lt.evaluate_answer('{"name":"Bob"}', '{"name":"Alice"}', "json") == 0.0


# ── answer evaluation: ordered list ─────────────────────────────────────

def test_list_ordered_match():
    assert lt.evaluate_answer(
        "1. Alice\n2. Bob\n3. Charlie",
        "1. Alice\n2. Bob\n3. Charlie", "list") == 1.0

def test_list_wrong_order():
    """Order matters unless tied."""
    assert lt.evaluate_answer(
        "1. Bob\n2. Alice\n3. Charlie",
        "1. Alice\n2. Bob\n3. Charlie", "list") == 0.0

def test_list_numeric_items():
    assert lt.evaluate_answer(
        "1. 100\n2. 200\n3. 300",
        "1. 100.0\n2. 200\n3. 300", "list") == 1.0

def test_list_length_mismatch():
    assert lt.evaluate_answer(
        "1. Alice\n2. Bob",
        "1. Alice\n2. Bob\n3. Charlie", "list") == 0.0


# ── auto-detection ──────────────────────────────────────────────────────

def test_auto_detect():
    assert lt.evaluate_answer("42", "42") == 1.0          # numeric
    assert lt.evaluate_answer("Hello", "Hello") == 1.0    # categorical
    assert lt.evaluate_answer('{"k":"v"}', '{"k":"v"}') == 1.0  # json


# ── prompt building ─────────────────────────────────────────────────────

def test_prompt_first_turn():
    prompt = lt.build_prompt(
        {"turn_id": 1, "context": "Analyze FIFA data.",
         "question": "Who scored the most goals?", "files_used": ["players.csv"]},
        first_turn=True,
    )
    assert "expert data scientist" in prompt
    assert "Analyze FIFA data." in prompt
    assert "Who scored the most goals?" in prompt
    assert "players.csv" in prompt
    assert "answer.json" in prompt
    assert "## Previous Turns" not in prompt

def test_prompt_later_turn():
    prompt = lt.build_prompt(
        {"turn_id": 2, "context": "Now compute correlation.",
         "question": "What is the Pearson r?", "files_used": ["players.csv"]},
        first_turn=False,
        history=[{"turn_id": 1, "question": "Most goals?", "answer": "Mbappe"}],
    )
    assert "expert data scientist" not in prompt   # system only on first turn
    assert "## Previous Turns" in prompt
    assert "Mbappe" in prompt
    assert "Pearson" in prompt
    assert "answer.json" in prompt


# ── metadata validation ─────────────────────────────────────────────────

def test_validate_ok():
    md = {"turns": [{"turn_id": 1, "question": "Q", "ground_truth": "A"}]}
    assert lt.validate_metadata(md) is None

def test_validate_empty():
    assert lt.validate_metadata({"turns": []}) == "empty_turns"

def test_validate_missing_gt():
    reason = lt.validate_metadata({"turns": [{"turn_id": 1, "question": "Q"}]})
    assert reason and "missing_ground_truth" in reason


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
                import traceback
                print(f"  ❌ {name}: {e}")
                traceback.print_exc()
    print(f"\nPhase 2: {results['passed']} passed, {results['failed']} failed")
    sys.exit(1 if results["failed"] else 0)
