"""LongDS task utilities — pure functions, no I/O dependencies.

These are the building blocks that the per-sample generate() will use at
training time. Separated here so they can be tested without Docker/GPU/SGLang.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

LONGDS_SYSTEM_PROMPT = """\
You are an expert data scientist and statistical analyst.

You have access to tools for reading files, writing code, and running shell
commands. Use them to explore data, run analyses, and answer questions.

**Data files** are in the `data/` directory. Read them with the Read tool.
Do NOT modify or delete files under `data/`.

**Working directory**: use the current directory as your workspace. Create
helper scripts and save intermediate results here.

**Rules**:
1. Solve only the current question. Use exact calculations from data —
   never mental arithmetic.
2. When you need to run Python, use: `python3 -c "..."` or create a .py file.
3. You may install packages with `pip install --quiet <package>`.
4. For numerical answers, round only when explicitly asked.
5. When you have the final answer, write it with the Bash tool:
   `echo '{"answer":"<your answer>","reasoning":"<brief method>"}' > answer.json`
6. Put the direct answer in the "answer" field. Keep it concise.
"""


def build_prompt(
    turn: dict,
    *,
    first_turn: bool = False,
    history: list[dict] | None = None,
) -> str:
    """Build a Claude Code prompt for one LongDS turn.

    Args:
        turn: {"turn_id", "context", "question", "files_used"}
        first_turn: Include system instructions only on first turn.
        history: Previous turns as [{"turn_id","question","answer"},...].
    """
    parts: list[str] = []
    if first_turn:
        parts.append(LONGDS_SYSTEM_PROMPT)
    if history:
        parts.append("## Previous Turns\n")
        for h in history:
            parts.append(f"### Turn {h['turn_id']}")
            parts.append(f"Question: {h['question']}")
            parts.append(f"Answer: {h['answer']}\n")

    ctx = turn.get("context", "")
    q = turn.get("question", "")
    files = turn.get("files_used", [])

    parts.append("## Current Task")
    if ctx:
        parts.append(f"\n{ctx}")
    parts.append(f"\nQuestion: {q}")
    if files:
        parts.append(f"\nRelevant files in data/: {', '.join(files)}")
    parts.append(
        "\n\nWhen you have the answer, write it with:\n"
        '  echo \'{"answer":"<your answer>","reasoning":"<method>"}\' > answer.json'
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Answer evaluation — mirrors LongDS JUDGE_PROMPT rules
# ---------------------------------------------------------------------------

def evaluate_answer(prediction: str, ground_truth: str, answer_type: str = "auto") -> float:
    """Score a prediction against ground truth. Returns 0.0 or 1.0.

    LongDS JUDGE_PROMPT rules implemented programmatically:
      1. Numeric: exact after stripping trailing zeros, no rounding.
      2. Categorical: strict equality after case/punctuation normalization.
      3. Ordered list: match items AND order (ties allowed).
      4. JSON: recursive field-by-field, only GT keys matter.
      5. Binary: all required fields correct → 1; anything wrong → 0.
    """
    p = (prediction or "").strip()
    g = (ground_truth or "").strip()
    if not p or not g:
        return 0.0

    at = answer_type if answer_type != "auto" else _detect_type(g)

    if at == "numeric":
        return _numeric_match(p, g)
    if at == "json":
        return _json_match(p, g)
    if at == "list":
        return _list_match(p, g)
    return _exact_normalized(p, g)


def _detect_type(gt: str) -> str:
    s = gt.strip()
    if s.startswith("{") or s.startswith("["):
        try:
            json.loads(s)
            return "json"
        except json.JSONDecodeError:
            pass
    cleaned = s.replace("$", "").replace("€", "").replace(",", "").replace("%", "").strip()
    try:
        float(cleaned)
        return "numeric"
    except ValueError:
        pass
    if re.match(r"^\d+[\.\)]\s", s) or ("," in s and len(s.split(",")) >= 3):
        return "list"
    return "categorical"


def _norm_trailing_zeros(s: str) -> str:
    """Strip insignificant trailing zeros: '22245.00' → '22245', '25.7600' → '25.76'."""
    s = s.strip()
    if "." not in s:
        return s
    left, right = s.rsplit(".", 1)
    right = right.rstrip("0")
    return left if not right else f"{left}.{right}"


def _parse_num(s: str) -> str:
    return s.replace("$", "").replace("€", "").replace(",", "").replace("%", "").strip()


def _numeric_match(pred: str, gt: str) -> float:
    return 1.0 if _norm_trailing_zeros(_parse_num(pred)) == _norm_trailing_zeros(_parse_num(gt)) else 0.0


def _exact_normalized(pred: str, gt: str) -> float:
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".,;:\"'\n\t ").strip()
    return 1.0 if _norm(pred) == _norm(gt) else 0.0


def _parse_list_items(s: str) -> list[str]:
    lines = [l.strip() for l in s.strip().split("\n") if l.strip()]
    items = []
    for line in lines:
        m = re.match(r"^[\d]+[\.\)]\s*(.+)", line)
        items.append(m.group(1).strip() if m else line)
    if len(items) <= 1 and "," in s:
        items = [i.strip() for i in s.split(",")]
    return items


def _list_match(pred: str, gt: str) -> float:
    p_items = _parse_list_items(pred)
    g_items = _parse_list_items(gt)
    if not p_items or not g_items:
        return _exact_normalized(pred, gt)
    if len(p_items) != len(g_items):
        return 0.0
    for pi, gi in zip(p_items, g_items):
        if _numeric_match(pi, gi) < 1.0 and _exact_normalized(pi, gi) < 1.0:
            return 0.0
    return 1.0


def _json_match(pred: str, gt: str) -> float:
    try:
        p_obj, g_obj = json.loads(pred), json.loads(gt)
    except json.JSONDecodeError:
        return _exact_normalized(pred, gt)
    return 1.0 if _dict_equal(p_obj, g_obj) else 0.0


def _dict_equal(d1: Any, d2: Any) -> bool:
    if isinstance(d1, dict) and isinstance(d2, dict):
        return all(_dict_equal(d1.get(k), d2.get(k)) for k in d2)
    if isinstance(d1, list) and isinstance(d2, list):
        return len(d1) == len(d2) and all(_dict_equal(a, b) for a, b in zip(d1, d2))
    # Numeric: allow int/float coercion (json parses 3.00→3.0, 3→3)
    if isinstance(d1, (int, float)) and isinstance(d2, (int, float)):
        return _numeric_match(str(d1), str(d2)) >= 1.0
    if type(d1) != type(d2):
        return False
    return str(d1).strip() == str(d2).strip()


# ---------------------------------------------------------------------------
# Metadata validation (used by generate() at rollout time)
# ---------------------------------------------------------------------------

def validate_metadata(md: dict) -> str | None:
    """Return None if task metadata is evaluable, or a diagnostic string."""
    turns = md.get("turns", [])
    if not turns:
        return "empty_turns"
    for t in turns:
        if not t.get("question"):
            return f"turn_{t.get('turn_id','?')}_missing_question"
        if not t.get("ground_truth"):
            return f"turn_{t.get('turn_id','?')}_missing_ground_truth"
    return None
