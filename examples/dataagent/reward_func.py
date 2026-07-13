"""Reward function for DataAgent — ground-truth-based scoring.

Two modes, selected by the ``label`` argument:

1. **Structured label** (JSON string with ``key_numbers``/``key_entities``):
   Reward is dominated by *numerical accuracy* — did the report cite the
   correct key numbers computed from the database?  This is the mode that
   produces a real gradient for RL: different rollouts issue different SQL,
   get different result sets, and cite different numbers, so the reward
   varies meaningfully across rollouts of the same query.

2. **Empty / plain-string label** (legacy): falls back to the format-based
   heuristics so old unlabeled queries still produce a number, but the
   signal is weak (format-only rewards are gameable).

Score breakdown (structured mode, totals to ~1.0):

    number_accuracy  0.45   matched ground-truth numbers / total
    entity_coverage  0.20   matched key entities / total
    sql_success      0.15   at least one non-empty RESULT_SET
    format_bonus     0.10   headings + table + reasonable length
    process_bonus    0.10   multi-step reasoning (≥3 substantial nodes)
    all_correct_bonus 0.10  extra when every key number is matched
    ─ penalties ─       −0.20 trivial output, −0.10 no SQL+short, …

Used by ``custom_generate.generate`` via ``sample.reward = score(nodes, label)``.
Credit assignment is episode-level: one score → all trainable tokens.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ── weights ─────────────────────────────────────────────────────────────
W_NUMBER_ACCURACY = 0.45
W_ENTITY_COVERAGE = 0.20
W_SQL_SUCCESS = 0.15
W_FORMAT = 0.10
W_PROCESS = 0.10
W_ALL_CORRECT_BONUS = 0.10


# ── number extraction & normalization ──────────────────────────────────

# Chinese magnitude suffixes: 万=1e4, 亿=1e8, 千=1e3 (rare in reports).
_MAGNITUDE = {"万": 1e4, "亿": 1e8, "千": 1e3, "百": 1e2, "十": 1e1}

# Matches a number with optional thousands separators and a trailing unit.
# Examples that match:  252811  252,811  25.28万  65.4%  ¥252811  252811元
_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9])"               # not part of a larger token (e.g. version)
    r"(\d{1,3}(?:,\d{3})+|\d+\.?\d*)" # the number itself
    r"\s*"
    r"(%|万|亿|千|百万|十万|元|￥|¥)?"  # optional unit / magnitude
)


def _normalize_number(raw: str, unit: str) -> float | None:
    """Convert a matched number string + optional unit to a plain float.

    Handles thousands separators (``252,811 → 252811``) and Chinese
    magnitude suffixes (``25.28万 → 252800``).  Returns ``None`` if the
    value cannot be parsed (e.g. a bare ``%`` with no digits).
    """
    s = raw.replace(",", "")
    try:
        val = float(s)
    except ValueError:
        return None
    if unit == "%":
        return val  # keep percentages as-is for comparison
    if unit in _MAGNITUDE:
        val *= _MAGNITUDE[unit]
    return val


def extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from ``text``, normalized to floats."""
    out: list[float] = []
    for m in _NUM_RE.finditer(text):
        val = _normalize_number(m.group(1), m.group(2) or "")
        if val is not None:
            out.append(val)
    return out


def _numbers_match(report_num: float, truth_str: str) -> bool:
    """Check whether ``report_num`` matches a ground-truth number string.

    Tolerates rounding and 万/亿 conversion: a truth of ``"252811.0"``
    matches a report value of ``252800`` (from ``25.28万``).

    Integer counts (no decimal point in truth) require near-exact match;
    floats (money, rates, percentages) get a 1% relative tolerance.
    """
    try:
        truth = float(truth_str)
    except (ValueError, TypeError):
        return False
    diff = abs(report_num - truth)
    # Integer counts: tight match (tolerate float-precision only).
    if "." not in truth_str and "e" not in truth_str.lower():
        return diff < 0.5
    # Floats: 1% relative + tiny absolute floor for rounding noise.
    return diff <= 0.01 * abs(truth) + 0.01


def _count_number_hits(report_text: str, key_numbers: list[str]) -> tuple[int, int]:
    """Return (matched_count, total_count) for ground-truth numbers."""
    if not key_numbers:
        return 0, 0
    report_nums = extract_numbers(report_text)
    if not report_nums:
        return 0, len(key_numbers)
    matched = 0
    for truth in key_numbers:
        if any(_numbers_match(rn, truth) for rn in report_nums):
            matched += 1
    return matched, len(key_numbers)


def _count_entity_hits(report_text: str, key_entities: list[str]) -> tuple[int, int]:
    """Return (matched_count, total_count) for key named entities."""
    if not key_entities:
        return 0, 0
    matched = sum(1 for e in key_entities if e and e in report_text)
    return matched, len(key_entities)


# ── label parsing ────────────────────────────────────────────────────────

def _parse_label(label: str) -> dict | None:
    """Parse a structured label JSON string; return None if not structured."""
    if not label or not isinstance(label, str):
        return None
    s = label.strip()
    if not s or not s.startswith("{"):
        return None
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or ("key_numbers" not in d and "key_entities" not in d):
        return None
    return d


# ── format / process heuristics (minor component) ───────────────────────

def _format_score(text: str) -> float:
    """Small format bonus so the model keeps producing readable reports."""
    s = 0.0
    if re.findall(r"^#{1,3}\s", text, re.MULTILINE):
        s += 0.04
    if len(re.findall(r"^\|.+\|$", text, re.MULTILINE)) >= 2:
        s += 0.03
    if 200 < len(text) < 8000:
        s += 0.03
    return min(s, W_FORMAT)


def _process_score(nodes: list[dict], successful_sql: list, text_len: int) -> float:
    """Reward multi-step reasoning: successful SQL + multiple substantial nodes."""
    s = 0.0
    if successful_sql and text_len > 200:
        s += 0.05
    substantial = [n for n in nodes
                   if n.get("type") == "TEXT" and len(n.get("text", "")) > 50]
    if len(substantial) >= 3:
        s += 0.05
    return min(s, W_PROCESS)


# ── main entry ──────────────────────────────────────────────────────────

def score(nodes: list[dict[str, Any]], label: str = "") -> float:
    """Score a DataAgent workflow trace.

    Args:
        nodes: Per-node outputs, each ``{"node", "type", "text"}``.
        label: Either a JSON string (structured ground truth, see module
            docstring) or a plain/empty string (legacy format-only mode).

    Returns:
        Score in [0.0, 1.0].
    """
    if not nodes:
        return 0.0

    final = nodes[-1]
    text = final.get("text", "")

    sql_results = [n for n in nodes if n.get("type") == "RESULT_SET"]
    successful_sql = [
        s for s in sql_results
        if "error" not in s.get("text", "").lower() and s.get("text", "").strip()
    ]
    error_sql = [s for s in sql_results if "error" in s.get("text", "").lower()]
    text_len = len(text)

    structured = _parse_label(label)

    # ── structured mode: ground-truth-based reward ─────────────────────
    if structured is not None:
        key_numbers = structured.get("key_numbers", []) or []
        key_entities = structured.get("key_entities", []) or []

        num_hit, num_total = _count_number_hits(text, key_numbers)
        ent_hit, ent_total = _count_entity_hits(text, key_entities)

        s = 0.0
        # Number accuracy — the dominant signal.
        if num_total:
            s += W_NUMBER_ACCURACY * (num_hit / num_total)
        # Entity coverage.
        if ent_total:
            s += W_ENTITY_COVERAGE * (ent_hit / ent_total)
        # SQL execution quality.
        if successful_sql:
            s += W_SQL_SUCCESS * min(1.0, len(successful_sql) / 2.0)
        # Format & process (minor).
        s += _format_score(text)
        s += _process_score(nodes, successful_sql, text_len)
        # All-correct bonus.
        if num_total and num_hit == num_total and ent_total and ent_hit == ent_total:
            s += W_ALL_CORRECT_BONUS

        # Penalties.
        s -= 0.04 * len(error_sql)
        if text_len < 50:
            s -= 0.20
        if not successful_sql and text_len < 200:
            s -= 0.10

        return max(0.0, min(1.0, s))

    # ── legacy mode: format-only (kept for backward compat) ─────────────
    return _legacy_score(nodes, text, successful_sql, error_sql, text_len, label)


def _legacy_score(
    nodes: list[dict], text: str, successful_sql: list,
    error_sql: list, text_len: int, label: str,
) -> float:
    """Original format-heuristic score (used when no structured label)."""
    text_nodes = [n for n in nodes if n.get("type") == "TEXT"]
    substantial_texts = [n for n in text_nodes if len(n.get("text", "")) > 50]

    s = 0.0
    if successful_sql:
        s += 0.08
        s += min(0.12, 0.03 * (len(successful_sql) - 1))
    non_empty_sql = [x for x in successful_sql if len(x.get("text", "").strip()) > 30]
    if non_empty_sql:
        s += 0.05
    s -= 0.08 * len(error_sql)

    if re.findall(r"^#{1,3}\s", text, re.MULTILINE):
        s += 0.10
    if len(re.findall(r"^\|.+\|$", text, re.MULTILINE)) >= 2:
        s += 0.04

    if text_len > 200:
        s += 0.04
        s += min(0.12, 0.00015 * (text_len - 200))

    insight_kw = ["建议", "结论", "总结", "趋势", "综上", "因此", "所以",
                  "主要原因", "关键", "值得关注", "异常", "显著"]
    if [kw for kw in insight_kw if kw in text]:
        s += 0.04
        s += min(0.06, 0.015 * len([kw for kw in insight_kw if kw in text]))

    numbers = re.findall(r"\d+\.?\d*%?", text)
    meaningful = [n for n in numbers if len(n.replace("%", "")) >= 2]
    if meaningful:
        s += 0.04
        s += min(0.08, 0.008 * len(meaningful))

    if label and label.lower() in text.lower():
        s += 0.25

    if successful_sql and text_len > 200:
        s += 0.05
    if len(substantial_texts) >= 3:
        s += 0.05
    if len(substantial_texts) >= 6:
        s += 0.05

    if text_len < 50:
        s -= 0.15
    if not successful_sql and text_len < 100:
        s -= 0.10

    return max(0.0, min(1.0, s))
