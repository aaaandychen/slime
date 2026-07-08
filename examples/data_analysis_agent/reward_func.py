"""Reward function for Data-Analysis-Agent — score the final analysis report.

Input comes from custom_generate:
  text: concatenated final answer (from text_delta SSE events)
  tool_events: list of tool_audit events from the agent run

Design principles for RL reward:
1. Many fine-grained dimensions (0.03–0.05) instead of few coarse ones (0.2)
   → higher variance across samples → stronger gradient signal.
2. Reward the *process* (SQL execution, multi-step reasoning), not just
   the final report structure.
3. Data grounding: Did the agent actually query data and cite numbers?
4. Negative signals: penalize SQL errors, empty results, trivial output.
"""

from __future__ import annotations

import re
from typing import Any


def score(text: str, tool_events: list[dict], label: str = "") -> float:
    """Score a DAA workflow trace.

    Args:
        text: Concatenated final answer from DAA SSE text_delta events.
        tool_events: List of tool_audit events, each containing
            ``{"tool", "ok", "error", "summary"}``.
        label: Optional ground-truth answer for exact-match bonus.

    Returns:
        Score in [0.0, 1.0].
    """
    if not text:
        return 0.0

    # ── categorize tool events ──────────────────────────────────────
    sql_events = [
        e for e in tool_events
        if e.get("tool") in ("query_data", "create_analysis_table")
    ]
    successful_sql = [e for e in sql_events if e.get("ok", False)]
    error_sql = [e for e in sql_events if not e.get("ok", False)]
    other_tools = [
        e for e in tool_events
        if e.get("tool") not in ("query_data", "create_analysis_table")
    ]

    s = 0.0

    # ── SQL execution quality (0.0 – 0.25) ──────────────────────────
    if successful_sql:
        s += 0.08
        s += min(0.12, 0.03 * (len(successful_sql) - 1))

    # Bonus: query returned non-trivial data
    non_empty = [
        e for e in successful_sql
        if (e.get("content_len") or 0) > 30
    ]
    if non_empty:
        s += 0.05

    s -= 0.08 * len(error_sql)

    # ── Report structure (0.0 – 0.20) ───────────────────────────────
    heading_matches = re.findall(r'^#{1,3}\s', text, re.MULTILINE)
    if heading_matches:
        s += min(0.10, 0.025 * len(heading_matches))

    table_rows = len(re.findall(r'^\|.+\|$', text, re.MULTILINE))
    if table_rows >= 2:
        s += 0.04
        s += min(0.06, 0.01 * (table_rows - 2))

    # ── Content depth (0.0 – 0.15) ──────────────────────────────────
    text_len = len(text)
    if text_len > 200:
        s += 0.04
        s += min(0.11, 0.00015 * (text_len - 200))

    insight_kw = ["建议", "结论", "总结", "趋势", "综上", "因此", "所以",
                   "主要原因", "关键", "值得关注", "异常", "显著"]
    found = [kw for kw in insight_kw if kw in text]
    if found:
        s += 0.04
        s += min(0.06, 0.015 * len(found))

    # ── Data grounding (0.0 – 0.20) ─────────────────────────────────
    numbers = re.findall(r'\d+\.?\d*%?', text)
    meaningful = [n for n in numbers if len(n.replace('%', '')) >= 2]
    if meaningful:
        s += 0.04
        s += min(0.08, 0.008 * len(meaningful))

    data_phrases = ["数据显示", "根据", "占比", "增长", "下降", "排名",
                    "最高", "最低", "平均", "同比", "环比", "趋势"]
    found_data = [p for p in data_phrases if p in text]
    if found_data:
        s += 0.04
        s += min(0.04, 0.01 * len(found_data))

    # ── Process quality (0.0 – 0.15) ────────────────────────────────
    if successful_sql and text_len > 200:
        s += 0.05

    if len(other_tools) >= 2:
        s += 0.05
    if len(other_tools) >= 5:
        s += 0.05

    # ── Label match (0.0 – 0.25) ────────────────────────────────────
    if label and label.lower() in text.lower():
        s += 0.25

    # ── Penalties ───────────────────────────────────────────────────
    if text_len < 50:
        s -= 0.15
    if not successful_sql and text_len < 100:
        s -= 0.10

    return max(0.0, min(1.0, s))
