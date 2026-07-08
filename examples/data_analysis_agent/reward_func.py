"""Reward function for DataAgent — score the final report output.

Used by custom_generate: the reward is applied to all tokens in the
sample (credit assignment is episode-level: one score → all tokens).

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


def score(nodes: list[dict[str, Any]], label: str = "") -> float:
    """Score a DataAgent workflow trace.

    Args:
        nodes: List of per-node outputs, each ``{"node", "type", "text"}``.
        label: Optional ground-truth answer for exact-match bonus.

    Returns:
        Score in [0.0, 1.0].
    """
    if not nodes:
        return 0.0

    final = nodes[-1]
    text = final.get("text", "")

    # ── categorize nodes ──────────────────────────────────────────────
    sql_results = [n for n in nodes if n.get("type") == "RESULT_SET"]
    text_nodes = [n for n in nodes if n.get("type") == "TEXT"]
    successful_sql = [
        s for s in sql_results
        if "error" not in s.get("text", "").lower() and s.get("text", "").strip()
    ]
    error_sql = [s for s in sql_results if "error" in s.get("text", "").lower()]
    substantial_texts = [n for n in text_nodes if len(n.get("text", "")) > 50]

    s = 0.0

    # ── SQL execution quality (0.0 – 0.25) ────────────────────────────
    # Base reward for at least one successful query
    if successful_sql:
        s += 0.08
        # Progressive bonus per additional successful query (diminishing)
        s += min(0.12, 0.03 * (len(successful_sql) - 1))

    # Bonus: query returned non-trivial data (not just 0 rows / empty)
    non_empty_sql = [
        s for s in successful_sql
        if len(s.get("text", "").strip()) > 30
    ]
    if non_empty_sql:
        s += 0.05

    # Penalty per SQL error
    s -= 0.08 * len(error_sql)

    # ── Report structure (0.0 – 0.20) ─────────────────────────────────
    # Markdown headings — reward count, not just existence
    heading_matches = re.findall(r'^#{1,3}\s', text, re.MULTILINE)
    if heading_matches:
        s += min(0.10, 0.025 * len(heading_matches))

    # Table — check for well-formed markdown table
    table_rows = len(re.findall(r'^\|.+\|$', text, re.MULTILINE))
    if table_rows >= 2:  # at least header + separator
        s += 0.04
        s += min(0.06, 0.01 * (table_rows - 2))

    # ── Content depth (0.0 – 0.20) ────────────────────────────────────
    text_len = len(text)
    if text_len > 200:
        s += 0.04
        # Progressive: ~0.08 bonus at 1000 chars, ~0.12 at 2000
        s += min(0.12, 0.00015 * (text_len - 200))

    # Conclusion / insight keywords
    insight_kw = ["建议", "结论", "总结", "趋势", "综上", "因此", "所以",
                   "主要原因", "关键", "值得关注", "异常", "显著"]
    found_insight = [kw for kw in insight_kw if kw in text]
    if found_insight:
        s += 0.04
        s += min(0.06, 0.015 * len(found_insight))

    # ── Data grounding (0.0 – 0.20) ───────────────────────────────────
    # Specific numbers cited → proxy for data-backed analysis
    numbers = re.findall(r'\d+\.?\d*%?', text)
    # Filter out trivial numbers (dates, single digits)
    meaningful_numbers = [n for n in numbers if len(n.replace('%', '')) >= 2]
    if meaningful_numbers:
        s += 0.04
        s += min(0.08, 0.008 * len(meaningful_numbers))

    # Data-analysis-specific phrases
    data_phrases = ["数据显示", "根据", "占比", "增长", "下降", "排名",
                    "最高", "最低", "平均", "同比", "环比", "趋势"]
    found_data = [p for p in data_phrases if p in text]
    if found_data:
        s += 0.04
        s += min(0.04, 0.01 * len(found_data))

    # ── Process quality (0.0 – 0.15) ──────────────────────────────────
    # Reward: queried data AND produced substantial report
    if successful_sql and text_len > 200:
        s += 0.05

    # Reward: multi-step analysis (multiple substantial text nodes)
    if len(substantial_texts) >= 3:
        s += 0.05
    if len(substantial_texts) >= 6:
        s += 0.05

    # ── Label match (0.0 – 0.25) ──────────────────────────────────────
    if label and label.lower() in text.lower():
        s += 0.25

    # ── Penalty: trivial / low-effort output ──────────────────────────
    if text_len < 50:
        s -= 0.15
    if not successful_sql and text_len < 100:
        s -= 0.10  # no data queried + short output = likely hallucinated

    return max(0.0, min(1.0, s))
