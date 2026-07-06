"""Reward function for DataAgent — score the final report output.

Used by custom_generate: the reward is applied to all tokens in the
sample (credit assignment is episode-level: one score → all tokens).

"""

from __future__ import annotations

from typing import Any


def score(nodes: list[dict[str, Any]], label: str = "") -> float:
    """Score a DataAgent workflow trace from its final report.

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

    s = 0.0

    # --- structural quality of final report ---
    if "##" in text:
        s += 0.2  # has markdown headings
    if "|" in text and text.count("|") >= 4:
        s += 0.2  # has a table (at least header + separator + 1 row)
    if len(text) > 200:
        s += 0.2  # substantive analysis
    if any(kw in text for kw in ("建议", "结论", "总结", "建议", "趋势")):
        s += 0.2  # has conclusion / recommendation

    # --- intermediate signal: SQL executed successfully ---
    sql_results = [n for n in nodes if n.get("type") == "RESULT_SET"]
    if sql_results and "error" not in sql_results[-1].get("text", "").lower():
        s += 0.2

    # --- label match ---
    if label and label.lower() in text.lower():
        s += 0.3

    return min(s, 1.0)
