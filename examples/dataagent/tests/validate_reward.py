"""Validate that reward_func discriminates good vs bad reports.

Builds synthetic DataAgent traces (nodes list) for each labeled query and
shows that:

  - A report citing the *correct* ground-truth numbers scores high.
  - A report citing *wrong* numbers scores low.
  - A report citing *some* numbers scores in between.
  - A format-only report with no correct numbers scores low.

This is the critical sanity check before training: if the reward doesn't
vary with answer quality, RL won't move it.

Usage::

    python examples/dataagent/tests/validate_reward.py
    python examples/dataagent/tests/validate_reward.py --jsonl examples/dataagent/queries_labeled.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from examples.dataagent.reward_func import score, extract_numbers, _numbers_match  # noqa: E402

# queries_labeled.jsonl lives in the parent directory (examples/dataagent/).
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..")


def _make_nodes(final_text: str, sql_text: str = "product_id|name\n1|iPhone") -> list[dict]:
    """Build a minimal DataAgent node trace with one SQL result + final report."""
    return [
        {"node": "IntentRecognition", "type": "TEXT", "text": "意图：分析销售额排名"},
        {"node": "SqlGenerate", "type": "TEXT", "text": "SELECT region, SUM(quantity*unit_price) FROM orders GROUP BY region"},
        {"node": "SqlExecute", "type": "RESULT_SET", "text": sql_text},
        {"node": "ReportGenerator", "type": "TEXT", "text": final_text},
    ]


def _good_report(label: dict) -> str:
    """A report that cites every ground-truth number and entity."""
    parts = ["## 分析报告"]
    parts.append(f"根据查询，结果如下：{label['summary']}")
    if label.get("key_entities"):
        parts.append("涉及：" + "、".join(label["key_entities"]))
    parts.append("结论：以上数据反映了真实的业务分布情况，建议关注排名靠前的项。")
    return "\n".join(parts)


def _partial_report(label: dict) -> str:
    """A realistic partial report: cites the first 1-2 correct numbers+entities,
    misses the rest.  Long enough to avoid the trivial-output penalty."""
    parts = ["## 分析报告"]
    parts.append("根据数据库查询，结果如下：")
    ents = label.get("key_entities", [])
    nums = label.get("key_numbers", [])
    if ents:
        parts.append(f"排名第一的是{ents[0]}，表现最为突出。")
    if nums:
        parts.append(f"其数值约为{nums[0]}，明显高于其他项。")
    parts.append("由于篇幅限制，其他排名项暂未展开。建议进一步关注后续分析。")
    parts.append("总体来看，数据反映出当前业务的分布特征。")
    return "\n".join(parts)


def _wrong_report(label: dict) -> str:
    """A well-formatted report with completely wrong entities AND numbers."""
    parts = ["## 分析报告"]
    parts.append("根据查询，XX区域销售额最高，约为99999元。")
    parts.append("其次是YY约88888元，ZZ约77777元。")
    parts.append("| 区域 | 销售额 |\n|------|--------|\n| XX | 99999 |\n| YY | 88888 |")
    parts.append("结论：建议关注XX区域的表现，优化运营策略。")
    return "\n".join(parts)


def _format_only_report(label: dict) -> str:
    """Long, well-structured report with zero correct numbers — the reward-hacking case."""
    parts = ["## 概述", "本报告对销售情况进行分析。"]
    parts.append("## 分析")
    parts.append("从数据来看，业务表现良好，各项指标稳定。")
    parts.append("## 结论")
    parts.append("建议持续关注各项指标变化趋势，优化运营策略。")
    parts.append("数据显示整体趋势向好，主要原因是多方面因素共同作用。")
    return "\n".join(parts)


def _no_sql_traces(label: dict) -> list[dict]:
    """Traces where the SQL step failed / returned nothing."""
    return [
        {"node": "IntentRecognition", "type": "TEXT", "text": "意图：分析"},
        {"node": "SqlGenerate", "type": "TEXT", "text": "SELECT * FROM missing_table"},
        {"node": "SqlExecute", "type": "RESULT_SET", "text": "ERROR: table not found"},
        {"node": "ReportGenerator", "type": "TEXT", "text": _wrong_report(label)},
    ]


# ── main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=os.path.join(DATA_DIR, "queries_labeled.jsonl"))
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    with open(args.jsonl, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]

    print(f"Loaded {len(items)} labeled queries from {args.jsonl}\n")
    print(f"{'Query':<40} {'good':>6} {'partial':>8} {'wrong':>6} {'fmt':>6} {'noSQL':>6}")
    print("─" * 80)

    good_total = partial_total = wrong_total = fmt_total = nosql_total = 0.0
    n = 0
    for it in items:
        label = it["label"]
        label_dict = json.loads(label)

        good = score(_make_nodes(_good_report(label_dict)), label)
        partial = score(_make_nodes(_partial_report(label_dict)), label)
        wrong = score(_make_nodes(_wrong_report(label_dict)), label)
        fmt = score(_make_nodes(_format_only_report(label_dict)), label)
        nosql = score(_no_sql_traces(label_dict), label)

        good_total += good
        partial_total += partial
        wrong_total += wrong
        fmt_total += fmt
        nosql_total += nosql
        n += 1

        if args.verbose:
            print(f"{it['query'][:38]:<40} {good:>6.2f} {partial:>8.2f} {wrong:>6.2f} {fmt:>6.2f} {nosql:>6.2f}")

    print("─" * 80)
    print(f"{'AVERAGE':<40} {good_total/n:>6.2f} {partial_total/n:>8.2f} {wrong_total/n:>6.2f} {fmt_total/n:>6.2f} {nosql_total/n:>6.2f}")
    print()
    print("Sanity checks:")
    print(f"  good > wrong ?    {good_total/n:.2f} > {wrong_total/n:.2f}  → {'YES ✓' if good_total/n > wrong_total/n else 'NO ✗'}")
    print(f"  good > partial ?  {good_total/n:.2f} > {partial_total/n:.2f}  → {'YES ✓' if good_total/n > partial_total/n else 'NO ✗'}")
    print(f"  partial > wrong ? {partial_total/n:.2f} > {wrong_total/n:.2f}  → {'YES ✓' if partial_total/n > wrong_total/n else 'NO ✗'}")
    print(f"  good > fmt-only ? {good_total/n:.2f} > {fmt_total/n:.2f}  → {'YES ✓' if good_total/n > fmt_total/n else 'NO ✗'}")
    print(f"  wrong ≈ fmt-only ? {wrong_total/n:.2f} vs {fmt_total/n:.2f}  → {'YES ✓' if abs(wrong_total/n - fmt_total/n) < 0.15 else 'check'}")

    # ── number extraction self-test ──────────────────────────────────
    print("\nNumber extraction self-test:")
    for text, expect in [
        ("销售额252811元", [252811.0]),
        ("约25.28万", [252800.0]),
        ("线上457,862元，线下176,328元", [457862.0, 176328.0]),
        ("转化率0.91%", [0.91]),
        ("电子产品占65.4%", [65.4]),
        ("库存35件", [35.0]),
    ]:
        got = extract_numbers(text)
        ok = "✓" if got == expect else f"✗ (got {got})"
        print(f"  {text:<30} expect {expect}  {ok}")

    # ── matching tolerance self-test ─────────────────────────────────
    print("\nMatching tolerance self-test:")
    for report_num, truth, expect_ok in [
        (252800.0, "252811.0", True),    # 25.28万 vs 252811 — within 1%
        (252811.0, "252811.0", True),    # exact
        (99999.0, "252811.0", False),    # wrong
        (7, "7", True),                  # exact count
        (6, "7", False),                 # wrong count
        (0.91, "0.91", True),           # percentage
        (1.0, "0.91", False),           # wrong percentage
    ]:
        got = _numbers_match(report_num, truth)
        ok = "✓" if got == expect_ok else f"✗"
        print(f"  report={report_num:<10} truth={truth:<10} match={got} expect={expect_ok} {ok}")


if __name__ == "__main__":
    main()
