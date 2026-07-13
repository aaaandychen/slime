"""Generate labeled training data for DataAgent RL.

Computes ground-truth answers for query templates against the demo_sales
database, so the reward function can score numerical accuracy instead of
just report format.

Two modes:
  1. --source mysql   reads from a live MariaDB (default if mysql works)
  2. --source offline  parses the INSERT statements embedded in Database.md
                      so you can generate labels without a running DB

Output: queries_labeled.jsonl, one JSON per line::

    {"query": "各区域销售额排名",
     "label": "{\"key_numbers\": [...], \"key_entities\": [...], ...}"}

The ``label`` is a JSON *string* (matches slime Sample.label: str|None).

Usage::

    python examples/dataagent/generate_training_data.py
    python examples/dataagent/generate_training_data.py --source mysql --out queries_labeled.jsonl
    python examples/dataagent/generate_training_data.py --source offline
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from decimal import Decimal
from typing import Any

# ── path setup ─────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DATABASE_MD = os.path.join(HERE, "Database.md")


# ── offline data: parse INSERT statements from Database.md ──────────────

def _parse_sql_inserts(text: str) -> dict[str, tuple[list[str] | None, list[tuple]]]:
    """Parse ``INSERT INTO `tbl` [(cols)] VALUES (...), (...)`` from SQL text.

    Returns ``{table: (col_names_or_None, [(v1, v2, ...), ...])}``.  Values
    are best-effort coerced to int / float / str / None.
    """
    tables: dict[str, tuple[list[str] | None, list[tuple]]] = {}
    insert_re = re.compile(
        r"INSERT\s+INTO\s+`(\w+)`\s*(?:\(([^)]*)\)\s*)?VALUES\s*(.*?);",
        re.IGNORECASE | re.DOTALL,
    )
    for m in insert_re.finditer(text):
        tbl = m.group(1)
        cols_str = m.group(2)
        cols = None
        if cols_str:
            cols = [c.strip().strip("`") for c in cols_str.split(",") if c.strip()]
        values_sql = m.group(3)
        rows = _parse_value_tuples(values_sql)
        if rows:
            tables[tbl] = (cols, rows)
    return tables


def _parse_value_tuples(s: str) -> list[tuple]:
    """Parse ``(...), (...), (...)`` into a list of tuples of Python values."""
    rows: list[tuple] = []
    i, n = 0, len(s)
    while i < n:
        # skip to '('
        while i < n and s[i] != "(":
            i += 1
        if i >= n:
            break
        i += 1  # past '('
        depth, buf, in_str, cur = 1, [], False, []
        while i < n and depth > 0:
            ch = s[i]
            if ch == "'" and (i + 1 >= n or s[i + 1] != "'"):
                in_str = not in_str
                cur.append(ch)
            elif ch == "'" and i + 1 < n and s[i + 1] == "'":
                cur.append("'")
                i += 1
            elif ch == "(" and not in_str:
                depth += 1
                cur.append(ch)
            elif ch == ")" and not in_str:
                depth -= 1
                if depth == 0:
                    break
                cur.append(ch)
            elif ch == "," and depth == 1 and not in_str:
                buf.append("".join(cur).strip())
                cur = []
            else:
                cur.append(ch)
            i += 1
        buf.append("".join(cur).strip())
        rows.append(tuple(_coerce(v) for v in buf))
        i += 1  # past ')'
    return rows


def _coerce(v: str) -> Any:
    v = v.strip()
    if v == "NULL":
        return None
    if v.startswith("'") and v.endswith("'"):
        return v[1:-1]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _load_offline() -> sqlite3.Connection:
    """Build an in-memory sqlite db from Database.md INSERT data."""
    with open(DATABASE_MD, encoding="utf-8") as f:
        text = f.read()
    tables = _parse_sql_inserts(text)

    con = sqlite3.connect(":memory:")
    cur = con.cursor()

    # Schema (mirrors Database.md, types simplified for sqlite)
    schema = {
        "suppliers": """CREATE TABLE suppliers (
            supplier_id INTEGER, name TEXT, category TEXT, contact_person TEXT,
            region TEXT, cooperation_start TEXT, rating REAL,
            on_time_delivery_rate REAL, total_supplied_amount REAL)""",
        "products": """CREATE TABLE products (
            product_id INTEGER, name TEXT, category TEXT, price REAL,
            cost REAL, stock INTEGER, supplier_id INTEGER)""",
        "customers": """CREATE TABLE customers (
            customer_id INTEGER, name TEXT, region TEXT, city TEXT, level TEXT,
            gender TEXT, age_group TEXT, register_date TEXT,
            total_orders INTEGER, total_amount REAL)""",
        "orders": """CREATE TABLE orders (
            order_id INTEGER, product_id INTEGER, customer_name TEXT,
            quantity INTEGER, unit_price REAL, order_date TEXT,
            region TEXT, channel TEXT)""",
        "campaigns": """CREATE TABLE campaigns (
            campaign_id INTEGER, name TEXT, channel TEXT, type TEXT,
            budget REAL, actual_cost REAL, start_date TEXT, end_date TEXT,
            impressions INTEGER, clicks INTEGER, conversions INTEGER,
            revenue_generated REAL)""",
        "returns": """CREATE TABLE returns (
            return_id INTEGER, order_id INTEGER, product_id INTEGER,
            customer_id INTEGER, quantity INTEGER, return_amount REAL,
            reason TEXT, return_date TEXT, status TEXT)""",
        "daily_traffic": """CREATE TABLE daily_traffic (
            id INTEGER, dt TEXT, channel TEXT, visits INTEGER,
            unique_visitors INTEGER, orders_count INTEGER, revenue REAL)""",
    }
    for ddl in schema.values():
        cur.execute(ddl)

    for tbl, (cols, rows) in tables.items():
        if not rows:
            continue
        ncols = len(rows[0])
        placeholders = ",".join("?" * ncols)
        if cols:
            col_list = ",".join(cols)
            cur.executemany(
                f"INSERT INTO {tbl} ({col_list}) VALUES ({placeholders})",
                rows,
            )
        else:
            cur.executemany(f"INSERT INTO {tbl} VALUES ({placeholders})", rows)
    con.commit()
    return con


def _load_mysql() -> Any:
    import pymysql  # type: ignore
    return pymysql.connect(host="127.0.0.1", port=3306, user="root",
                          password="", database="demo_sales", charset="utf8mb4")


# ── helpers ─────────────────────────────────────────────────────────────

def _q(con, sql: str) -> list[dict]:
    cur = con.cursor()
    cur.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _scalar(con, sql: str) -> Any:
    cur = con.cursor()
    cur.execute(sql)
    row = cur.fetchone()
    return row[0] if row else None


def _num(x: Any) -> float:
    if x is None:
        return 0.0
    if isinstance(x, Decimal):
        return float(x)
    return float(x)


def _round(x: Any, ndigits: int = 2) -> float:
    return round(_num(x), ndigits)


# ── query templates → (query_text, label_dict) ──────────────────────────
# Each template returns a list of (query, label) pairs.
# ``label`` is a dict with keys:
#   key_numbers   — list of numeric strings that should appear in a good report
#   key_entities  — list of named entities (product/region/customer/etc.)
#   expected_sql  — the canonical SQL (for audit; not used by reward directly)
#   summary       — short human-readable expected answer (for debugging)

def _t_region_sales_ranking(con) -> list[tuple[str, dict]]:
    rows = _q(con, """
        SELECT region, SUM(quantity * unit_price) AS sales
        FROM orders GROUP BY region ORDER BY sales DESC
    """)
    ranked = [(r["region"], _num(r["sales"])) for r in rows]
    label = {
        "key_numbers": [str(_round(s, 2)) for _, s in ranked],
        "key_entities": [name for name, _ in ranked],
        "expected_sql": "SELECT region, SUM(quantity*unit_price) FROM orders GROUP BY region ORDER BY 2 DESC",
        "summary": "、".join(f"{name}({s:.0f})" for name, s in ranked),
    }
    return [("各区域销售额排名", label)]


def _t_channel_sales(con) -> list[tuple[str, dict]]:
    rows = _q(con, """
        SELECT channel, SUM(quantity * unit_price) AS sales
        FROM orders GROUP BY channel ORDER BY sales DESC
    """)
    label = {
        "key_numbers": [str(_round(r["sales"], 2)) for r in rows],
        "key_entities": [r["channel"] for r in rows],
        "expected_sql": "SELECT channel, SUM(quantity*unit_price) FROM orders GROUP BY channel",
        "summary": "、".join(f"{r['channel']}({r['sales']:.0f})" for r in rows),
    }
    return [("各渠道销售额对比分析", label)]


def _t_monthly_sales_trend(con) -> list[tuple[str, dict]]:
    rows = _q(con, """
        SELECT strftime('%Y-%m', order_date) AS ym,
               SUM(quantity * unit_price) AS sales,
               COUNT(*) AS orders
        FROM orders GROUP BY ym ORDER BY ym
    """)
    label = {
        "key_numbers": [str(_round(r["sales"], 2)) for r in rows],
        "key_entities": [r["ym"] for r in rows],
        "expected_sql": "SELECT strftime('%Y-%m', order_date), SUM(quantity*unit_price) FROM orders GROUP BY 1",
        "summary": "、".join(f"{r['ym']}({r['sales']:.0f})" for r in rows),
    }
    return [("每月的总销售额和订单量趋势", label)]


def _t_category_sales_share(con) -> list[tuple[str, dict]]:
    rows = _q(con, """
        SELECT p.category, SUM(o.quantity * o.unit_price) AS sales
        FROM orders o JOIN products p ON o.product_id = p.product_id
        GROUP BY p.category ORDER BY sales DESC
    """)
    total = sum(_num(r["sales"]) for r in rows)
    label = {
        "key_numbers": [str(_round(r["sales"], 2)) for r in rows]
                     + [str(_round(total, 2))],
        "key_entities": [r["category"] for r in rows],
        "expected_sql": "SELECT p.category, SUM(o.quantity*o.unit_price) FROM orders o JOIN products p ON o.product_id=p.product_id GROUP BY p.category",
        "summary": "、".join(f"{r['category']}({r['sales']:.0f}, {r['sales']/total*100:.1f}%)" for r in rows),
    }
    return [("各品类销售额占比分析", label)]


def _t_top_products_by_revenue(con) -> list[tuple[str, dict]]:
    rows = _q(con, """
        SELECT p.name, SUM(o.quantity * o.unit_price) AS revenue
        FROM orders o JOIN products p ON o.product_id = p.product_id
        GROUP BY p.name ORDER BY revenue DESC LIMIT 5
    """)
    label = {
        "key_numbers": [str(_round(r["revenue"], 2)) for r in rows],
        "key_entities": [r["name"] for r in rows],
        "expected_sql": "SELECT p.name, SUM(o.quantity*o.unit_price) FROM orders o JOIN products p ON o.product_id=p.product_id GROUP BY p.name ORDER BY 2 DESC LIMIT 5",
        "summary": "、".join(f"{r['name']}({r['revenue']:.0f})" for r in rows),
    }
    return [("销售额最高的前5个商品", label)]


def _t_top_customers_by_spend(con) -> list[tuple[str, dict]]:
    rows = _q(con, """
        SELECT customer_name, SUM(quantity * unit_price) AS spend
        FROM orders GROUP BY customer_name ORDER BY spend DESC LIMIT 5
    """)
    label = {
        "key_numbers": [str(_round(r["spend"], 2)) for r in rows],
        "key_entities": [r["customer_name"] for r in rows],
        "expected_sql": "SELECT customer_name, SUM(quantity*unit_price) FROM orders GROUP BY customer_name ORDER BY 2 DESC LIMIT 5",
        "summary": "、".join(f"{r['customer_name']}({r['spend']:.0f})" for r in rows),
    }
    return [("2026年上半年消费最高的前5个客户", label)]


def _t_profit_margin_top(con) -> list[tuple[str, dict]]:
    rows = _q(con, """
        SELECT name, (price - cost) / price * 100 AS margin
        FROM products ORDER BY margin DESC LIMIT 5
    """)
    label = {
        "key_numbers": [str(_round(r["margin"], 2)) for r in rows],
        "key_entities": [r["name"] for r in rows],
        "expected_sql": "SELECT name, (price-cost)/price*100 FROM products ORDER BY 2 DESC LIMIT 5",
        "summary": "、".join(f"{r['name']}({r['margin']:.1f}%)" for r in rows),
    }
    return [("利润率最高的5个商品", label)]


def _t_campaign_roi(con) -> list[tuple[str, dict]]:
    rows = _q(con, """
        SELECT name, revenue_generated / actual_cost AS roi,
               revenue_generated, actual_cost
        FROM campaigns ORDER BY roi DESC
    """)
    label = {
        "key_numbers": [str(_round(r["roi"], 2)) for r in rows],
        "key_entities": [r["name"] for r in rows],
        "expected_sql": "SELECT name, revenue_generated/actual_cost FROM campaigns ORDER BY 2 DESC",
        "summary": "、".join(f"{r['name']}(ROI {r['roi']:.2f})" for r in rows),
    }
    return [("各营销活动ROI排名", label)]


def _t_campaign_type_roi(con) -> list[tuple[str, dict]]:
    rows = _q(con, """
        SELECT type, AVG(revenue_generated / actual_cost) AS avg_roi,
               SUM(revenue_generated) AS total_rev, SUM(actual_cost) AS total_cost
        FROM campaigns GROUP BY type ORDER BY avg_roi DESC
    """)
    label = {
        "key_numbers": [str(_round(r["avg_roi"], 2)) for r in rows],
        "key_entities": [r["type"] for r in rows],
        "expected_sql": "SELECT type, AVG(revenue_generated/actual_cost) FROM campaigns GROUP BY type ORDER BY 2 DESC",
        "summary": "、".join(f"{r['type']}(ROI {r['avg_roi']:.2f})" for r in rows),
    }
    return [("折扣类和满减类活动的ROI对比", label)]


def _t_inventory_low(con) -> list[tuple[str, dict]]:
    rows = _q(con, """
        SELECT name, stock, category FROM products
        WHERE stock < 50 ORDER BY stock ASC
    """)
    label = {
        "key_numbers": [str(r["stock"]) for r in rows],
        "key_entities": [r["name"] for r in rows],
        "expected_sql": "SELECT name, stock FROM products WHERE stock < 50 ORDER BY stock",
        "summary": "、".join(f"{r['name']}(库存{r['stock']})" for r in rows),
    }
    return [("库存低于50的商品有哪些", label)]


def _t_supplier_rating(con) -> list[tuple[str, dict]]:
    rows = _q(con, """
        SELECT name, rating, on_time_delivery_rate
        FROM suppliers ORDER BY rating DESC
    """)
    label = {
        "key_numbers": [str(_round(r["rating"], 2)) for r in rows],
        "key_entities": [r["name"] for r in rows],
        "expected_sql": "SELECT name, rating FROM suppliers ORDER BY rating DESC",
        "summary": "、".join(f"{r['name']}(评分{r['rating']:.2f})" for r in rows),
    }
    return [("各供应商评分排名", label)]


def _t_customer_level_count(con) -> list[tuple[str, dict]]:
    rows = _q(con, """
        SELECT level, COUNT(*) AS cnt FROM customers
        GROUP BY level ORDER BY cnt DESC
    """)
    label = {
        "key_numbers": [str(r["cnt"]) for r in rows],
        "key_entities": [r["level"] for r in rows],
        "expected_sql": "SELECT level, COUNT(*) FROM customers GROUP BY level ORDER BY 2 DESC",
        "summary": "、".join(f"{r['level']}({r['cnt']}人)" for r in rows),
    }
    return [("各等级会员数量分布", label)]


def _t_region_customer_count(con) -> list[tuple[str, dict]]:
    rows = _q(con, """
        SELECT region, COUNT(*) AS cnt FROM customers
        GROUP BY region ORDER BY cnt DESC
    """)
    label = {
        "key_numbers": [str(r["cnt"]) for r in rows],
        "key_entities": [r["region"] for r in rows],
        "expected_sql": "SELECT region, COUNT(*) FROM customers GROUP BY region ORDER BY 2 DESC",
        "summary": "、".join(f"{r['region']}({r['cnt']}人)" for r in rows),
    }
    return [("各区域客户数量分布", label)]


def _t_return_reason_count(con) -> list[tuple[str, dict]]:
    rows = _q(con, """
        SELECT reason, COUNT(*) AS cnt, SUM(return_amount) AS total
        FROM returns GROUP BY reason ORDER BY cnt DESC
    """)
    label = {
        "key_numbers": [str(r["cnt"]) for r in rows],
        "key_entities": [r["reason"] for r in rows],
        "expected_sql": "SELECT reason, COUNT(*) FROM returns GROUP BY reason ORDER BY 2 DESC",
        "summary": "、".join(f"{r['reason']}({r['cnt']}次)" for r in rows),
    }
    return [("各退货原因分布及占比", label)]


def _t_return_amount_by_reason(con) -> list[tuple[str, dict]]:
    rows = _q(con, """
        SELECT reason, SUM(return_amount) AS total
        FROM returns GROUP BY reason ORDER BY total DESC
    """)
    label = {
        "key_numbers": [str(_round(r["total"], 2)) for r in rows],
        "key_entities": [r["reason"] for r in rows],
        "expected_sql": "SELECT reason, SUM(return_amount) FROM returns GROUP BY reason ORDER BY 2 DESC",
        "summary": "、".join(f"{r['reason']}({r['total']:.0f}元)" for r in rows),
    }
    return [("哪种退货原因导致的经济损失最大", label)]


def _t_traffic_channel_conversion(con) -> list[tuple[str, dict]]:
    rows = _q(con, """
        SELECT channel, SUM(orders_count) AS orders, SUM(unique_visitors) AS uv,
               SUM(orders_count) * 1.0 / SUM(unique_visitors) AS conv
        FROM daily_traffic GROUP BY channel ORDER BY conv DESC
    """)
    label = {
        "key_numbers": [str(_round(r["conv"] * 100, 2)) for r in rows],
        "key_entities": [r["channel"] for r in rows],
        "expected_sql": "SELECT channel, SUM(orders_count)/SUM(unique_visitors) FROM daily_traffic GROUP BY channel",
        "summary": "、".join(f"{r['channel']}(转化率{r['conv']*100:.2f}%)" for r in rows),
    }
    return [("各渠道访客到订单的转化率对比", label)]


def _t_category_product_count(con) -> list[tuple[str, dict]]:
    rows = _q(con, """
        SELECT category, COUNT(*) AS cnt FROM products
        GROUP BY category ORDER BY cnt DESC
    """)
    label = {
        "key_numbers": [str(r["cnt"]) for r in rows],
        "key_entities": [r["category"] for r in rows],
        "expected_sql": "SELECT category, COUNT(*) FROM products GROUP BY category ORDER BY 2 DESC",
        "summary": "、".join(f"{r['category']}({r['cnt']}个)" for r in rows),
    }
    return [("各品类商品数量分布", label)]


def _t_top_supplier_by_supplied(con) -> list[tuple[str, dict]]:
    rows = _q(con, """
        SELECT name, total_supplied_amount, category
        FROM suppliers ORDER BY total_supplied_amount DESC LIMIT 5
    """)
    label = {
        "key_numbers": [str(_round(r["total_supplied_amount"], 2)) for r in rows],
        "key_entities": [r["name"] for r in rows],
        "expected_sql": "SELECT name, total_supplied_amount FROM suppliers ORDER BY 2 DESC LIMIT 5",
        "summary": "、".join(f"{r['name']}({r['total_supplied_amount']:.0f})" for r in rows),
    }
    return [("累计供货金额最高的5个供应商", label)]


def _t_orders_by_channel(con) -> list[tuple[str, dict]]:
    rows = _q(con, """
        SELECT channel, COUNT(*) AS cnt, SUM(quantity * unit_price) AS sales
        FROM orders GROUP BY channel ORDER BY sales DESC
    """)
    label = {
        "key_numbers": [str(r["cnt"]) for r in rows] + [str(_round(r["sales"], 2)) for r in rows],
        "key_entities": [r["channel"] for r in rows],
        "expected_sql": "SELECT channel, COUNT(*), SUM(quantity*unit_price) FROM orders GROUP BY channel",
        "summary": "、".join(f"{r['channel']}({r['cnt']}单,{r['sales']:.0f}元)" for r in rows),
    }
    return [("线上和线下渠道的订单数量和销售额对比", label)]


def _t_top_product_by_quantity(con) -> list[tuple[str, dict]]:
    rows = _q(con, """
        SELECT p.name, SUM(o.quantity) AS qty
        FROM orders o JOIN products p ON o.product_id = p.product_id
        GROUP BY p.name ORDER BY qty DESC LIMIT 5
    """)
    label = {
        "key_numbers": [str(r["qty"]) for r in rows],
        "key_entities": [r["name"] for r in rows],
        "expected_sql": "SELECT p.name, SUM(o.quantity) FROM orders o JOIN products p ON o.product_id=p.product_id GROUP BY p.name ORDER BY 2 DESC LIMIT 5",
        "summary": "、".join(f"{r['name']}({r['qty']}件)" for r in rows),
    }
    return [("销量最高的5个商品（按件数）", label)]


# ── registry ────────────────────────────────────────────────────────────

TEMPLATES = [
    _t_region_sales_ranking,
    _t_channel_sales,
    _t_monthly_sales_trend,
    _t_category_sales_share,
    _t_top_products_by_revenue,
    _t_top_customers_by_spend,
    _t_profit_margin_top,
    _t_campaign_roi,
    _t_campaign_type_roi,
    _t_inventory_low,
    _t_supplier_rating,
    _t_customer_level_count,
    _t_region_customer_count,
    _t_return_reason_count,
    _t_return_amount_by_reason,
    _t_traffic_channel_conversion,
    _t_category_product_count,
    _t_top_supplier_by_supplied,
    _t_orders_by_channel,
    _t_top_product_by_quantity,
]


def generate(con) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for tpl in TEMPLATES:
        for query, label in tpl(con):
            if query in seen:
                continue
            seen.add(query)
            out.append({
                "query": query,
                "label": json.dumps(label, ensure_ascii=False),
            })
    return out


# ── main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["offline", "mysql"], default="offline",
                    help="offline = parse Database.md (no DB needed); mysql = live MariaDB")
    ap.add_argument("--out", default=os.path.join(HERE, "queries_labeled.jsonl"))
    ap.add_argument("--print-summary", action="store_true",
                    help="print each query + expected answer for quick eyeball check")
    args = ap.parse_args()

    if args.source == "mysql":
        try:
            con = _load_mysql()
        except Exception as e:
            print(f"mysql connect failed: {e}; falling back to offline", file=sys.stderr)
            con = _load_offline()
    else:
        con = _load_offline()

    data = generate(con)
    with open(args.out, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"Wrote {len(data)} labeled queries → {args.out}")
    if args.print_summary:
        for d in data:
            label = json.loads(d["label"])
            print(f"\nQ: {d['query']}")
            print(f"  numbers : {label['key_numbers']}")
            print(f"  entities: {label['key_entities']}")
            print(f"  summary : {label['summary']}")


if __name__ == "__main__":
    main()
