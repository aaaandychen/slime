#!/usr/bin/env python3
"""Convert LongDS task.json files into slime-compatible JSONL training data.

Usage:
    python3 convert_longds_to_slime.py \
        --task-root /path/to/dataset/task/longds \
        --data-root /path/to/dataset/data/longds \
        --image registry.example.com/longds:latest \
        --output longds_train.jsonl

Input:
    {task_root}/task_list.json           — [{task_domain, dataset_name, task_id}, ...]
    {task_root}/{domain}/{dataset}/{id}/task.json — [{turn_id, context, question, answer, ...}]
    {data_root}/{domain}/{dataset}/{id}/data/*    — CSV/Parquet/JSON files

Output (one JSON line per task):
    {"prompt": "", "label": "domain__dataset__task_id",
     "metadata": {"task_id": "...", "domain": "...", "image": "...",
                  "workdir": "/workspace", "data_root": "...", "data_files": [...],
                  "turns": [{"turn_id":1, "context":"...", "question":"...",
                             "ground_truth":"...", "answer_type":"...", "files_used":[...]}]}}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def detect_answer_type(answer: str) -> str:
    """Detect answer type for reward function routing.

    Returns: numeric | categorical | json | list | exact
    """
    if not answer or not answer.strip():
        return "exact"
    s = answer.strip()
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


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_data_files(data_dir: Path) -> list[str]:
    if not data_dir.is_dir():
        return []
    return sorted(
        p.name for p in data_dir.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )


def convert_one_task(
    task_info: dict[str, str],
    task_root: Path,
    data_root: Path,
    image: str,
    workdir: str = "/workspace",
) -> dict[str, Any] | None:
    domain = task_info["task_domain"]
    dataset = task_info["dataset_name"]
    task_id = task_info["task_id"]
    task_key = f"{domain}/{dataset}/{task_id}"

    task_json_path = task_root / domain / dataset / task_id / "task.json"
    if not task_json_path.is_file():
        print(f"WARNING: missing {task_json_path}, skipping {task_key}", file=sys.stderr)
        return None

    data_dir = data_root / domain / dataset / task_id / "data"
    data_files = get_data_files(data_dir)

    turns_raw = load_json(task_json_path)
    if not isinstance(turns_raw, list):
        print(f"WARNING: {task_json_path} is not a list, skipping", file=sys.stderr)
        return None

    turns = []
    for t in turns_raw:
        answer = t.get("answer", "")
        if not isinstance(answer, str):
            answer = json.dumps(answer, ensure_ascii=False)
        turns.append({
            "turn_id": t.get("turn_id", 0),
            "context": t.get("context", ""),
            "question": t.get("question", ""),
            "ground_truth": answer,
            "answer_type": detect_answer_type(answer),
            "files_used": t.get("files_used", []),
        })

    return {
        "prompt": "",
        "label": task_key.replace("/", "__"),
        "metadata": {
            "task_id": task_key,
            "domain": domain,
            "dataset": dataset,
            "image": image,
            "workdir": workdir,
            "data_root": str(data_dir.resolve()),
            "data_files": data_files,
            "turns": turns,
            "num_turns": len(turns),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert LongDS tasks to slime JSONL")
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--image", default="longds-default:latest")
    parser.add_argument("--workdir", default="/workspace")
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args()

    task_list_path = args.task_root / "task_list.json"
    if not task_list_path.is_file():
        print(f"ERROR: {task_list_path} not found", file=sys.stderr)
        return 1

    task_list = load_json(task_list_path)
    rows, skipped = [], 0
    for info in task_list:
        row = convert_one_task(info, args.task_root, args.data_root,
                               args.image, args.workdir)
        if row is None:
            skipped += 1
        else:
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_turns = sum(r["metadata"]["num_turns"] for r in rows)
    atypes: dict[str, int] = {}
    for r in rows:
        for t in r["metadata"]["turns"]:
            at = t["answer_type"]
            atypes[at] = atypes.get(at, 0) + 1

    print(f"Converted {len(rows)} tasks ({total_turns} turns) → {args.output}")
    print(f"Skipped: {skipped}")
    print(f"Answer types: {atypes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
