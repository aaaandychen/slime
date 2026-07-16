"""Convert DataMind RL parquet data to slime JSONL format.

Usage:
    python convert_data.py \
        --input /mnt/cephfs/chenzhenyang/datasets/DataMind-Data/rl/train.parquet \
        --output /mnt/cephfs/chenzhenyang/datasets/DataMind-Data/rl/train.jsonl \
        --data-root /mnt/cephfs/chenzhenyang/datasets/DataMind-Data/rl/train_files
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def build_slime_record(row, args):
    """Convert one DataMind RL row to slime JSONL record."""
    prompt_msgs = row["prompt"]
    extra = row.get("extra_info", {})
    reward_model = row.get("reward_model", {})

    import re
    # Extract user question + data file path. Claude Code handles its own system prompt.
    user_content = ""
    data_path = ""
    for msg in prompt_msgs:
        if msg["role"] == "system":
            # Extract the data file path from the verl system prompt
            m = re.search(r"\*\*The data source path is '([^']+)'\.\*\*", msg["content"])
            if m:
                data_path = m.group(1)
        elif msg["role"] == "user":
            user_content = msg["content"]
            break

    if data_path:
        full_prompt = f"The data file is at {data_path}. {user_content}"
    else:
        full_prompt = user_content

    # Ground truth from reward_model
    gt = reward_model.get("ground_truth", {})
    if isinstance(gt, dict):
        ground_truth = gt.get("ground_truth", "")
    else:
        ground_truth = str(gt)

    # Determine which data files are needed
    db_id = extra.get("db_id", row.get("db_id", ""))
    task_id = row.get("task_id", "")
    data_source = row.get("data_source", "")

    return {
        "prompt": full_prompt,
        "label": task_id,
        "metadata": {
            "task_id": task_id,
            "db_id": db_id,
            "data_source": data_source,
            "ground_truth": ground_truth,
            "style": reward_model.get("style", "auto"),
            "data_root": args.data_root,
            "question": extra.get("question", user_content),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Convert DataMind RL parquet to slime JSONL")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--data-root", required=True, help="Root directory for task data files")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    if args.limit:
        df = df.head(args.limit)

    print(f"Converting {len(df)} records from {args.input}", file=sys.stderr)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        for _, row in df.iterrows():
            record = build_slime_record(row, args)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
