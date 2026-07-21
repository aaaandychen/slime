"""Extract teacher trajectories from DataMind RL parquet → sharegpt SFT JSON.

Converts DataMind's <code>/<interpreter>/<answer> format to Claude Code's
Bash tool call format, so the SFT model learns data analysis + tool calling
in one step.

Usage:
    python examples/datamind_rl/prepare_sft_data.py \
        --input /mnt/cephfs/chenzhenyang/datasets/DataMind-Data/rl/train.parquet \
        --output /mnt/cephfs/chenzhenyang/datasets/DataMind-Data/rl/sft_train.json \
        --limit 5000
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# Brief system prompt that mimics what Claude Code expects.
# The full tool definitions are added by Claude Code at runtime.
SYSTEM_PROMPT = (
    "You are an expert data analyst. Solve data analysis tasks step by step "
    "using the available tools (Bash to run code, Read to inspect files, "
    "Write to save results). Work through the problem methodically, read the "
    "data files, write and execute code, inspect results, and provide a final "
    "answer when done."
)


def _extract_think(text: str) -> str:
    """Extract <think>...</think> content."""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_code(text: str) -> str | None:
    """Extract Python/SQL code from <code>```python/code```</code> blocks."""
    m = re.search(r"<code>\s*```(?:python|sql)?\s*\n(.*?)```\s*</code>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try without language marker
    m = re.search(r"<code>\s*```\s*\n(.*?)```\s*</code>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_answer(text: str) -> str | None:
    """Extract <answer>...</answer> content."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _escape_for_bash(code: str) -> str:
    """Escape Python code for single-line bash -c execution."""
    # Use triple-quoted heredoc approach for multi-line code
    return code


# Function definitions injected into SFT data so the model never sees bare calls.
# Must stay in sync with the adapter-side definitions in anthropic.py.
_GET_DB_INFO_DEF = '''
def get_db_info():
    """Display database schema."""
    import sqlite3, os
    for f in sorted(os.listdir('data/files')):
        if f.endswith('.sqlite'):
            conn = sqlite3.connect(os.path.join('data/files', f))
            for row in conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"):
                print(row[0])
            conn.close()
'''.strip()

_EXECUTE_SQL_DEF = '''
def execute_sql(sql, output_path):
    """Execute SQL and save results to CSV."""
    import sqlite3, os, csv
    for f in sorted(os.listdir('data/files')):
        if f.endswith('.sqlite'):
            conn = sqlite3.connect(os.path.join('data/files', f))
            cur = conn.execute(sql)
            rows = cur.fetchall()
            if rows:
                cols = [d[0] for d in cur.description]
                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                with open(output_path, 'w', newline='') as fp:
                    w = csv.writer(fp)
                    w.writerow(cols)
                    w.writerows(rows)
                print(f'Output saved to: {output_path}')
                for row in rows:
                    print(','.join(str(x) for x in row))
            else:
                print(f'Output saved to: {output_path}')
                print('(empty result)')
            conn.close()
            break
'''.strip()


def _inject_function_defs(code: str) -> str:
    """Prepend function definitions if the code calls them, so SFT data has no bare calls."""
    parts = []
    if 'get_db_info()' in code:
        parts.append(_GET_DB_INFO_DEF)
    if 'execute_sql(' in code:
        parts.append(_EXECUTE_SQL_DEF)
    if not parts:
        return code
    return '\n\n'.join(parts) + '\n\n' + code


def _code_to_claude_format(code: str) -> str:
    """Wrap code as a Bash heredoc command for Claude Code execution.

    Always uses heredoc and injects function definitions for get_db_info /
    execute_sql so the SFT model never sees bare function calls.
    """
    code = _inject_function_defs(code.strip())
    return f"```bash\npython3 << 'PYEOF'\n{code}\nPYEOF\n```"


def build_sharegpt_conversation(row) -> dict | None:
    """Convert one DataMind RL row to sharegpt messages in Claude Code format."""
    traj = row.get("trajectory", [])

    # Get the original user question from the prompt field
    prompt_msgs = row.get("prompt", [])
    user_question = ""
    data_file = ""
    for msg in prompt_msgs:
        if msg["role"] == "system":
            # Extract data file path
            m = re.search(r"\*\*The data source path is '([^']+)'\.\*\*", msg["content"])
            if m:
                data_file = m.group(1)
        elif msg["role"] == "user" and not user_question:
            user_question = msg["content"]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add data file context to user message
    if data_file:
        user_msg = f"The data file is at {data_file}. {user_question}"
    else:
        user_msg = user_question
    messages.append({"role": "user", "content": user_msg})

    # Process trajectory turns: extract think + code pairs, build tool call turns
    for msg in traj:
        role = msg.get("role", "")
        content = str(msg.get("content", ""))

        if role == "system":
            continue  # skip, we already have our own system prompt

        if role == "assistant":
            think = _extract_think(content)
            code = _extract_code(content)
            answer = _extract_answer(content)

            if answer:
                # Final answer: echo to answer.json (matching RL generate.py expectation)
                escaped = answer.replace("\\", "\\\\").replace("'", "'\\''")
                answer_cmd = f"echo '{{\"answer\":\"{escaped}\",\"reasoning\":\"done\"}}' > answer.json"
                messages.append({"role": "assistant", "content": f"```bash\n{answer_cmd}\n```"})
            elif code:
                # Tool call turn: think + Bash command
                parts = []
                if think:
                    parts.append(f"<think>\n{think}\n</think>")
                parts.append(_code_to_claude_format(code))
                messages.append({"role": "assistant", "content": "\n\n".join(parts)})
            elif think:
                # Think only, no code
                messages.append({"role": "assistant", "content": f"<think>\n{think}\n</think>"})

        elif role == "user":
            # Interpreter output → tool result
            interpreter = re.search(r"<interpreter>(.*?)</interpreter>", content, re.DOTALL)
            if interpreter:
                result = interpreter.group(1).strip()
                if len(result) > 4096:
                    result = result[:4096] + "\n... (truncated)"
                messages.append({"role": "user", "content": f"**Output:**\n\n```\n{result}\n```"})

    if len(messages) < 3:
        return None
    return {"messages": messages}


def build_sharegpt_conversation_old(row) -> list[dict]:  # kept for reference, unused
    """Original converter without format transformation."""
    traj = row.get("trajectory", [])
    messages = []
    for msg in traj:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user" and "<interpreter>" in str(content):
            content = str(content).replace("<interpreter>\n", "").replace("\n</interpreter>", "").strip()
            if len(content) > 2000:
                content = content[:2000] + "\n... (truncated)"
            role = "user"
        elif role == "system":
            if messages:
                continue
        messages.append({"role": role, "content": str(content)})
    return messages


def main():
    parser = argparse.ArgumentParser(
        description="Extract DataMind teacher trajectories for SFT warmup"
    )
    parser.add_argument("--input", required=True, help="Input parquet file (train.parquet)")
    parser.add_argument("--output", required=True, help="Output JSON file for LLaMA-Factory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)

    # Only use samples that have a trajectory (teacher demonstration)
    has_traj = df["trajectory"].notna()
    df = df[has_traj]
    print(f"Samples with trajectory: {len(df)} / {len(df) + has_traj.sum()}", file=sys.stderr)

    if args.limit:
        df = df.head(args.limit)

    conversations = []
    skipped = 0
    for _, row in df.iterrows():
        task_id = row.get("task_id", "unknown")
        if not isinstance(task_id, str):
            task_id = str(row.get("extra_info", {}).get("index", "unknown"))

        conv = build_sharegpt_conversation(row)
        if conv is None:
            skipped += 1
            continue

        conversations.append(conv)

    print(f"Valid conversations: {len(conversations)}, skipped: {skipped}", file=sys.stderr)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

    print(f"Written {len(conversations)} conversations to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
