"""Check script: call custom_generate directly and inspect the output samples.

Usage:
  cd /path/to/slime
  DATAAGENT_BASE_URL=http://localhost:8065 python examples/dataagent/tests/check_output.py "各区域销售额排名"
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from unittest.mock import MagicMock

from examples.dataagent.custom_generate import generate
from slime.utils.types import Sample


async def main(query: str):
    # Build minimal fake args — the adapter needs tokenizer + sglang url.
    args = MagicMock()
    args.hf_checkpoint = os.environ.get("HF_CHECKPOINT", "")
    args.sglang_router_ip = os.environ.get("SGLANG_HOST", "127.0.0.1")
    args.sglang_router_port = int(os.environ.get("SGLANG_PORT", "30000"))
    args.sglang_tool_call_parser = None
    args.sglang_reasoning_parser = os.environ.get("SGLANG_REASONING_PARSER", None)
    args.rollout_max_context_len = 32768
    args.partial_rollout = False
    args.custom_generate_function_path = None
    args.group_rm = False

    # Build the sample.
    sample = Sample()
    sample.prompt = query
    sample.index = 0
    sample.status = Sample.Status.PENDING

    # Call the generate function (returns list[Sample]).
    samples = await generate(args, sample, sampling_params={})

    if not samples:
        print("No samples returned.")
        return 1

    # ── Print results ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Samples returned: {len(samples)}")
    for i, result in enumerate(samples):
        print(f"\n--- Sample {i} ---")
        print(f"Status   : {result.status}")
        print(f"Reward   : {result.reward}")
        print(f"Response len: {len(result.response or '')}")
        print(f"Thread ID: {result.metadata.get('dataagent_thread_id', 'N/A')}")
        print(f"Nodes ({len(result.metadata.get('dataagent_nodes', []))}):")
        for n in result.metadata.get("dataagent_nodes", []):
            text_preview = n["text"][:200].replace("\n", "\\n")
            print(f"  [{n['type']:12s}] {n['node']:35s} | {text_preview}...")
        print(f"\nResponse (first 500 chars):")
        print((result.response or "")[:500])
        if result.loss_mask:
            zeros = sum(1 for m in result.loss_mask if m == 0)
            ones = sum(1 for m in result.loss_mask if m == 1)
            print(f"\nloss_mask: prompt={zeros} output={ones} total={len(result.loss_mask)}")
    print(f"\n{'='*60}")

    failed = any(s.status == Sample.Status.FAILED for s in samples)
    if failed:
        print("\nERROR: one or more samples failed.")
        return 1
    return 0


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "各区域销售额排名"
    exit_code = asyncio.run(main(query))
    sys.exit(exit_code)
