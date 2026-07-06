"""Check script: call custom_generate directly and inspect the output sample.

Usage:
  cd /path/to/slime
  DATAAGENT_BASE_URL=http://localhost:8065 python examples/dataagent/check_output.py "各区域销售额排名"
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, ".")

from unittest.mock import MagicMock

from examples.dataagent.custom_generate import generate
from slime.utils.types import Sample


async def main(query: str):
    # Build minimal fake args — the custom_generate function doesn't use
    # most of these for Phase 1, but generate_and_rm expects them to exist.
    args = MagicMock()
    args.partial_rollout = False
    args.custom_generate_function_path = None
    args.group_rm = False

    # Build the sample.
    sample = Sample()
    sample.prompt = query
    sample.index = 0
    sample.status = Sample.Status.PENDING

    # Call the generate function.
    result = await generate(args, sample, sampling_params={})

    # ── Print results ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Status   : {result.status}")
    print(f"Reward   : {result.reward}")
    print(f"Response len: {len(result.response or '')}")
    print(f"Thread ID: {result.metadata.get('dataagent_thread_id', 'N/A')}")
    print(f"\nNodes ({len(result.metadata.get('dataagent_nodes', []))}):")
    for n in result.metadata.get("dataagent_nodes", []):
        text_preview = n["text"][:200].replace("\n", "\\n")
        print(f"  [{n['type']:12s}] {n['node']:35s} | {text_preview}...")
    print(f"\n{'='*60}")
    print("Response (first 500 chars):")
    print((result.response or "")[:500])
    print(f"{'='*60}")

    if result.status == Sample.Status.FAILED:
        print(f"\nERROR: {result.metadata.get('dataagent_error', 'unknown')}")
        return 1
    return 0


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "各区域销售额排名"
    exit_code = asyncio.run(main(query))
    sys.exit(exit_code)
