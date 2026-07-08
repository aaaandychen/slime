"""Check script: call DAA custom_generate directly and inspect the output sample.

Usage:
  cd /path/to/slime
  DAA_BASE_URL=http://localhost:5001 python examples/data_analysis_agent/check_output.py "各区域销售额排名"
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, ".")

from unittest.mock import MagicMock

from examples.data_analysis_agent.custom_generate import generate
from slime.utils.types import Sample


async def main(query: str):
    args = MagicMock()
    args.partial_rollout = False
    args.custom_generate_function_path = None
    args.group_rm = False

    sample = Sample()
    sample.prompt = query
    sample.index = 0
    sample.status = Sample.Status.PENDING

    result = await generate(args, sample, sampling_params={})

    # ── Print results ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Status   : {result.status}")
    print(f"Reward   : {result.reward}")
    print(f"Thread ID: {result.metadata.get('daa_thread_id', 'N/A')}")
    print(f"Text len : {len(result.metadata.get('daa_text', ''))}")
    print(f"Tool events: {len(result.metadata.get('daa_tool_events', []))}")
    for e in result.metadata.get("daa_tool_events", []):
        status = "OK" if e.get("ok") else "FAIL"
        print(f"  [{status}] {e['tool']}: {e.get('summary', '')[:100]}")
    print(f"\n{'='*60}")
    print("Text (first 800 chars):")
    print((result.metadata.get("daa_text", ""))[:800])
    print(f"{'='*60}")

    if result.status == Sample.Status.FAILED:
        print(f"\nERROR: generation failed")
        return 1
    return 0


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "各区域销售额排名"
    exit_code = asyncio.run(main(query))
    sys.exit(exit_code)
