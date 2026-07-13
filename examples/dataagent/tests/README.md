# Tests & Validation Scripts

Testing and debugging utilities for the DataAgent training pipeline. Not
required for training — use these to verify correctness before/after changes.

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `validate_reward.py` | Verify the reward function discriminates good vs bad answers (5 sanity checks + number extraction self-tests) | `python examples/dataagent/tests/validate_reward.py -v` |
| `check_output.py` | Call `custom_generate.generate()` directly and inspect the returned `list[Sample]` (requires running DataAgent + SGLang) | `DATAAGENT_BASE_URL=http://localhost:8065 python examples/dataagent/tests/check_output.py "各区域销售额排名"` |
| `test_end_to_end.sh` | Standalone DataAgent test with DeepSeek (no slime/SGLang) — verifies DataAgent + MariaDB + Embedding work | `DEEPSEEK_API_KEY=sk-xxx bash examples/dataagent/tests/test_end_to_end.sh` |

## When to run

- **After changing `reward_func.py`** → run `validate_reward.py` to confirm
  the 5 sanity checks still pass (good > wrong, good > partial, etc.).
- **After changing `custom_generate.py` or `adapter.py`** → run
  `check_output.py` with a live DataAgent to verify the full SSE → adapter →
  SGLang chain produces samples with correct reward.
- **After restarting DataAgent** (H2 in-memory DB is wiped) → run
  `test_end_to_end.sh` to re-verify DataAgent itself works before training.

## Path notes

All scripts resolve paths relative to `__file__` (or `BASH_SOURCE`), so they
work regardless of the current working directory. Run from the slime root
for consistency:

```bash
cd /path/to/slime
python examples/dataagent/tests/validate_reward.py -v
```
