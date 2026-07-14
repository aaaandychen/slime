# LongDS RL — Black-box RL training for data analysis agents

This example trains a coding agent on [LongDS](https://huggingface.co/datasets/zjunlp/LongDS)
benchmark tasks using slime's black-box RL pipeline (AnthropicAdapter +
ClaudeCodeHarness + TrajectoryManager).

## Architecture

```
LongDS tasks (68 tasks, 2225 turns)
  │
  │  convert_longds_to_slime.py   ← Phase 1
  ▼
slime JSONL (one row per task)
  │
  │  generate.py (per-sample rollout)   ← Phase 3+
  ▼
Claude Code (in E2B sandbox) → AnthropicAdapter → SGLang → token capture
  │
  │  longds_task.evaluate_answer()   ← Phase 2
  ▼
reward → TrajectoryManager.get_trajectory() → Sample → PPO/GRPO training
```

## Files

| File | Phase | Purpose |
|------|-------|---------|
| `convert_longds_to_slime.py` | 1 | Convert LongDS task.json → slime JSONL |
| `longds_task.py` | 2 | Prompt building, answer evaluation, metadata validation |
| `longds_harness.py` | 3 | ClaudeCodeHarness with --resume for multi-turn |
| `generate.py` | 3 | Per-sample rollout orchestrator |
| `run_longds_8nodes.sh` | 5 | Ray training launch script |

Sandbox images are **not** built here — reuse an existing Python data-analysis
image (DSGym's, or any image with pandas/scipy). ClaudeCodeHarness installs
Node.js + Claude Code CLI at boot time from host tarballs.

## Phase 1: Data conversion

```bash
pip install zjunlp/LongDS   # or: huggingface-cli download zjunlp/LongDS --repo-type dataset

python3 convert_longds_to_slime.py \
    --task-root /path/to/dataset/task/longds \
    --data-root /path/to/dataset/data/longds \
    --output longds_train.jsonl

# --image is optional — only needed when you're ready to train and know
# which sandbox image to use
```

Verify:
```bash
python3 tests/test_phase1_convert.py
```

## Phase 2: Task utilities

Verify:
```bash
python3 tests/test_phase2_longds_task.py
```

## Phase 3: Dry run (FakeSandbox, no Docker)

Verifies the full orchestration: sandbox boot → per-turn `claude -p`
→ answer.json parsing → reward computation.

```bash
python3 tests/test_phase3_dry_run.py
```

## Answer evaluation

`longds_task.evaluate_answer()` mirrors LongDS's `JUDGE_PROMPT` rules
programmatically (no LLM call needed at training time):

| Type | Rule |
|------|------|
| numeric | Exact after trailing-zero normalization. No rounding. |
| categorical | Strict equality after case/punctuation normalization. |
| ordered list | Match items AND order (ties allowed). |
| JSON | Recursive field-by-field; only ground-truth keys matter. |
| Score | Binary: 1.0 if ALL required fields correct, else 0.0. |

## Reused slime components

- `slime.agent.harness.ClaudeCodeHarness` — install and run Claude Code CLI
- `slime.agent.adapters.AnthropicAdapter` — intercept API calls, capture token logprobs
- `slime.agent.trajectory.TrajectoryManager` — message tree, loss_mask, token folding
- `slime.agent.sandbox.E2BSandbox` — sandbox lifecycle
- `slime.utils.types.Sample` — training sample format
