"""DataMind reward function for slime.

    --custom-rm-path examples.datamind_rl.reward.datamind_reward

Mirrors DataMind's reward design:
  - template_score: +1 if output follows the expected structure
    (<think> blocks, code, <answer> tags)
  - answer_score: +1 if answer matches ground_truth, -1 otherwise
  - final = answer_score > 0 ? answer_score
            : template_score == 1 ? 0.0
            : -0.1

Slime signature: ``def datamind_reward(args, sample) -> float``
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── Public entry point ────────────────────────────────────────────────


async def datamind_reward(args: Any, samples: list) -> float:
    """Compute reward for one sample's trajectory.

    Parses the model's full response and scores:
      1. Template compliance (think tags, structure)
      2. Answer correctness (vs ground_truth)
    """
    # slime passes a list of samples; take the last one with metadata
    if isinstance(samples, list):
        if not samples:
            return 0.0
        sample = samples[-1]
    else:
        sample = samples
    md = sample.metadata or {}
    answer = str(md.get("answer", ""))
    reasoning = str(md.get("reasoning", ""))
    ground_truth = str(md.get("ground_truth", ""))
    data_source = str(md.get("data_source", ""))

    # Build a synthetic "full response" from the trajectory tokens when available,
    # otherwise fall back to response string.
    full_response = _extract_full_response(sample)

    template_score = _evaluate_template(full_response)
    answer_score = _evaluate_answer(answer, ground_truth, data_source)
    tool_score = _evaluate_tool_usage(full_response)

    # DataMind reward logic
    if answer_score > 0.0:
        final_score = answer_score + 0.1 * template_score + 0.05 * tool_score
    elif template_score > 0.0:
        final_score = 0.0 + 0.1 * template_score + 0.05 * tool_score  # partial credit for trying
    else:
        final_score = -0.1

    final_score = max(-1.0, min(1.0, final_score))

    logger.info(
        "datamind_reward: instance=%s answer=%.2f template=%.2f tool=%.2f final=%.2f gt_snippet=%s",
        md.get("instance_id", "?"),
        answer_score,
        template_score,
        tool_score,
        final_score,
        ground_truth[:100],
    )

    # Store component scores for logging
    sample.metadata = {
        **md,
        "reward_answer_score": answer_score,
        "reward_template_score": template_score,
        "reward_tool_score": tool_score,
    }

    return final_score


# ── Template evaluation ───────────────────────────────────────────────


def _evaluate_template(full_response: str) -> float:
    """Score whether the model followed the expected response structure.

    Checks:
      - Contains <think> or <｜end▁of▁thinking｜> includes reasoning markers
      - Contains code blocks or tool call traces
      - Contains a final answer in some recognizable form
    """
    if not full_response.strip():
        return -1.0

    score = 0.0
    # Reasoning / think block
    if re.search(r"<think>|## Thought:", full_response, re.IGNORECASE):
        score += 0.3
    elif len(full_response) > 200:
        score += 0.1  # some thought, just not tagged

    # Code or tool usage
    if re.search(r"```(python|sql|bash)|<tool_call>|<function=", full_response, re.IGNORECASE):
        score += 0.3

    # Answer container
    if re.search(r"<answer>|answer\.json|Final Answer|final answer", full_response, re.IGNORECASE):
        score += 0.4
    elif len(full_response.strip()) > 100:
        score += 0.2

    # Penalize completely empty/shallow responses
    if len(full_response.strip()) < 50:
        score = -1.0

    return min(1.0, max(-1.0, score))


# ── Answer evaluation ─────────────────────────────────────────────────


def _evaluate_answer(answer: str, ground_truth: str, data_source: str = "") -> float:
    """Compare the model's answer against ground truth.

    Returns:
        1.0 for exact match, 0.5 for partial, -1.0 for wrong/missing.
    """
    if not answer.strip():
        return -1.0
    if not ground_truth.strip():
        logger.warning("datamind_reward: empty ground_truth, cannot evaluate answer")
        return 0.0

    a = _normalize(answer)
    g = _normalize(ground_truth)

    # Exact match
    if a == g:
        return 1.0

    # Containment: answer contains the ground truth or vice versa
    if len(a) > 5 and len(g) > 5:
        if g in a or a in g:
            return 0.8

    # Numeric comparison: try to extract numbers
    a_nums = _extract_numbers(answer)
    g_nums = _extract_numbers(ground_truth)
    if a_nums and g_nums:
        if a_nums == g_nums:
            return 1.0
        if len(a_nums) == len(g_nums) and all(abs(x - y) < 1e-3 for x, y in zip(a_nums, g_nums)):
            return 1.0
        if any(abs(x - y) < 1e-3 for x in a_nums for y in g_nums):
            return 0.5

    # Fuzzy string match for text answers
    ratio = difflib.SequenceMatcher(None, a, g).ratio()
    if ratio > 0.95:
        return 1.0
    if ratio > 0.7:
        return 0.5

    return -1.0


# ── Tool usage evaluation ─────────────────────────────────────────────


def _evaluate_tool_usage(full_response: str) -> int:
    """Count tool interaction rounds. Each detected tool call = +0.1,

    capped at 0.5.
    """
    tool_count = len(re.findall(r"<tool_call>", full_response))
    tool_count += len(re.findall(r"<function=", full_response))
    tool_count += len(re.findall(r"```(python|sql|bash)", full_response))
    return min(5, tool_count) * 0.1


# ── Helpers ───────────────────────────────────────────────────────────


def _extract_full_response(sample: Any) -> str:
    """Reconstruct the full text response from sample data."""
    response = getattr(sample, "response", "")
    if response:
        return response

    md = sample.metadata or {}
    answer = str(md.get("answer", ""))
    reasoning = str(md.get("reasoning", ""))
    if reasoning or answer:
        return f"{reasoning}\n\n{answer}"

    return ""


def _normalize(s: str) -> str:
    """Collapse whitespace and normalize for comparison."""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".,;:\"' ")
    return s


def _extract_numbers(text: str) -> list[float]:
    """Extract all floating-point numbers from text."""
    nums = re.findall(r"[-+]?\d+\.?\d*", text.replace(",", ""))
    try:
        return [float(n) for n in nums]
    except ValueError:
        return []
