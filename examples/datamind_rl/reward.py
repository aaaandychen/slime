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
            return [0.0]
        sample = samples[-1]
    else:
        sample = samples
    md = sample.metadata or {}
    reasoning = str(md.get("reasoning", ""))
    ground_truth = str(md.get("ground_truth", ""))
    data_source = str(md.get("data_source", ""))

    # Extract answer: answer.json takes priority over response text extraction
    full_response = getattr(sample, "response", "") or str(md.get("answer", ""))
    extracted_answer = _extract_answer_from_response(sample)
    answer = str(md.get("answer", "")) or extracted_answer
    logger.warning("DEBUG_REWARD: sid=%s n_samples=%d resp_len=%d answer_json_len=%d answer_extract_len=%d resp_tail=%s full_resp=%s",
        getattr(sample, "session_id", "?") or "?",
        len(samples) if isinstance(samples, list) else 1,
        len(full_response),
        len(str(md.get("answer", ""))),
        len(extracted_answer),
        full_response[-300:],
        full_response if len(full_response) <= 2000 else full_response[:2000] + "…")

    template_score = _evaluate_template(full_response)
    answer_score = _evaluate_answer(answer, ground_truth, data_source)
    tool_score = _evaluate_tool_usage(full_response)

    # DataMind reward logic
    # answer.json bonus: writing answer.json (even if wrong) beats text-only responses
    has_answer_json = bool(str(md.get("answer", "")).strip())
    answer_json_bonus = 0.15 if has_answer_json else 0.0

    if answer_score > 0.0:
        final_score = answer_score + 0.1 * template_score + 0.05 * tool_score
    elif template_score > 0.0:
        final_score = 0.0 + 0.1 * template_score + 0.05 * tool_score + answer_json_bonus
    else:
        final_score = -0.1 + answer_json_bonus

    final_score = max(-1.0, min(1.0, final_score))

    # Length penalty: discourage long exploration without producing an answer
    length_penalty = 0.0
    if answer_score <= 0:
        resp_len = len(full_response)
        threshold = 4000
        if resp_len > threshold:
            length_penalty = 0.3 * min(1.0, (resp_len - threshold) / 8192)
            final_score -= length_penalty
            final_score = max(-1.0, final_score)

    logger.info(
        "datamind_reward: instance=%s answer=%.2f template=%.2f tool=%.2f aj_bonus=%.2f len_pen=%.3f final=%.2f gt_snippet=%s",
        md.get("instance_id", "?"),
        answer_score,
        template_score,
        tool_score,
        answer_json_bonus,
        length_penalty,
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

    # slime expects a list of floats, one per sample
    if isinstance(samples, list):
        return [final_score] * len(samples)
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

    Uses multi-level comparison:
      1. Exact match after normalization
      2. Key number overlap (how many ground-truth numbers appear in the answer)
      3. Substring containment for descriptive answers
      4. Fuzzy token overlap for semantic similarity
    """
    if not answer.strip():
        return -1.0
    if not ground_truth.strip():
        return 0.0

    a = _normalize(answer)
    g = _normalize(ground_truth)

    # Exact match
    if a == g:
        return 1.0

    # Extract numbers from both
    a_nums = set(_extract_numbers(answer))
    g_nums = set(_extract_numbers(ground_truth))

    # Numeric overlap: what fraction of ground-truth numbers are in the answer?
    if g_nums:
        if a_nums == g_nums:
            return 1.0
        overlap = len(g_nums & a_nums)
        ratio = overlap / len(g_nums)
        if ratio >= 0.8:
            return 0.8
        elif ratio >= 0.5:
            return 0.5
        elif ratio >= 0.2:
            return 0.3

    # Semantic containment: check if key ground-truth sentences appear in answer
    g_sentences = [s.strip() for s in re.split(r'[.;]\s*', ground_truth) if len(s.strip()) > 10]
    if g_sentences:
        matches = sum(1 for gs in g_sentences if _normalize(gs)[:30] in a)
        sentence_ratio = matches / len(g_sentences)
        if sentence_ratio >= 0.6:
            return 0.6
        elif sentence_ratio >= 0.3:
            return 0.3

    # Token overlap for semantic similarity (more robust than char-level)
    a_tokens = set(re.findall(r'\w+', a.lower()))
    g_tokens = set(re.findall(r'\w+', g.lower()))
    if g_tokens and len(g_tokens) > 5:
        token_overlap = len(g_tokens & a_tokens) / len(g_tokens)
        if token_overlap > 0.6:
            return 0.3
        if token_overlap > 0.4:
            return 0.1

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


def _extract_answer_from_response(sample: Any) -> str:
    """Extract just the final answer from the model's response text.

    Tries multiple strategies, ordered by priority:
      1. <answer> tags (SFT format from teacher trajectories)
      2. answer.json content from metadata
      3. Last meaningful paragraph from the response
    """
    response = getattr(sample, "response", "")
    md = sample.metadata or {}

    # 1. Look for <answer> tags in the response (SFT model uses this format)
    if response:
        m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if m:
            return m.group(1).strip()

    # 2. Look for JSON answer in the response (Claude Code format)
    if response:
        m = re.search(r'"answer"\s*:\s*"([^"]*)"', response)
        if m:
            return m.group(1).strip()

    # 3. Fall back to metadata answer
    answer = str(md.get("answer", ""))
    if answer.strip():
        return answer.strip()

    # 4. Last resort: last non-empty paragraph before any tool call
    if response:
        # Strip tool calls and take the last substantial text block
        clean = re.sub(r"<tool_call>.*?</tool_call>", "", response, flags=re.DOTALL)
        clean = re.sub(r"```[^`]*```", "", clean)
        paragraphs = [p.strip() for p in clean.split("\n\n") if len(p.strip()) > 20]
        if paragraphs:
            return paragraphs[-1]

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
