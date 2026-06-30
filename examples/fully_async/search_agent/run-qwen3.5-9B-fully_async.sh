#!/bin/bash
# Fully-async Search Agent RL training script with Qwen3.5-9B.
#
# Architecture:
#   train_async.py (training loop)
#     → StalenessDataSource (staleness-aware data source, pauses when threshold exceeded)
#     → AsyncRolloutWorker (background thread, continuous generation)
#       → fully_async_agent_loop.generate() (multi-turn search agent, per-sample)
#         → SGLang /generate (input_ids mode, non-streaming)
#         → _execute_search() (search backend call)
#     → generate_rollout_fully_async() (collects exactly rollout_batch_size groups)
#
# Prerequisites:
#   [TODO] /path/to/Qwen3.5-9B/                    (HF checkpoint)
#   [TODO] /path/to/Qwen3.5-9B_torch_dist/          (from tools/convert_hf_to_torch_dist.py)
#   [TODO] /path/to/search_train_data.jsonl         (search training data with "prompt" and "label" keys)
#   [TODO] Search retrieval server running at SEARCH_URL (default: http://127.0.0.1:8000/retrieve)
#   [TODO] Megatron-LM at /path/to/Megatron-LM/
#
# Usage:
#   bash examples/fully_async/search_agent/run-qwen3.5-9B-fully_async.sh
#
# Key differences from base fully-async script:
#   --rollout-function-path   → custom rollout (exact batch size, no discard)
#   --custom-generate-function-path → multi-turn search agent loop
#   --data-source-path        → StalenessDataSource (pause/resume on staleness)
#   --staleness-threshold     → max stale batches before pause (default 1)

# ============================================================
# [TODO] Fill these paths before running
# ============================================================
MODEL_DIR="${MODEL_DIR:-/path/to/Qwen3.5-9B}"                       # TODO: HF model checkpoint
DATA_PATH="${DATA_PATH:-/path/to/search_train_data.jsonl}"          # TODO: search training data
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"          # TODO: search retrieval server URL
MEGATRON_LM_DIR="${MEGATRON_LM_DIR:-/path/to/Megatron-LM}"          # TODO: Megatron-LM root
SAVE_DIR="${SAVE_DIR:-/path/to/slime_checkpoints/}"                 # TODO: checkpoint save directory

# ============================================================
# Cleanup
# ============================================================
pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray python 2>/dev/null || true
sleep 3

set -ex

export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../../scripts/models/qwen3.5-9B.sh"

# ============================================================
# Checkpoint args
# ============================================================
CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}"
    --ref-load "${MODEL_DIR}_torch_dist"
    --save "${SAVE_DIR}"
    --save-interval 20
)

# ============================================================
# Rollout args — search agent specific
# ============================================================
ROLLOUT_ARGS=(
    # ↓↓↓ Fully-async mode ↓↓↓
    --rollout-function-path examples.fully_async.search_agent.fully_async_agent_loop.generate_rollout_fully_async

    # ↓↓↓ Data source with staleness control ↓↓↓
    --data-source-path examples.fully_async.search_agent.stale_data_source.StalenessDataSource
    --staleness-threshold 1

    # ↓↓↓ Training data ↓↓↓
    --prompt-data "${DATA_PATH}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle

    # ↓↓↓ Reward model — [TODO] choose or implement your search reward function ↓↓↓
    # Option A: use search-r1 reward (F1/EM-based, requires --custom-rm-path)
    #   --rm-type custom
    #   --custom-rm-path examples.search-r1.generate_with_search.reward_func
    # Option B: use deepscaler RM for math (not recommended for search)
    #   --rm-type deepscaler
    --rm-type custom
    --custom-rm-path examples.search-r1.generate_with_search.reward_func

    --num-rollout 3000
    --rollout-batch-size 32
    --n-samples-per-prompt 8
    --rollout-max-response-len 2048
    --rollout-temperature 1

    --global-batch-size 256
    --balance-data
)

# ============================================================
# Performance args (tuned for 8×GPU, adjust for your setup)
# ============================================================
PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --use-dynamic-batch-size
    --max-tokens-per-gpu 4096
)

# ============================================================
# GRPO args
# ============================================================
GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
)

# ============================================================
# Optimizer args
# ============================================================
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

# ============================================================
# SGLang args
# ============================================================
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.7
)

# ============================================================
# Custom generate / search agent args
# ============================================================
CUSTOM_ARGS=(
    # ↓↓↓ Multi-turn search agent loop (per-sample) ↓↓↓
    --custom-generate-function-path examples.fully_async.search_agent.fully_async_agent_loop.generate

    # ↓↓↓ Search config — [TODO] adjust in SEARCH_CONFIG inside fully_async_agent_loop.py ↓↓↓
    # Current defaults:
    #   max_turns=3, topk=3, search_concurrency=256
    #   search_backend="local" → SEARCH_CONFIG["local"]["search_url"]
    #     default: http://127.0.0.1:8000/retrieve
    # For Google Search, set search_backend="google" and add your API key.
)

# ============================================================
# Misc args
# ============================================================
MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

# ============================================================
# Launch Ray + training
# ============================================================

# [TODO] Adjust GPU counts for your cluster
NUM_GPUS=${NUM_GPUS:-8}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_DIR}:${SCRIPT_DIR}/../../..\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

# fully-async splits actor / rollout onto disjoint GPUs (no colocation).
# [TODO] Adjust based on your GPU layout.
#   Recommended for 9B on 8×GPU: actor=4, rollout=4
#   For smaller models: actor=2, rollout=6
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-$((NUM_GPUS - ACTOR_GPUS))}

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train_async.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${ACTOR_GPUS}" \
    --rollout-num-gpus "${ROLLOUT_GPUS}" \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${CUSTOM_ARGS[@]} \
    ${MISC_ARGS[@]}
