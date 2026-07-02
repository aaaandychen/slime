#!/bin/bash
# Tau-bench fully async training with Qwen3-14B on 8xH200.
#
# Usage:
#   cd /path/to/slime
#   bash examples/tau-bench/run_qwen3_14B_fully_async.sh
#
# Prerequisites:
#   - Qwen3-14B HF checkpoint
#   - tau-bench installed (pip install -e . --no-deps)
#   - tau-bench prompt data generated (python tau1_mock.py --local_dir /root/tau-bench/)
#   - GEMINI_API_KEY set or configured in generate_with_tau.py

set -ex

export PYTHONUNBUFFERED=1

# --- cleanup -----------------------------------------------------------------
pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray python 2>/dev/null || true
sleep 3

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-14B.sh"

# --- paths -------------------------------------------------------------------
MODEL_DIR=${MODEL_DIR:-/root/models/Qwen3-14B}
DATA_DIR=${DATA_DIR:-/root/tau-bench}

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}"
    --ref-load "${MODEL_DIR}_torch_dist"
    --save "${MODEL_DIR}_slime_async"
    --save-interval 20
)

ROLLOUT_ARGS=(
    # ↓↓↓ Fully async core ↓↓↓
    --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
    --update-weights-interval 2
    --staleness-threshold 1

    # ↓↓↓ Tau-bench agent ↓↓↓
    --custom-generate-function-path generate_with_tau.generate
    --prompt-data "${DATA_DIR}/retail_train_tasks.jsonl"
    --input-key index
    --rollout-shuffle

    --num-rollout 500
    --rollout-batch-size 32
    --n-samples-per-prompt 8
    --rollout-max-response-len 1024
    --rollout-temperature 1.0

    --global-batch-size 128
    --balance-data
    --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)

# Evaluation is not yet supported in fully-async mode
# (generate_rollout_fully_async raises ValueError for evaluation=True).
# Run eval separately with the synchronous train.py path if needed.
# EVAL_ARGS=(
#     --eval-interval 5
#     --eval-prompt-data retail-dev "${DATA_DIR}/retail_dev_tasks.jsonl"
#     --n-samples-per-eval-prompt 1
#     --eval-max-response-len 1024
#     --eval-top-k 1
# )
EVAL_ARGS=()

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
    --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.7
    --sglang-server-concurrency 128
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --transformer-impl transformer_engine
    --log-passrate
)

# --- GPU split: 4 actor + 4 rollout (8xH200 total) ---------------------------
NUM_GPUS=8
ACTOR_GPUS=4
ROLLOUT_GPUS=4

# --- launch ------------------------------------------------------------------
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:${SCRIPT_DIR}/../..\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"DEEPSEEK_API_KEY\": \"${DEEPSEEK_API_KEY:-}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train_async.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${ACTOR_GPUS}" \
    --rollout-num-gpus "${ROLLOUT_GPUS}" \
    ${MODEL_ARGS[@]} \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"
