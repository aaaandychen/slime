#!/bin/bash
# Based on official scripts/run-qwen3-4B.sh + Decoupled PPO (--use-tis).
#
# Usage:
#   cd /mnt/cephfs/chenzhenyang/slime
#   bash examples/fully_async/run-qwen3-4B-fully_async.sh

set -ex

export PYTHONUNBUFFERED=1
source /mnt/cephfs/chenzhenyang/slime/.venv/bin/activate

# --- cleanup -----------------------------------------------------------------
pkill -9 sglang 2>/dev/null || true
sleep 3
uv run ray stop --force 2>/dev/null || true
pkill -9 ray python 2>/dev/null || true
sleep 3

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B.sh"

# --- paths -------------------------------------------------------------------
MODEL_DIR=${MODEL_DIR:-/mnt/cephfs/chenzhenyang/models/Qwen3-4B}
DATA_PATH=${DATA_PATH:-/mnt/cephfs/chenzhenyang/datasets/dapo-math-17k/dapo-math-17k.jsonl}
SAVE_DIR=${SAVE_DIR:-/mnt/cephfs/chenzhenyang/models/Qwen3-4B_slime_async}

# --- arguments ---------------------------------------------------------------

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}"
    --ref-load "${MODEL_DIR}_torch_dist"
    --save "${SAVE_DIR}"
    --save-interval 20
)

ROLLOUT_ARGS=(
    --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
    --update-weights-interval 2

    --prompt-data "${DATA_PATH}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rm-type deepscaler

    --num-rollout 250
    --rollout-batch-size 32
    --n-samples-per-prompt 8
    --rollout-max-response-len 8192
    --rollout-temperature 1.0

    --global-batch-size 128
    --balance-data
)

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
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28

    --use-tis
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project slime-areal-repro
    --wandb-group qwen3-4B-async
    --wandb-key "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"
    --disable-wandb-random-suffix
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
    --no-gradient-accumulation-fusion
    --no-rope-fusion
    --no-masked-softmax-fusion

    --log-passrate
)

CUSTOM_ARGS=(
    --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
    --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

# --- GPU split: 2 actor + 6 rollout (matching AReaL 1:3 ratio) --------------
NUM_GPUS=8
ACTOR_GPUS=2
ROLLOUT_GPUS=6

# --- launch ------------------------------------------------------------------
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
uv run ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/mnt/cephfs/chenzhenyang/Megatron-LM/:${SCRIPT_DIR}/../..\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_DISABLE_MEMORY_SAVER\": \"1\"
  }
}"

uv run ray job submit --address="http://127.0.0.1:8265" \
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
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}"