#!/usr/bin/env bash
# End-to-end DataMind RL training on Qwen2.5-Coder-14B-Instruct.
# Run from tmux/screen on the Ray head node.
#
# Prerequisites:
#   1. Qwen2.5-Coder-14B-Instruct at HF_CHECKPOINT
#   2. DataMind RL data converted to JSONL via convert_data.py
#   3. DataMind train_files.zip extracted to DATAMIND_DATA_ROOT
#   4. Claude Code CLI installed on all nodes

set -ex

# Best-effort cleanup
pkill -9 sglang || true
sleep 3
ray stop --force || true
sleep 3

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# ============ model ============
HF_CHECKPOINT="${HF_CHECKPOINT:-/mnt/cephfs/chenzhenyang/models/Qwen2.5-Coder-14B-Instruct}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/mnt/cephfs/chenzhenyang/models/Qwen2.5-Coder-14B-Instruct_torch_dist}"

# ============ data ============
PROMPT_DATA="${PROMPT_DATA:-/mnt/cephfs/chenzhenyang/datasets/DataMind-Data/rl/train.jsonl}"
DATAMIND_DATA_ROOT="${DATAMIND_DATA_ROOT:-/mnt/cephfs/chenzhenyang/datasets/DataMind-Data/rl/train_files}"

# ============ model parallelism ============
TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-2}"

# ============ rollout ============
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-2}"
ROLLOUT_MEM_UTILIZATION="${ROLLOUT_MEM_UTILIZATION:-0.75}"

# ============ context ============
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-32768}"
MAX_GEN_LEN="${MAX_GEN_LEN:-8192}"

# ============ run tag ============
EXP_TAG="${EXP_TAG:-datamind_qwen25_coder_14b}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${SLIME_DIR}/runs/${EXP_TAG}_${STAMP}}"
mkdir -p "${RUN_ROOT}/rollout_dumps"
LOG_FILE="${RUN_ROOT}/run.log"
echo "Log: ${LOG_FILE}"

# Model args built from existing model config
source "${SCRIPT_DIR}/../../scripts/models/qwen25-coder-14B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_MODEL_PATH}"
)

# ============ rollout ============
ROLLOUT_ARGS=(
   --custom-generate-function-path examples.datamind_rl.generate.generate
   --custom-rm-path examples.datamind_rl.reward.datamind_reward
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --num-rollout 500
   --rollout-batch-size 32
   --n-samples-per-prompt 4
   --rollout-max-context-len ${MAX_CONTEXT_LEN}
   --rollout-max-response-len ${MAX_GEN_LEN}
   --rollout-temperature 0.7
   --rollout-top-p 1.0
   --num-steps-per-rollout 1
   --global-batch-size 128
   --micro-batch-size 1
   --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --sequence-parallel
   --pipeline-model-parallel-size ${PP_SIZE}
   --context-parallel-size ${CP_SIZE}
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --max-tokens-per-gpu $((MAX_CONTEXT_LEN / CP_SIZE))
   --log-probs-chunk-size 1024
   --use-dynamic-batch-size
)

ALGO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --eps-clip 0.2
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus 4
   --rollout-num-gpus-per-engine ${ROLLOUT_TP_SIZE}
   --sglang-mem-fraction-static ${ROLLOUT_MEM_UTILIZATION}
   --sglang-tool-call-parser qwen3_coder
   --sglang-reasoning-parser qwen3
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-wandb
   --wandb-project datamind-rl
   --wandb-group qwen25-coder-14b
   --wandb-key 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
)

# ============ network ============
export MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"
export MASTER_PORT="${MASTER_PORT:-6379}"
_DEFAULT_IFNAME="$(ip -o link show 2>/dev/null | awk -F': ' '/eth/ && !/lo/{print $2; exit}')"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${_DEFAULT_IFNAME}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${_DEFAULT_IFNAME}}"

# ============ DataMind rollout env ============
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-${MASTER_ADDR}}"
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export ADAPTER_PORT="${ADAPTER_PORT:-18001}"
export DATAMIND_TIMEOUT_SEC="${DATAMIND_TIMEOUT_SEC:-1200}"

export SLIME_AGENT_CC_EXTRA_ARGS="--settings '{\"permissions\":{\"defaultMode\":\"bypassPermissions\"},\"autoCompactEnabled\":true}' --disallowedTools WebFetch WebSearch"

# ============ proxy ============
export no_proxy="127.0.0.1,${MASTER_ADDR},${ADAPTER_PUBLIC_HOST}"

cd "${SLIME_DIR}"

# ============ Ray cluster ============
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
TRAIN_GPUS_PER_NODE="${TRAIN_GPUS_PER_NODE:-4}"

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

sleep 10
ray status

# ============ runtime env → Ray workers ============
export SLIME_DIR
RUNTIME_ENV_JSON=$(python3 - <<PY
import json, os
keys = (
    "no_proxy", "NO_PROXY",
    "ADAPTER_PUBLIC_HOST", "ADAPTER_BIND_HOST", "ADAPTER_PORT",
    "DATAMIND_TIMEOUT_SEC",
    "SLIME_AGENT_CC_EXTRA_ARGS",
)
env = {k: os.environ[k] for k in keys if k in os.environ}
env["MASTER_ADDR"] = os.environ["MASTER_ADDR"]
env["MASTER_PORT"] = os.environ.get("MASTER_PORT", "")
env["GLOO_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["TP_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["NCCL_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"]
env["PYTHONPATH"] = f"{os.environ['SLIME_DIR']}:{os.environ['SLIME_DIR']}/.venv/lib/python3.12/site-packages"
env["PATH"] = f"{os.environ['SLIME_DIR']}/.venv/bin:/root/.nvm/versions/node/v22.23.1/bin:" + os.environ.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
env["NCCL_NVLS_ENABLE"] = "0"
print(json.dumps({"env_vars": env}))
PY
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- /mnt/cephfs/chenzhenyang/czy/slime/.venv/bin/python3 -u train.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${TRAIN_GPUS_PER_NODE}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${ALGO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   2>&1 | tee "${LOG_FILE}"

echo "RUN_ROOT=${RUN_ROOT}"
