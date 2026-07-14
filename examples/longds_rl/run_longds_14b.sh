#!/usr/bin/env bash
# End-to-end LongDS RL training on Qwen3-14B.
# Run from tmux/screen on the Ray head node.
#
# Prerequisites:
#   1. LongDS dataset converted to JSONL via convert_longds_to_slime.py
#   2. Claude Code CLI installed on all nodes (in PATH)
#   3. Qwen3-14B checkpoint accessible at HF_CHECKPOINT

# Best-effort cleanup
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
sleep 3

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# ============ model ============
HF_CHECKPOINT="${HF_CHECKPOINT:-/mnt/cephfs/chenzhenyang/models/Qwen3-14B}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/mnt/cephfs/chenzhenyang/models/Qwen3-14B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-/path/to/longds_train.jsonl}"

# ============ model parallelism ============
TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-4}"

# ============ rollout engine ============
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-2}"
ROLLOUT_DP_SIZE="${ROLLOUT_DP_SIZE:-4}"
ROLLOUT_MEM_UTILIZATION="${ROLLOUT_MEM_UTILIZATION:-0.75}"

# ============ context ============
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-48000}"
MAX_GEN_LEN="${MAX_GEN_LEN:-16384}"

# ============ run tag ============
EXP_TAG="${EXP_TAG:-longds_qwen3_14b}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${SLIME_DIR}/runs/${EXP_TAG}_${STAMP}}"
mkdir -p "${RUN_ROOT}/rollout_dumps"
LOG_FILE="${RUN_ROOT}/run.log"
echo "Log: ${LOG_FILE}"

# ============ model architecture (Qwen3 dense) ============
# --spec reads hidden_size/num_layers/heads from HF config.json.
# Qwen3-14B is dense (no MoE), so MoE args from the SWE script are omitted.
MODEL_ARGS=(
   --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"
   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --normalization RMSNorm
   --apply-layernorm-1p
   --position-embedding-type rope
   --swiglu
   --untie-embeddings-and-output-weights
   --attention-output-gate
)

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_MODEL_PATH}"
)

# ============ rollout ============
ROLLOUT_ARGS=(
   --custom-generate-function-path examples.longds_rl.generate.generate
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --num-rollout 100
   --rollout-batch-size 4
   --n-samples-per-prompt 4
   --rollout-max-context-len ${MAX_CONTEXT_LEN}
   --rollout-max-response-len ${MAX_GEN_LEN}
   --rollout-temperature 1.0
   --num-steps-per-rollout 1
   --global-batch-size 32
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
   --eps-clip-high 0.28
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
   --rollout-num-gpus 32
   --rollout-num-gpus-per-engine ${ROLLOUT_TP_SIZE}
   --sglang-mem-fraction-static ${ROLLOUT_MEM_UTILIZATION}
   --sglang-enable-dp-attention
   --sglang-dp-size ${ROLLOUT_DP_SIZE}
   --sglang-tool-call-parser qwen3_coder
   --sglang-reasoning-parser qwen3
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --colocate
)

# ============ network ============
export MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-$(hostname -I | awk '{print $1}')}}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-6379}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"

# ============ LongDS rollout env ============
# Claude Code must be in PATH on all nodes. No sandbox/E2B needed —
# each rollout runs in a local temp directory.
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-${MASTER_ADDR}}"
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export ADAPTER_PORT="${ADAPTER_PORT:-18001}"

export LONGDS_TURN_TIMEOUT_SEC="${LONGDS_TURN_TIMEOUT_SEC:-600}"
export LONGDS_ROLLOUT_TIMEOUT_SEC="${LONGDS_ROLLOUT_TIMEOUT_SEC:-7200}"

# CC settings: bypass permissions, enable auto-compact
SETTINGS_JSON='{"permissions":{"defaultMode":"bypassPermissions"},"autoCompactEnabled":true,"autoCompactWindow":40000}'
export SLIME_AGENT_CC_EXTRA_ARGS="--settings '${SETTINGS_JSON}' --disallowedTools WebFetch WebSearch"

# ============ proxy ============
export no_proxy="127.0.0.1,${MASTER_ADDR},${ADAPTER_PUBLIC_HOST}"
export NO_PROXY="${no_proxy}"

cd "${SLIME_DIR}"

# ============ Ray cluster ============
HOSTFILE="${HOSTFILE:-/root/mpi_rack_hostfile}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

if [[ -f "${HOSTFILE}" ]]; then
  for WORKER_IP in $(awk '{print $1}' "${HOSTFILE}"); do
    [[ -z "${WORKER_IP}" ]] && continue
    [[ "${WORKER_IP}" == "${MASTER_ADDR}" ]] && continue
    echo "Starting Ray worker on ${WORKER_IP}"
    ssh -o StrictHostKeyChecking=no "root@${WORKER_IP}" \
      "pkill -9 sglang ; ray stop --force ; \
       ray start --address=${MASTER_ADDR}:6379 --num-gpus ${ACTOR_NUM_GPUS_PER_NODE} \
         --node-ip-address ${WORKER_IP} --disable-usage-stats" &
  done
  wait
fi

sleep 30
ray status

# ============ runtime env → Ray workers ============
export SLIME_DIR
RUNTIME_ENV_JSON=$(python3 - <<PY
import json, os
keys = (
    "no_proxy", "NO_PROXY",
    "ADAPTER_PUBLIC_HOST", "ADAPTER_BIND_HOST", "ADAPTER_PORT",
    "LONGDS_TURN_TIMEOUT_SEC", "LONGDS_ROLLOUT_TIMEOUT_SEC",
    "SLIME_AGENT_CC_EXTRA_ARGS",
)
env = {k: os.environ[k] for k in keys if k in os.environ}
env["MASTER_ADDR"] = os.environ["MASTER_ADDR"]
env["MASTER_PORT"] = os.environ.get("MASTER_PORT", "")
env["GLOO_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["TP_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["NCCL_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"]
env["PYTHONPATH"] = f"{os.environ['SLIME_DIR']}"
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
env["NCCL_NVLS_ENABLE"] = "0"
print(json.dumps({"env_vars": env}))
PY
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -u train.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
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
