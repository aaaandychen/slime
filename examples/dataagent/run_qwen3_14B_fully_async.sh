#!/bin/bash
# DataAgent fully async RL training with Qwen3-14B on 8xH200.
#
# Usage:
#   bash examples/dataagent/run_qwen3_14B_fully_async.sh
#
# Prerequisites:
#   - Qwen3-14B HF checkpoint at /mnt/cephfs/chenzhenyang/models/Qwen3-14B
#   - Megatron-LM dist checkpoint at .../Qwen3-14B_torch_dist
#   - DataAgent built at /mnt/cephfs/chenzhenyang/DataAgent
#   - MariaDB running with demo_sales database (see DataAgent/demo.sh)
#   - Embedding model downloaded (BAAI/bge-small-zh-v1.5)

set -ex

export PYTHONUNBUFFERED=1

# ── cleanup ────────────────────────────────────────────────────────────
pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray python 2>/dev/null || true
sleep 3

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-${SCRIPT_DIR}/../..}"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-14B.sh"

# ── config ────────────────────────────────────────────────────────────
MODEL_DIR="${MODEL_DIR:-/mnt/cephfs/chenzhenyang/models/Qwen3-14B}"
DATAAGENT_DIR="${DATAAGENT_DIR:-/mnt/cephfs/chenzhenyang/DataAgent}"
PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/queries_labeled.jsonl}"
SAVE_DIR="${SAVE_DIR:-/mnt/cephfs/chenzhenyang/models/Qwen3-14B_slime_dataagent}"

DATAAGENT_PORT="${DATAAGENT_PORT:-8065}"
EMBEDDING_PORT="${EMBEDDING_PORT:-8765}"
API_PORT="${API_PORT:-18080}"

# WandB
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
WANDB_PROJECT="${WANDB_PROJECT:-slime-dataagent}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-qwen3-14B_$(date +%Y%m%d_%H%M%S)}"

# ── helpers ────────────────────────────────────────────────────────────
wait_for_url() {
    local url="$1" desc="$2" max="${3:-60}"
    echo -n "  Waiting for ${desc} (${url}) "
    for _ in $(seq 1 "${max}"); do
        if curl -sf -o /dev/null "${url}" 2>/dev/null; then echo " OK"; return 0; fi
        sleep 2; echo -n "."
    done
    echo " FAILED"; return 1
}

api_curl()  { curl -s --max-time 10 "$@"; }
api_post()  { curl -s -X POST --max-time 10 "$@"; }
json_val()  { python3 -c "import sys,json; print(json.load(sys.stdin)${1})" 2>/dev/null; }

# ── step 1: services ──────────────────────────────────────────────────
echo "=== [1/5] Starting services ==="

# MariaDB
if ! mysql -u root -e "SELECT 1 FROM demo_sales.orders LIMIT 1" &>/dev/null; then
    echo "  Starting MariaDB..."
    mysqld_safe & sleep 3
fi

# Embedding
if ! curl -sf "http://127.0.0.1:${EMBEDDING_PORT}/health" &>/dev/null; then
    echo "  Starting Embedding service..."
    cd "${DATAAGENT_DIR}"
    nohup .venv/bin/python3 embedding_server.py ${EMBEDDING_PORT} > /tmp/embedding.log 2>&1 &
fi
wait_for_url "http://127.0.0.1:${EMBEDDING_PORT}/health" "Embedding"

# DataAgent backend
# Always restart backend (H2 in-memory — old config is lost on restart).
pkill -f "spring-ai-alibaba-data-agent-management" 2>/dev/null || true
sleep 2
echo "  Starting DataAgent backend..."
cd "${DATAAGENT_DIR}"
export PATH="${DATAAGENT_DIR}/.venv/bin:$PATH"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-17-openjdk-amd64}"
nohup java -jar "${DATAAGENT_DIR}/data-agent-management/target/spring-ai-alibaba-data-agent-management-1.0.0-SNAPSHOT.jar" \
    --spring.profiles.active=h2 --server.port="${DATAAGENT_PORT}" \
    > /tmp/dataagent.log 2>&1 &
wait_for_url "http://127.0.0.1:${DATAAGENT_PORT}/echo/ok" "DataAgent backend"

# ── step 2: configure DataAgent ───────────────────────────────────────
echo "=== [2/5] Configuring DataAgent ==="
API="http://127.0.0.1:${DATAAGENT_PORT}"

# Clean up old CHAT models, register fresh.
echo "  Registering CHAT model (→ Slime API :${API_PORT})..."
LIST=$(api_curl "${API}/api/model-config/list")
OLD=$(echo "${LIST}" | python3 -c "import sys,json; print(','.join(str(m['id']) for m in json.load(sys.stdin).get('data',[]) if m.get('modelType')=='CHAT'))" 2>/dev/null || true)
if [[ -n "${OLD}" ]]; then
    IFS=',' read -ra IDS <<< "${OLD}"
    for oid in "${IDS[@]}"; do api_post "${API}/api/model-config/${oid}" -X DELETE > /dev/null; done
fi
api_post "${API}/api/model-config/add" -H "Content-Type: application/json" \
    -d "{\"provider\":\"custom\",\"baseUrl\":\"http://127.0.0.1:${API_PORT}\",\"apiKey\":\"\",\"modelName\":\"slime-rl\",\"modelType\":\"CHAT\",\"maxTokens\":8192,\"completionsPath\":\"/v1/chat/completions\",\"proxyEnabled\":false}" > /dev/null
CHAT_ID=$(api_curl "${API}/api/model-config/list" | python3 -c "import sys,json; [print(m['id']) for m in json.load(sys.stdin).get('data',[]) if m.get('modelType')=='CHAT']" | tail -1)
api_post "${API}/api/model-config/activate/${CHAT_ID}" > /dev/null
echo "  CHAT model id=${CHAT_ID} activated → baseUrl=http://127.0.0.1:${API_PORT}"

# Embedding
echo "  Registering Embedding model..."
LIST=$(api_curl "${API}/api/model-config/list")
EMB_ID=$(echo "${LIST}" | python3 -c "import sys,json; [print(m['id']) for m in json.load(sys.stdin).get('data',[]) if m.get('modelType')=='EMBEDDING']" | tail -1)
if [[ -z "${EMB_ID}" ]]; then
    api_post "${API}/api/model-config/add" -H "Content-Type: application/json" \
        -d "{\"provider\":\"custom\",\"baseUrl\":\"http://127.0.0.1:${EMBEDDING_PORT}\",\"apiKey\":\"\",\"modelName\":\"bge-small-zh-v1.5\",\"modelType\":\"EMBEDDING\",\"embeddingsPath\":\"/v1/embeddings\",\"proxyEnabled\":false}" > /dev/null
    EMB_ID=$(api_curl "${API}/api/model-config/list" | python3 -c "import sys,json; [print(m['id']) for m in json.load(sys.stdin).get('data',[]) if m.get('modelType')=='EMBEDDING']" | tail -1)
fi
api_post "${API}/api/model-config/activate/${EMB_ID}" > /dev/null
echo "  EMBEDDING model id=${EMB_ID} activated"

# Datasource
echo "  Setting up datasource..."
DS_NAME="demo_sales"
DSID=$(api_curl "${API}/api/datasource" | python3 -c "
import sys,json; items=json.load(sys.stdin); items=items if isinstance(items,list) else items.get('data',[])
for ds in items:
    if ds.get('name')=='${DS_NAME}' and ds.get('status')=='active': print(ds['id']); break
" 2>/dev/null)
if [[ -z "${DSID}" ]]; then
    DSID=$(api_post "${API}/api/datasource" -H "Content-Type: application/json" \
        -d '{"name":"demo_sales","type":"mysql","host":"127.0.0.1","port":3306,"databaseName":"demo_sales","username":"root","password":""}' \
        | json_val "['id']")
fi
echo "  Datasource id=${DSID}"

# Agent — always delete+recreate for clean state.
echo "  Setting up agent..."
api_curl "${API}/api/agent/list" | python3 -c "
import sys,json
for a in json.load(sys.stdin).get('data',[]):
    if a.get('name')=='Demo': print(a['id'])
" 2>/dev/null | while read oid; do
    api_post "${API}/api/agent/${oid}" -X DELETE > /dev/null
done
AID=$(api_post "${API}/api/agent" -H "Content-Type: application/json" \
    -d '{"name":"Demo","description":"Demo Agent","category":"demo"}' \
    | json_val "['id']")
api_post "${API}/api/agent/${AID}/datasources/${DSID}" > /dev/null
api_post "${API}/api/agent/${AID}/datasources/tables" -H "Content-Type: application/json" \
    -d "{\"datasourceId\":${DSID},\"tables\":[\"products\",\"orders\"]}" > /dev/null
api_post "${API}/api/agent/${AID}/datasources/init" > /dev/null
api_post "${API}/api/agent/${AID}/publish" > /dev/null
echo "  Agent id=${AID} ready"

# Verify: active CHAT model must point to the correct Slime API port.
ACTIVE_URL=$(api_curl "${API}/api/model-config/list" | python3 -c "
import sys,json
for m in json.load(sys.stdin).get('data',[]):
    if m.get('modelType')=='CHAT' and m.get('isActive'):
        print(m['baseUrl']); break
")
if [[ "${ACTIVE_URL}" != "http://127.0.0.1:${API_PORT}" ]]; then
    echo "  ERROR: active CHAT model points to ${ACTIVE_URL}, expected http://127.0.0.1:${API_PORT}"
    exit 1
fi
echo "  Verified: active CHAT model → ${ACTIVE_URL}"

export DATAAGENT_AGENT_ID="${AID}"
export DATAAGENT_BASE_URL="${API}"
echo "  DATAAGENT_AGENT_ID=${DATAAGENT_AGENT_ID}"

# ── step 3: prompt data ───────────────────────────────────────────────
echo "=== [3/5] Preparing prompt data ==="
if [[ ! -f "${PROMPT_DATA}" ]]; then
    echo "  ${PROMPT_DATA} not found, generating labeled queries..."
    python3 "${SCRIPT_DIR}/generate_training_data.py" --source offline --out "${PROMPT_DATA}"
fi
echo "  Prompt data: ${PROMPT_DATA} ($(wc -l < "${PROMPT_DATA}") queries)"

# ── step 4: slime args ────────────────────────────────────────────────
echo "=== [4/5] Building training args ==="

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}"
    --ref-load "${MODEL_DIR}_torch_dist"
    --save "${SAVE_DIR}"
    --save-interval 20
)

ROLLOUT_ARGS=(
    --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
    --custom-generate-function-path examples.dataagent.custom_generate.generate
    --update-weights-interval 1

    --prompt-data "${PROMPT_DATA}"
    --input-key query
    --label-key label
    --rollout-shuffle

    --num-rollout 20  # TODO: increase for production
    --rollout-batch-size 8
    --n-samples-per-prompt 4
    --rollout-max-context-len 32768
    --rollout-max-response-len 2048
    --rollout-temperature 1.0

    --global-batch-size 8
    --balance-data
)

PERF_ARGS=(
    --tensor-model-parallel-size 4
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
    --use-tis
    --tis-clip 2.0
    --tis-clip-low 0.0
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
    --rollout-num-gpus-per-engine 4
    --sglang-mem-fraction-static 0.7
    --sglang-server-concurrency 32
    --sglang-reasoning-parser qwen3
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --transformer-impl transformer_engine
    --cross-entropy-loss-fusion
    # NOTE: --log-passrate removed — compute_pass_rate asserts
    # len(flat_rewards) == num_groups * group_size, but DataAgent's
    # TrajectoryManager fans out into variable segments per query
    # (5~27), so total rewards (412) != expected (8*4=32).  The
    # pass@k metric also checks reward==1.0 which never holds after
    # reward/K split across fan-out segments.  coding_agent_rl doesn't
    # use this flag either.
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-exp-name "${WANDB_EXP_NAME}"
)

# ── step 5: launch ────────────────────────────────────────────────────
echo "=== [5/5] Launching training ==="

NUM_GPUS="${NUM_GPUS:-8}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || echo 0)

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats

RUNTIME_ENV_JSON=$(python3 -c "
import json, os
env = {
    'PYTHONPATH': '${MEGATRON_DIR:-/mnt/cephfs/chenzhenyang/Megatron-LM}:${SCRIPT_DIR}:${SCRIPT_DIR}/../..',
    'CUDA_DEVICE_MAX_CONNECTIONS': '1',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    'NCCL_NVLS_ENABLE': '${NVLINK_COUNT}',
    'DATAAGENT_AGENT_ID': '${DATAAGENT_AGENT_ID}',
    'DATAAGENT_BASE_URL': '${DATAAGENT_BASE_URL}',
    'DATAAGENT_TIMEOUT': '600',
    'DATAAGENT_ROLLOUT_GUARD_SEC': '${DATAAGENT_ROLLOUT_GUARD_SEC:-900}',
    'SLIME_API_PORT': '${API_PORT}',
    'WANDB_API_KEY': '${WANDB_API_KEY}',
    'WANDB_PROJECT': '${WANDB_PROJECT}',
    'WANDB_EXP_NAME': '${WANDB_EXP_NAME}',
}
# Explicit working_dir so the Ray job resolves train_async.py regardless
# of the current shell cwd.
print(json.dumps({'env_vars': env, 'working_dir': '${SLIME_DIR}'}))
")
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- ${SLIME_DIR}/.venv/bin/python3 -u train_async.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${ACTOR_GPUS}" \
    --rollout-num-gpus "${ROLLOUT_GPUS}" \
    ${MODEL_ARGS[@]} \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"

echo "=== Launched.  Logs: ray job logs ==="
