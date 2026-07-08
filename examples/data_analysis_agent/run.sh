#!/bin/bash
# ===========================================================================
# DAA + Slime RL 训练启动脚本
#
# 用法:
#   在远程 GPU 集群上执行:
#     DAA_DIR=/path/to/Data-Analysis-Agent \
#     SLIME_DIR=/path/to/slime \
#     bash examples/data_analysis_agent/run.sh
#
# 环境变量:
#   DAA_DIR             Data-Analysis-Agent 项目路径 (必需)
#   SLIME_DIR           Slime 项目路径 (必需，默认当前目录)
#   DAA_PORT            DAA 服务端口 (默认 5001)
#   SLIME_API_PORT      slime_api 端口 (默认 18080)
#   PROMPT_DATA         查询数据 JSONL (默认 examples/data_analysis_agent/queries.jsonl)
#   HF_CHECKPOINT       HF 模型路径 (仅需 config.json + tokenizer, 权重无所谓——会被 NCCL 覆盖)
#   REF_LOAD            Megatron 分布式 checkpoint 路径 (唯一权重来源)
#   SAVE_DIR            训练保存目录 (默认 /tmp/slime_daa_checkpoint)
#   N_GPUS              GPU 数量 (默认 8)
#   TRAIN_STEPS         训练步数 (默认 10)
# ===========================================================================

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DAA_DIR="${DAA_DIR:-/mnt/cephfs/chenzhenyang/Data-Analysis-Agent}"
DAA_PORT="${DAA_PORT:-5001}"
SLIME_API_PORT="${SLIME_API_PORT:-18080}"
PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/queries.jsonl}"
HF_CHECKPOINT="${HF_CHECKPOINT}"
REF_LOAD="${REF_LOAD}"
SAVE_DIR="${SAVE_DIR:-/tmp/slime_daa_checkpoint}"
N_GPUS="${N_GPUS:-8}"
TRAIN_STEPS="${TRAIN_STEPS:-10}"

if [ -z "$HF_CHECKPOINT" ] || [ -z "$REF_LOAD" ]; then
    echo "ERROR: HF_CHECKPOINT and REF_LOAD must be set."
    echo "  HF_CHECKPOINT:  HF model (只需要 config.json + tokenizer, 权重无所谓)"
    echo "                  训练前 NCCL broadcast 会将 Megatron 权重覆盖到 SGLang"
    echo "                  e.g. /mnt/cephfs/models/Qwen3-14B"
    echo "  REF_LOAD:       Megatron dist checkpoint (唯一权重来源)"
    echo "                  e.g. /mnt/cephfs/models/Qwen3-14B_torch_dist"
    exit 1
fi

# =========================================================================
# Step 1: Generate SQLite database (if not exists)
# =========================================================================
if [ ! -f "${SCRIPT_DIR}/demo_sales.db" ]; then
    echo "=== Generating demo_sales.db ==="
    python3 "${SCRIPT_DIR}/generate_demo_sales_db.py"
fi

# =========================================================================
# Step 1.5: Install DAA dependencies (skip auto-install at runtime)
# =========================================================================
echo "=== Installing DAA dependencies ==="
cd "$DAA_DIR"
pip install -r requirements.txt -q 2>&1 | tail -3
export BAA_SKIP_DEPENDENCY_CHECK=1
echo "  DAA dependencies ready"

# =========================================================================
# Step 2: Configure DAA LLM (slime-rl → slime_api)
# =========================================================================
echo "=== Configuring DAA LLM provider ==="
DAA_CONFIG_DIR="${DAA_DIR}/LLM"
mkdir -p "$DAA_CONFIG_DIR"

cat > "${DAA_CONFIG_DIR}/llm_config.json" << EOF
{
  "slime-rl": {
    "provider": "slime-rl",
    "api_key": "not-needed",
    "base_url": "http://127.0.0.1:${SLIME_API_PORT}",
    "model": "slime-rl",
    "name": "Slime RL",
    "enabled": true,
    "is_custom": true,
    "context_window": 131072,
    "max_output_tokens": 8192,
    "enable_thinking": false
  }
}
EOF
echo "  DAA LLM config written: ${DAA_CONFIG_DIR}/llm_config.json"

# =========================================================================
# Step 3: Start DAA
# =========================================================================
echo "=== Starting DAA on port ${DAA_PORT} ==="
# Only kill DAA's own app.py, not other python processes
pkill -f "Data-Analysis-Agent.*app.py" 2>/dev/null || true
sleep 2

cd "$DAA_DIR"
PORT="${DAA_PORT}" BAA_WSGI=flask \
    nohup python3 app.py \
    > /tmp/daa.log 2>&1 &
DAA_PID=$!
echo "  DAA PID=$DAA_PID"

# Wait for DAA readiness
for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${DAA_PORT}/api/health" > /dev/null 2>&1; then
        echo "  DAA ready (port ${DAA_PORT})"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: DAA failed to start. Check /tmp/daa.log"
        exit 1
    fi
    sleep 1
done

# =========================================================================
# Step 4: Launch Slime training (via Ray)
# =========================================================================
echo "=== Launching Slime training ==="
cd "$SLIME_DIR"

export DAA_BASE_URL="http://127.0.0.1:${DAA_PORT}"
export DAA_PROVIDER="slime-rl"
export DEMO_SALES_DB="${SCRIPT_DIR}/demo_sales.db"
export SLIME_API_PORT="${SLIME_API_PORT}"

# GPU split: actor uses TP=4 (first 4 GPUs), rollout uses TP=4 (last 4 GPUs)
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || echo 0)

# Start Ray head node
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${N_GPUS}" --disable-usage-stats 2>/dev/null || true

# Build runtime env JSON for Ray job
RUNTIME_ENV_JSON=$(python3 -c "
import json, os
env = {
    'PYTHONPATH': '${SLIME_DIR}',
    'CUDA_DEVICE_MAX_CONNECTIONS': '1',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    'NCCL_NVLS_ENABLE': '${NVLINK_COUNT}',
    'DAA_BASE_URL': '${DAA_BASE_URL}',
    'DAA_PROVIDER': '${DAA_PROVIDER}',
    'DEMO_SALES_DB': '${DEMO_SALES_DB}',
    'DAA_TIMEOUT': '600',
    'SLIME_API_PORT': '${SLIME_API_PORT}',
}
print(json.dumps({'env_vars': env, 'working_dir': '${SLIME_DIR}'}))
")

# Submit Ray job
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- ${SLIME_DIR}/.venv/bin/python3 -u train_async.py \
    --hf-checkpoint "$HF_CHECKPOINT" \
    --ref-load "$REF_LOAD" \
    --save "$SAVE_DIR" \
    --save-interval 20 \
    --custom-generate-function-path examples.data_analysis_agent.custom_generate.generate \
    --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async \
    --update-weights-interval 1 \
    --prompt-data "$PROMPT_DATA" \
    --input-key query \
    --rollout-shuffle \
    --num-rollout 20 \
    --rollout-batch-size 8 \
    --n-samples-per-prompt 4 \
    --rollout-max-response-len 2048 \
    --rollout-temperature 1.0 \
    --global-batch-size 8 \
    --balance-data \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "$ACTOR_GPUS" \
    --rollout-num-gpus "$ROLLOUT_GPUS" \
    --rollout-num-gpus-per-engine 4 \
    --tensor-model-parallel-size 4 \
    --sequence-parallel \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu 9216 \
    --advantage-estimator grpo \
    --use-kl-loss \
    --kl-loss-coef 0.00 \
    --kl-loss-type low_var_kl \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    --use-tis \
    --tis-clip 2.0 \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --slime-api-host 0.0.0.0 \
    --slime-api-port "$SLIME_API_PORT" \
    --rollout-steps "$TRAIN_STEPS" \
    "$@"

echo "=== Launched. Logs: ray job logs ==="

# =========================================================================
# Cleanup
# =========================================================================
kill $DAA_PID 2>/dev/null || true
echo "=== Done ==="