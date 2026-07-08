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
#   MODEL_PATH          模型路径 (必需)
#   CHECKPOINT_PATH     Megatron 分布式 checkpoint 路径 (必需)
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
MODEL_PATH="${MODEL_PATH}"
CHECKPOINT_PATH="${CHECKPOINT_PATH}"
SAVE_DIR="${SAVE_DIR:-/tmp/slime_daa_checkpoint}"
N_GPUS="${N_GPUS:-8}"
TRAIN_STEPS="${TRAIN_STEPS:-10}"

if [ -z "$MODEL_PATH" ] || [ -z "$CHECKPOINT_PATH" ]; then
    echo "ERROR: MODEL_PATH and CHECKPOINT_PATH must be set."
    echo "  MODEL_PATH:       path to HF model (e.g. /mnt/cephfs/models/Qwen3-14B)"
    echo "  CHECKPOINT_PATH:  path to Megatron dist checkpoint"
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
pkill -f "app.py" 2>/dev/null || true
sleep 2

cd "$DAA_DIR"
BAA_WSGI=flask DAA_PROVIDER=slime-rl \
    nohup python3 app.py --port "$DAA_PORT" \
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
# Step 4: Launch Slime training
# =========================================================================
echo "=== Launching Slime training ==="
cd "$SLIME_DIR"

export DAA_BASE_URL="http://127.0.0.1:${DAA_PORT}"
export DAA_PROVIDER="slime-rl"
export DEMO_SALES_DB="${SCRIPT_DIR}/demo_sales.db"
export SLIME_API_PORT="${SLIME_API_PORT}"

# GPU split: actor uses TP=4 (first 4 GPUs), rollout uses TP=4 (last 4 GPUs)
N_ACTOR=$((N_GPUS / 2))
N_ROLLOUT=$((N_GPUS - N_ACTOR))

python3 -m slime.run_train \
    --model-path "$MODEL_PATH" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --prompt-data "$PROMPT_DATA" \
    --input-key query \
    --custom-generate-function-path examples.data_analysis_agent.custom_generate.generate \
    --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async \
    --save-dir "$SAVE_DIR" \
    --train-steps "$TRAIN_STEPS" \
    --n-samples-per-prompt 4 \
    --num-rollout 20 \
    --rollout-batch-size 8 \
    --n-actor "$N_ACTOR" \
    --n-rollout "$N_ROLLOUT" \
    --tensor-model-parallel-size 4 \
    --lr 1e-6 \
    --lr-warmup-steps 5 \
    --kl-loss-coef 0.001 \
    --use-tis \
    --tis-clip-rho 1.2 \
    --pg-clip-eps 0.2 \
    --slime-api-host 0.0.0.0 \
    --slime-api-port "$SLIME_API_PORT" \
    "$@"

# =========================================================================
# Cleanup
# =========================================================================
kill $DAA_PID 2>/dev/null || true
echo "=== Done ==="