
#!/bin/bash
# End-to-end test: DataAgent + DeepSeek + Slime custom_generate.
#
# Usage:
#   export DEEPSEEK_API_KEY=sk-xxxxxxxx
#   bash examples/dataagent/test_end_to_end.sh
#
# Optional env vars:
#   DATAAGENT_DIR    — DataAgent project root (default /mnt/cephfs/chenzhenyang/DataAgent)
#   QUERY            — test query (default "各区域销售额排名")
#   HTTP_PROXY       — proxy for DeepSeek API access

set -e

# ── config ────────────────────────────────────────────────────────────
DATAAGENT_DIR="${DATAAGENT_DIR:-/mnt/cephfs/chenzhenyang/DataAgent}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
QUERY="${QUERY:-各区域销售额排名}"
DATAAGENT_PORT=8065
EMBEDDING_PORT=8765

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY not set."
    echo "  export DEEPSEEK_API_KEY=sk-xxxxxxxx"
    exit 1
fi

# Proxy for DeepSeek API access (required if machine has no direct internet)
HTTP_PROXY="${HTTP_PROXY:-http://10.74.176.8:11080}"
PROXY_HOST="${PROXY_HOST:-10.74.176.8}"
PROXY_PORT="${PROXY_PORT:-11080}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTP_PROXY}"
export HTTP_PROXY="${HTTP_PROXY}"
export HTTPS_PROXY="${HTTP_PROXY}"
export no_proxy="localhost,127.0.0.1,localaddress,localdomain.com,.internal"
export NO_PROXY="${no_proxy}"

# ── helpers ───────────────────────────────────────────────────────────
die() { echo "ERROR: $1"; exit 1; }

wait_for_url() {
    local url="$1" desc="$2" max_retries="${3:-60}"
    echo -n "  Waiting for ${desc} (${url}) "
    for i in $(seq 1 "${max_retries}"); do
        if curl -sf -o /dev/null "${url}" 2>/dev/null; then
            echo " OK"
            return 0
        fi
        sleep 2
        echo -n "."
    done
    echo " FAILED"
    return 1
}

# ── step 1: MariaDB ──────────────────────────────────────────────────
echo "=== [1/7] Checking MariaDB ==="
if ! mysql -u root -e "SELECT 1" &>/dev/null; then
    echo "  Starting MariaDB..."
    mysqld_safe &
    sleep 3
fi
mysql -u root -e "SELECT 1" && echo "  MariaDB OK"

# ── step 2: Embedding service ────────────────────────────────────────
echo "=== [2/7] Starting Embedding service (:${EMBEDDING_PORT}) ==="
cd "${DATAAGENT_DIR}"

if ! curl -sf "http://localhost:${EMBEDDING_PORT}/health" &>/dev/null; then
    pkill -f "embedding_server.py ${EMBEDDING_PORT}" 2>/dev/null || true
    sleep 1
    nohup .venv/bin/python3 embedding_server.py ${EMBEDDING_PORT} \
        > /tmp/embedding_server.log 2>&1 &
fi
wait_for_url "http://localhost:${EMBEDDING_PORT}/health" "Embedding service"

# ── step 3: DataAgent backend ────────────────────────────────────────
echo "=== [3/7] Starting DataAgent backend (:${DATAAGENT_PORT}) ==="

export PATH="${DATAAGENT_DIR}/.venv/bin:$PATH"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-17-openjdk-amd64}"

JAR="${DATAAGENT_DIR}/data-agent-management/target/spring-ai-alibaba-data-agent-management-1.0.0-SNAPSHOT.jar"

if ! curl -sf "http://localhost:${DATAAGENT_PORT}/echo/ok" &>/dev/null; then
    pkill -f "spring-ai-alibaba-data-agent-management" 2>/dev/null || true
    sleep 2
    nohup java -jar "${JAR}" --spring.profiles.active=h2 \
        --server.port="${DATAAGENT_PORT}" \
        > /tmp/dataagent_backend.log 2>&1 &
fi
wait_for_url "http://localhost:${DATAAGENT_PORT}/echo/ok" "DataAgent backend"

# ── step 4: Register DeepSeek model ──────────────────────────────────
echo "=== [4/7] Registering DeepSeek model ==="
API_BASE="http://localhost:${DATAAGENT_PORT}"

# Clean up old CHAT models first, then register fresh.
echo "  Cleaning up old CHAT models..."
LIST=$(curl -s "${API_BASE}/api/model-config/list")
OLD_IDS=$(echo "${LIST}" | python3 -c "
import sys,json
models = json.load(sys.stdin).get('data', [])
ids = [str(m['id']) for m in models if m.get('modelType') == 'CHAT']
print(','.join(ids))
" 2>/dev/null)
if [[ -n "${OLD_IDS}" ]]; then
    IFS=',' read -ra IDS <<< "${OLD_IDS}"
    for OID in "${IDS[@]}"; do
        curl -s -X DELETE "${API_BASE}/api/model-config/${OID}" > /dev/null || true
        echo "  Deleted old model id=${OID}"
    done
fi

echo "  Registering DeepSeek chat model..."
curl -s -X POST "${API_BASE}/api/model-config/add" \
    -H "Content-Type: application/json" \
    -d "{\"provider\":\"deepseek\",\"baseUrl\":\"https://api.deepseek.com\",\"apiKey\":\"${DEEPSEEK_API_KEY}\",\"modelName\":\"deepseek-chat\",\"modelType\":\"CHAT\",\"maxTokens\":8192,\"completionsPath\":\"/v1/chat/completions\",\"proxyEnabled\":true,\"proxyHost\":\"${PROXY_HOST}\",\"proxyPort\":${PROXY_PORT}}" \
    > /dev/null

# add 接口返回 data:null, 从 list 接口拿 ID
LIST=$(curl -s "${API_BASE}/api/model-config/list")
MODEL_ID=$(echo "${LIST}" | python3 -c "
import sys,json
models = json.load(sys.stdin).get('data', [])
chat = [m for m in models if m.get('modelType') == 'CHAT']
print(chat[-1]['id'] if chat else '')
")
if [[ -z "${MODEL_ID}" ]]; then
    echo "  ERROR: Could not find model id. list response: ${LIST}"
    exit 1
fi
echo "  Model id=${MODEL_ID}"

curl -s -X POST "${API_BASE}/api/model-config/activate/${MODEL_ID}" > /dev/null || true
echo "  Activated model id=${MODEL_ID}"

# Register local Embedding model (required for schema init)
echo "  Registering Embedding model..."
curl -s -X POST "${API_BASE}/api/model-config/add" \
    -H "Content-Type: application/json" \
    -d "{\"provider\":\"custom\",\"baseUrl\":\"http://localhost:${EMBEDDING_PORT}\",\"apiKey\":\"\",\"modelName\":\"bge-small-zh-v1.5\",\"modelType\":\"EMBEDDING\",\"embeddingsPath\":\"/v1/embeddings\",\"proxyEnabled\":false}" \
    > /dev/null

EMBED_ID=$(curl -s "${API_BASE}/api/model-config/list" | python3 -c "
import sys,json
for m in json.load(sys.stdin).get('data',[]):
    if m.get('modelType')=='EMBEDDING': print(m['id']); break
")
[[ -z "${EMBED_ID}" ]] && die "Embedding model registration failed"
curl -s -X POST "${API_BASE}/api/model-config/activate/${EMBED_ID}" > /dev/null || true
echo "  Embedding model id=${EMBED_ID} activated"

# ── step 5: Datasource (idempotent) ──────────────────────────────────
echo "=== [5/7] Setting up Datasource ==="
DS_NAME="demo_sales"
DSID=$(curl -s "${API_BASE}/api/datasource" | python3 -c "
import sys,json; d=json.load(sys.stdin); items=d if isinstance(d,list) else d.get('data',[])
for ds in items:
    if ds.get('name')=='${DS_NAME}' and ds.get('status')=='active': print(ds['id']); break
" 2>/dev/null)
if [[ -z "${DSID}" ]]; then
    DSID=$(curl -s -X POST "${API_BASE}/api/datasource" -H "Content-Type: application/json" \
        -d '{"name":"demo_sales","type":"mysql","host":"127.0.0.1","port":3306,"databaseName":"demo_sales","username":"root","password":""}' \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null)
    [[ -z "${DSID}" ]] && die "Datasource creation failed"
fi
echo "  Datasource id=${DSID}"

# ── step 6: Agent (always recreate to ensure clean state) ─────────────
echo "=== [6/7] Setting up Agent ==="
# Delete old Demo agent if exists
curl -s "${API_BASE}/api/agent/list" | python3 -c "
import sys,json; d=json.load(sys.stdin); items=d if isinstance(d,list) else d.get('data',[])
for a in items:
    if a.get('name')=='Demo': print(a['id'])
" 2>/dev/null | while read old_id; do
    curl -s -X DELETE "${API_BASE}/api/agent/${old_id}" > /dev/null 2>&1 || true
done

AID=$(curl -s -X POST "${API_BASE}/api/agent" -H "Content-Type: application/json" \
    -d '{"name":"Demo","description":"Demo Agent","category":"demo"}' \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null)
[[ -z "${AID}" ]] && die "Agent creation failed"

curl -s -X POST "${API_BASE}/api/agent/${AID}/datasources/${DSID}" > /dev/null
curl -s -X POST "${API_BASE}/api/agent/${AID}/datasources/tables" -H "Content-Type: application/json" \
    -d "{\"datasourceId\":${DSID},\"tables\":[\"products\",\"orders\"]}" > /dev/null
INIT=$(curl -s -X POST "${API_BASE}/api/agent/${AID}/datasources/init" | python3 -c "import sys,json; print(json.load(sys.stdin).get('success',''))" 2>/dev/null)
[[ "${INIT}" != "True" ]] && die "Schema init failed"
curl -s -X POST "${API_BASE}/api/agent/${AID}/publish" > /dev/null
echo "  Agent id=${AID} ready"

# ── step 7: Slime → DataAgent test ───────────────────────────────────
echo "=== [7/7] Running Slime → DataAgent test ==="
export DATAAGENT_BASE_URL="http://localhost:${DATAAGENT_PORT}"
export DATAAGENT_AGENT_ID="${AID}"

cd "${SLIME_DIR}"
python examples/dataagent/check_output.py "${QUERY}"

echo
echo "=== All 7 steps completed successfully ==="