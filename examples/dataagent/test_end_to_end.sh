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

# ── helper ────────────────────────────────────────────────────────────
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
echo "=== [1/4] Checking MariaDB ==="
if ! mysql -u root -e "SELECT 1" &>/dev/null; then
    echo "  Starting MariaDB..."
    mysqld_safe &
    sleep 3
fi
mysql -u root -e "SELECT 1" && echo "  MariaDB OK"

# ── step 2: Embedding service ────────────────────────────────────────
echo "=== [2/4] Starting Embedding service (:${EMBEDDING_PORT}) ==="
cd "${DATAAGENT_DIR}"

if ! curl -sf "http://localhost:${EMBEDDING_PORT}/health" &>/dev/null; then
    pkill -f "embedding_server.py ${EMBEDDING_PORT}" 2>/dev/null || true
    sleep 1
    nohup .venv/bin/python3 embedding_server.py ${EMBEDDING_PORT} \
        > /tmp/embedding_server.log 2>&1 &
fi
wait_for_url "http://localhost:${EMBEDDING_PORT}/health" "Embedding service"

# ── step 3: DataAgent backend ────────────────────────────────────────
echo "=== [3/4] Starting DataAgent backend (:${DATAAGENT_PORT}) ==="

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

# ── step 4: Slime → DataAgent test ───────────────────────────────────
echo "=== [4/4] Running Slime → DataAgent test ==="
cd "${SLIME_DIR}"

DATAAGENT_BASE_URL="http://localhost:${DATAAGENT_PORT}" \
    python examples/dataagent/check_output.py "${QUERY}"

echo
echo "=== All 4 steps completed successfully ==="
