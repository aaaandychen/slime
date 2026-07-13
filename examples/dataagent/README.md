# DataAgent + Slime 对接文档

## 架构

```
Slime Worker
  │
  ├─ custom_generate.generate()
  │     ├─ 启动 Slime API (:19876)
  │     ├─ 注册 sample → threadId
  │     └─ SSE 请求 → DataAgent (:8065)
  │
  ▼
DataAgent (:8065)
  │
  ├─ 内部串行 8~12 轮 LLM 调用
  │     └─ POST /v1/chat/completions → Slime API (:19876)
  │           └─ 转发 SGLang → HF checkpoint 推理
  │
  └─ 每轮 HTTP 请求头带 X-DataAgent-Thread-Id（Slime 侧关联 sample）
```

## 数据库

### DataAgent 有两层数据库

| 层 | 数据库 | 存储内容 | 生命周期 |
|----|--------|---------|---------|
| 业务库 | H2 内存 | Agent、Datasource、ModelConfig、Session | 后端重启即清空 |
| 分析库 | MariaDB | 业务数据（products、orders） | 持久化，独立于 DataAgent |

### 业务库（H2）

DataAgent 自身用 H2 内存数据库（开发模式）。存储：
- `agent` — Agent 配置
- `datasource` — 外部数据库连接信息
- `agent_datasource` — Agent 与数据源的关联
- `agent_datasource_tables` — 选中的表
- `model_config` — LLM 模型配置（baseUrl、apiKey 等）
- `chat_session` / `chat_message` — 对话历史

**每次后端重启全部清空**。因此 `test_end_to_end.sh` 和 `test_abort_resume.py` 都必须在每次启动后重新配置。

### 分析库（MariaDB）

存放被分析的真实业务数据。独立于 DataAgent，不会因后端重启而丢失。

**创建方式**：

```bash
mysql -u root <<'EOSQL'
CREATE DATABASE IF NOT EXISTS demo_sales DEFAULT CHARSET utf8mb4;
USE demo_sales;

CREATE TABLE IF NOT EXISTS products (
    product_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100),
    category VARCHAR(50),
    price DECIMAL(10,2),
    cost DECIMAL(10,2),
    stock INT
);

CREATE TABLE IF NOT EXISTS orders (
    order_id INT PRIMARY KEY AUTO_INCREMENT,
    product_id INT,
    customer_name VARCHAR(50),
    quantity INT,
    unit_price DECIMAL(10,2),
    order_date DATE,
    region VARCHAR(20),
    channel VARCHAR(20),
    FOREIGN KEY(product_id) REFERENCES products(product_id)
);

INSERT INTO products VALUES ...   -- 15 条商品
INSERT INTO orders VALUES ...     -- 47 条订单（2026 H1）
EOSQL
```

完整数据见 `DataAgent/demo.sh`。

### 数据库如何注册到 DataAgent

MariaDB 本身 DataAgent 不知道。需要通过 API 告诉它：

```
1. POST /api/datasource     ← 注册连接信息
2. POST /api/agent/{id}/datasources/{dsId}   ← 关联到 Agent
3. POST /api/agent/{id}/datasources/tables    ← 选择哪些表
4. POST /api/agent/{id}/datasources/init      ← Schema 向量化（Embedding）
5. POST /api/agent/{id}/publish               ← 发布 Agent
```

`test_abort_resume.py` 的 `configure_dataagent()` 和 `test_end_to_end.sh` 的 step 5-6 都在做这件事。

## 两个测试脚本

### test_end_to_end.sh（bash，DeepSeek 直连）

用途：验证 DataAgent 本身能正常工作。

```
MariaDB → Embedding → 后端 → 注册 DeepSeek 模型 → 数据源 → Agent → SSE 查询
```

DataAgent 的 LLM 请求发往 DeepSeek API。

### test_abort_resume.py（Python，Slime + SGLang）

用途：验证 Slime → DataAgent → Slime API → SGLang 全链路 + pause/resume。

```
MariaDB → Embedding → 后端 → 注册 Slime API 为模型 → 数据源 → Agent
  → custom_generate → SSE → DataAgent → Slime API → SGLang
                                                   → pause_generation
                                                   → continue_generation
```

DataAgent 的 LLM 请求发往 Slime API（`localhost:19876`），Slime API 转发 SGLang。

### 关键区别

| | test_end_to_end.sh | test_abort_resume.py |
|------|------|------|
| LLM 后端 | DeepSeek API | Slime API → SGLang |
| 模型注册 | `baseUrl: https://api.deepseek.com` | `baseUrl: http://127.0.0.1:19876` |
| ThreadId 透传 | 不需要 | 需要（`X-DataAgent-Thread-Id` header） |
| pause/resume | 不支持 | 支持（SGLang pause_generation） |
| 启动 MariaDB | ✅ step 1 | ❌ 缺失（需补） |

## test_abort_resume.py 需要的改动

### 1. ensure_dataagent_ready() 补 MariaDB 检查

```python
def ensure_dataagent_ready():
    import subprocess, time

    # MariaDB
    r = subprocess.run(["mysql", "-u", "root", "-e", "SELECT 1 FROM demo_sales.orders LIMIT 1"],
                       capture_output=True)
    if r.returncode != 0:
        print("  Starting MariaDB...")
        subprocess.Popen(["mysqld_safe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        # demo_sales 库和数据的创建逻辑内嵌或引用 demo.sh 的 heredoc
        # （太长不展开，完整 SQL 见 DataAgent/demo.sh）

    # Embedding ...（不变）
    # Backend ...（不变）
```

### 2. _ensure_agent() 删旧建新

删掉幂等分支，改成总是删除重建。避免"Agent 已存在但 datasource 未激活"之类的状态不一致：

```python
def _ensure_agent(api, ds_id):
    # 删除旧 Demo agent
    resp = httpx.get(f"{api}/api/agent/list", timeout=httpx.Timeout(10))
    items = resp.json() if isinstance(resp.json(), list) else resp.json().get("data", [])
    for a in items:
        if a.get("name") == "Demo":
            httpx.delete(f"{api}/api/agent/{a['id']}", timeout=httpx.Timeout(10))

    # 创建新 Agent
    resp = httpx.post(f"{api}/api/agent", json={
        "name": "Demo", "description": "Test", "category": "demo",
    }, timeout=httpx.Timeout(10))
    aid = resp.json()["id"]

    # 关联 → 选表 → 初始化 → 发布
    httpx.post(f"{api}/api/agent/{aid}/datasources/{ds_id}", timeout=httpx.Timeout(10))
    httpx.post(f"{api}/api/agent/{aid}/datasources/tables", json={
        "datasourceId": ds_id, "tables": ["products", "orders"],
    }, timeout=httpx.Timeout(10))
    resp = httpx.post(f"{api}/api/agent/{aid}/datasources/init", timeout=httpx.Timeout(120))
    if not resp.json().get("success"):
        raise RuntimeError(f"Schema init failed: {resp.json()}")
    httpx.post(f"{api}/api/agent/{aid}/publish", timeout=httpx.Timeout(10))

    return aid
```

### 3. configure_dataagent() 返回 agent_id

```python
def configure_dataagent():
    # ... 模型、Embedding、数据源配置不变 ...
    agent_id = _ensure_agent(api, ds_id)
    print(f"  Agent id={agent_id} ready")
    return agent_id   # 不要用 global DATAAGENT_AGENT_ID

# main() 里：
agent_id = configure_dataagent()
os.environ["DATAAGENT_AGENT_ID"] = str(agent_id)
```

## DataAgent 侧改动

为支持 Slime 感知多轮 LLM 调用，DataAgent 做了 ThreadId 透传：

| 文件 | 改动 |
|------|------|
| `util/ThreadContext.java` | 新建，ThreadLocal 持有当前 SSE 请求的 threadId |
| `service/graph/GraphServiceImpl.java` | 入口 `setThreadId()`，出口 `clear()` |
| `service/aimodelconfig/DynamicModelFactory.java` | WebClient/RestClient 加 filter，每次 LLM 请求注入 `X-DataAgent-Thread-Id` header |

Slime 侧使用：

```python
# slime_api.py
thread_id = request.headers.get("X-DataAgent-Thread-Id")
sample = _thread_samples.get(thread_id)
sample.tokens.extend(response_tokens)
```

## 数据流完整示例

以 "各区域销售额排名" 为例：

```
Worker (custom_generate)
  │  thread_id = "slime-0-abc12345"
  │
  ├─ SSE: GET /api/stream/search?agentId=1&query=各区域销售额排名&threadId=slime-0-abc12345
  │
  ▼
DataAgent
  │  ThreadContext.setThreadId("slime-0-abc12345")
  │
  ├─ IntentRecognitionNode
  │    POST /v1/chat/completions   Header: X-DataAgent-Thread-Id: slime-0-abc12345
  │    → Slime API → SGLang → assistant response
  │
  ├─ EvidenceRecallNode
  │    POST /v1/chat/completions   Header: X-DataAgent-Thread-Id: slime-0-abc12345
  │    → Slime API → SGLang → assistant response
  │
  ├─ ... (6 more turns, same threadId)
  │
  ├─ ReportGeneratorNode
  │    POST /v1/chat/completions   Header: X-DataAgent-Thread-Id: slime-0-abc12345
  │    → Slime API → SGLang → assistant response
  │
  └─ SSE: event:complete
       ThreadContext.clear()
```

Slime API 通过 `X-DataAgent-Thread-Id` 把每轮 LLM 调用产生的 token 追加到同一个 `sample.tokens` 中。

## API 格式注意点

DataAgent 不同接口返回格式不统一：

| 接口 | 返回格式 |
|------|---------|
| `POST /api/model-config/add` | `{"success": true, "data": null}` — **data 是 null，没有 id** |
| `GET /api/model-config/list` | `{"success": true, "data": [{id, ...}, ...]}` |
| `POST /api/agent` | `{"id": N, "name": "..."}` — **直接顶层有 id** |
| `GET /api/agent/list` | `[{id, ...}, ...]` — **裸数组** |
| `GET /api/datasource` | `[{id, ...}, ...]` — **裸数组** |
| `POST /api/datasource` | `{"id": N, ...}` — 直接顶层有 id |
| `POST /api/agent/{id}/datasources/init` | `{"success": true/false, ...}` |

处理时注意区分。

## 训练数据 + 奖励机制（核心）

要让 reward 真正上升，奖励必须满足两个条件：
1. **可区分**：不同质量的回答要有不同的 reward（否则无梯度信号）
2. **不可作弊**：不能只靠格式（标题、表格）刷分（否则模型会 reward hacking）

### 设计：基于 ground-truth 的数值准确性奖励

每个 query 都带结构化 label（JSON 字符串），包含从 `demo_sales` 数据库实际计算出的：
- `key_numbers` — 正确报告中必须出现的关键数字
- `key_entities` — 必须提及的实体（区域、商品名等）
- `expected_sql` — 标准 SQL（审计用）
- `summary` — 人类可读的预期答案

奖励函数 `reward_func.score(nodes, label)` 在结构化模式下：

| 维度 | 权重 | 说明 |
|------|------|------|
| 数值准确性 | 0.45 | 报告中是否引用了正确的 ground-truth 数字（主导信号）|
| 实体覆盖 | 0.20 | 是否提及了关键实体 |
| SQL 执行成功 | 0.15 | 至少有一条非空 RESULT_SET |
| 格式 | 0.10 | 标题、表格、长度合理（次要，防退化）|
| 过程 | 0.10 | 多步推理（≥3 个 substantial 节点）|
| 全对加成 | +0.10 | 所有关键数字和实体都命中时额外奖励 |
| 惩罚 | −0.10~0.20 | 平凡输出、SQL 全错+短文本 |

数值匹配容忍度：整数计数近精确匹配（±0.5），浮点数（金额、比率）1% 相对误差
（兼容 `25.28万` ↔ `252811` 的万/亿单位换算）。

### 生成训练数据

```bash
# 默认 offline 模式：从 Database.md 解析 INSERT 数据，无需运行 MariaDB
python examples/dataagent/generate_training_data.py --print-summary

# 连接 MariaDB 重新计算（验证一致性）
python examples/dataagent/generate_training_data.py --source mysql --out queries_labeled.jsonl
```

输出 `queries_labeled.jsonl`，每行：
```json
{"query": "各区域销售额排名",
 "label": "{\"key_numbers\": [\"252811.0\", \"213744.0\", \"167635.0\"], \"key_entities\": [\"华东\", \"华北\", \"华南\"], ...}"}
```

当前内置 20 个 query 模板（区域排名、渠道对比、月度趋势、品类占比、Top-N 商品/客户、
利润率、ROI、库存、供应商评分、退货分析等）。要扩展：在 `generate_training_data.py`
的 `TEMPLATES` 列表中加一个返回 `(query, label_dict)` 的函数即可。

### 验证奖励可区分性

```bash
python examples/dataagent/validate_reward.py -v
```

典型输出（20 个 query 的平均 reward）：

```
Query                                      good  partial  wrong    fmt  noSQL
────────────────────────────────────────────────────────────────────────────
各区域销售额排名                                   0.86     0.33   0.15   0.11   0.00
...
────────────────────────────────────────────────────────────────────────────
AVERAGE                                    0.88     0.30   0.15   0.11   0.00

Sanity checks:
  good > wrong ?    0.88 > 0.15  → YES ✓   ← 主梯度方向
  good > partial ?  0.88 > 0.30  → YES ✓
  partial > wrong ? 0.30 > 0.15  → YES ✓   ← 部分正确有部分分
  good > fmt-only ? 0.88 > 0.11  → YES ✓   ← 格式无法刷分
  wrong ≈ fmt-only ? 0.15 vs 0.11  → YES ✓ ← 错答案 ≈ 没答案
```

五个 check 全过 → reward 信号有梯度、不可作弊 → RL 能让 reward 上升。

### 训练命令

`run_qwen3_14B_fully_async.sh` 已更新为使用 labeled 数据：

```bash
--prompt-data queries_labeled.jsonl
--input-key query
--label-key label          # ← 新增，把 label 字段加载到 sample.label
```

`custom_generate.generate()` 把 `sample.label` 透传给 `reward_func.score()`，
后者解析 JSON 得到 ground-truth，计算数值准确性奖励。

