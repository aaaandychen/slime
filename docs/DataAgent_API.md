
# DataAgent 接口规格

本文档描述 DataAgent 的三个对外接口：上游 SSE 请求、下游 LLM Chat 请求、下游 Embedding 请求。用于 Slime / RL 训练系统对接。

---

## 1. 上游：Worker → DataAgent（SSE 流式查询）

### 请求

```
GET /api/stream/search?agentId=20&query=各区域销售额排名

Headers:
  Accept: text/event-stream
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `agentId` | 是 | Agent ID |
| `query` | 是 | 自然语言查询（URL 编码） |
| `threadId` | 否 | 多轮对话 ID，不传则自动生成 UUID |

### 响应（SSE 事件流）

普通数据事件：

```
data:{"agentId":"20","threadId":"fa080d16-...","nodeName":"IntentRecognitionNode","textType":"TEXT","text":"正在进行意图识别...\n","error":false,"complete":false}
```

完成事件：

```
event:complete
data:{"agentId":"20","threadId":"fa080d16-...","nodeName":null,"textType":"TEXT","text":null,"error":false,"complete":true}
```

错误事件：

```
event:error
data:{"agentId":"20","threadId":"fa080d16-...","text":"Error in stream processing: ...","error":true,"complete":false}
```

### GraphNodeResponse 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `agentId` | String | Agent ID |
| `threadId` | String | 会话 ID，多轮复用 |
| `nodeName` | String | 当前执行的节点名称 |
| `textType` | String | TEXT / JSON / SQL / PYTHON / PLAN / RESULT_SET / MARK_DOWN / HTML |
| `text` | String | 实际内容（流式分片，需拼接） |
| `error` | boolean | 是否异常 |
| `complete` | boolean | 是否结束 |

### textType 说明

| textType | 含义 | 出现节点 |
|----------|------|----------|
| `TEXT` | 纯文本提示 | 所有节点 |
| `JSON` | 结构化输出（流式拼接后是合法 JSON） | IntentRecognition / EvidenceRecall / QueryEnhance / Planner |
| `SQL` | SQL 语句 | SqlGenerateNode |
| `RESULT_SET` | JSON 格式的查询结果 | SqlExecuteNode |
| `MARK_DOWN` | Markdown 报告 | ReportGeneratorNode |
| `HTML` | HTML 报告 | ReportGeneratorNode |
| `PYTHON` | Python 代码 | PythonGenerateNode |

### Worker 消费伪代码

```python
import requests, json

def call_dataagent(query: str, agent_id: int = 20) -> dict:
    params = {"agentId": agent_id, "query": query}
    nodes, thread_id = {}, None

    with requests.get(
        "http://localhost:8065/api/stream/search",
        params=params,
        headers={"Accept": "text/event-stream"},
        stream=True
    ) as resp:
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            d = json.loads(line[5:])
            if d.get("complete"):
                break
            if d.get("error"):
                raise Exception(d.get("text"))
            thread_id = d["threadId"]
            key = f"{d['nodeName']}|{d['textType']}"
            nodes.setdefault(key, {"node": d["nodeName"], "type": d["textType"], "text": ""})
            nodes[key]["text"] += d.get("text", "")

    return {"threadId": thread_id, "nodes": nodes}
```

---

## 2. 下游：DataAgent → LLM Chat

### 示例：PlannerNode

```json
POST /v1/chat/completions

{
  "model": "deepseek-chat",
  "temperature": 0.0,
  "max_tokens": 8192,
  "stream": true,
  "messages": [
    {
      "role": "system",
      "content": "# ROLE: 专业数据分析师\n\n你是一位资深数据分析师，需要根据用户需求和可用数据库表结构，制定清晰的数据分析执行计划。\n\n## 输出格式\n{\n  \"thought_process\": \"...\",\n  \"execution_plan\": [\n    {\"step\": 1, \"tool_to_use\": \"SQL_GENERATE_NODE\", \"tool_parameters\": {...}}\n  ]\n}\n\n## 可用表结构\n- orders: order_id, region, unit_price, quantity, order_date\n- products: product_id, name, category, price, cost"
    },
    {
      "role": "user",
      "content": "用户需求：查询2026年上半年各区域的销售额排名\n\n请制定执行计划。"
    }
  ]
}
```

### 示例：SqlGenerateNode

```json
POST /v1/chat/completions

{
  "model": "deepseek-chat",
  "temperature": 0.0,
  "max_tokens": 8192,
  "stream": true,
  "messages": [
    {
      "role": "system",
      "content": "# ROLE: SQL生成专家\n\n编写可执行的SQL语句。\n\n## 规则\n1. 只输出SQL，不要任何解释\n2. 使用`table_name`格式引用表名\n3. 日期范围使用 BETWEEN\n\n## 表结构\nCREATE TABLE orders (\n  order_id INT PRIMARY KEY,\n  region VARCHAR(20),\n  unit_price DECIMAL(10,2),\n  quantity INT,\n  order_date DATE\n);"
    },
    {
      "role": "user",
      "content": "查询各区域销售额排名。按region分组，计算unit_price*quantity总和，降序排列。"
    }
  ]
}
```

### 示例：ReportGeneratorNode

```json
POST /v1/chat/completions

{
  "model": "deepseek-chat",
  "temperature": 0.0,
  "max_tokens": 8192,
  "stream": true,
  "messages": [
    {
      "role": "system",
      "content": "# ROLE: 专业数据分析顾问\n\n生成结构化Markdown报告，包含执行摘要、分析背景、分析过程、结果解读和建议。"
    },
    {
      "role": "user",
      "content": "用户查询：各区域销售额排名\n\nSQL执行结果：\n{\"resultSet\":{\"column\":[\"region\",\"total_sales\"],\"data\":[{\"region\":\"华东\",\"total_sales\":\"123038.00\"},{\"region\":\"华北\",\"total_sales\":\"123031.00\"},{\"region\":\"华南\",\"total_sales\":\"59360.00\"}]}}"
    }
  ]
}
```

### LLM 调用特征

| 特征 | 值 |
|------|-----|
| 入向地址 | `baseUrl` + `completionsPath`（生产配置为 `https://api.deepseek.com/v1/chat/completions`，测试配置为 Slime API） |
| stream | `true`（逐 token 返回，SSE 格式的子流） |
| HTTP 连接 | 每次调用独立的 `POST`，无 keep-alive 复用 |
| system prompt | 来自 `resources/prompts/*.txt`，可通过 PromptConfig API 自定义 |
| user prompt | 由上一个 Node 的输出 + 表结构信息拼接 |
| 顺序 | 同一 SSE 请求内的 LLM 调用是严格串行的 |
| 关联标识 | 无。所有 LLM 请求无 `X-Thread-Id` 或类似 header，完全独立 HTTP 调用 |

### 同一 SSE 流内的完整 LLM 调用序列

```
节点                              system prompt 来源

1. IntentRecognitionNode         prompts/intent-recognition.txt
2. EvidenceRecallNode            prompts/evidence-recall.txt
3. QueryEnhanceNode              prompts/query-enhance.txt
4. FeasibilityAssessmentNode     prompts/feasibility-assessment.txt
5. PlannerNode                   prompts/planner.txt
6. SqlGenerateNode               prompts/sql-generator.txt
7. SemanticConsistencyNode       prompts/semantic-consistency.txt
8. ReportGeneratorNode           prompts/report-generator.txt

（如果 Planner 决定了 Python 分析，还会插入：
   PythonGenerateNode            prompts/python-generator.txt
   PythonAnalyzeNode             prompts/python-analyze.txt）
```

---

## 3. 下游：DataAgent → Embedding

### Schema 初始化 / SchemaRecallNode

```json
POST /v1/embeddings

{
  "model": "bge-small-zh-v1.5",
  "input": [
    "orders表：order_id, region, unit_price, quantity, order_date - 订单信息表",
    "products表：product_id, name, category, price, cost, stock - 商品信息表"
  ]
}
```

### 响应

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.023190947, -0.037621278, 0.044130608, ...]
    },
    {
      "object": "embedding",
      "index": 1,
      "embedding": [0.018234561, -0.041234567, 0.039876543, ...]
    }
  ],
  "model": "bge-small-zh-v1.5",
  "usage": {
    "prompt_tokens": 156,
    "total_tokens": 156
  }
}
```

### Embedding 调用特征

| 特征 | 值 |
|------|-----|
| 入向地址 | `baseUrl` + `embeddingsPath`（测试配置为 `http://localhost:8765/v1/embeddings`） |
| stream | 不支持，普通 POST |
| `input` 字段 | 字符串或字符串数组 |
| 维度 | 本地 BGE-small-zh 为 512，OpenAI text-embedding-3-small 为 1536 |
| 调用时机 | Schema 初始化（批量）、SchemaRecallNode（查询时检索） |
| 结果使用 | 存入 SimpleVectorStore，通过余弦相似度做表/字段语义检索 |

---

## 4. 与 Slime 的对接点

| 接口 | Slime 需要做什么 |
|------|-----------------|
| SSE 查询 | Worker 发起 GET，消费 SSE 事件流，感知 `event:complete` → 样本结束 |
| LLM Chat | Slime API 替代 `baseUrl`，接收 Chat Completions 请求，用自己的 RL 模型推理；通过调用顺序（或注入 header）关联到同一 SSE 流 |
| Embedding | **不经过 Slime**。保持独立部署（本地 BGE 或外部服务），Embedding 模型不需要 RL 训练 |