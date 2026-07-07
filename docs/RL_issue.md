
# DataAgent RL 训练设计文档

## 背景

DataAgent 每次 SSE 请求内部会串行发起 8~12 轮独立的 LLM Chat Completions 调用。Slime 介入后，这些调用不再发往 DeepSeek 而是发往 Slime API，由 RL 模型推理。本文档描述 Slime 侧需要处理的关键问题。

---

## 1. 调用模型

```
Worker → GET /api/stream/search?agentId=X&query=各区域销售额排名
  │
  └── DataAgent 内部串行发起 (每轮 = 一次独立的 HTTP POST)：
        [1] IntentRecognitionNode      POST /v1/chat/completions
        [2] EvidenceRecallNode         POST /v1/chat/completions
        [3] QueryEnhanceNode           POST /v1/chat/completions
        [4] FeasibilityAssessmentNode  POST /v1/chat/completions
        [5] PlannerNode                POST /v1/chat/completions
        [6] SqlGenerateNode            POST /v1/chat/completions
        [7] SemanticConsistencyNode    POST /v1/chat/completions
        [8] ReportGeneratorNode        POST /v1/chat/completions
      (根据 Planner 决策，可能额外插入 SqlExecute 重试 / PythonGenerate / PythonAnalyze)
```

每轮调用特征：
- 独立的 `POST /v1/chat/completions`，每次新 HTTP 连接
- `stream: true`，逐 token 返回
- 无 header 或字段标记"属于同一个 SSE 请求"

---

## 2. 多轮归属感知

### 问题

DataAgent 发给 Slime API 的 8 次 POST 请求没有任何显式关联标识。Slime API 端需要知道哪些请求属于同一个 sample（同一 SSE 请求），才能把它们的 token 序列拼接成一个 RL episode。

### 当前可行方案：时序推断

同一 SSE 请求内的 LLM 调用是严格串行的。Slime API 如果单线程处理请求，连续到达的 post 天然就是同一个 sample。但并发场景下不可靠。

### 推荐方案：注入 threadId header

在 DataAgent 侧改一行代码，`DynamicModelFactory.createChatModel()`：

```java
// 注入 threadId，Slime API 从 header 获取
apiBuilder.defaultHeader("X-DataAgent-Thread-Id", threadContextHolder.get());
```

Slime API 侧：

```python
thread_id = request.headers.get("X-DataAgent-Thread-Id")
# 同一 thread_id → 同一 sample/trajectory
```

不改代码的情况下，可以通过 SSE 请求的到达时间和 Slime API 请求的到达时间做关联（同一 Worker 进程内串行调用），但不够健壮。

---

## 3. 上下文传递

### DataAgent 的上下文机制

**每轮 LLM 的 prompt 不包含前几轮的原始 LLM 输出。** 上下文通过 workflow state 传递：

```
PlannerNode 看到的:
  system: "你是数据分析师，根据表结构制定执行计划..."
  user:   "用户需求：各区域销售额排名
           可用表：orders(region, unit_price, quantity, order_date)"
         ↑ 来自 QueryEnhanceNode 的 canonical_query + SchemaRecallNode 的表信息

SqlGenerateNode 看到的:
  system: "你是 SQL 专家，只输出 SQL..."
  user:   "执行计划第一步：按region分组计算unit_price*quantity总和
           表结构：CREATE TABLE orders (...)"
         ↑ 来自 PlannerNode 的 execution_plan + TableRelationNode 的 schema

ReportGeneratorNode 看到的:
  system: "你是数据分析顾问，生成 Markdown 报告..."
  user:   "用户查询：各区域销售额排名
           SQL 执行结果：华东 123038, 华北 123031, 华南 59360"
         ↑ 来自 SqlExecuteNode 的结果 + 原始 query
```

**对 RL 训练的影响**：
- 每轮 LLM 收到的 input 是**结构化任务描述**，不是冗长的对话历史
- 不需要你做上下文窗口截断、历史压缩等处理
- 每轮的 input 自包含，可独立作为 RL 训练的 state

---

## 4. Reward 分配

### 核心挑战

一次 SSE 请求产生 8 轮 LLM 调用，只有一个最终观察（报告文本），但需要给每轮的每个 token 分配 reward。这是典型的 credit assignment 问题。

### 方案 A：Per-turn reward（推荐）

利用 DataAgent 每个 Node 输出的可评判性，对每轮独立打分：

| Turn | Node | 可观测输出 | Reward 信号 |
|------|------|-----------|-------------|
| 1 | IntentRecognition | `{"classification": "数据分析请求"}` | 分类是否正确 |
| 2 | EvidenceRecall | `{"standalone_query": "..."}` | 查询重写是否保留语义 |
| 3 | QueryEnhance | `{"expanded_queries": [...]}` | 扩展查询是否相关 |
| 4 | FeasibilityAssessment | 可行性评估文本 | 是否正确判断了可行性 |
| 5 | Planner | `{"execution_plan": [...]}` | 步骤选择是否正确（SQL vs Python） |
| 6 | SqlGenerate | `SELECT ...` | **SQL 语法是否正确、执行是否成功、结果行数是否 >0（最强信号）** |
| 7 | SemanticConsistency | 校验结果 | SQL 语义是否正确 |
| 8 | ReportGenerator | Markdown 报告 | 报告是否包含计划要求的所有内容 |

每轮 token 的 reward = 该轮 reward / 该轮 token 数（或最后一个 token 拿全部）。

**优势**：不需要从最终 report 反向传播。SQL 执行失败直接给负 reward，不需要等到报告生成才知道 SQL 错了。

### 方案 B：Episode-level reward

所有 8 轮的 token 共享同一个最终 reward。需要人工或自动对最终报告打分。

**劣势**：SQL 生成错但报告写得好，reward 无法区分——credit assignment 很差。

### 方案 C：混合（最佳）

Per-turn 即时 reward 作为 dense reward 用于在线训练，episode-level reward 作为 sparse reward 用于最终评估和模型选择。

---

## 5. Sample 结构

Slime 拼装一个 episode 的样本结构：

```python
sample = {
    "thread_id": "330ecdef-...",
    "query": "各区域销售额排名",
    "turns": [
        {
            "node": "IntentRecognitionNode",
            "system_prompt": "你是意图识别专家...",
            "user_prompt": "各区域销售额排名",
            "tokens": [...],
            "reward": 1.0  # 分类正确
        },
        {
            "node": "PlannerNode",
            "system_prompt": "你是数据分析师...",
            "user_prompt": "用户需求：...\n表结构：...",
            "tokens": [...],
            "reward": 1.0  # 计划合理
        },
        {
            "node": "SqlGenerateNode",
            "system_prompt": "你是SQL专家...",
            "user_prompt": "按region分组计算销售额...\n表结构：...",
            "tokens": [...],
            "reward": 1.0  # SQL 执行成功
        },
        {
            "node": "ReportGeneratorNode",
            "system_prompt": "你是数据分析顾问...",
            "user_prompt": "SQL结果：...\n请生成报告",
            "tokens": [...],
            "reward": 1.0  # 报告完整
        },
    ],
    "episode_reward": 4.0,  # 可选：端到端评估分
}
```

---

## 6. Summary

| 问题 | 现状 | Slime 侧对策 |
|------|------|-------------|
| 多轮归属感知 | 无显式标识 | 注入 `X-Thread-Id` header（改一行 Java），或靠时序推断 |
| 上下文传递 | 通过 workflow state 传递，不依赖 LLM 对话历史 | 每轮 input 自包含，直接作为 RL state |
| Reward 分配 | — | Per-turn reward：SQL 执行结果、报告完整性等可自动打分 |