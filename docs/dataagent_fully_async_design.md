# DataAgent × Slime Fully Async RL 训练 — 详细设计文档

## 1. 架构对齐

### 1.1 核心定位

DataAgent 的 StateGraph 整体作为 slime 的 **custom generate function**（即 agent loop 本身），而非 agent loop 调用的一个 tool。

| slime agent loop 概念 | DataAgent 对应 |
|----------------------|---------------|
| `post(url, payload)` — 一次 LLM 生成 | 图节点内的 `LlmService.call()` — 节点的一次推理 |
| `_execute_search()` — tool 执行 | `SqlExecuteNode` / `PythonExecuteNode` / RAG 检索 |
| `sample.tokens` — 完整对话历史 | `OverAllState` + token 序列 — 图执行状态 |
| `append_response_tokens(trainable=False)` — 工具结果 | tool 执行结果写入 state，标记不可训 |
| `sample.status = ABORTED` + `return` — 中断退出 | LlmService 捕获 abort 信号 → 节点返回 → 图状态序列化 |
| requeue → `_prepare_prompt_ids` 恢复 | requeue → 反序列化 `OverAllState` + token 序列 → 图 continue |

### 1.2 DataAgent StateGraph 节点分类

**LLM 节点**（每次调用走 SGLang，可被 `pause_generation` 中断）：

| 节点 | LLM 用途 | 文件 |
|------|---------|------|
| `IntentRecognitionNode` | 意图识别 | `workflow/node/IntentRecognitionNode.java` |
| `EvidenceRecallNode` | RAG 查询生成 | `workflow/node/EvidenceRecallNode.java` |
| `QueryEnhanceNode` | Query 改写 | `workflow/node/QueryEnhanceNode.java` |
| `FeasibilityAssessmentNode` | 可行性评估 | `workflow/node/FeasibilityAssessmentNode.java` |
| `PlannerNode` | 执行计划生成 | `workflow/node/PlannerNode.java` |
| `SqlGenerateNode` | SQL 生成 | `workflow/node/SqlGenerateNode.java` |
| `PythonGenerateNode` | Python 代码生成 | `workflow/node/PythonGenerateNode.java` |
| `PythonAnalyzeNode` | Python 结果分析 | `workflow/node/PythonAnalyzeNode.java` |
| `ReportGeneratorNode` | 报告生成 | `workflow/node/ReportGeneratorNode.java` |

**Tool 节点**（执行环境交互，不可中断）：

| 节点 | Tool 操作 | 文件 |
|------|----------|------|
| `EvidenceRecallNode` | 向量检索 (vector store) | 同上 |
| `SqlExecuteNode` | SQL 执行 (JDBC) | `workflow/node/SqlExecuteNode.java` |
| `PythonExecuteNode` | Python 执行 (Docker/Local) | `workflow/node/PythonExecuteNode.java` |

**Dispatcher 节点**（纯路由逻辑，无 LLM 调用，无副作用）：

| 节点 | 文件 |
|------|------|
| `PlanExecutorNode` | `workflow/node/PlanExecutorNode.java` |
| `SchemaRecallNode` | `workflow/node/SchemaRecallNode.java` |
| `TableRelationNode` | `workflow/node/TableRelationNode.java` |
| `SemanticConsistencyNode` | `workflow/node/SemanticConsistencyNode.java` |
| `HumanFeedbackNode` | `workflow/node/HumanFeedbackNode.java` |

### 1.3 图执行流程

```
START
  → IntentRecognition (LLM)
  → EvidenceRecall (LLM + RAG tool)
  → QueryEnhance (LLM)
  → SchemaRecall (dispatcher)
  → TableRelation (dispatcher)
  → FeasibilityAssessment (LLM)
  → Planner (LLM)
  → PlanExecutor (dispatcher)
       ├→ SqlGenerate (LLM) → SemanticConsistency (dispatcher) → SqlExecute (tool)
       │       ↑                                                    │
       │       └────────── retry ───────────────────────────────────┘
       ├→ PythonGenerate (LLM) → PythonExecute (tool) → PythonAnalyze (LLM)
       │       ↑                                                    │
       │       └────────── retry ───────────────────────────────────┘
       └→ ReportGenerator (LLM)
  → END
```

---

## 2. Partial Rollout 机制回顾

### 2.1 Slime 侧的完整流程

```
训练循环 (train_async.py):
  rollout_data = ray.get(rollout_manager.generate.remote(rollout_id))   // 取已完成的样本
  actor_model.async_train(rollout_id, rollout_data)                     // 训练
  if step % update_weights_interval == 0:
      actor_model.update_weights()   // ① 触发权重同步
```

```
权重同步 (update_weight_from_distributed.py):
  engine.pause_generation()    // ② 中断所有在飞的 SGLang /generate 请求
  engine.flush_cache()         // ③ 清空 KV cache
  _send_weights(...)           // ④ 广播新权重到所有推理引擎
  engine.continue_generation() // ⑤ 恢复推理
```

```
后台 worker (fully_async_rollout.py):
  task = asyncio.create_task(generate_and_rm_group(...))   // 持续生成样本
  callback:
    if ABORTED → data_buffer.add_samples(group)            // ⑥ requeue
    else       → output_queue.put((gid, group))            // ⑦ 进入训练队列
```

```
custom generate function (fully_async_agent_loop.py / dataagent):
  output = await post(url, payload)
  if finish_reason == "abort" → sample.status = ABORTED; return sample   // ⑧ 中断退出
  append_response_tokens(trainable=True/False)                           // ⑨ 累积 token
  execute_tool(...)                                                       // ⑩ 环境交互
```

```
恢复 (requeue 后再次调用 custom generate function):
  _prepare_prompt_ids(sample) → sample.tokens   // ⑪ 恢复全部历史 token
  post(url, {"input_ids": sample.tokens})       // ⑫ SGLang prefix caching 跳过已生成部分
  → 用新权重从断点续写
```

### 2.2 中断发生的精确位置

`pause_generation()` 只中断 SGLang `/generate` 端点正在处理的 HTTP 请求。因此 abort 只可能发生在 **LLM 节点内部的 `LlmService.call()` 调用期间**。Tool 节点（SQL 执行、Python 执行）和 Dispatcher 节点不受影响。

---

## 3. 改动设计

### 3.1 改动总览

| # | 位置 | 类型 | 内容 | 行数 |
|---|------|------|------|------|
| 1 | DataAgent `OverAllState` | 框架修改 | Jackson 序列化支持 | ~50 |
| 2 | DataAgent `LlmService` | 应用修改 | 指向 SGLang + abort 感知 + 内部重试 | ~80 |
| 3 | Slime 新增 custom generate function | 新增文件 | DataAgent agent loop | ~250 |
| 4 | Slime 配置 | 新增文件 | YAML + 启动脚本 | ~30 |

**总计约 400 行，其余零改动。**

### 3.2 改动 1：OverAllState 序列化

**文件**：`spring-ai-alibaba-graph-core` 依赖包

**原因**：abort → requeue 后需要从序列化状态恢复图执行。当前 `OverAllState` 为纯内存对象，不支持持久化。

**方案**：

```java
// OverAllState 增加序列化支持
public class OverAllState {
    // 序列化为 JSON
    public String toJson() {
        return objectMapper.writeValueAsString(this.stateMap);
    }

    // 从 JSON 恢复
    public static OverAllState fromJson(String json, KeyStrategyFactory factory) {
        Map<String, Object> map = objectMapper.readValue(json, ...);
        OverAllState state = new OverAllState(factory);
        state.updateState(map);
        return state;
    }
}
```

**需序列化的值类型**：
- `String`、`Integer`、`Boolean` — 原生支持
- `Plan` / `ExecutionStep` — 已有 Jackson 注解，直接可用
- `SchemaDTO` — 已有 Jackson 注解
- `SqlRetryDto` — 已有 Jackson 注解
- `List<Map<String, Object>>`（SQL 结果集）— Jackson 原生支持
- `HashMap` — Jackson 原生支持

**影响面**：仅 `OverAllState` 类本身，不涉及节点逻辑、dispatcher、服务层。

### 3.3 改动 2：LlmService abort 感知

**文件**：`data-agent-management/src/main/java/.../service/llm/impls/StreamLlmService.java`

**原因**：当前 `Flux<ChatResponse>` 被中断时异常直接穿透节点 → 图崩溃。需要在 LlmService 层内部消化 abort，做透明重试。

**核心逻辑**：

```java
public class StreamLlmService implements LlmService {

    private final AiModelRegistry registry;

    @Override
    public Flux<ChatResponse> call(String system, String user) {
        return Flux.defer(() -> {
            // 尝试调用 LLM
            return registry.getChatClient()
                .prompt().system(system).user(user)
                .stream().chatResponse()
                .onErrorResume(this::isAbortError, e -> {
                    // abort 时：refresh ChatClient（指向更新后权重的 SGLang）
                    // 重新发起完全相同的请求
                    // prefix caching 跳过已生成的 token
                    return registry.refresh().getChatClient()
                        .prompt().system(system).user(user)
                        .stream().chatResponse();
                });
        });
    }

    private boolean isAbortError(Throwable e) {
        // SGLang abort 表现为连接断开或特定错误响应
        // 需要根据 SGLang 的实际行为调整判断逻辑
        return e instanceof IOException
            || (e.getMessage() != null && e.getMessage().contains("abort"));
    }
}
```

**关键点**：
- 图节点完全不感知 abort。对节点来说，`Flux<ChatResponse>` 只是一个"偶尔有短暂延迟"的流。
- 重试不需要改变 prompt。SGLang prefix caching 自动跳过已生成的 token，只生成增量部分。
- 旧权重生成的 token 需要在训练侧通过 `mask_offpolicy` 处理。由于 partial rollout 下 `mask_offpolicy` 整段清零，这部分自然被覆盖。
- `AiModelRegistry.refresh()` 是已有的模型热切换能力（文档记载），只需确保底层 ChatClient 的 HTTP endpoint 指向 slime 管理的 SGLang。

**配置变更**：
```yaml
# application.yml — 训练环境
spring:
  ai:
    openai:
      base-url: ${SGLANG_ROUTER_URL:http://127.0.0.1:30000}
      api-key: unused
```

### 3.4 改动 3：DataAgent Custom Generate Function

**文件**：`slime/examples/dataagent/generate_with_dataagent.py`（新建）

**模式**：与 `examples/fully_async/search_agent/fully_async_agent_loop.py` 同构，区别仅在于：
- Tool 执行换成 DataAgent 的 SQL/Python/RAG
- 图状态管理（推进图节点、记录执行进度）
- Prompt 构建（DataAgent 的分析任务描述）

**核心结构**：

```python
"""
DataAgent multi-turn agent loop for slime fully async training.

Wire via::

    --custom-generate-function-path examples.dataagent.generate_with_dataagent.generate
    --rollout-function-path examples.fully_async.search_agent.fully_async_agent_loop.generate_rollout_fully_async
"""

from __future__ import annotations

import asyncio
import logging
import time
from argparse import Namespace
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

__all__ = ["generate"]

logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

DATAAGENT_CONFIG: dict = {
    "max_turns": 30,         # 最多 30 轮（对应图节点执行步数）
    "data_agent_url": "http://127.0.0.1:8065",
    # DataAgent tool 端点（SQL 执行、Python 执行、RAG 检索）
    "sql_execute_url": "http://127.0.0.1:8065/api/tool/sql-execute",
    "python_execute_url": "http://127.0.0.1:8065/api/tool/python-execute",
    "rag_retrieve_url": "http://127.0.0.1:8065/api/tool/rag-retrieve",
}

# ============================================================
# Graph state (persisted in sample.metadata["__graph_state__"])
# ============================================================

_GRAPH_STATE_KEY = "__graph_state__"

# DataAgent StateGraph 节点的 LLM prompt 模板
# 每个节点有自己特定的 prompt 结构，需要从 DataAgent 的 prompts/ 目录导出

NODE_PROMPTS: dict[str, dict] = {
    "INTENT_RECOGNITION_NODE": {
        "system": "...",  # 从 DataAgent prompts/intent_recognition.txt 导出
        "user_template": "{query}",
    },
    "PLANNER_NODE": {
        "system": "...",  # 从 DataAgent prompts/planner.txt 导出
        "user_template": "...",
    },
    "SQL_GENERATE_NODE": {
        "system": "...",
        "user_template": "...",
    },
    "PYTHON_GENERATE_NODE": {
        "system": "...",
        "user_template": "...",
    },
    "REPORT_GENERATOR_NODE": {
        "system": "...",
        "user_template": "...",
    },
    # ... 其余节点
}

# 各节点的 tool 执行器
NODE_TOOL_EXECUTORS: dict[str, str] = {
    "SQL_EXECUTE_NODE": "sql",
    "PYTHON_EXECUTE_NODE": "python",
    "EVIDENCE_RECALL_NODE": "rag",
}


def _init_graph_state() -> dict:
    return {
        "current_node": "INTENT_RECOGNITION_NODE",
        "overall_state": {},     # 序列化后的 OverAllState
        "plan_steps": [],        # Planner 生成的执行计划步骤
        "current_step": 0,       # 当前计划步骤
        "completed_steps": [],   # 已完成步骤记录
    }


def _get_graph_state(sample: Sample) -> dict:
    state = sample.metadata.get(_GRAPH_STATE_KEY)
    if state is None:
        state = _init_graph_state()
        sample.metadata[_GRAPH_STATE_KEY] = state
    return state


# ============================================================
# Action parsing (模型输出 → 图节点路由)
# ============================================================

def _build_node_prompt(gs: dict, sample: Sample) -> tuple[str, str]:
    """根据当前图节点构建 (system_prompt, user_prompt)。
    
    对于 LLM 节点：返回该节点的 prompt 模板
    对于 Tool 节点：跳过，返回 (None, None)
    对于 Dispatcher 节点：不需要 LLM 调用，直接推进
    """
    node = gs["current_node"]
    prompt_config = NODE_PROMPTS.get(node)

    if prompt_config is None:
        # Tool 节点或 Dispatcher 节点 — 不需要 LLM 调用
        return None, None

    system = prompt_config["system"]
    user = prompt_config["user_template"].format(
        query=sample.prompt,
        state=gs["overall_state"],
        # ... 其他变量填充
    )
    return system, user


def _parse_and_route(gs: dict, output_text: str) -> dict:
    """解析 LLM 输出，决定下一步：继续 LLM / 执行 tool / 结束。
    
    当前 DataAgent 的 planner prompt 输出 JSON 执行计划：
    {"execution_plan": [{"step": 1, "tool_to_use": "SQL_GENERATE_NODE", ...}]}
    
    图节点的输出格式由各节点的 prompt 定义。解析逻辑需要与 DataAgent
    各节点的 prompt 模板保持一致。
    """
    return {
        "next_node": "...",
        "tool_params": {...},
    }


# ============================================================
# Tool execution (调用 DataAgent 服务)
# ============================================================

async def _execute_sql(params: dict, state: GenerateState) -> str:
    """执行 SQL，返回结果文本。"""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.post(
            DATAAGENT_CONFIG["sql_execute_url"],
            json={"sql": params["sql"], "datasource_id": params["datasource_id"]},
        ) as resp:
            if resp.status != 200:
                logger.warning("SQL execute HTTP %d", resp.status)
                return f"SQL execution failed: {await resp.text()}"
            result = await resp.json()
    return _format_sql_result(result)


async def _execute_python(params: dict, state: GenerateState) -> str:
    """执行 Python 代码，返回执行输出。"""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.post(
            DATAAGENT_CONFIG["python_execute_url"],
            json={"code": params["code"], "agent_id": params["agent_id"]},
        ) as resp:
            if resp.status != 200:
                logger.warning("Python execute HTTP %d", resp.status)
                return f"Python execution failed: {await resp.text()}"
            result = await resp.json()
    return _format_python_result(result)


async def _execute_rag(params: dict, state: GenerateState) -> str:
    """执行 RAG 检索，返回检索结果文本。"""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.post(
            DATAAGENT_CONFIG["rag_retrieve_url"],
            json={"query": params["query"], "agent_id": params["agent_id"]},
        ) as resp:
            if resp.status != 200:
                logger.warning("RAG retrieve HTTP %d", resp.status)
                return ""
            result = await resp.json()
    return _format_rag_result(result)


TOOL_EXECUTORS = {
    "sql": _execute_sql,
    "python": _execute_python,
    "rag": _execute_rag,
}


def _format_sql_result(result: dict) -> str:
    """格式化 SQL 结果为 token 序列文本。"""
    if not result.get("rows"):
        return "\n<sql_result>No rows returned.</sql_result>\n"
    rows = result["rows"]
    cols = result.get("columns", [])
    # 格式化为 ES|表格| 形式，便于模型解析
    parts = ["\n<sql_result>"]
    parts.append("| " + " | ".join(cols) + " |")
    parts.append("|" + "|".join(["---" for _ in cols]) + "|")
    for row in rows[:50]:  # 限制行数
        parts.append("| " + " | ".join(str(v) for v in row) + " |")
    if len(rows) > 50:
        parts.append(f"\n... ({len(rows) - 50} more rows)")
    parts.append("</sql_result>\n")
    return "\n".join(parts)


def _format_python_result(result: dict) -> str:
    """格式化 Python 执行结果为文本。"""
    return f"\n<python_output>{result.get('output', '')}</python_output>\n"


def _format_rag_result(result: dict) -> str:
    """格式化 RAG 检索结果为文本。"""
    docs = result.get("documents", [])
    if not docs:
        return ""
    parts = ["\n<rag_results>"]
    for i, doc in enumerate(docs[:10]):
        parts.append(f"Doc {i+1}: {doc.get('content', '')}")
    parts.append("</rag_results>\n")
    return "\n".join(parts)


# ============================================================
# Step function — 执行一个图节点
# ============================================================

async def _execute_node(
    gs: dict, sample: Sample, state: GenerateState,
    url: str, sampling_params: dict, turn_sp: dict,
) -> tuple[bool, str | None]:
    """执行当前图节点，返回 (should_continue, tool_output_text)。

    一个节点可能是：
    - LLM 节点：需要调用 post(url, payload) 获取生成结果
    - Tool 节点：调用 DataAgent 工具服务
    - Dispatcher 节点：纯计算路由，不需要 LLM 或 Tool
    """
    node = gs["current_node"]

    # ---- 检查是否是 Tool 节点 ----
    tool_type = NODE_TOOL_EXECUTORS.get(node)
    if tool_type:
        # 从 gs 中获取上一步 LLM 生成的 tool 参数
        tool_params = gs.get("pending_tool_params", {})
        executor = TOOL_EXECUTORS.get(tool_type)
        if executor:
            tool_output = await executor(tool_params, state)
            return True, tool_output
        return True, None

    # ---- 检查是否是 Dispatcher 节点（纯路由） ----
    if node in ("PLAN_EXECUTOR_NODE", "SCHEMA_RECALL_NODE", "TABLE_RELATION_NODE",
                 "SEMANTIC_CONSISTENCY_NODE", "HUMAN_FEEDBACK_NODE"):
        # Dispatcher 节点不需要 LLM 调用或 tool 调用
        # 直接根据当前状态决定下一个节点
        _route_dispatcher(gs)
        return True, None

    # ---- LLM 节点 ----
    system, user = _build_node_prompt(gs, sample)
    if system is None:
        # 未知节点，跳过
        return False, None

    # 构建 prompt（首次编码，后续复用 sample.tokens）
    if gs["current_node"] == "INTENT_RECOGNITION_NODE" and not sample.tokens:
        prompt_text = f"<|system|>{system}<|user|>{user}"
        prompt_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        sample.tokens = list(prompt_ids)

    payload: dict = {
        "input_ids": sample.tokens,
        "sampling_params": {**turn_sp, "max_new_tokens": sampling_params.get("max_new_tokens", 2048)},
        "return_logprob": True,
    }

    output = await post(url, payload)
    last_meta_info = output.get("meta_info", {})

    # ---- Abort 检查 ----
    if last_meta_info.get("finish_reason", {}).get("type") == "abort":
        sample.status = Sample.Status.ABORTED
        return False, None  # 立即返回，worker callback 会 requeue

    # ---- 累积 token ----
    if "output_token_logprobs" in last_meta_info:
        cur_tokens = [item[1] for item in last_meta_info["output_token_logprobs"]]
        cur_logprobs = [item[0] for item in last_meta_info["output_token_logprobs"]]
    else:
        cur_tokens, cur_logprobs = [], []

    if not cur_tokens:
        return False, None

    cur_text: str = output.get("text", "")
    sample.append_response_tokens(
        args=state.args,            # 注意：generate 函数签名是 async def generate(args, sample, sampling_params, evaluation=False)
        tokens=cur_tokens,          # args 从外部传入
        log_probs=cur_logprobs,
        trainable=True,
        meta_info=last_meta_info,
        text=cur_text,
    )

    # ---- Length limit ----
    if last_meta_info.get("finish_reason", {}).get("type") == "length":
        return False, None

    # ---- 解析输出，路由到下一个节点 ----
    route_result = _parse_and_route(gs, cur_text)
    gs["current_node"] = route_result["next_node"]
    gs["pending_tool_params"] = route_result.get("tool_params", {})

    if route_result["next_node"] == "END":
        return False, None

    return True, None


def _route_dispatcher(gs: dict) -> None:
    """Dispatcher 节点的纯计算路由逻辑。

    映射自 DataAgent 各 Dispatcher 的 apply() 方法（纯 Java 逻辑，
    无 LLM 调用，无副作用）。在 Python 侧重建同样的判断逻辑。
    """
    node = gs["current_node"]

    if node == "PLAN_EXECUTOR_NODE":
        plan_steps = gs.get("plan_steps", [])
        step = gs.get("current_step", 0)
        if step >= len(plan_steps):
            gs["current_node"] = "REPORT_GENERATOR_NODE"
        else:
            step_info = plan_steps[step]
            gs["current_node"] = step_info["tool_to_use"]
            gs["current_step"] = step + 1

    elif node == "SEMANTIC_CONSISTENCY_NODE":
        # 语义校验通过 → SQL 执行；失败 → 重新生成 SQL
        # 简化：先总是通过
        gs["current_node"] = "SQL_EXECUTE_NODE"

    elif node == "SCHEMA_RECALL_NODE":
        gs["current_node"] = "TABLE_RELATION_NODE"

    elif node == "TABLE_RELATION_NODE":
        gs["current_node"] = "FEASIBILITY_ASSESSMENT_NODE"

    # ... 其余 dispatcher 路由


# ============================================================
# Main entry point
# ============================================================

async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict,
    evaluation: bool = False,
) -> Sample:
    """DataAgent multi-turn agent loop。

    由 ``generate_and_rm`` → ``generate_and_rm_group`` 为每个 sample 调用。
    必须遵守 ``--custom-generate-function-path`` 契约。
    """
    gs = _get_graph_state(sample)
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # ---- 构建 stop tokens ----
    _stop_tags = ["</sql>", "</python>", "</answer>", "```"]
    _existing = sampling_params.get("stop") or []
    if isinstance(_existing, str):
        _existing = [_existing]
    turn_sp: dict = {
        **sampling_params,
        "stop": list(dict.fromkeys([*_existing, *_stop_tags])),
    }

    max_turns: int = DATAAGENT_CONFIG.get("max_turns", 30)

    last_meta_info: dict = {}
    non_generation_start: float = time.time()

    # ---- main graph loop ----
    turn = 0
    while turn < max_turns:
        should_continue, tool_output = await _execute_node(
            gs, sample, state, url, sampling_params, turn_sp
        )

        # Abort 传播
        if sample.status == Sample.Status.ABORTED:
            return sample

        if not should_continue:
            break

        # ---- Tool 结果写入 ----
        if tool_output:
            tool_ids = state.tokenizer(tool_output, add_special_tokens=False)["input_ids"]
            sample.append_response_tokens(
                args=args,
                tokens=tool_ids,
                trainable=False,
                text=tool_output,
            )

        turn += 1

    # ---- finalize ----
    sample.non_generation_time += time.time() - non_generation_start

    if sample.status != Sample.Status.ABORTED:
        if gs["current_node"] == "END" or turn >= max_turns:
            sample.status = Sample.Status.COMPLETED
        else:
            sample.status = Sample.Status.TRUNCATED

    return sample
```

### 3.5 改动 4：配置与启动

**新建文件**：`slime/examples/dataagent/run_dataagent_fully_async.sh`

```bash
#!/bin/bash
# DataAgent fully async RL training

# 启动 DataAgent 服务（独立进程，非训练 GPU）
# java -jar data-agent-management.jar --spring.profiles.active=h2 &

python train_async.py \
    --rollout-function-path examples.dataagent.generate_with_dataagent.generate \
    --custom-generate-function-path examples.dataagent.generate_with_dataagent.generate \
    --partial-rollout \
    --mask_offpolicy_in_partial_rollout \
    --staleness_threshold 0.5 \
    --sglang_server_concurrency 128 \
    --rollout_batch_size 32 \
    --hf_checkpoint /path/to/model \
    --rollout_max_response_len 8192 \
    # ... 其余标准参数
```

---

## 4. Abort 恢复的正确性分析

### 4.1 中断场景全覆盖

```
场景 A：LLM 节点内部的 post() 执行中被 abort
  → finish_reason == "abort"
  → sample.status = ABORTED
  → return sample（当前 turn token 未 append，全部丢弃）
  → worker callback → requeue → _prepare_prompt_ids 返回完整 sample.tokens
  → 重新进入 generate() → prefix caching 跳过已生成 → 新权重续写
  ✅ 正确

场景 B：Tool 节点执行期间触发权重同步
  → pause_generation() 调用，但没有在飞的 SGLang 请求
  → Tool 继续执行，不受影响
  → Tool 完成后 → 下一个 _execute_node 进入 LLM 节点
  → 此时权重已更新，post() 直接走新权重
  ✅ 正确（无额外延迟，无需 requeue）

场景 C：Dispatcher 节点（纯计算）期间触发权重同步
  → 同场景 B：没有在飞的 SGLang 请求
  → Dispatcher 瞬间完成 → 下一个 LLM 节点自然走新权重
  ✅ 正确

场景 D：Tool 执行 + LLM 生成交替（流式 tool call 提前 dispatch）
  → LLM 流式生成中检测到 tool call → dispatch tool（asyncio task）
  → LLM 继续流式生成 → abort → post() 返回 abort
  → 此时 tool 异步任务可能还在执行
  → 必须 await tool_tasks → 写入 tool token → 再标记 ABORTED
  → requeue 后 sample.tokens 完整（含已完成的 tool 结果）
  ✅ 正确
```

### 4.2 Tool Token 保护

```
sample.tokens:
  [prompt] [turn1_model (trainable=1)] [turn1_tool (trainable=0)]
           [turn2_model (trainable=1)] [turn2_tool (trainable=0)]
           [turn3_partial (trainable=1, 被 abort 丢弃)]
                        ↑
              requeue 后：mask_offpolicy 将 turn1_model + turn2_model 全部 mask
              turn1_tool + turn2_tool 的 loss_mask=0 保持不变
              turn3 用新权重重新生成，trainable=1
```

### 4.3 Prefix Caching 正确性

requeue 后：`post(url, {"input_ids": sample.tokens})`

- SGLang 对所有前缀 token（包括旧权重生成的 model token 和 tool token）用**新权重**重新做 prefill
- KV cache 全部以新权重计算
- 模型看到的上下文语义与"新权重生成所有 token"一致
- 权重同步时 `flush_cache()` 已清空旧 KV cache，不存在混用

---

## 5. 不需要改动的部分

### 5.1 DataAgent 侧

- **图拓扑**：节点注册、边定义、dispatcher 路由逻辑 — 不变
- **Tool 执行器**：`SqlExecuteNode` 的 JDBC 调用、`PythonExecuteNode` 的 Docker 执行、`EvidenceRecallNode` 的向量检索 — 不变
- **Prompt 模板**：`resources/prompts/*.txt` — 内容不变，仅需被 Python 侧引用
- **Entity / DTO / Mapper**：全部不变
- **Controller**：不变（DataAgent 作为独立 HTTP 服务运行）
- **多轮上下文管理**：`MultiTurnContextManager` — 训练模式下不使用，由 slime 的 `sample.tokens` 管理对话历史

### 5.2 Slime 侧

- `fully_async_rollout.py`：AsyncRolloutWorker — 不变
- `sglang_rollout.py`：generate_and_rm / generate_and_rm_group / abort — 不变
- `update_weight_from_distributed.py`：pause/flush/broadcast/continue — 不变
- `train_async.py`：训练主循环 — 不变
- `Sample` 类型：`append_response_tokens` / `loss_mask` / `tokens` — 不变
- `mask_offpolicy` / `staleness_samples` — 不变

---

## 6. 风险点与缓解

| 风险 | 等级 | 缓解 |
|------|------|------|
| `OverAllState` 序列化在框架层不可行 | 中 | 退化方案：不序列化图状态，改由 Python 侧重建。因为图的路由逻辑是确定的，可以通过 from-scratch replay 恢复。代价是额外推理开销。 |
| SGLang abort 的错误签名不明确 | 低 | 需要在实际环境中抓取 abort 时的异常类型。当前预留 `IOException` + `"abort"` 关键词两个匹配条件，后续可扩展。 |
| Tool 执行超时（SQL 跑 5 分钟） | 中 | 给 `_execute_sql` / `_execute_python` 加 `asyncio.wait_for(timeout=N)`。超时后 tool 结果写入失败信息，不做 rollback。 |
| DataAgent Prompt 同步维护 | 低 | DataAgent prompt 模板变更时需同步更新 Python 侧的 `NODE_PROMPTS`。建议从 DataAgent 的 `resources/prompts/` 自动导出到 Python dict 文件。 |
| Prefix caching 命中率 | 低 | requeue 后 prompt 完全不变，SGLang 的 radix tree-based prefix caching 应 100% 命中。实际受 `max_total_tokens` 限制，超长轨迹可能部分未命中。

---

## 7. 与 verl 的对比

| | verl | slime（本方案） |
|---|------|----------------|
| Agent loop 位置 | `verl/experimental/agent_loop/` 框架内部 | 自定义 `generate()` 函数，通过 `--custom-generate-function-path` 接入 |
| LLM 调用方式 | `AsyncLLMServerManager.generate()` (vLLM) | `post(url, payload)` (SGLang HTTP) |
| State 管理 | `ToolAgentLoop.AgentData` | Python dict 存入 `sample.metadata["__graph_state__"]` |
| Partial rollout 恢复 | `FullyAsyncLLMServerManager` 的 `while True` 分段恢复 | slime 的 requeue + `_prepare_prompt_ids` |
| 改动量 | ~660 行（含框架改动） | ~400 行（框架改动 ~50 行） |
